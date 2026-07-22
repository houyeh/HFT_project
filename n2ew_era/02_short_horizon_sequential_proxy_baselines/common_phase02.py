from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE01_ROOT = PROJECT_ROOT / "n2ew_era" / "01_full_l2_action_universe_builder"
PHASE01_DATA = PHASE01_ROOT / "data"
PHASE01_RESULTS = PHASE01_ROOT / "results"

ROOT = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

HORIZONS = [1, 5, 10, 20, 50, 100]
FEATURE_COLUMNS = [
    "spread_ticks",
    "queue_imbalance",
    "deep_imbalance",
    "top_depth",
    "volatility_past_50",
]
IDENTITY_COLUMNS = ["split", "session_id", "symbol", "event_index", "timestamp"]
ACTION_SLOT_BID = 0
ACTION_SLOT_ASK = 1
ACTION_SLOT_NO_QUOTE = 2

VAL_SELECTION_SPLIT = "val"
FROZEN_EVAL_SPLITS = ["test", "new4"]
RANDOM_SEED = 20260528


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def split_dataset_dir(split: str) -> Path:
    return PHASE01_DATA / f"full_l2_action_universe_{split}.parquet"


def available_splits() -> list[str]:
    counts = pd.read_csv(PHASE01_RESULTS / "action_universe_by_split.csv")
    preferred = ["train", "val", "test", "new4"]
    present = counts["split"].astype(str).tolist()
    return [s for s in preferred if s in present] + [s for s in present if s not in preferred]


def split_counts() -> pd.DataFrame:
    return pd.read_csv(PHASE01_RESULTS / "action_universe_by_split.csv")


def part_paths(split: str) -> list[Path]:
    path = split_dataset_dir(split)
    return sorted(path.glob("*.parquet"), key=lambda p: p.name)


def event_frame_from_part(path: Path, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    pnl_cols = [f"pnl_H{h}" for h in horizons]
    cols = [*IDENTITY_COLUMNS, "action_slot", *FEATURE_COLUMNS, *pnl_cols]
    raw = pd.read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int) == ACTION_SLOT_BID].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int) == ACTION_SLOT_ASK].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int) == ACTION_SLOT_NO_QUOTE].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet count mismatch in {path}")
    if not bid["event_index"].reset_index(drop=True).equals(ask["event_index"].reset_index(drop=True)):
        raise RuntimeError(f"Bid/ask event alignment mismatch in {path}")

    out = bid[[*IDENTITY_COLUMNS, *FEATURE_COLUMNS]].copy()
    for h in horizons:
        out[f"bid_pnl_H{h}"] = pd.to_numeric(bid[f"pnl_H{h}"], errors="coerce").to_numpy(dtype=float)
        out[f"ask_pnl_H{h}"] = pd.to_numeric(ask[f"pnl_H{h}"], errors="coerce").to_numpy(dtype=float)
    return out


def iter_event_frames(split: str, horizons: Iterable[int] = HORIZONS) -> Iterable[pd.DataFrame]:
    for path in part_paths(split):
        yield event_frame_from_part(path, horizons=horizons)


def val_feature_quantiles() -> dict[str, dict[str, float]]:
    cols = ["action_slot", *FEATURE_COLUMNS]
    frames: list[pd.DataFrame] = []
    for path in part_paths(VAL_SELECTION_SPLIT):
        raw = pd.read_parquet(path, columns=cols)
        frames.append(raw[raw["action_slot"].astype(int) == ACTION_SLOT_BID][FEATURE_COLUMNS])
    val = pd.concat(frames, ignore_index=True)
    out: dict[str, dict[str, float]] = {}
    for feature in ["top_depth", "volatility_past_50"]:
        s = pd.to_numeric(val[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        q = s.quantile([0.25, 0.50, 0.75])
        out[feature] = {"q25": float(q.loc[0.25]), "q50": float(q.loc[0.50]), "q75": float(q.loc[0.75])}
    return out


def policy_catalog() -> list[dict[str, Any]]:
    q = val_feature_quantiles()
    policies: list[dict[str, Any]] = [
        {"policy_id": "no_quote", "family": "control", "policy_type": "no_quote", "complexity_rank": 0},
        {"policy_id": "bid_all", "family": "control", "policy_type": "bid_all", "complexity_rank": 1},
        {"policy_id": "ask_all", "family": "control", "policy_type": "ask_all", "complexity_rank": 1},
        {
            "policy_id": "old_bid_only_spread_ge_1",
            "family": "control",
            "policy_type": "old_bid_only_spread_ge_1",
            "min_spread_ticks": 1.0,
            "complexity_rank": 1,
        },
    ]
    signed_thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5]
    abs_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for feature, prefix in [("queue_imbalance", "queue_imbalance"), ("deep_imbalance", "deep_imbalance")]:
        for t in signed_thresholds:
            policies.append(
                {
                    "policy_id": f"bid_if_{prefix}_gt_{t:g}".replace("-", "neg"),
                    "family": prefix,
                    "policy_type": "bid_if_feature_gt",
                    "feature": feature,
                    "threshold": float(t),
                    "threshold_source": "fixed_grid",
                    "complexity_rank": 2,
                }
            )
        for t in abs_thresholds:
            policies.append(
                {
                    "policy_id": f"side_by_{prefix}_abs_{t:g}".replace(".", "p"),
                    "family": prefix,
                    "policy_type": "side_by_feature_abs",
                    "feature": feature,
                    "threshold": float(t),
                    "threshold_source": "fixed_grid",
                    "complexity_rank": 3,
                }
            )
    for q_name, threshold in q["top_depth"].items():
        policies.append(
            {
                "policy_id": f"bid_if_top_depth_ge_{q_name}",
                "family": "depth_liquidity",
                "policy_type": "bid_if_feature_ge",
                "feature": "top_depth",
                "threshold": float(threshold),
                "quantile": q_name,
                "threshold_source": "validation_quantile",
                "complexity_rank": 2,
            }
        )
    for q_name, threshold in q["volatility_past_50"].items():
        policies.append(
            {
                "policy_id": f"bid_only_low_volatility_{q_name}",
                "family": "volatility",
                "policy_type": "bid_if_feature_le",
                "feature": "volatility_past_50",
                "threshold": float(threshold),
                "quantile": q_name,
                "threshold_source": "validation_quantile",
                "complexity_rank": 2,
            }
        )
    return policies


def policy_decision(policy: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    decision = np.zeros(n, dtype=np.int8)
    policy_type = policy["policy_type"]
    if policy_type == "no_quote":
        return decision
    if policy_type == "bid_all":
        decision[:] = ACTION_SLOT_BID + 1
        return decision
    if policy_type == "ask_all":
        decision[:] = ACTION_SLOT_ASK + 1
        return decision
    if policy_type == "old_bid_only_spread_ge_1":
        spread = pd.to_numeric(frame["spread_ticks"], errors="coerce").to_numpy(dtype=float)
        decision[np.isfinite(spread) & (spread >= float(policy.get("min_spread_ticks", 1.0)))] = ACTION_SLOT_BID + 1
        return decision

    feature = str(policy["feature"])
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
    threshold = float(policy["threshold"])
    finite = np.isfinite(values)
    if policy_type == "bid_if_feature_gt":
        decision[finite & (values > threshold)] = ACTION_SLOT_BID + 1
    elif policy_type == "bid_if_feature_ge":
        decision[finite & (values >= threshold)] = ACTION_SLOT_BID + 1
    elif policy_type == "bid_if_feature_le":
        decision[finite & (values <= threshold)] = ACTION_SLOT_BID + 1
    elif policy_type == "side_by_feature_abs":
        decision[finite & (values > threshold)] = ACTION_SLOT_BID + 1
        decision[finite & (values < -threshold)] = ACTION_SLOT_ASK + 1
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")
    return decision


@dataclass
class MetricState:
    state_count: int = 0
    selected_action_count: int = 0
    bid_count: int = 0
    ask_count: int = 0
    pnl_count: int = 0
    missing_label_selected_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    cum_pnl: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0


def update_metric_state(state: MetricState, decision: np.ndarray, frame: pd.DataFrame, horizon: int) -> None:
    bid_mask = decision == (ACTION_SLOT_BID + 1)
    ask_mask = decision == (ACTION_SLOT_ASK + 1)
    selected = bid_mask | ask_mask
    bid_pnl = frame[f"bid_pnl_H{horizon}"].to_numpy(dtype=float)
    ask_pnl = frame[f"ask_pnl_H{horizon}"].to_numpy(dtype=float)
    selected_pnl = np.zeros(len(frame), dtype=float)
    selected_values = np.empty(len(frame), dtype=float)
    selected_values[:] = np.nan
    selected_values[bid_mask] = bid_pnl[bid_mask]
    selected_values[ask_mask] = ask_pnl[ask_mask]
    finite_selected = selected & np.isfinite(selected_values)
    selected_pnl[finite_selected] = selected_values[finite_selected]

    state.state_count += int(len(frame))
    state.selected_action_count += int(selected.sum())
    state.bid_count += int(bid_mask.sum())
    state.ask_count += int(ask_mask.sum())
    state.pnl_count += int(finite_selected.sum())
    state.missing_label_selected_count += int((selected & ~np.isfinite(selected_values)).sum())
    state.win_count += int((selected_values[finite_selected] > 0.0).sum())
    state.loss_count += int((selected_values[finite_selected] < 0.0).sum())
    part_sum = float(selected_pnl.sum())
    state.total_pnl += part_sum

    if len(selected_pnl):
        equity = state.cum_pnl + np.cumsum(selected_pnl)
        running_peak = np.maximum.accumulate(np.maximum(equity, state.peak))
        dd = running_peak - equity
        if len(dd):
            state.max_drawdown = max(state.max_drawdown, float(dd.max()))
            state.peak = max(state.peak, float(running_peak[-1]))
            state.cum_pnl = float(equity[-1])


def finalize_metric(policy: dict[str, Any], split: str, horizon: int, state: MetricState) -> dict[str, Any]:
    pnl_count = max(state.pnl_count, 1)
    selected_count = max(state.selected_action_count, 1)
    return {
        "split": split,
        "horizon": int(horizon),
        "policy_id": policy["policy_id"],
        "family": policy.get("family", ""),
        "policy_type": policy.get("policy_type", ""),
        "feature": policy.get("feature", ""),
        "threshold": policy.get("threshold", np.nan),
        "threshold_source": policy.get("threshold_source", "fixed"),
        "quantile": policy.get("quantile", ""),
        "complexity_rank": int(policy.get("complexity_rank", 99)),
        "state_count": int(state.state_count),
        "selected_action_count": int(state.selected_action_count),
        "selection_rate": float(state.selected_action_count / state.state_count) if state.state_count else 0.0,
        "bid_count": int(state.bid_count),
        "ask_count": int(state.ask_count),
        "pnl_count": int(state.pnl_count),
        "missing_label_selected_count": int(state.missing_label_selected_count),
        "total_pnl": float(state.total_pnl),
        "avg_pnl_per_selected_action": float(state.total_pnl / selected_count) if state.selected_action_count else 0.0,
        "avg_pnl_per_evaluable_action": float(state.total_pnl / pnl_count) if state.pnl_count else 0.0,
        "win_rate": float(state.win_count / pnl_count) if state.pnl_count else 0.0,
        "loss_rate": float(state.loss_count / pnl_count) if state.pnl_count else 0.0,
        "max_drawdown": float(state.max_drawdown),
        "drawdown_to_pnl_ratio": float(state.max_drawdown / state.total_pnl) if state.total_pnl > 0 else np.nan,
    }


def evaluate_grid(policies: list[dict[str, Any]], splits: list[str], horizons: list[int] = HORIZONS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        states = {
            (policy["policy_id"], h): MetricState()
            for policy in policies
            for h in horizons
        }
        policy_by_id = {p["policy_id"]: p for p in policies}
        for frame in iter_event_frames(split, horizons=horizons):
            decisions = {policy["policy_id"]: policy_decision(policy, frame) for policy in policies}
            for policy_id, decision in decisions.items():
                for h in horizons:
                    update_metric_state(states[(policy_id, h)], decision, frame, h)
        for policy_id, policy in policy_by_id.items():
            for h in horizons:
                rows.append(finalize_metric(policy, split, h, states[(policy_id, h)]))
        print(f"evaluated split={split}: policies={len(policies)} horizons={len(horizons)}")
    return pd.DataFrame(rows)


def selected_policy_entries() -> list[dict[str, Any]]:
    payload = read_json(RESULTS_DIR / "selected_phase02_policy.json")
    return [payload["main_policy"], *payload.get("backup_policies", [])]


def chosen_pnl_array(policy: dict[str, Any], frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    decision = policy_decision(policy, frame)
    bid_mask = decision == (ACTION_SLOT_BID + 1)
    ask_mask = decision == (ACTION_SLOT_ASK + 1)
    bid_pnl = frame[f"bid_pnl_H{horizon}"].to_numpy(dtype=float)
    ask_pnl = frame[f"ask_pnl_H{horizon}"].to_numpy(dtype=float)
    pnl = np.zeros(len(frame), dtype=float)
    selected_values = np.empty(len(frame), dtype=float)
    selected_values[:] = np.nan
    selected_values[bid_mask] = bid_pnl[bid_mask]
    selected_values[ask_mask] = ask_pnl[ask_mask]
    finite = np.isfinite(selected_values)
    pnl[finite] = selected_values[finite]
    return pnl, decision


def session_pnls(policy: dict[str, Any], split: str, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in iter_event_frames(split, horizons=[horizon]):
        pnl, decision = chosen_pnl_array(policy, frame, horizon)
        session_id = str(frame["session_id"].iloc[0]) if len(frame) else ""
        rows.append(
            {
                "split": split,
                "session_id": session_id,
                "policy_id": policy["policy_id"],
                "horizon": int(horizon),
                "state_count": int(len(frame)),
                "selected_action_count": int((decision != 0).sum()),
                "bid_count": int((decision == 1).sum()),
                "ask_count": int((decision == 2).sum()),
                "session_pnl": float(pnl.sum()),
                "win_rate": float((pnl[decision != 0] > 0.0).mean()) if (decision != 0).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def equity_curve(policy: dict[str, Any], split: str, horizon: int, max_points: int = 200_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    offset = 0
    for frame in iter_event_frames(split, horizons=[horizon]):
        pnl, decision = chosen_pnl_array(policy, frame, horizon)
        eq = np.cumsum(pnl) + offset
        offset = float(eq[-1]) if len(eq) else offset
        local = pd.DataFrame(
            {
                "split": split,
                "session_id": frame["session_id"].astype(str).to_numpy(),
                "event_index": frame["event_index"].to_numpy(dtype=np.int64),
                "selected": decision != 0,
                "pnl": pnl,
                "equity": eq,
            }
        )
        chunks.append(local)
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(out) > max_points:
        idx = np.linspace(0, len(out) - 1, max_points).astype(int)
        out = out.iloc[idx].reset_index(drop=True)
    if not out.empty:
        out["drawdown"] = out["equity"].cummax() - out["equity"]
    return out
