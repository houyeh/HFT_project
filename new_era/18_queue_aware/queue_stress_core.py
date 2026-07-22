from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
if str(V16_DIR) not in sys.path:
    sys.path.insert(0, str(V16_DIR))

from round2_audit_utils import load_test_frame


ROOT = Path(__file__).resolve().parent
COMMON_015_DIR = ROOT / "015_common_audit"
POINT_VALUE = 200.0
MAKER_FEE = 15.0
MAX_INV = 3
MASK_SEED = 20260525
RANDOM_V3_REFERENCE_TRIALS = 16
_RANDOM_V3_RAW_BASE_CACHE: dict[tuple[int, int], float] = {}


@dataclass(frozen=True)
class StressConfig:
    neutral_fill_haircut: float = 1.0
    good_fill_haircut: float = 1.0
    toxic_multiplier: float = 1.0
    cancel_delay_ticks: int = 0


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_016 = load_test_frame()
    path_015 = COMMON_015_DIR / "015_common_predictions.parquet"
    if not path_015.exists():
        raise FileNotFoundError(f"Missing {path_015}. Run new_era/15_models_playaround/10_audit_to_016_standard.py first.")
    frame_015 = pd.read_parquet(path_015)
    return frame_016, frame_015


def _mask_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def build_action_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    action_id = 0
    for session_id, group in frame.groupby("session_id", sort=True):
        g = group.sort_values("local_row_id")
        idx = g.index.to_numpy()
        bid = g["bid_px_1"].to_numpy(dtype=float)
        ask = g["ask_px_1"].to_numpy(dtype=float)
        fut = g["future_mid_return_H_ticks"].to_numpy(dtype=float)
        for pos in range(max(len(g) - 1, 0)):
            if ask[pos + 1] <= bid[pos]:
                rows.append(
                    {
                        "action_id": action_id,
                        "row_index": int(idx[pos]),
                        "session_id": str(session_id),
                        "side": "bid",
                        "price": float(bid[pos]),
                        "future_mid_return_H_ticks": float(fut[pos]),
                    }
                )
                action_id += 1
            if bid[pos + 1] >= ask[pos]:
                rows.append(
                    {
                        "action_id": action_id,
                        "row_index": int(idx[pos]),
                        "session_id": str(session_id),
                        "side": "ask",
                        "price": float(ask[pos]),
                        "future_mid_return_H_ticks": float(fut[pos]),
                    }
                )
                action_id += 1
    return pd.DataFrame(rows)


def ensure_random_masks(frame: pd.DataFrame, out_dir: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = build_action_table(frame)
    rng = np.random.default_rng(MASK_SEED)
    neutral_u = rng.random(len(actions), dtype=np.float64)
    good_u = rng.random(len(actions), dtype=np.float64)
    np.savez_compressed(out_dir / "random_masks.npz", action_id=actions["action_id"].to_numpy(), neutral_u=neutral_u, good_u=good_u)
    manifest = {
        "mask_seed": MASK_SEED,
        "action_count": int(len(actions)),
        "neutral_u_sha256": _mask_hash(neutral_u),
        "good_u_sha256": _mask_hash(good_u),
        "random_v3_reference_trials": RANDOM_V3_REFERENCE_TRIALS,
        "random_v3_reference_seed": MASK_SEED + 33,
        "fixed_mask_policy": "The same neutral_u/good_u arrays and deterministic random_v3 reference masks are used for every stress setting.",
    }
    (out_dir / "random_mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    actions["neutral_u"] = neutral_u
    actions["good_u"] = good_u
    return actions, manifest


def _effective_cancel(mask: np.ndarray, delay: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    delay = int(delay)
    if delay <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    out[delay:] = mask[:-delay]
    return out


def simulate_policy(frame: pd.DataFrame, pull_bid: np.ndarray, pull_ask: np.ndarray, actions: pd.DataFrame, stress: StressConfig) -> dict[str, Any]:
    pull_bid_eff = _effective_cancel(pull_bid, stress.cancel_delay_ticks)
    pull_ask_eff = _effective_cancel(pull_ask, stress.cancel_delay_ticks)
    action_lookup = actions.set_index("action_id")
    action_by_row_side = {(int(r.row_index), str(r.side)): int(r.action_id) for r in actions.itertuples()}

    total_trades = 0
    total_pnl = 0.0
    max_dd = 0.0
    session_rows = []
    for session_id, group in frame.groupby("session_id", sort=True):
        g = group.sort_values("local_row_id")
        idx = g.index.to_numpy()
        bid = g["bid_px_1"].to_numpy(dtype=float)
        ask = g["ask_px_1"].to_numpy(dtype=float)
        mid = g["mid_price"].to_numpy(dtype=float)
        fut = g["future_mid_return_H_ticks"].to_numpy(dtype=float)

        cash = 0.0
        fees = 0.0
        inventory = 0
        trades = 0
        equity_curve: list[float] = []
        for j in range(max(len(g) - 1, 0)):
            row_idx = int(idx[j])
            if not pull_bid_eff[row_idx] and inventory < MAX_INV and ask[j + 1] <= bid[j]:
                action_id = action_by_row_side.get((row_idx, "bid"))
                if action_id is not None and _fill_kept(action_lookup.loc[action_id], "bid", stress):
                    cash -= bid[j] * POINT_VALUE
                    penalty = _toxic_penalty_ticks("bid", float(fut[j]), stress.toxic_multiplier)
                    cash -= penalty * POINT_VALUE
                    fees += MAKER_FEE
                    inventory += 1
                    trades += 1
            if not pull_ask_eff[row_idx] and inventory > -MAX_INV and bid[j + 1] >= ask[j]:
                action_id = action_by_row_side.get((row_idx, "ask"))
                if action_id is not None and _fill_kept(action_lookup.loc[action_id], "ask", stress):
                    cash += ask[j] * POINT_VALUE
                    penalty = _toxic_penalty_ticks("ask", float(fut[j]), stress.toxic_multiplier)
                    cash -= penalty * POINT_VALUE
                    fees += MAKER_FEE
                    inventory -= 1
                    trades += 1
            equity_curve.append(cash - fees + inventory * mid[j] * POINT_VALUE)

        final_equity = cash - fees + inventory * mid[-1] * POINT_VALUE if len(g) else 0.0
        if equity_curve:
            equity_curve[-1] = final_equity
        dd = max_drawdown(np.asarray(equity_curve, dtype=float))
        total_trades += trades
        total_pnl += final_equity
        max_dd = max(max_dd, dd)
        session_rows.append({"session_id": str(session_id), "net_pnl": float(final_equity), "trade_count": int(trades), "max_drawdown": dd})

    return {
        "net_pnl": float(total_pnl),
        "trade_count": int(total_trades),
        "max_drawdown": float(max_dd),
        "session_rows": session_rows,
    }


def _fill_kept(action: pd.Series, side: str, stress: StressConfig) -> bool:
    if float(action["neutral_u"]) > float(stress.neutral_fill_haircut):
        return False
    fut = float(action["future_mid_return_H_ticks"])
    favorable = (side == "bid" and fut > 0) or (side == "ask" and fut < 0)
    if favorable and float(action["good_u"]) > float(stress.good_fill_haircut):
        return False
    return True


def _toxic_penalty_ticks(side: str, future_mid_return_ticks: float, toxic_multiplier: float) -> float:
    extra_mult = max(float(toxic_multiplier) - 1.0, 0.0)
    if extra_mult <= 0:
        return 0.0
    if side == "bid":
        adverse = max(0.0, -float(future_mid_return_ticks))
    else:
        adverse = max(0.0, float(future_mid_return_ticks))
    return float(extra_mult * adverse)


def policy_masks(frame_016: pd.DataFrame, frame_015: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    return {
        "no_cancel": (frame_016, np.zeros(len(frame_016), dtype=bool), np.zeros(len(frame_016), dtype=bool)),
        "016_v2_binary": (
            frame_016,
            frame_016["pull_bid"].to_numpy(dtype=bool),
            frame_016["pull_ask"].to_numpy(dtype=bool),
        ),
        "015_common": (
            frame_015,
            frame_015["pull_bid_015_common"].to_numpy(dtype=bool),
            frame_015["pull_ask_015_common"].to_numpy(dtype=bool),
        ),
    }


def random_v3_reference_pnl(frame: pd.DataFrame, actions: pd.DataFrame, stress: StressConfig) -> float:
    strict = pd.read_csv(V16_DIR / "results" / "strict_random_baseline_distribution.csv")
    base_exact = float(strict[strict["method"] == "random_v3"].iloc[0]["mean"])
    if stress == StressConfig():
        return base_exact
    base_stress = StressConfig()
    cache_key = (id(frame), id(actions))
    if cache_key not in _RANDOM_V3_RAW_BASE_CACHE:
        _RANDOM_V3_RAW_BASE_CACHE[cache_key] = _random_v3_raw_mean(frame, actions, base_stress)
    raw_base = _RANDOM_V3_RAW_BASE_CACHE[cache_key]
    raw_stress = _random_v3_raw_mean(frame, actions, stress)
    return float(raw_stress + (base_exact - raw_base))


def _random_v3_raw_mean(frame: pd.DataFrame, actions: pd.DataFrame, stress: StressConfig) -> float:
    rng = np.random.default_rng(MASK_SEED + 33)
    pnls = []
    grouped = frame.groupby(["session_id", "spread_bucket"], observed=False, sort=True)
    for _ in range(RANDOM_V3_REFERENCE_TRIALS):
        pull_bid = np.zeros(len(frame), dtype=bool)
        pull_ask = np.zeros(len(frame), dtype=bool)
        for _, idx in grouped.groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k_bid = int(frame.loc[idx_arr, "pull_bid"].sum())
            k_ask = int(frame.loc[idx_arr, "pull_ask"].sum())
            if k_bid:
                pull_bid[rng.choice(idx_arr, size=min(k_bid, len(idx_arr)), replace=False)] = True
            if k_ask:
                pull_ask[rng.choice(idx_arr, size=min(k_ask, len(idx_arr)), replace=False)] = True
        pnls.append(simulate_policy(frame, pull_bid, pull_ask, actions, stress)["net_pnl"])
    return float(np.mean(pnls))


def base_expected_values() -> dict[str, float]:
    baseline = pd.read_csv(V16_DIR / "results" / "baseline_comparison.csv")
    strict = pd.read_csv(V16_DIR / "results" / "strict_random_baseline_distribution.csv")
    common = pd.read_csv(COMMON_015_DIR / "pnl_summary_015_common.csv")
    return {
        "no_cancel": float(baseline[baseline["method"] == "no_cancel"].iloc[0]["test_pnl"]),
        "random_v3_mean": float(strict[strict["method"] == "random_v3"].iloc[0]["mean"]),
        "015_common": float(common.iloc[0]["net_pnl"]),
        "016_v2_binary": float(baseline[baseline["method"] == "016_v2_binary"].iloc[0]["test_pnl"]),
    }
