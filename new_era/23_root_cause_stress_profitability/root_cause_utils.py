from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "23_root_cause_stress_profitability"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V20_DIR = PROJECT_ROOT / "new_era" / "20_policy_search"
V21_DIR = PROJECT_ROOT / "new_era" / "21_stress_robust_policy"
V22_DIR = PROJECT_ROOT / "new_era" / "22_signal_generalization_ev"
for path in [V16_DIR, V20_DIR, V21_DIR, V22_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stress_robust_utils import (  # noqa: E402
    MAKER_FEE,
    POINT_VALUE,
    STRESSES,
    apply_policy,
    baseline_policy_masks,
    load_frames_actions,
)


MODERATE = STRESSES["moderate"]
QUOTE_EV_METRICS = V22_DIR / "results" / "quote_ev_model_metrics.csv"
SELECTED_021_POLICY = V21_DIR / "search" / "selected_stress_robust_policy.json"
_SPREAD_ONLY_SPEC_CACHE: dict[str, Any] | None = None


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS_DIR, PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_context(split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, actions = load_frames_actions(split)
    actions = actions.copy()
    actions["session_id"] = actions["session_id"].astype(str)
    actions["side"] = actions["side"].astype(str)
    actions["spread_bucket"] = actions["spread_bucket"].astype(str)
    actions["coarse_spread_bucket"] = actions["coarse_spread_bucket"].astype(str)
    actions["quote_value_moderate"] = quote_value_vector(actions)
    actions["oracle_positive_quote"] = actions["quote_value_moderate"] > 0.0
    actions["oracle_selected_quote"] = actions["oracle_positive_quote"]
    actions["spread_only_score"] = spread_only_scores(actions)
    actions["ev_021_score"] = (
        0.5 * actions["spread_ticks"].astype(float).to_numpy() * POINT_VALUE
        - 5.0 * actions["side_score_016"].astype(float).to_numpy() * POINT_VALUE
        - MAKER_FEE
    )
    actions["ensemble_risk_score"] = 0.75 * actions["side_score_016"].astype(float) + 0.25 * actions["side_score_015"].astype(float)
    return frame, actions


def quote_value_vector(actions: pd.DataFrame) -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(
        side == "bid",
        actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(),
        actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy(),
    )
    adverse = np.maximum(0.0, -side_markout)
    eligible = actions["neutral_u"].astype(float).to_numpy() <= float(MODERATE.neutral_fill_haircut)
    favorable = side_markout > 0.0
    eligible &= ~(favorable & (actions["good_u"].astype(float).to_numpy() > float(MODERATE.good_fill_haircut)))
    value = terminal_ticks * POINT_VALUE - MAKER_FEE - max(float(MODERATE.toxic_multiplier) - 1.0, 0.0) * adverse * POINT_VALUE
    return np.where(eligible, value, 0.0).astype(float)


def spread_only_threshold() -> float:
    return float(spread_only_selection_spec()["threshold"])


def spread_only_higher_is_better() -> bool:
    return bool(spread_only_selection_spec()["higher_is_better"])


def spread_only_selection_spec() -> dict[str, Any]:
    global _SPREAD_ONLY_SPEC_CACHE
    if _SPREAD_ONLY_SPEC_CACHE is not None:
        return _SPREAD_ONLY_SPEC_CACHE
    _, val_actions = load_frames_actions("val")
    val_actions = val_actions.copy()
    val_actions["quote_value_moderate"] = quote_value_vector(val_actions)
    scores = val_actions["spread_ticks"].astype(float).to_numpy()
    values = val_actions["quote_value_moderate"].to_numpy(dtype=float)
    candidates = np.unique(np.nanquantile(scores, np.linspace(0.05, 0.95, 37)))
    best = {"selected_quote_pnl": -np.inf, "threshold": np.nan, "higher_is_better": True, "selection_split": "validation"}
    for higher in [True, False]:
        for thr in candidates:
            selected = scores >= thr if higher else scores <= thr
            pnl = float(values[selected].sum())
            if pnl > best["selected_quote_pnl"]:
                best = {
                    "selected_quote_pnl": pnl,
                    "threshold": float(thr),
                    "higher_is_better": bool(higher),
                    "selection_split": "validation",
                    "score_definition": "raw spread_ticks fallback because the 022 sklearn pickle is version-incompatible in this runtime",
                    "test_used_for_selection": False,
                }
    _SPREAD_ONLY_SPEC_CACHE = best
    return best


def spread_only_scores(actions: pd.DataFrame) -> np.ndarray:
    return actions["spread_ticks"].astype(float).to_numpy()


def load_021_policy() -> dict[str, Any]:
    payload = json.loads(SELECTED_021_POLICY.read_text(encoding="utf-8"))
    return payload["selected_policy"]


def action_selected_from_event_pulls(actions: pd.DataFrame, pull_bid: np.ndarray, pull_ask: np.ndarray) -> np.ndarray:
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    canceled = np.where(side == "bid", pull_bid[idx], pull_ask[idx])
    return ~np.asarray(canceled, dtype=bool)


def selected_quotes_for_policy(frame: pd.DataFrame, actions: pd.DataFrame, policy_name: str) -> np.ndarray:
    n = len(actions)
    if policy_name == "no_cancel":
        return np.ones(n, dtype=bool)
    if policy_name == "015":
        pull_bid, pull_ask = baseline_policy_masks(frame, "015_common")
        return action_selected_from_event_pulls(actions, pull_bid, pull_ask)
    if policy_name == "016_v2":
        pull_bid, pull_ask = baseline_policy_masks(frame, "016_v2")
        return action_selected_from_event_pulls(actions, pull_bid, pull_ask)
    if policy_name == "020":
        pull_bid, pull_ask = baseline_policy_masks(frame, "020_selected_policy")
        return action_selected_from_event_pulls(actions, pull_bid, pull_ask)
    if policy_name == "021":
        pull_bid, pull_ask = apply_policy(frame, load_021_policy())
        return action_selected_from_event_pulls(actions, pull_bid, pull_ask)
    if policy_name == "spread_only":
        return select_by_threshold(
            actions["spread_only_score"].astype(float).to_numpy(),
            spread_only_threshold(),
            higher_is_better=spread_only_higher_is_better(),
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def score_specs() -> dict[str, dict[str, Any]]:
    return {
        "015": {"column": "side_score_015", "higher_is_better": False},
        "016_v2": {"column": "side_score_016", "higher_is_better": False},
        "ensemble_015_016": {"column": "ensemble_risk_score", "higher_is_better": False},
        "021_ev_score": {"column": "ev_021_score", "higher_is_better": True},
        "spread_only": {"column": "spread_only_score", "higher_is_better": True},
    }


def alignment_metrics(actions: pd.DataFrame, selected: np.ndarray, policy_name: str, group_cols: list[str]) -> pd.DataFrame:
    selected = np.asarray(selected, dtype=bool)
    df = actions[group_cols + ["quote_value_moderate", "oracle_positive_quote"]].copy()
    df["selected_quote"] = selected
    df["positive_value"] = np.where(df["quote_value_moderate"] > 0.0, df["quote_value_moderate"], 0.0)
    df["negative_value"] = np.where(df["quote_value_moderate"] < 0.0, df["quote_value_moderate"], 0.0)
    rows: list[dict[str, Any]] = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, observed=False, sort=True)
    for key, g in grouped:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(group_cols, key))
        else:
            base = {"group": "overall"}
        sel = g["selected_quote"].to_numpy(dtype=bool)
        pos = g["oracle_positive_quote"].to_numpy(dtype=bool)
        values = g["quote_value_moderate"].to_numpy(dtype=float)
        selected_count = int(sel.sum())
        oracle_count = int(pos.sum())
        hit = sel & pos
        missed = (~sel) & pos
        false_pos = sel & (values < 0.0)
        avoided_bad = (~sel) & (values < 0.0)
        row = {
            **base,
            "policy": policy_name,
            "num_quote_actions": int(len(g)),
            "selected_quote_count": selected_count,
            "oracle_positive_quote_count": oracle_count,
            "oracle_recall": float(hit.sum() / oracle_count) if oracle_count else np.nan,
            "oracle_precision": float(hit.sum() / selected_count) if selected_count else np.nan,
            "captured_oracle_pnl": float(values[hit].sum()),
            "missed_oracle_pnl": float(values[missed].sum()),
            "false_positive_loss": float(-values[false_pos].sum()),
            "avoided_bad_pnl": float(-values[avoided_bad].sum()),
            "missed_good_pnl": float(values[missed].sum()),
            "selected_quote_pnl": float(values[sel].sum()),
            "oracle_positive_pnl": float(values[pos].sum()),
            "selection_rate": float(selected_count / len(g)) if len(g) else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def matched_random_selected_pnl(
    actions: pd.DataFrame,
    selected: np.ndarray,
    *,
    group_cols: list[str],
    trials: int = 500,
    seed: int = 20260525,
) -> tuple[float, float]:
    selected = np.asarray(selected, dtype=bool)
    work = actions.reset_index(drop=True).copy()
    work["_selected"] = selected
    values = work["quote_value_moderate"].to_numpy(dtype=float)
    groups = work.groupby(group_cols, observed=False, sort=True).groups
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(int(trials)):
        mask = np.zeros(len(actions), dtype=bool)
        for _, idx in groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k = int(selected[idx_arr].sum())
            if k:
                mask[rng.choice(idx_arr, size=min(k, len(idx_arr)), replace=False)] = True
        totals.append(float(values[mask].sum()))
    arr = np.asarray(totals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fixed_threshold_from_validation(
    val_actions: pd.DataFrame,
    score_col: str,
    *,
    higher_is_better: bool,
    quantile: float,
    spread_bucket: str | None = None,
) -> float:
    source = val_actions if spread_bucket is None else val_actions[val_actions["spread_bucket"].astype(str) == str(spread_bucket)]
    scores = source[score_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(scores) == 0:
        return np.nan
    q = float(quantile if higher_is_better else 1.0 - quantile)
    return float(np.quantile(scores, q))


def select_by_threshold(scores: np.ndarray, threshold: float, *, higher_is_better: bool) -> np.ndarray:
    if np.isnan(threshold):
        return np.zeros(len(scores), dtype=bool)
    scores = np.asarray(scores, dtype=float)
    return scores >= threshold if higher_is_better else scores <= threshold


def eval_quote_selection(actions: pd.DataFrame, selected: np.ndarray, name: str) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    values = actions["quote_value_moderate"].to_numpy(dtype=float)
    pos = values > 0.0
    return {
        "policy": name,
        "selected_quote_count": int(selected.sum()),
        "selection_rate": float(selected.mean()) if len(selected) else 0.0,
        "selected_quote_pnl": float(values[selected].sum()),
        "avg_selected_quote_pnl": float(values[selected].mean()) if selected.any() else np.nan,
        "oracle_recall": float((selected & pos).sum() / pos.sum()) if pos.sum() else np.nan,
        "oracle_precision": float((selected & pos).sum() / selected.sum()) if selected.sum() else np.nan,
        "captured_oracle_pnl": float(values[selected & pos].sum()),
        "missed_oracle_pnl": float(values[(~selected) & pos].sum()),
        "false_positive_loss": float(-values[selected & (values < 0.0)].sum()),
        "avoided_bad_pnl": float(-values[(~selected) & (values < 0.0)].sum()),
        "missed_good_pnl": float(values[(~selected) & pos].sum()),
    }


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
