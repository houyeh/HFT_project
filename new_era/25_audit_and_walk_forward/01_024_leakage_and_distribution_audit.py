from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "25_audit_and_walk_forward"
RESULTS = ROOT / "results"
V24 = PROJECT_ROOT / "new_era" / "24_calibration_robust_policy"
if str(V24) not in sys.path:
    sys.path.insert(0, str(V24))

from calibration_utils import write_markdown  # noqa: E402


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def recompute_thresholds(actions: pd.DataFrame, policy: dict) -> tuple[np.ndarray, np.ndarray]:
    work = actions.reset_index(drop=True).copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    score_col = str(policy["score_column"])
    if score_col not in work.columns and "score" in work.columns:
        score_col = "score"
    higher = bool(policy["higher_is_better"])
    window = int(policy["window"])
    min_periods = max(5, window // 5)
    side_fracs = {"bid": float(policy["bid_quote_fraction"]), "ask": float(policy["ask_quote_fraction"])}
    thresholds = np.full(len(work), np.nan, dtype=float)
    selected = np.zeros(len(work), dtype=bool)
    for _, g in work.sort_values(["session_id", "row_index", "action_id"]).groupby(["session_id", "side"], observed=False, sort=False):
        side = str(g["side"].iloc[0])
        q = side_fracs[side] if not higher else 1.0 - side_fracs[side]
        hist = g[score_col].astype(float).shift(1).rolling(window=window, min_periods=min_periods)
        thr = hist.quantile(q)
        sel = g[score_col].astype(float) >= thr if higher else g[score_col].astype(float) <= thr
        pos = g["_pos"].to_numpy(dtype=np.int64)
        thresholds[pos] = thr.to_numpy(dtype=float)
        selected[pos] = sel.fillna(False).to_numpy(dtype=bool)
    return thresholds, selected


def pnl_distribution(actions: pd.DataFrame) -> pd.DataFrame:
    selected = actions[actions["selected_quote"].astype(bool)].copy()
    pnl = selected["quote_value_moderate"].astype(float).to_numpy()
    total = float(pnl.sum())
    pos_rank = np.sort(np.maximum(pnl, 0.0))[::-1]
    rows = []
    base = {
        "selected_quote_count": int(len(pnl)),
        "total_selected_quote_pnl": total,
        "mean_quote_pnl": float(np.mean(pnl)) if len(pnl) else np.nan,
        "median_quote_pnl": float(np.median(pnl)) if len(pnl) else np.nan,
        "std_quote_pnl": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        "max_win": float(np.max(pnl)) if len(pnl) else np.nan,
        "max_loss": float(np.min(pnl)) if len(pnl) else np.nan,
        "positive_quote_count": int(np.sum(pnl > 0.0)),
        "negative_quote_count": int(np.sum(pnl < 0.0)),
    }
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(np.ceil(frac * len(pnl)))) if len(pnl) else 0
        contrib = float(pos_rank[:k].sum()) if k else 0.0
        rows.append(
            {
                **base,
                "top_fraction": frac,
                "top_quote_count": k,
                "top_positive_contribution": contrib,
                "top_positive_contribution_pct_of_total_pnl": contrib / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    ensure_dirs()
    policy_payload = json.loads((V24 / "selected_calibration_robust_policy.json").read_text(encoding="utf-8"))
    policy = policy_payload["selected_policy"]
    actions = pd.read_csv(V24 / "results" / "frozen_calibration_robust_test_actions.csv")
    source = (V24 / "calibration_utils.py").read_text(encoding="utf-8")

    thresholds, selected = recompute_thresholds(actions, policy)
    stored_thr = actions["past_only_threshold"].astype(float).to_numpy()
    stored_sel = actions["selected_quote"].astype(bool).to_numpy()
    finite = np.isfinite(stored_thr) | np.isfinite(thresholds)
    threshold_max_abs_diff = float(np.nanmax(np.abs(stored_thr[finite] - thresholds[finite]))) if finite.any() else 0.0
    selected_mismatch_count = int(np.sum(stored_sel != selected))
    first_by_group = actions.sort_values(["session_id", "side", "row_index", "action_id"]).groupby(["session_id", "side"], observed=False).head(1)
    first_rows_unselected = bool((~first_by_group["selected_quote"].astype(bool)).all())
    first_rows_threshold_nan = bool(first_by_group["past_only_threshold"].isna().all())
    session_boundary_ok = first_rows_unselected and first_rows_threshold_nan
    shift_present = ".shift(1)" in source or "shift(1)" in source

    pnl_distribution(actions).to_csv(RESULTS / "selected_quote_pnl_distribution.csv", index=False)
    lines = [
        "# Rolling Threshold Leakage Audit",
        "",
        f"- Selected policy: `{policy['policy_id']}`.",
        f"- Required window: `{policy['window']}`.",
        f"- Source contains explicit `shift(1)`: `{shift_present}`.",
        "- Threshold formula audited: `threshold_t = quantile(score[t-N:t-1], q)` within `session_id|side` groups.",
        f"- Recomputed threshold max absolute difference: `{threshold_max_abs_diff:.12g}`.",
        f"- Recomputed selected-mask mismatches: `{selected_mismatch_count}`.",
        f"- First row of each session/side group has no threshold and is not selected: `{session_boundary_ok}`.",
        "",
        "Conclusion: " + (
            "No lookahead leakage detected in the frozen 024 action file."
            if shift_present and selected_mismatch_count == 0 and threshold_max_abs_diff < 1e-12 and session_boundary_ok
            else "Potential leakage or implementation mismatch detected; inspect before making claims."
        ),
    ]
    write_markdown(RESULTS / "rolling_threshold_leakage_audit.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote leakage audit outputs to {RESULTS}")
