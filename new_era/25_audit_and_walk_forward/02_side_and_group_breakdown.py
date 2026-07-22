from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "25_audit_and_walk_forward"
RESULTS = ROOT / "results"
V24 = PROJECT_ROOT / "new_era" / "24_calibration_robust_policy"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def run() -> None:
    ensure_dirs()
    actions = pd.read_csv(V24 / "results" / "frozen_calibration_robust_test_actions.csv")
    detail = pd.read_csv(V24 / "results" / "frozen_calibration_robust_test_by_session_side_spread.csv")
    selected = actions[actions["selected_quote"].astype(bool)].copy()
    side = (
        selected.groupby("side", observed=False)
        .agg(
            selected_quote_count=("action_id", "count"),
            selected_quote_pnl=("quote_value_moderate", "sum"),
            avg_quote_pnl=("quote_value_moderate", "mean"),
            median_quote_pnl=("quote_value_moderate", "median"),
            positive_quote_count=("quote_value_moderate", lambda s: int((s > 0).sum())),
            negative_quote_count=("quote_value_moderate", lambda s: int((s < 0).sum())),
            max_win=("quote_value_moderate", "max"),
            max_loss=("quote_value_moderate", "min"),
        )
        .reset_index()
    )
    all_side = (
        actions.groupby("side", observed=False)
        .agg(
            all_quote_count=("action_id", "count"),
            all_quote_pnl=("quote_value_moderate", "sum"),
            all_negative_quote_count=("quote_value_moderate", lambda s: int((s < 0).sum())),
            all_negative_quote_rate=("quote_value_moderate", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )
    side = side.merge(all_side, on="side", how="left")
    side["selection_rate_within_side"] = side["selected_quote_count"] / side["all_quote_count"]
    side["pnl_share"] = side["selected_quote_pnl"] / side["selected_quote_pnl"].sum()
    side["ask_toxic_interpretation"] = np.where(
        side["side"].astype(str) == "ask",
        "Ask had low selected fraction under validation-selected ask0.10; inspect negative-rate and selected PnL as toxicity evidence.",
        "",
    )
    side.to_csv(RESULTS / "side_specific_breakdown.csv", index=False)

    group = detail.copy()
    group["positive_group"] = group["selected_quote_pnl"].astype(float) > 0.0
    group["pnl_share"] = group["selected_quote_pnl"].astype(float) / group["selected_quote_pnl"].astype(float).sum()
    group.to_csv(RESULTS / "group_level_performance.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote side/group breakdown outputs to {RESULTS}")

