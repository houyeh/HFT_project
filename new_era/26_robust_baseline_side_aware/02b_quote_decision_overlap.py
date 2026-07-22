from __future__ import annotations

import pandas as pd

from baseline_utils import RESULTS_DIR, ensure_dirs


def run() -> None:
    ensure_dirs()
    actions = pd.read_parquet(RESULTS_DIR / "ml_veto_test_actions.parquet")
    base = actions["baseline_selected"].astype(bool)
    ml = actions["ml_final_selected"].astype(bool)
    labels = []
    for b, m in zip(base, ml):
        if b and m:
            labels.append("kept_by_both")
        elif (not b) and (not m):
            labels.append("blocked_by_both")
        elif b and (not m):
            labels.append("blocked_by_ML_only")
        else:
            labels.append("kept_by_ML_only")
    actions["decision_overlap"] = labels
    out = (
        actions.groupby(["decision_overlap", "side"], observed=False)
        .agg(
            quote_count=("action_id", "count"),
            total_pnl=("quote_value_moderate", "sum"),
            avg_pnl=("quote_value_moderate", "mean"),
            toxic_rate=("side_toxic", "mean"),
            positive_pnl_rate=("quote_value_moderate", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    out.to_csv(RESULTS_DIR / "quote_decision_overlap.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote quote decision overlap to {RESULTS_DIR}")

