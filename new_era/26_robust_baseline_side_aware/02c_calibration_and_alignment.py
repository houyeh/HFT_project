from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_utils import RESULTS_DIR, ensure_dirs, val_test_actions


def bucket_by_val_quantiles(val_scores: np.ndarray, scores: np.ndarray, bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(val_scores[np.isfinite(val_scores)], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(scores), np.nanmax(scores), bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(scores, edges[1:-1], right=False) + 1


def run() -> None:
    ensure_dirs()
    val = val_test_actions("val")
    test = val_test_actions("test")
    rows = []
    for side in ["bid", "ask"]:
        val_side = val[val["side"].astype(str) == side].copy()
        for split, df in [("validation", val), ("test_diagnostic_burned", test)]:
            sub = df[df["side"].astype(str) == side].copy()
            sub["score_bucket_val_quantile"] = bucket_by_val_quantiles(val_side["side_score_016"].astype(float).to_numpy(), sub["side_score_016"].astype(float).to_numpy())
            agg = (
                sub.groupby("score_bucket_val_quantile", observed=False)
                .agg(
                    quote_count=("action_id", "count"),
                    avg_pnl=("quote_value_moderate", "mean"),
                    total_pnl=("quote_value_moderate", "sum"),
                    toxic_rate=("side_toxic", "mean"),
                    avg_score=("side_score_016", "mean"),
                )
                .reset_index()
            )
            agg["side"] = side
            agg["split"] = split
            rows.append(agg)
    pd.concat(rows, ignore_index=True).to_csv(RESULTS_DIR / "score_bucket_transfer.csv", index=False)

    align_rows = []
    for split, df in [("validation", val), ("test_diagnostic_burned", test)]:
        work = df.copy()
        work["pnl_positive"] = work["quote_value_moderate"].astype(float) > 0.0
        table = (
            work.groupby(["side", "side_toxic", "pnl_positive"], observed=False)
            .agg(quote_count=("action_id", "count"), total_pnl=("quote_value_moderate", "sum"), avg_pnl=("quote_value_moderate", "mean"))
            .reset_index()
        )
        table["split"] = split
        align_rows.append(table)
    pd.concat(align_rows, ignore_index=True).to_csv(RESULTS_DIR / "toxic_vs_pnl_alignment.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote calibration/alignment outputs to {RESULTS_DIR}")

