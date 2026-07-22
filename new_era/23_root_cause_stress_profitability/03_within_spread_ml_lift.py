from __future__ import annotations

import numpy as np
import pandas as pd

from root_cause_utils import (
    RESULTS_DIR,
    alignment_metrics,
    ensure_dirs,
    fixed_threshold_from_validation,
    load_context,
    matched_random_selected_pnl,
    score_specs,
    select_by_threshold,
    write_markdown,
)


def run() -> None:
    ensure_dirs()
    _, val_actions = load_context("val")
    _, test_actions = load_context("test")
    rows = []
    detail_rows = []
    quote_fraction = 0.20
    for score_name, spec in score_specs().items():
        col = spec["column"]
        higher = bool(spec["higher_is_better"])
        for spread_bucket in sorted(test_actions["spread_bucket"].astype(str).unique()):
            val_thr = fixed_threshold_from_validation(
                val_actions,
                col,
                higher_is_better=higher,
                quantile=1.0 - quote_fraction if higher else quote_fraction,
                spread_bucket=spread_bucket,
            )
            sub = test_actions[test_actions["spread_bucket"].astype(str) == spread_bucket].copy()
            if sub.empty:
                continue
            selected = select_by_threshold(sub[col].to_numpy(dtype=float), val_thr, higher_is_better=higher)
            rand_mean, rand_std = matched_random_selected_pnl(
                sub,
                selected,
                group_cols=["session_id", "side", "spread_bucket"],
                trials=500,
                seed=20260525 + len(rows),
            )
            values = sub["quote_value_moderate"].to_numpy(dtype=float)
            oracle_pos = values > 0.0
            selected_pnl = float(values[selected].sum())
            all_pnl = float(values.sum())
            rows.append(
                {
                    "split": "test_diagnostic_burned",
                    "score_name": score_name,
                    "spread_bucket": spread_bucket,
                    "validation_threshold": val_thr,
                    "threshold_selected_on": "validation",
                    "test_used_for_threshold_selection": False,
                    "selected_quote_count": int(selected.sum()),
                    "bucket_quote_count": int(len(sub)),
                    "ml_selected_pnl": selected_pnl,
                    "all_quotes_same_bucket_pnl": all_pnl,
                    "all_quotes_same_bucket_avg_pnl": float(values.mean()) if len(values) else np.nan,
                    "random_matched_side_session_bucket_mean_pnl": rand_mean,
                    "random_matched_side_session_bucket_std_pnl": rand_std,
                    "incremental_pnl_vs_same_bucket_random": selected_pnl - rand_mean,
                    "oracle_positive_quote_count": int(oracle_pos.sum()),
                    "oracle_positive_pnl_same_bucket": float(values[oracle_pos].sum()),
                    "oracle_recall": float((selected & oracle_pos).sum() / oracle_pos.sum()) if oracle_pos.sum() else np.nan,
                    "oracle_precision": float((selected & oracle_pos).sum() / selected.sum()) if selected.sum() else np.nan,
                }
            )
            detail = alignment_metrics(sub, selected, score_name, ["session_id", "side", "spread_bucket"])
            detail["validation_threshold"] = val_thr
            detail["threshold_selected_on"] = "validation"
            detail["test_used_for_threshold_selection"] = False
            detail_rows.append(detail)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "within_spread_ml_lift.csv", index=False)
    if detail_rows:
        pd.concat(detail_rows, ignore_index=True).to_csv(RESULTS_DIR / "within_spread_ml_lift_by_session_side_spread.csv", index=False)
    summary = out.groupby("score_name", observed=False)["incremental_pnl_vs_same_bucket_random"].sum().sort_values(ascending=False)
    lines = [
        "# Within-Spread ML Incremental Value",
        "",
        "Each ML-selected set uses validation-defined within-spread thresholds and is compared against random quotes matched by side/session/spread bucket.",
        "",
    ]
    for name, val in summary.items():
        lines.append(f"- `{name}` total incremental PnL vs matched same-bucket random: `{val:.3f}`.")
    write_markdown(RESULTS_DIR / "within_spread_ml_lift_report.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote within-spread ML lift outputs to {RESULTS_DIR}")
