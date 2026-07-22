# Residual Universe Filtering Waterfall Report

This is an explanation audit only. It does not tune ML, alter thresholds, or change the final non-ML controller.

## Executive Answer

Train shrinks from `3,593,426` raw rows to exactly `820` residual ML rows because the residual experiment is not trained on all market rows. It first turns quote-eligible rows into sparse passive-fill candidate actions, then applies the final baseline policy `bid_only_spread_ge_1`.

The largest train-stage drop is `sparse_fill_proxy_candidate_action_construction`: `3,592,764` input rows/actions to `1,726` retained, removing `3,591,038`.

## Split Waterfall

| split | raw_rows | label_rows | valid_book_rows | quote_eligible_rows | candidate_action_rows | bid_candidate_action_rows | ask_candidate_action_rows | baseline_selected_rows | residual_ml_rows | negative_ev_moderate_rows | severe_negative_ev_moderate_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3593426 | 3592764 | 3593424 | 3592764 | 1726 | 820 | 906 | 820 | 820 | 226 | 173 |
| val | 563593 | 563477 | 563587 | 563477 | 262 | 134 | 128 | 134 | 134 | 17 | 3 |
| test | 1166541 | 1166321 | 1166540 | 1166321 | 934 | 419 | 515 | 419 | 419 | 16 | 0 |
| NEW4 | 2274678 | 2197106 | 2197106 | 2197106 | 1306 | 714 | 592 | 714 | 714 | 103 | 36 |

## Why Train Becomes 820 Samples

- Quote eligibility is not the main bottleneck: train keeps `3,592,764` quote-eligible rows out of `3,593,426` raw rows.
- Sparse action construction is the dominant bottleneck: only `1,726` candidate actions remain, a `0.048041%` retention rate from quote-eligible rows.
- The baseline policy then keeps `820` bid-side `spread_ticks >= 1` actions, a `47.51%` retention rate from candidate actions.
- The residual ML parquet contains `820` rows, matching the baseline-selected count; residual materialization itself does not add another filter.
- End-to-end train retention is `0.022819%` from raw rows.

## What Caused the Small Sample Size?

The small residual training set is primarily caused by sparse fill-proxy candidate action construction, not by quote eligibility. The baseline filter is the second major reduction because it removes all ask-side candidate actions and keeps only bid-side candidates satisfying `spread_ticks >= 1`.

## Source Reconstruction Notes

- Historical train/val/test raw and quote-eligible counts are reconstructed from `new_era/16_binary_label_v2/outputs/quote_eligible_filter_report.csv`.
- Historical label rows equal the rows loaded by `common_028.load_label_split`, which reads `binary_labels_v2.parquet`; that parquet is already quote-eligible.
- The requested table includes `label_rows`, but the causal filter order is `raw -> valid_book -> quote_eligible -> label_frame -> candidate_actions -> baseline -> residual`; retention-from-previous-stage columns use this actual execution order.
- Candidate action and baseline-selected counts are read from the frozen Phase 031 full-action universe generated from the Phase 028/027 pipeline.
- NEW4 has no `binary_labels_v2.parquet` entry. Its `label_rows` and `quote_eligible_rows` are explicitly marked as a valid-book OOS proxy from the `build_new4_frame` pathway, not guessed.
- Severe negative EV uses the Phase 030 convention: `quote_ev_moderate <= -5000`.

## Impact on Residual ML Interpretation

The residual ML experiment is a valid incremental veto audit, but it is not a full test of all L2-based ML potential. Because the residual universe contains only baseline-selected quote samples, the training set is very small. Therefore, the failure of residual ML should be interpreted as failure to add value after the non-ML baseline, not as proof that all ML approaches are useless.

## Outputs

- `filtering_waterfall_by_split.csv`
- `filtering_waterfall_by_session.csv`
- `filtering_waterfall_by_reason.csv`
- `filtering_waterfall_table.md`
- `filtering_code_path_audit.md`
- `figures/filtering_waterfall_by_split.png`
- `figures/residual_sample_count_vs_raw_rows.png`
- `figures/negative_ev_label_counts_by_split.png`
