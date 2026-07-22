# NEW5 TXF Action Universe Report

This is the one-shot final holdout action universe for `NEW5 20260128` only. The frozen TXF filter `^TXF[A-Z0-9]*$` was applied and malformed `info_time` rows were dropped/logged before feature or label construction.

No tuning, retraining, threshold sweep, or NEW5-derived model selection was performed.

## Build Summary

| source_group   |   session | product_id   |   raw_txf_rows |   source_rows_after_valid_book_boundary |   action_rows |   bid_actions |   ask_actions |   no_quote_actions |   malformed_info_time_rows_dropped | new5_scope    |
|:---------------|----------:|:-------------|---------------:|----------------------------------------:|--------------:|--------------:|--------------:|-------------------:|-----------------------------------:|:--------------|
| NEW5           |  20260128 | TXF          |           3190 |                                       0 |             0 |             0 |             0 |                  0 |                                  0 | 20260128_only |
| NEW5           |  20260128 | TXFB6        |         845793 |                                  815912 |       2447736 |        815912 |        815912 |             815912 |                                  0 | 20260128_only |

## Causal Feature Columns

spread_ticks, queue_imbalance, deep_imbalance, top_depth, depth_ratio, event_rate_past_50, inter_arrival_ms, volatility_past_50, abs_return_past_50, imbalance_change_past, spread_change_past, hawkes_excite, bid_sz_1, ask_sz_1

## Label Columns

future_mid_H20, raw_pnl_H20, half_spread_component_H20, excess_pnl_H20

Output parquet: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\04c_new5_one_shot_final_holdout\data\new5_txf_only_action_universe.parquet`.
