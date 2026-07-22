# Phase 03B Feature Schema

Phase 03B uses only verified causal L2 and rebuilt dynamic features. Raw absolute price columns exist in the source parquet for label construction lineage, but they are excluded from the ML feature schema by the Phase 03A.6 decision.

## Allowed Feature Columns

| feature                    |
|:---------------------------|
| spread_ticks               |
| bid_sz_1                   |
| ask_sz_1                   |
| top_depth                  |
| queue_imbalance            |
| deep_imbalance             |
| volatility_past_50         |
| event_rate_past_50         |
| inter_arrival_ms           |
| imbalance_change_past      |
| spread_change_past         |
| action_is_bid              |
| action_is_ask              |
| action_is_no_quote         |
| side_sign                  |
| side_queue_imbalance       |
| side_deep_imbalance        |
| side_imbalance_change_past |

## Forbidden Columns

`best_bid`, `best_ask`, `mid_price`, `quote_price`, all `future_mid_*`, all raw/excess PnL label columns, and all target columns are forbidden as ML features.
