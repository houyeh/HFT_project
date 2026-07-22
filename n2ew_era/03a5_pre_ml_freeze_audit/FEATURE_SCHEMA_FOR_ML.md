# Feature Schema For ML

This file freezes the pre-ML schema. Future labels and targets are forbidden as features.

## Allowed Features

| allowed_feature    |
|:-------------------|
| action_side        |
| action_slot        |
| quote_price        |
| best_bid           |
| best_ask           |
| mid_price          |
| tick_size          |
| spread_ticks       |
| bid_sz_1           |
| ask_sz_1           |
| top_depth          |
| queue_imbalance    |
| deep_imbalance     |
| volatility_past_50 |

## Forbidden Columns / Patterns

| forbidden_column_or_pattern   |
|:------------------------------|
| split                         |
| session_id                    |
| symbol                        |
| event_index                   |
| timestamp                     |
| future_mid_H1                 |
| future_mid_H5                 |
| future_mid_H10                |
| future_mid_H20                |
| future_mid_H50                |
| future_mid_H100               |
| excess_bid_pnl_H1             |
| excess_ask_pnl_H1             |
| excess_pnl_H1                 |
| positive_excess_pnl_H1        |
| negative_excess_pnl_H1        |
| excess_bid_pnl_H5             |
| excess_ask_pnl_H5             |
| excess_pnl_H5                 |
| positive_excess_pnl_H5        |
| negative_excess_pnl_H5        |
| excess_bid_pnl_H10            |
| excess_ask_pnl_H10            |
| excess_pnl_H10                |
| positive_excess_pnl_H10       |
| negative_excess_pnl_H10       |
| excess_bid_pnl_H20            |
| excess_ask_pnl_H20            |
| excess_pnl_H20                |
| positive_excess_pnl_H20       |
| negative_excess_pnl_H20       |
| excess_bid_pnl_H50            |
| excess_ask_pnl_H50            |
| excess_pnl_H50                |
| positive_excess_pnl_H50       |
| negative_excess_pnl_H50       |
| excess_bid_pnl_H100           |
| excess_ask_pnl_H100           |
| excess_pnl_H100               |
| positive_excess_pnl_H100      |
| negative_excess_pnl_H100      |
| event_rate_past_50            |
| spread_change_past            |
| imbalance_change_past         |
| raw_pnl_H*                    |
| pnl_H*                        |
| future_mid_H*                 |
| excess_pnl_H*                 |
| excess_bid_pnl_H*             |
| excess_ask_pnl_H*             |
| positive_excess_pnl_H*        |
| negative_excess_pnl_H*        |
| target_*                      |

Hard rule: no feature in `allowed_features.json` may have `uses_future_data = true` in `results/feature_causality_audit.csv`.
