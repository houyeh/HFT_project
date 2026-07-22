# Feature Causality Audit

Hard rule: no feature with `uses_future_data = true` may be allowed for ML.

Violation count: `0`.

Allowed features are current-row or past-only causal fields. Identity columns are not future-leaking but are not model features.

| column                   | category                   | uses_future_data   | allowed_as_feature   | hard_rule_pass   | notes                                                              |
|:-------------------------|:---------------------------|:-------------------|:---------------------|:-----------------|:-------------------------------------------------------------------|
| split                    | identity_not_model_feature | False              | False                | True             |                                                                    |
| session_id               | identity_not_model_feature | False              | False                | True             |                                                                    |
| symbol                   | identity_not_model_feature | False              | False                | True             |                                                                    |
| event_index              | identity_not_model_feature | False              | False                | True             |                                                                    |
| timestamp                | identity_not_model_feature | False              | False                | True             |                                                                    |
| action_side              | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| action_slot              | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| quote_price              | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| best_bid                 | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| best_ask                 | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| mid_price                | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| tick_size                | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| spread_ticks             | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| bid_sz_1                 | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| ask_sz_1                 | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| top_depth                | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| queue_imbalance          | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| deep_imbalance           | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| volatility_past_50       | allowed_causal_feature     | False              | True                 | True             | current/past-only by construction                                  |
| future_mid_H1            | future_or_target_label     | True               | False                | True             |                                                                    |
| future_mid_H5            | future_or_target_label     | True               | False                | True             |                                                                    |
| future_mid_H10           | future_or_target_label     | True               | False                | True             |                                                                    |
| future_mid_H20           | future_or_target_label     | True               | False                | True             |                                                                    |
| future_mid_H50           | future_or_target_label     | True               | False                | True             |                                                                    |
| future_mid_H100          | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H1        | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H1        | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H1            | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H1   | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H1   | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H5        | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H5        | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H5            | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H5   | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H5   | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H10       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H10       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H10           | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H10  | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H10  | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H20       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H20       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H20           | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H20  | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H20  | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H50       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H50       | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H50           | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H50  | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H50  | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_bid_pnl_H100      | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_ask_pnl_H100      | future_or_target_label     | True               | False                | True             |                                                                    |
| excess_pnl_H100          | future_or_target_label     | True               | False                | True             |                                                                    |
| positive_excess_pnl_H100 | future_or_target_label     | True               | False                | True             |                                                                    |
| negative_excess_pnl_H100 | future_or_target_label     | True               | False                | True             |                                                                    |
| event_rate_past_50       | requested_but_missing      | False              | False                | True             | Missing from Phase 02F parquet; do not use until rebuilt causally. |
| spread_change_past       | requested_but_missing      | False              | False                | True             | Missing from Phase 02F parquet; do not use until rebuilt causally. |
| imbalance_change_past    | requested_but_missing      | False              | False                | True             | Missing from Phase 02F parquet; do not use until rebuilt causally. |
