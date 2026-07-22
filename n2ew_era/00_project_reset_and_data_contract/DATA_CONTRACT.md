# Data Contract

All future outcome labels must be treated as labels only. They are never allowed as model features or rule inputs.

| field_name | category | description | causal_or_future | allowed_as_feature | required_phase |
| --- | --- | --- | --- | --- | --- |
| spread_ticks | causal_feature | Current best ask minus best bid in ticks. | causal | yes | 01_full_l2_action_universe_builder |
| queue_imbalance | causal_feature | L1 bid/ask queue imbalance at the current event. | causal | yes | 01_full_l2_action_universe_builder |
| deep_imbalance | causal_feature | L2 depth imbalance across multiple book levels. | causal | yes | 01_full_l2_action_universe_builder |
| top_depth | causal_feature | Sum or available size at the top of book. | causal | yes | 01_full_l2_action_universe_builder |
| depth_ratio | causal_feature | Bid depth divided by ask depth, with safe zero handling. | causal | yes | 01_full_l2_action_universe_builder |
| event_rate_past_50 | causal_feature | Past-only event rate over the previous 50 events. | causal | yes | 01_full_l2_action_universe_builder |
| inter_arrival_ms | causal_feature | Time since prior event in milliseconds. | causal | yes | 01_full_l2_action_universe_builder |
| volatility_past_50 | causal_feature | Past-only rolling volatility over previous 50 events. | causal | yes | 01_full_l2_action_universe_builder |
| abs_return_past_50 | causal_feature | Past-only rolling absolute return over previous 50 events. | causal | yes | 01_full_l2_action_universe_builder |
| imbalance_change_past | causal_feature | Past-only change in imbalance, excluding the future/current label horizon. | causal | yes | 01_full_l2_action_universe_builder |
| spread_change_past | causal_feature | Past-only spread change feature. | causal | yes | 01_full_l2_action_universe_builder |
| hawkes_excite | causal_feature_optional | Causal excitation or event intensity proxy, if built only from past events. | causal | yes | 03_l2_imbalance_rule_search |
| pnl_H1 | future_outcome_label | Sequential proxy PnL for horizon H=1. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| pnl_H5 | future_outcome_label | Sequential proxy PnL for horizon H=5. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| pnl_H10 | future_outcome_label | Sequential proxy PnL for horizon H=10. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| pnl_H20 | future_outcome_label | Sequential proxy PnL for horizon H=20. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| pnl_H50 | future_outcome_label | Sequential proxy PnL for horizon H=50. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| pnl_H100 | future_outcome_label | Sequential proxy PnL for horizon H=100. | future_label | no | 02_short_horizon_sequential_proxy_baselines |
| positive_pnl_H1 | future_outcome_label | Classification label: pnl_H1 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| positive_pnl_H5 | future_outcome_label | Classification label: pnl_H5 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| positive_pnl_H10 | future_outcome_label | Classification label: pnl_H10 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| positive_pnl_H20 | future_outcome_label | Classification label: pnl_H20 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| positive_pnl_H50 | future_outcome_label | Classification label: pnl_H50 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| positive_pnl_H100 | future_outcome_label | Classification label: pnl_H100 greater than zero. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H1 | future_outcome_label | Classification label for severe negative pnl_H1. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H5 | future_outcome_label | Classification label for severe negative pnl_H5. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H10 | future_outcome_label | Classification label for severe negative pnl_H10. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H20 | future_outcome_label | Classification label for severe negative pnl_H20. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H50 | future_outcome_label | Classification label for severe negative pnl_H50. | future_label | no | 04_full_action_ml_ev_models |
| severe_negative_pnl_H100 | future_outcome_label | Classification label for severe negative pnl_H100. | future_label | no | 04_full_action_ml_ev_models |
| terminal_quote_ev | future_outcome_label_diagnostic | Session-final or terminal quote EV diagnostic from prior work. | future_label | no | 02_short_horizon_sequential_proxy_baselines |

## Hard Rule

All future labels must have `causal_or_future = future_label` and `allowed_as_feature = no`.
