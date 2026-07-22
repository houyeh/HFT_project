# PHC NEW5 Window Full Oracle Report

For each PHC sequence terminal state, oracle action is `argmax(excess_bid, excess_ask, 0)`. Oracle uses future labels and is non-causal.

| split                                      | policy_id                     |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   negative_rate |   max_drawdown |   positive_session_count |   session_count | oracle_uses_future_labels   | post_hoc_diagnostic_only   | new5_used_for_tuning   |
|:-------------------------------------------|:------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|----------------:|---------------:|-------------------------:|----------------:|:----------------------------|:---------------------------|:-----------------------|
| post_hoc_diagnostic_only_not_fresh_holdout | oracle_phc_window_full_action |        20 |         25494 |                   18110 |         0.710363 |        9194 |        8916 |             7384 |            2.64665e+06 |               146.143 |          1 |               0 |              0 |                        1 |               1 | True                        | True                       | False                  |

This is post-hoc diagnostic because NEW5 was already used in Phase 04C.
