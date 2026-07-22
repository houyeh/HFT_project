# PHC NEW5 Post-hoc Diagnostic Report

Final status: `phc_posthoc_beats_hgb_on_new5`.

NEW5 was already used in Phase 04C as the final HGB holdout. Therefore this PHC-lite NEW5 evaluation is post-hoc diagnostic only, not an independent fresh final holdout. A new fresh holdout such as NEW6 is required for a formal PHC final claim.

No tuning on NEW5. No architecture change. No threshold change. No fresh holdout claim.

Comparison caveat: the frozen Phase 04B.1 HGB artifact emitted a local sklearn `InconsistentVersionWarning` when loaded. PHC-lite evaluation does not depend on that pickle; the caveat applies only to the HGB comparison row.

## Policy Evaluation

| split                                      | policy_id                                     |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   negative_rate |   max_drawdown |   positive_session_count |   session_count | post_hoc_diagnostic_only   | new5_used_for_selection   |
|:-------------------------------------------|:----------------------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|----------------:|---------------:|-------------------------:|----------------:|:---------------------------|:--------------------------|
| post_hoc_diagnostic_only_not_fresh_holdout | no_quote                                      |        20 |         25494 |                       0 |         0        |           0 |           0 |            25494 |                      0 |                0      |   0        |        0        |              0 |                        0 |               1 | True                       | False                     |
| post_hoc_diagnostic_only_not_fresh_holdout | old_bid_only_spread_ge_1                      |        20 |         25494 |                   25494 |         1        |       25494 |           0 |                0 |                -333895 |              -13.097  |   0.360634 |        0.639327 |         335450 |                        0 |               1 | True                       | False                     |
| post_hoc_diagnostic_only_not_fresh_holdout | side_by_queue_imbalance_abs_0p1               |        20 |         25494 |                   20979 |         0.8229   |       10259 |       10720 |             4515 |                 361030 |               17.2091 |   0.430287 |        0.569665 |           4640 |                        1 |               1 | True                       | False                     |
| post_hoc_diagnostic_only_not_fresh_holdout | ml_hgb_regressor_shallow_convergence_selected |        20 |         25494 |                   19219 |         0.753864 |        9996 |        9223 |             6275 |                 487430 |               25.3619 |   0.457568 |        0.54238  |           3595 |                        1 |               1 | True                       | False                     |
| post_hoc_diagnostic_only_not_fresh_holdout | phc_lite_posthoc_new5                         |        20 |         25494 |                   19819 |         0.777399 |        9982 |        9837 |             5675 |                 582715 |               29.4018 |   0.471971 |        0.528029 |           3580 |                        1 |               1 | True                       | False                     |

## Matched Random

| split                                      | policy_id             |   model_total_excess_pnl_H20 |   model_selected_action_count |   random_trials |   random_mean |   random_std |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   | matching                                                    | post_hoc_diagnostic_only   |
|:-------------------------------------------|:----------------------|-----------------------------:|------------------------------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------------------------------------:|:----------------------------|:-------------------|:------------------------------------------------------------|:---------------------------|
| post_hoc_diagnostic_only_not_fresh_holdout | phc_lite_posthoc_new5 |                       582715 |                         19819 |            1000 |       -295702 |      17413.1 |      -267261 |      -254541 |                                            0 | < 0.001                     | True               | count_session_side_spread_bucket_with_session_side_fallback | True                       |

## Causal-Fill Sequential Proxy

| split                                      | policy_id                                     |   horizon | fill_model             |   inventory_cap |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   | post_hoc_diagnostic_only   |
|:-------------------------------------------|:----------------------------------------------|----------:|:-----------------------|----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|:---------------------------|
| post_hoc_diagnostic_only_not_fresh_holdout | side_by_queue_imbalance_abs_0p1               |        20 | conservative_l2_stress |               1 |                   40880 |         2248 |                       0 |                   1 |           2355 |   0.433274 |    0.566726 |                18.1851 | False                  | True                       |
| post_hoc_diagnostic_only_not_fresh_holdout | ml_hgb_regressor_shallow_convergence_selected |        20 | conservative_l2_stress |               1 |                   59670 |         2222 |                       0 |                   1 |           1840 |   0.464446 |    0.535554 |                26.8542 | False                  | True                       |
| post_hoc_diagnostic_only_not_fresh_holdout | phc_lite_posthoc_new5                         |        20 | conservative_l2_stress |               1 |                   68670 |         2422 |                       0 |                   1 |           2485 |   0.47853  |    0.52147  |                28.3526 | False                  | True                       |

This does not prove real-world profitability and is not true exchange replay.
