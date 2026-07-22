# TXF-only Final Development Decision Report Patched

Final patched decision: `ml_selected_for_new5`.
Selected strategy for NEW5 holdout: `ml_hgb_regressor_shallow_convergence_selected`.

This final patched report uses the convergence-selected exact ML configuration, not merely the earlier 80-iteration Phase 04B frozen diagnostic model.

NEW5 was not loaded or evaluated. NEW3/NEW4 were not used for tuning, thresholding, model selection, or HGB iteration selection.

## Candidate Completion Scorecard

| candidate                                     | exact_config                                             | beats_matched_random_95p_val_new3_new4   | positive_causal_fill_sequential_val_new3_new4   | competitive_vs_queue_in_sequential_val_new3_new4   |   hgb_convergence_closed | selected_for_new5   |
|:----------------------------------------------|:---------------------------------------------------------|:-----------------------------------------|:------------------------------------------------|:---------------------------------------------------|-------------------------:|:--------------------|
| ml_hgb_regressor_shallow_convergence_selected | hgb_regressor max_iter=120 threshold=0.22419440373778343 | True                                     | True                                            | True                                               |                        1 | True                |
| side_by_queue_imbalance_abs_0p1               | queue_imbalance abs threshold 0.1                        | True                                     | True                                            | True                                               |                      nan | False               |

## Final ML Matched Random Summary

| split   | policy_id                                     | model_label                     |   threshold_selected_on_val |   model_total_excess_pnl_H20 |   model_selected_action_count |   random_trials |   random_mean |   random_std |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model | beats_random_95p   | matching                 | selection_split   | test_new3_new4_used_for_selection   | new5_used   |
|:--------|:----------------------------------------------|:--------------------------------|----------------------------:|-----------------------------:|------------------------------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------------------------------------:|:-------------------|:-------------------------|:------------------|:------------------------------------|:------------|
| val     | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |                  5.45502e+07 |                       1980301 |            1000 |  -2.96072e+07 |       167619 | -2.934e+07   | -2.92264e+07 |                                            0 | True               | exact_count_side_session | val               | False                               | False       |
| new3    | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |                  1.29325e+08 |                       4294276 |            1000 |  -6.43206e+07 |       259084 | -6.38861e+07 | -6.37162e+07 |                                            0 | True               | exact_count_side_session | val               | False                               | False       |
| new4    | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |                  4.60936e+07 |                       1652822 |            1000 |  -2.47448e+07 |       163681 | -2.44729e+07 | -2.43664e+07 |                                            0 | True               | exact_count_side_session | val               | False                               | False       |

Advisor-safe p-value wording: if the empirical fraction is `0`, then in 1000 matched random trials no random trial had PnL >= model PnL, so empirical p-value `< 0.001`; it is not an exact zero probability claim.

## Final ML vs Queue Causal-Fill Sequential Summary

| split   | policy_id                                     | model_label                     |   threshold_selected_on_val |   horizon | fill_model             |   inventory_cap |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   | selection_split   | test_new3_new4_used_for_selection   | new5_used   |   ml_minus_queue_rule |   queue_sequential_excess_pnl | ml_beats_or_matches_queue   |
|:--------|:----------------------------------------------|:--------------------------------|----------------------------:|----------:|:-----------------------|----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|:------------------|:------------------------------------|:------------|----------------------:|------------------------------:|:----------------------------|
| val     | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |        20 | conservative_l2_stress |               1 |             2.7386e+06  |       124692 |                   79757 |                   1 |          16490 |   0.4495   |    0.55046  |                21.9629 | False                  | val               | False                               | False       |      681390           |                   2.0572e+06  | True                        |
| new3    | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |        20 | conservative_l2_stress |               1 |             7.45686e+06 |       289888 |                  218633 |                   1 |          18215 |   0.473514 |    0.526469 |                25.7232 | False                  | val               | False                               | False       |           1.72326e+06 |                   5.7336e+06  | True                        |
| new4    | ml_hgb_regressor_shallow_convergence_selected | hgb_iter_120_threshold_0.224194 |                    0.224194 |        20 | conservative_l2_stress |               1 |             2.72002e+06 |       111277 |                   81292 |                   1 |          10215 |   0.46865  |    0.531305 |                24.4437 | False                  | val               | False                               | False       |      684870           |                   2.03515e+06 | True                        |

## HGB Convergence Closure

|   selected_budget_on_validation |   selected_threshold_on_validation |   iter80_validation_policy_pnl |   best_validation_policy_pnl |   best_minus_iter80_val_pnl | best_at_max_grid_boundary   | convergence_closed_for_candidate_decision   | selection_split   | test_new3_new4_used_for_selection   | new5_used   |
|--------------------------------:|-----------------------------------:|-------------------------------:|-----------------------------:|----------------------------:|:----------------------------|:--------------------------------------------|:------------------|:------------------------------------|:------------|
|                             120 |                           0.224194 |                    5.43826e+07 |                  5.45502e+07 |                      167650 | False                       | True                                        | val               | False                               | False       |

Decision rule: ML can be selected only if the exact convergence-selected configuration beats matched random 95p, remains positive and competitive against the queue rule in causal-fill sequential evaluation, and convergence is closed under validation-only iteration selection.

Allowed claim: TXF-only ML candidate completion patch selected one exact candidate under frozen development protocol.
Forbidden claims: real-world profitability, true exchange replay, FPGA PnL improvement, TCN/PHC result, or any NEW5 result.
