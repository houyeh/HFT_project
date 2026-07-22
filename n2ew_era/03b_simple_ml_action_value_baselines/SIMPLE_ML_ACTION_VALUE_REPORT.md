# Simple ML Action-Value Baselines Report

Final classification: `ml_signal_exists_but_no_strategy_gain`.

No deep MLP, TCN, or massive grid search was used. No threshold, model, or feature decision was selected on `test`, `NEW3`, or `NEW4`.

## 1. Input Contract

Phase 03B used only verified causal L2 features plus rebuilt causal dynamic features. Raw absolute price columns were forbidden by Phase 03A.6 and are absent from the ML feature matrix.

Feature schema: `feature_schema_03b.json`.

## 2. Simple Model Training

| model_id               | model_family   |   train_rows |   val_rows |   feature_count |   val_rmse_vs_excess |   val_mae_vs_excess |   val_pr_auc_positive_excess |   val_roc_auc_positive_excess |   val_spearman_score_vs_excess | test_new3_new4_used_for_training   |
|:-----------------------|:---------------|-------------:|-----------:|----------------:|---------------------:|--------------------:|-----------------------------:|------------------------------:|-------------------------------:|:-----------------------------------|
| ridge_l2               | regressor      |       900000 |    1690431 |              18 |              240.777 |             74.3852 |                     0.328145 |                      0.541892 |                      0.244506  | False                              |
| lasso_l1               | regressor      |       900000 |    1690431 |              18 |              240.777 |             74.385  |                     0.328149 |                      0.541897 |                      0.244502  | False                              |
| logistic_l2            | classifier     |       900000 |    1690431 |              18 |              nan     |            nan      |                     0.430893 |                      0.778233 |                     -0.0912874 | False                              |
| logistic_l1            | classifier     |       900000 |    1690431 |              18 |              nan     |            nan      |                     0.430893 |                      0.778233 |                     -0.0912874 | False                              |
| hgb_regressor_shallow  | regressor      |       900000 |    1690431 |              18 |              239.726 |             72.416  |                     0.374129 |                      0.578924 |                      0.297754  | False                              |
| hgb_classifier_shallow | classifier     |       900000 |    1690431 |              18 |              nan     |            nan      |                     0.474425 |                      0.798055 |                     -0.0630614 | False                              |

## 3. Validation-Only Threshold Selection

Selected model: `hgb_regressor_shallow`.

Selected threshold: `0.0`.

Selection split: `val`; selection metric: `validation_total_excess_pnl_H20`.

| split   | policy_id                 | model_id               |   horizon |   threshold |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_excess_pnl |   win_rate |   loss_rate |   max_drawdown | selection_split   | test_new3_new4_used_for_selection   |
|:--------|:--------------------------|:-----------------------|----------:|------------:|--------------:|------------------------:|-----------------:|------------:|------------:|-------------------:|-----------:|------------:|---------------:|:------------------|:------------------------------------|
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |    0        |        563477 |                  411760 |         0.730749 |      202665 |      209095 |        1.2159e+07  |   0.468593 |    0.531368 |         186385 | val               | False                               |
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |   -3.0818   |        563477 |                  450488 |         0.799479 |      221051 |      229437 |        1.20112e+07 |   0.460842 |    0.539118 |         188650 | val               | False                               |
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |    7.9513   |        563477 |                  337899 |         0.599668 |      165462 |      172437 |        1.18838e+07 |   0.483943 |    0.516033 |         123220 | val               | False                               |
| val     | ml_action_value_threshold | hgb_classifier_shallow |        20 |    0.4      |        563477 |                  355303 |         0.630555 |      175510 |      179793 |        1.15092e+07 |   0.483371 |    0.516601 |         123520 | val               | False                               |
| val     | ml_action_value_threshold | hgb_classifier_shallow |        20 |    0.40629  |        563477 |                  337112 |         0.598271 |      166346 |      170766 |        1.14785e+07 |   0.487295 |    0.512675 |          92790 | val               | False                               |
| val     | ml_action_value_threshold | hgb_classifier_shallow |        20 |    0.374326 |        563477 |                  440136 |         0.781107 |      218159 |      221977 |        1.14638e+07 |   0.465915 |    0.534051 |         169495 | val               | False                               |
| val     | ml_action_value_threshold | hgb_classifier_shallow |        20 |    0.336856 |        563477 |                  517174 |         0.917826 |      256974 |      260200 |        1.09462e+07 |   0.450342 |    0.549627 |         217845 | val               | False                               |
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |  -13.6688   |        563477 |                  549577 |         0.975332 |      271194 |      278383 |        1.08359e+07 |   0.443299 |    0.556666 |         221010 | val               | False                               |
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |  -80.1367   |        563477 |                  563477 |         1        |      278029 |      285448 |        1.06365e+07 |   0.440644 |    0.55932  |         220940 | val               | False                               |
| val     | ml_action_value_threshold | hgb_regressor_shallow  |        20 |  -50.1448   |        563477 |                  563477 |         1        |      278029 |      285448 |        1.06365e+07 |   0.440644 |    0.55932  |         220940 | val               | False                               |

## 4. Frozen OOS Evaluation

| split   | policy_id                                | model_id              |   horizon |   threshold |   state_count |   selected_action_count |   bid_count |   ask_count |   total_excess_pnl |   max_drawdown |   selection_rate |
|:--------|:-----------------------------------------|:----------------------|----------:|------------:|--------------:|------------------------:|------------:|------------:|-------------------:|---------------:|-----------------:|
| test    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 |           0 |       1166321 |                  870509 |      428530 |      441979 |        2.07099e+07 |          26375 |         0.746372 |
| new4    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 |           0 |       2196640 |                 1621093 |      802623 |      818470 |        2.70813e+07 |          68420 |         0.737988 |
| new3    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 |           0 |       5657362 |                 4185481 |     2078114 |     2107367 |        7.68689e+07 |          72175 |         0.739829 |

## 5. Baseline Controls

| split   | policy_id                       |   horizon |   state_count |   selected_action_count |   bid_count |   ask_count |   total_excess_pnl |     max_drawdown |   selection_rate |
|:--------|:--------------------------------|----------:|--------------:|------------------------:|------------:|------------:|-------------------:|-----------------:|-----------------:|
| test    | no_quote                        |        20 |       1166321 |                       0 |           0 |           0 |        0           |      0           |         0        |
| test    | old_bid_only_spread_ge_1        |        20 |       1166321 |                 1166321 |     1166321 |           0 |       -1.59608e+07 |      1.09752e+07 |         1        |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |       1166321 |                  972497 |      480551 |      491946 |        1.63357e+07 |  44890           |         0.833816 |
| new4    | no_quote                        |        20 |       2196640 |                       0 |           0 |           0 |        0           |      0           |         0        |
| new4    | old_bid_only_spread_ge_1        |        20 |       2196640 |                 2196640 |     2196640 |           0 |       -3.18859e+07 |      1.50318e+07 |         1        |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |       2196640 |                 1799777 |      887768 |      912009 |        2.9901e+07  |  85710           |         0.819332 |
| new3    | no_quote                        |        20 |       5657362 |                       0 |           0 |           0 |        0           |      0           |         0        |
| new3    | old_bid_only_spread_ge_1        |        20 |       5657362 |                 5657362 |     5657362 |           0 |       -8.28546e+07 |      2.0007e+07  |         1        |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 |       5657362 |                 4632107 |     2292063 |     2340044 |        8.67324e+07 | 115340           |         0.818775 |

## 6. Matched Random and Queue-Rule Hurdles

| split   | policy_id                                | model_id              |   model_total_excess_pnl |   queue_rule_total_excess_pnl |   old_bid_only_total_excess_pnl |   matched_random_mean |   matched_random_95p |   matched_random_99p |   empirical_p_value_fraction_random_ge_model | beats_matched_random_95p   | beats_queue_imbalance_rule   |   random_trials | random_method                                                   | test_new3_new4_used_for_selection   |
|:--------|:-----------------------------------------|:----------------------|-------------------------:|------------------------------:|--------------------------------:|----------------------:|---------------------:|---------------------:|---------------------------------------------:|:---------------------------|:-----------------------------|----------------:|:----------------------------------------------------------------|:------------------------------------|
| new3    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              7.68689e+07 |                   8.67324e+07 |                    -8.28546e+07 |          -6.27973e+07 |         -6.23779e+07 |         -6.21954e+07 |                                            0 | True                       | False                        |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |
| new4    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              2.70813e+07 |                   2.9901e+07  |                    -3.18859e+07 |          -2.43171e+07 |         -2.40416e+07 |         -2.39159e+07 |                                            0 | True                       | False                        |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |
| test    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              2.07099e+07 |                   1.63357e+07 |                    -1.59608e+07 |          -1.30839e+07 |         -1.2924e+07  |         -1.28802e+07 |                                            0 | True                       | True                         |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |

Matched random uses 1000 finite-population random trials matched by selected count, session, and side. This is a hurdle for signal value, not a strategy tuning mechanism.

If `empirical_p_value_fraction_random_ge_model` is displayed as `0`, interpret it as: in 1000 matched random trials, no random trial exceeded the selected ML policy; therefore empirical p-value `< 0.001`, not exactly zero.

## 7. Causal-Fill Sequential Proxy

| split   | policy_id                                | model_id              |   horizon | fill_model             |   inventory_cap |   fee_multiplier |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   | test_new3_new4_used_for_selection   |
|:--------|:-----------------------------------------|:----------------------|----------:|:-----------------------|----------------:|-----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|:------------------------------------|
| test    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 | conservative_l2_stress |               1 |                1 |             1.0108e+06  |        52473 |                   35426 |                   1 |           3845 |   0.416252 |    0.583748 |                19.2633 | False                  | False                               |
| new4    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 | conservative_l2_stress |               1 |                1 |             1.59584e+06 |       113448 |                   70466 |                   1 |          13845 |   0.432339 |    0.567626 |                14.0667 | False                  | False                               |
| new3    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |        20 | conservative_l2_stress |               1 |                1 |             4.13092e+06 |       297351 |                  190094 |                   1 |           9555 |   0.431315 |    0.568665 |                13.8924 | False                  | False                               |

The sequential fill proxy uses current L2 state only, conservative L2 stress fill probabilities, inventory cap = 1, fee multiplier = 1, and H20 scheduled exit.

## 8. Interpretation

The ML model shows some evidence of signal value, but it does not clear every deployment-style hurdle against the queue imbalance rule and causal-fill sequential proxy.

The correct interpretation is signal discovery, not final strategy replacement.

Allowed wording: `simple causal L2 ML baseline`, `validation-selected action-value proxy`, `frozen OOS hurdle result`.

Forbidden wording: real-world profitability, true queue replay, FPGA-proven PnL improvement, or test-tuned threshold success.
