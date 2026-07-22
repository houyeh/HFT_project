# Phase 03B ML Training Diagnostics Report

Final diagnostic conclusion: `implementation_issue_requires_fix`.

No TCN, large MLP, or frozen-OOS tuning was performed. Diagnostic ablations select thresholds on validation only.

## 1. Bid/Ask Pairing Hard Asserts

| split   |   parts | status   |
|:--------|--------:|:---------|
| new3    |       6 | PASS     |
| new4    |       4 | PASS     |
| test    |       2 | PASS     |
| train   |       6 | PASS     |
| val     |       1 | PASS     |

## 2. Training Convergence Diagnostics

| model_id               | score_kind   |   train_rows |   val_rows | n_iter   |   max_iter | converged_flag   |   hgb_do_early_stopping |   hgb_train_score_len |   hgb_validation_score_len |   train_pr_auc |   val_pr_auc |   train_roc_auc |   val_roc_auc |   train_log_loss |   val_log_loss |   train_rmse |   val_rmse |   train_mae |   val_mae |   train_spearman_vs_excess |   val_spearman_vs_excess |
|:-----------------------|:-------------|-------------:|-----------:|:---------|-----------:|:-----------------|------------------------:|----------------------:|---------------------------:|---------------:|-------------:|----------------:|--------------:|-----------------:|---------------:|-------------:|-----------:|------------:|----------:|---------------------------:|-------------------------:|
| ridge_l2               | regression   |       900000 |    1690431 | nan      |        nan | True             |                     nan |                   nan |                        nan |       0.330056 |     0.328145 |        0.542423 |      0.541892 |       nan        |     nan        |      150.016 |    240.777 |     76.974  |   74.3852 |                  0.2464    |                0.244506  |
| lasso_l1               | regression   |       900000 |    1690431 | 2380     |       2500 | True             |                     nan |                   nan |                        nan |       0.330059 |     0.328149 |        0.542428 |      0.541897 |       nan        |     nan        |      150.016 |    240.777 |     76.9737 |   74.385  |                  0.246397  |                0.244502  |
| logistic_l2            | probability  |       900000 |    1690431 | [15]     |        120 | True             |                     nan |                   nan |                        nan |       0.436047 |     0.430893 |        0.779963 |      0.778233 |         0.43481  |       0.433501 |      nan     |    nan     |    nan      |  nan      |                 -0.0840813 |               -0.0912874 |
| logistic_l1            | probability  |       900000 |    1690431 | [15]     |        120 | True             |                     nan |                   nan |                        nan |       0.436047 |     0.430893 |        0.779963 |      0.778233 |         0.43481  |       0.433501 |      nan     |    nan     |    nan      |  nan      |                 -0.0840813 |               -0.0912874 |
| hgb_regressor_shallow  | regression   |       900000 |    1690431 | 80       |         80 | False            |                       1 |                    81 |                         81 |       0.385693 |     0.374129 |        0.586149 |      0.578924 |       nan        |     nan        |      147.836 |    239.726 |     74.9171 |   72.416  |                  0.298262  |                0.297754  |
| hgb_classifier_shallow | probability  |       900000 |    1690431 | 80       |         80 | False            |                       1 |                    81 |                         81 |       0.488424 |     0.474425 |        0.802533 |      0.798055 |         0.413471 |       0.413855 |      nan     |    nan     |    nan      |  nan      |                 -0.0527335 |               -0.0630614 |

All simple model convergence flags pass: `False`.

## 3. Confusion and Oracle Matrices

Full confusion matrices are in `results/oracle_action_confusion.csv`. They compare policy quote/no-quote against any positive quote availability and policy action against the oracle argmax over bid/ask/zero.

## 4. No-Quote Training Ablation

| variant                  |        new3 |        new4 |        test |
|:-------------------------|------------:|------------:|------------:|
| train_bid_ask_only       | 7.98994e+07 | 2.82022e+07 | 2.05091e+07 |
| train_with_no_quote_rows | 7.90727e+07 | 2.76598e+07 | 2.05484e+07 |

No-quote zero-row diagnosis: `not_primary_bottleneck`.

## 5. Train Sample Stability

|   seed |        new3 |        new4 |        test |
|-------:|------------:|------------:|------------:|
|      0 | 7.77732e+07 | 2.71813e+07 | 2.06076e+07 |
|      1 | 7.7787e+07  | 2.73529e+07 | 2.0621e+07  |
|      2 | 7.82569e+07 | 2.7498e+07  | 2.06934e+07 |

Seed instability detected at CV > 25%: `False`.

## 6. Exact Sampled Matched Random

| split   |   model_total_excess_pnl |   random_mean |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model |   random_trials |
|:--------|-------------------------:|--------------:|-------------:|-------------:|---------------------------------------------:|----------------:|
| new3    |              7.68689e+07 |  -6.2773e+07  | -6.23333e+07 | -6.22005e+07 |                                            0 |            1000 |
| new4    |              2.70813e+07 |  -2.43255e+07 | -2.40391e+07 | -2.39144e+07 |                                            0 |            1000 |
| test    |              2.07099e+07 |  -1.30883e+07 | -1.29307e+07 | -1.28742e+07 |                                            0 |            1000 |

Selected ML beats exact sampled matched random 95p on all frozen OOS splits: `True`.

If empirical p-value displays as 0, read it as no random trial exceeded the model in 1000 trials, i.e. empirical p-value `< 0.001`, not exactly zero probability.

## 7. HGB Feature Importance

| metric_type                       | feature                    |   baseline_spearman |   permuted_spearman |   importance_spearman_drop |   sample_rows |   bucket |   mean_score |   mean_excess_pnl |   base_mean_excess_pnl |   economic_lift |
|:----------------------------------|:---------------------------|--------------------:|--------------------:|---------------------------:|--------------:|---------:|-------------:|------------------:|-----------------------:|----------------:|
| permutation_importance_val_sample | side_queue_imbalance       |            0.289036 |            0.108677 |                0.180358    |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | side_imbalance_change_past |            0.289036 |            0.191723 |                0.0973132   |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | spread_change_past         |            0.289036 |            0.272592 |                0.0164441   |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | top_depth                  |            0.289036 |            0.28277  |                0.00626556  |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | spread_ticks               |            0.289036 |            0.285576 |                0.00345946  |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | side_deep_imbalance        |            0.289036 |            0.286275 |                0.00276083  |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | volatility_past_50         |            0.289036 |            0.287788 |                0.0012477   |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | ask_sz_1                   |            0.289036 |            0.288891 |                0.000144446 |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | queue_imbalance            |            0.289036 |            0.288897 |                0.000138837 |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | inter_arrival_ms           |            0.289036 |            0.288976 |                6.01818e-05 |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | bid_sz_1                   |            0.289036 |            0.288997 |                3.87077e-05 |        200000 |      nan |          nan |               nan |                    nan |             nan |
| permutation_importance_val_sample | imbalance_change_past      |            0.289036 |            0.289016 |                1.95542e-05 |        200000 |      nan |          nan |               nan |                    nan |             nan |

## 8. Incremental Gain Diagnosis

| split   | policy_id                                | model_id              |   model_total_excess_pnl |   queue_rule_total_excess_pnl |   old_bid_only_total_excess_pnl |   matched_random_mean |   matched_random_95p |   matched_random_99p |   empirical_p_value_fraction_random_ge_model | beats_matched_random_95p   | beats_queue_imbalance_rule   |   random_trials | random_method                                                   | test_new3_new4_used_for_selection   |
|:--------|:-----------------------------------------|:----------------------|-------------------------:|------------------------------:|--------------------------------:|----------------------:|---------------------:|---------------------:|---------------------------------------------:|:---------------------------|:-----------------------------|----------------:|:----------------------------------------------------------------|:------------------------------------|
| new3    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              7.68689e+07 |                   8.67324e+07 |                    -8.28546e+07 |          -6.27973e+07 |         -6.23779e+07 |         -6.21954e+07 |                                            0 | True                       | False                        |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |
| new4    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              2.70813e+07 |                   2.9901e+07  |                    -3.18859e+07 |          -2.43171e+07 |         -2.40416e+07 |         -2.39159e+07 |                                            0 | True                       | False                        |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |
| test    | phase03b_selected_simple_ml_action_value | hgb_regressor_shallow |              2.07099e+07 |                   1.63357e+07 |                    -1.59608e+07 |          -1.30839e+07 |         -1.2924e+07  |         -1.28802e+07 |                                            0 | True                       | True                         |            1000 | finite_population_normal_1000_trials_matched_count_session_side | False                               |

At least one diagnostic indicates a possible implementation or training-design issue. Phase 03C should not proceed as a model-expansion phase until the flagged issue is fixed with a targeted root-cause audit.
