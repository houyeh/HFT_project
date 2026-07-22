# PHC Dale Ablation Report

Final status: `dale_hurts_performance`.

## What is Dale's Law?

Dale's Law constrains each presynaptic hidden unit to have outgoing synaptic weights of one fixed sign: excitatory or inhibitory.

## Implementation

`message = z @ W_eff.T + bias`, with `W_eff = softplus(raw_weight) * sign_vector`. The sign vector is applied to columns of `W_eff`, so each presynaptic unit's outgoing fan-out has one sign. Ratio is 80% excitatory / 20% inhibitory.

This is a controlled Dale's Law ablation only. It is not full PHCSSM, not ALIF, not STP, not STDP, not true exchange replay, and not real-world profitability proof.

Caveat: the HGB comparison is loaded as a frozen Phase 04B/05A artifact. The local runtime emits a scikit-learn version warning when deserializing that joblib; no HGB retraining or retuning is performed in this phase.

## Sign Audit

|   hidden_dim |   presynaptic_unit | unit_type   |   sign_value |   outgoing_weight_min |   outgoing_weight_max |   sign_violations | constraint_enforced_during_training   |   excitatory_units |   inhibitory_units |
|-------------:|-------------------:|:------------|-------------:|----------------------:|----------------------:|------------------:|:--------------------------------------|-------------------:|-------------------:|
|           16 |                  0 | excitatory  |            1 |              0.575701 |              0.815184 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  1 | excitatory  |            1 |              0.555551 |              0.858377 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  2 | excitatory  |            1 |              0.572034 |              0.832651 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  3 | excitatory  |            1 |              0.5714   |              0.802735 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  4 | excitatory  |            1 |              0.562696 |              0.804973 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  5 | excitatory  |            1 |              0.575315 |              0.794936 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  6 | excitatory  |            1 |              0.578134 |              0.822722 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  7 | excitatory  |            1 |              0.606655 |              0.803075 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  8 | excitatory  |            1 |              0.561293 |              0.827655 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  9 | excitatory  |            1 |              0.579217 |              0.807626 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 10 | excitatory  |            1 |              0.560044 |              0.822426 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 11 | excitatory  |            1 |              0.584952 |              0.837847 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 12 | excitatory  |            1 |              0.494075 |              0.86131  |                 0 | True                                  |                 13 |                  3 |
|           16 |                 13 | inhibitory  |           -1 |             -0.817954 |             -0.563712 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 14 | inhibitory  |           -1 |             -0.835665 |             -0.538733 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 15 | inhibitory  |           -1 |             -0.878153 |             -0.548977 |                 0 | True                                  |                 13 |                  3 |

## OOS Evaluation

| split   | policy_id                                     |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   negative_rate |   max_drawdown |   positive_session_count |   session_count |   seed_selected_on_val |   threshold_selected_on_val |   new3_used_for_selection |   new4_used_for_selection |   new5_used_for_selection |
|:--------|:----------------------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|----------------:|---------------:|-------------------------:|----------------:|-----------------------:|----------------------------:|--------------------------:|--------------------------:|--------------------------:|
| val     | side_by_queue_imbalance_abs_0p1               |        20 |         82960 |                   68981 |         0.831497 |       34027 |       34954 |            13979 |            1.23714e+06 |               17.9346 |   0.426451 |        0.573491 |          18000 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| val     | ml_hgb_regressor_shallow_convergence_selected |        20 |         82960 |                   61679 |         0.743479 |       32479 |       29200 |            21281 |            1.67168e+06 |               27.1028 |   0.459816 |        0.540119 |          16255 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| val     | phc_lite                                      |        20 |         82960 |                   63766 |         0.768635 |       32177 |       31589 |            19194 |            1.92717e+06 |               30.2225 |   0.469451 |        0.530486 |          15585 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new3    | side_by_queue_imbalance_abs_0p1               |        20 |        176754 |                  144745 |         0.818907 |       71697 |       73048 |            32009 |            2.95477e+06 |               20.4136 |   0.446737 |        0.553242 |           6385 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new3    | ml_hgb_regressor_shallow_convergence_selected |        20 |        176754 |                  134210 |         0.759304 |       70944 |       63266 |            42544 |            4.01058e+06 |               29.8829 |   0.478966 |        0.521019 |           4540 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new3    | phc_lite                                      |        20 |        176754 |                  138261 |         0.782223 |       70340 |       67921 |            38493 |            5.02152e+06 |               36.3191 |   0.501045 |        0.49894  |           4855 |                        6 |               6 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new4    | side_by_queue_imbalance_abs_0p1               |        20 |         68620 |                   56147 |         0.818231 |       27616 |       28531 |            12473 |            1.00391e+06 |               17.88   |   0.438777 |        0.561205 |           6835 |                        4 |               4 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new4    | ml_hgb_regressor_shallow_convergence_selected |        20 |         68620 |                   51753 |         0.754197 |       27325 |       24428 |            16867 |            1.39542e+06 |               26.9631 |   0.469615 |        0.530365 |           6625 |                        4 |               4 |                    nan |                         nan |                       nan |                       nan |                       nan |
| new4    | phc_lite                                      |        20 |         68620 |                   53169 |         0.774832 |       27155 |       26014 |            15451 |            1.7625e+06  |               33.1489 |   0.490605 |        0.509357 |           5020 |                        4 |               4 |                    nan |                         nan |                       nan |                       nan |                       nan |
| val     | phc_dale                                      |        20 |         82960 |                   63237 |         0.762259 |       31811 |       31426 |            19723 |            1.89099e+06 |               29.9032 |   0.470642 |        0.52931  |          16175 |                        6 |               6 |                      2 |                           0 |                         0 |                         0 |                         0 |
| new3    | phc_dale                                      |        20 |        176754 |                  136429 |         0.771858 |       69316 |       67113 |            40325 |            4.97038e+06 |               36.432  |   0.503779 |        0.496214 |           4025 |                        6 |               6 |                      2 |                           0 |                         0 |                         0 |                         0 |
| new4    | phc_dale                                      |        20 |         68620 |                   52331 |         0.76262  |       26794 |       25537 |            16289 |            1.71635e+06 |               32.798  |   0.491735 |        0.508246 |           5120 |                        4 |               4 |                      2 |                           0 |                         0 |                         0 |                         0 |

## Matched Random

| split   | policy_id   |   model_total_excess_pnl_H20 |       random_mean |   random_std |        random_95p |        random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   |   n_trials | matching                                                    |   seed_selected_on_val |   threshold_selected_on_val |
|:--------|:------------|-----------------------------:|------------------:|-------------:|------------------:|------------------:|---------------------------------------------:|:----------------------------|:-------------------|-----------:|:------------------------------------------------------------|-----------------------:|----------------------------:|
| val     | phc_dale    |                  1.89099e+06 | -949253           |      28323.1 | -903757           | -879425           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 |
| new3    | phc_dale    |                  4.97038e+06 |      -2.04456e+06 |      45873.6 |      -1.96819e+06 |      -1.94489e+06 |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 |
| new4    | phc_dale    |                  1.71635e+06 | -783142           |      28599.9 | -736850           | -720631           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 |

## Causal-fill Sequential

| split   | policy_id                                     |   horizon | fill_model             |   inventory_cap |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   |   new5_used |   fee_multiplier |   new5_used_for_selection |
|:--------|:----------------------------------------------|----------:|:-----------------------|----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|------------:|-----------------:|--------------------------:|
| val     | side_by_queue_imbalance_abs_0p1               |        20 | conservative_l2_stress |               1 |                  103260 |         6416 |                       0 |                   1 |           2875 |   0.432824 |    0.567176 |                16.0941 | False                  |           0 |              nan |                       nan |
| val     | ml_hgb_regressor_shallow_convergence_selected |        20 | conservative_l2_stress |               1 |                  142435 |         6231 |                       0 |                   1 |           1845 |   0.45723  |    0.54277  |                22.8591 | False                  |           0 |              nan |                       nan |
| val     | phc_lite                                      |        20 | conservative_l2_stress |               1 |                  170310 |         6746 |                       0 |                   1 |           2365 |   0.464572 |    0.535428 |                25.2461 | False                  |           0 |              nan |                       nan |
| new3    | side_by_queue_imbalance_abs_0p1               |        20 | conservative_l2_stress |               1 |                  303700 |        13821 |                    2113 |                   1 |           3615 |   0.458939 |    0.540988 |                21.9738 | False                  |           0 |              nan |                       nan |
| new3    | ml_hgb_regressor_shallow_convergence_selected |        20 | conservative_l2_stress |               1 |                  411090 |        13955 |                    1993 |                   1 |           2615 |   0.484558 |    0.515371 |                29.4583 | False                  |           0 |              nan |                       nan |
| new3    | phc_lite                                      |        20 | conservative_l2_stress |               1 |                  488870 |        15123 |                    2344 |                   1 |           3335 |   0.498578 |    0.501356 |                32.3263 | False                  |           0 |              nan |                       nan |
| new4    | side_by_queue_imbalance_abs_0p1               |        20 | conservative_l2_stress |               1 |                   72415 |         3820 |                    2216 |                   1 |           3560 |   0.442932 |    0.557068 |                18.9568 | False                  |           0 |              nan |                       nan |
| new4    | ml_hgb_regressor_shallow_convergence_selected |        20 | conservative_l2_stress |               1 |                   97680 |         3629 |                    2479 |                   1 |           2105 |   0.472306 |    0.527694 |                26.9165 | False                  |           0 |              nan |                       nan |
| new4    | phc_lite                                      |        20 | conservative_l2_stress |               1 |                  121765 |         4110 |                    2574 |                   1 |           1925 |   0.485401 |    0.514599 |                29.6265 | False                  |           0 |              nan |                       nan |
| val     | phc_dale                                      |        20 | conservative_l2_stress |               1 |                  160875 |         6695 |                       0 |                   1 |           2745 |   0.461837 |    0.538163 |                24.0291 | False                  |         nan |                1 |                         0 |
| new3    | phc_dale                                      |        20 | conservative_l2_stress |               1 |                  473440 |        14805 |                    2253 |                   1 |           2760 |   0.500169 |    0.499764 |                31.9784 | False                  |         nan |                1 |                         0 |
| new4    | phc_dale                                      |        20 | conservative_l2_stress |               1 |                  115410 |         3927 |                    2523 |                   1 |           2045 |   0.49096  |    0.50904  |                29.3888 | False                  |         nan |                1 |                         0 |

## Seed Stability

|   seed | split   |   best_epoch |   threshold |         pnl |   selected_action_count | positive   | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|-------:|:--------|-------------:|------------:|------------:|------------------------:|:-----------|:--------------------------|:--------------------------|:--------------------------|
|      0 | val     |           50 |     1.88522 | 1.88294e+06 |                   62220 | True       | False                     | False                     | False                     |
|      0 | new3    |           50 |     1.88522 | 4.85693e+06 |                  134600 | True       | False                     | False                     | False                     |
|      0 | new4    |           50 |     1.88522 | 1.6976e+06  |                   51622 | True       | False                     | False                     | False                     |
|      1 | val     |           49 |     0       | 1.86216e+06 |                   63047 | True       | False                     | False                     | False                     |
|      1 | new3    |           49 |     0       | 4.87784e+06 |                  137300 | True       | False                     | False                     | False                     |
|      1 | new4    |           49 |     0       | 1.6875e+06  |                   52735 | True       | False                     | False                     | False                     |
|      2 | val     |           49 |     0       | 1.89099e+06 |                   63237 | True       | False                     | False                     | False                     |
|      2 | new3    |           49 |     0       | 4.97038e+06 |                  136429 | True       | False                     | False                     | False                     |
|      2 | new4    |           49 |     0       | 1.71635e+06 |                   52331 | True       | False                     | False                     | False                     |

## Oracle Capture

| split   | policy_id                                     |   policy_selected_count |   policy_pnl |   matched_count_oracle_pnl |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | new5_used   |
|:--------|:----------------------------------------------|------------------------:|-------------:|---------------------------:|----------------:|------------------:|:----------------------------|:------------|
| val     | side_by_queue_imbalance_abs_0p1               |                   68981 |  1.23714e+06 |                7.64484e+06 |        0.161827 |       6.4077e+06  | True                        | False       |
| val     | ml_hgb_regressor_shallow_convergence_selected |                   61679 |  1.67168e+06 |                7.75438e+06 |        0.215578 |       6.0827e+06  | True                        | False       |
| val     | phc_lite                                      |                   63766 |  1.92717e+06 |                7.72307e+06 |        0.249534 |       5.7959e+06  | True                        | False       |
| val     | phc_dale                                      |                   63237 |  1.89099e+06 |                7.731e+06   |        0.244598 |       5.84002e+06 | True                        | False       |
| new3    | side_by_queue_imbalance_abs_0p1               |                  144745 |  2.95477e+06 |                1.8744e+07  |        0.157638 |       1.57892e+07 | True                        | False       |
| new3    | ml_hgb_regressor_shallow_convergence_selected |                  134210 |  4.01058e+06 |                1.8902e+07  |        0.212178 |       1.48914e+07 | True                        | False       |
| new3    | phc_lite                                      |                  138261 |  5.02152e+06 |                1.88412e+07 |        0.266517 |       1.38197e+07 | True                        | False       |
| new3    | phc_dale                                      |                  136429 |  4.97038e+06 |                1.88687e+07 |        0.263419 |       1.38983e+07 | True                        | False       |
| new4    | side_by_queue_imbalance_abs_0p1               |                   56147 |  1.00391e+06 |                7.32802e+06 |        0.136996 |       6.32412e+06 | True                        | False       |
| new4    | ml_hgb_regressor_shallow_convergence_selected |                   51753 |  1.39542e+06 |                7.39394e+06 |        0.188725 |       5.99852e+06 | True                        | False       |
| new4    | phc_lite                                      |                   53169 |  1.7625e+06  |                7.3727e+06  |        0.239057 |       5.6102e+06  | True                        | False       |
| new4    | phc_dale                                      |                   52331 |  1.71635e+06 |                7.38526e+06 |        0.232402 |       5.66892e+06 | True                        | False       |

## Interpretation

Dale sign audit passed: `True`.
Matched random passed all splits: `True`.
Seed stability positive all splits/seeds: `True`.
Validation action-level matches/improves PHC-lite v1: `False`.
NEW3/NEW4 action-level matches/improves PHC-lite v1: `False`.
Causal-fill sequential matches/improves PHC-lite v1 on val/NEW3/NEW4: `False`.

Forbidden claims remain: Dale variant is not full PHCSSM, not a fresh NEW5 holdout result, not real-world profitability proof, not true exchange replay, and not FPGA PnL proof.
