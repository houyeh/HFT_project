# PHC STP-like Final Decision Report

Final status: `stp_inconclusive_due_to_epoch_cap`.

Decision rule: STP is not adopted unless it matches or beats frozen PHC-lite v1 on validation, NEW3/NEW4 action-level PnL, causal-fill sequential PnL, and oracle capture. NEW3/NEW4/NEW5 are not used for selection. Because the extended run improves action-level/OOS/oracle metrics but still misses causal-fill on NEW4 and hits max_epoch for all seeds, the model remains not adopted.

## Code Audit

| audit_item                               | status   | evidence                                                                                          | file_path                                                                                                       |
|:-----------------------------------------|:---------|:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| train_val_new3_new4_split_handling       | PASS     | Training uses train payload; evaluation loops over val/new3/new4.                                 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| no_new5_usage                            | PASS     | No NEW5 sequence loading found; selected JSON marks new5_used_for_selection=false.                | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| validation_only_epoch_selection          | PASS     | Checkpoint selected by validation policy excess PnL.                                              | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| validation_only_threshold_seed_selection | PASS     | Threshold grid CSV contains only selection_split=val and false OOS selection flags.               | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| feature_list                             | PASS     | Model feature_dim is len(p05a.PHC_FEATURES); inherited forbidden-feature contract from Phase 05A. | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| target_excess_pnl_H20                    | PASS     | Selected policy target is excess_pnl_H20; metrics use Phase 05A horizon constant H20.             | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| matched_random_implementation            | PASS     | Matched random delegates to Phase 05A count/session/side/spread-bucket matcher.                   | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| causal_fill_implementation               | PASS     | Causal-fill uses train-derived norms, stable uniforms, and no future-PnL fill decision.           | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| oracle_capture_computation               | PASS     | Oracle capture explicitly uses future labels and is diagnostic only.                              | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| stp_gain_definition                      | PASS     | gain_t = sigmoid(W_g encoded_t + U_g state_{t-1}) exists in model source.                         | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\models\phc_lite_stp.py |
| not_full_biological_stp                  | PASS     | Model docstring states it is not full biological STP.                                             | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\models\phc_lite_stp.py |
| new5_not_in_outputs                      | PASS     | OOS/oracle output splits exclude NEW5.                                                            | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |

## Original 50 Epoch Action-level Comparison

| split   | policy_id   |   total_excess_pnl_H20 |   selected_action_count |   win_rate |   max_drawdown |
|:--------|:------------|-----------------------:|------------------------:|-----------:|---------------:|
| val     | phc_lite    |            1.92717e+06 |                   63766 |   0.469451 |          15585 |
| new3    | phc_lite    |            5.02152e+06 |                  138261 |   0.501045 |           4855 |
| new4    | phc_lite    |            1.7625e+06  |                   53169 |   0.490605 |           5020 |
| val     | phc_stp     |            1.9086e+06  |                   64670 |   0.467682 |          16115 |
| new3    | phc_stp     |            4.88642e+06 |                  140115 |   0.497748 |           4500 |
| new4    | phc_stp     |            1.68752e+06 |                   53794 |   0.486281 |           7420 |

## Original 50 Epoch Causal-fill Comparison

| split   | policy_id   |   sequential_excess_pnl |   fill_count |   win_rate |   max_drawdown |
|:--------|:------------|------------------------:|-------------:|-----------:|---------------:|
| val     | phc_lite    |                  170310 |         6746 |   0.464572 |           2365 |
| new3    | phc_lite    |                  488870 |        15123 |   0.498578 |           3335 |
| new4    | phc_lite    |                  121765 |         4110 |   0.485401 |           1925 |
| val     | phc_stp     |                  164020 |         6852 |   0.459428 |           2920 |
| new3    | phc_stp     |                  477230 |        15379 |   0.496976 |           2935 |
| new4    | phc_stp     |                  111480 |         3949 |   0.481134 |           2530 |

## Original 50 Epoch Oracle Capture

| split   | policy_id   |   policy_pnl |   capture_ratio |   opportunity_gap |
|:--------|:------------|-------------:|----------------:|------------------:|
| val     | phc_lite    |  1.92717e+06 |        0.249534 |       5.7959e+06  |
| val     | phc_stp     |  1.9086e+06  |        0.247564 |       5.80092e+06 |
| new3    | phc_lite    |  5.02152e+06 |        0.266517 |       1.38197e+07 |
| new3    | phc_stp     |  4.88642e+06 |        0.259731 |       1.3927e+07  |
| new4    | phc_lite    |  1.7625e+06  |        0.239057 |       5.6102e+06  |
| new4    | phc_stp     |  1.68752e+06 |        0.229179 |       5.6758e+06  |

## Extended Epoch Seed Summary

|   seed |   best_epoch_t0 |   best_val_pnl_at_threshold0 |   best_val_pnl_epoch_le_50 |   best_val_pnl_epoch_gt_50 | improved_after_epoch50   | max_epoch_hit   |
|-------:|----------------:|-----------------------------:|---------------------------:|---------------------------:|:-------------------------|:----------------|
|      0 |              75 |                  1.94044e+06 |                1.86789e+06 |                1.94044e+06 | True                     | True            |
|      1 |              75 |                  1.9346e+06  |                1.87642e+06 |                1.9346e+06  | True                     | True            |
|      2 |              78 |                  2.03234e+06 |                1.9086e+06  |                2.03234e+06 | True                     | True            |

## Extended Epoch Selected Evaluation

| split   |   seed_selected_on_val |   threshold_selected_on_val |   best_epoch |   action_pnl |   v1_action_pnl |   action_delta_vs_v1 |   selected_action_count |   causal_fill_pnl |   v1_causal_fill_pnl |   causal_delta_vs_v1 |   causal_fill_count |   causal_max_drawdown |   oracle_capture_ratio |   v1_oracle_capture_ratio |   oracle_capture_delta_vs_v1 | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|:--------|-----------------------:|----------------------------:|-------------:|-------------:|----------------:|---------------------:|------------------------:|------------------:|---------------------:|---------------------:|--------------------:|----------------------:|-----------------------:|--------------------------:|-----------------------------:|:--------------------------|:--------------------------|:--------------------------|
| val     |                      2 |                           0 |           78 |  2.03234e+06 |     1.92717e+06 |               105170 |                   63248 |            174370 |               170310 |                 4060 |                6742 |                  2060 |               0.262887 |                  0.249534 |                   0.0133532  | False                     | False                     | False                     |
| new3    |                      2 |                           0 |           78 |  5.12445e+06 |     5.02152e+06 |               102935 |                  138012 |            582810 |               488870 |                93940 |               17486 |                  3935 |               0.271927 |                  0.266517 |                   0.00540938 | False                     | False                     | False                     |
| new4    |                      2 |                           0 |           78 |  1.8104e+06  |     1.7625e+06  |                47905 |                   52921 |            117470 |               121765 |                -4295 |                3943 |                  1940 |               0.245431 |                  0.239057 |                   0.00637379 | False                     | False                     | False                     |

## Gain Behavior Highlights

| split   | group_type              | group_value       |   num_samples |   gain_mean |   gain_std |   gain_min |   gain_q01 |   gain_q05 |   gain_q25 |   gain_q50 |   gain_q75 |   gain_q95 |   gain_q99 |   gain_max |
|:--------|:------------------------|:------------------|--------------:|------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| val     | overall_state_mean_gain | all               |         82960 |    0.527751 |  0.0161895 |   0.457767 |   0.487394 |   0.499953 |   0.517081 |   0.528519 |   0.539157 |   0.55305  |   0.561967 |   0.587029 |
| val     | selected_status         | non_selected      |         18290 |    0.528703 |  0.0160333 |   0.461223 |   0.489088 |   0.501246 |   0.518101 |   0.529531 |   0.540062 |   0.553544 |   0.562714 |   0.585239 |
| val     | selected_status         | selected          |         64670 |    0.527481 |  0.0162232 |   0.457767 |   0.487045 |   0.499549 |   0.516793 |   0.528232 |   0.538896 |   0.552919 |   0.561778 |   0.587029 |
| val     | selected_outcome        | non_selected      |         18290 |    0.528703 |  0.0160333 |   0.461223 |   0.489088 |   0.501246 |   0.518101 |   0.529531 |   0.540062 |   0.553544 |   0.562714 |   0.585239 |
| val     | selected_outcome        | selected_negative |         34425 |    0.527618 |  0.0162339 |   0.464208 |   0.487568 |   0.499724 |   0.516838 |   0.528362 |   0.539007 |   0.553198 |   0.562044 |   0.587029 |
| val     | selected_outcome        | selected_positive |         30245 |    0.527325 |  0.0162097 |   0.457767 |   0.4864   |   0.499394 |   0.516734 |   0.528113 |   0.538748 |   0.55269  |   0.561449 |   0.578865 |
| new3    | overall_state_mean_gain | all               |        176754 |    0.522957 |  0.0161836 |   0.454306 |   0.483473 |   0.495504 |   0.512219 |   0.52346  |   0.534145 |   0.548711 |   0.558374 |   0.589219 |
| new3    | selected_status         | non_selected      |         36639 |    0.523261 |  0.01619   |   0.457628 |   0.483617 |   0.495844 |   0.512434 |   0.523715 |   0.534437 |   0.549094 |   0.558894 |   0.589219 |
| new3    | selected_status         | selected          |        140115 |    0.522878 |  0.016181  |   0.454306 |   0.483431 |   0.49543  |   0.512157 |   0.523392 |   0.534064 |   0.54862  |   0.558217 |   0.584611 |
| new3    | selected_outcome        | non_selected      |         36639 |    0.523261 |  0.01619   |   0.457628 |   0.483617 |   0.495844 |   0.512434 |   0.523715 |   0.534437 |   0.549094 |   0.558894 |   0.589219 |
| new3    | selected_outcome        | selected_negative |         70373 |    0.522814 |  0.0161999 |   0.454306 |   0.483419 |   0.495409 |   0.512082 |   0.52327  |   0.533992 |   0.548626 |   0.558361 |   0.583594 |
| new3    | selected_outcome        | selected_positive |         69742 |    0.522943 |  0.0161616 |   0.458185 |   0.483489 |   0.495451 |   0.512236 |   0.523499 |   0.534132 |   0.548615 |   0.558039 |   0.584611 |
| new4    | overall_state_mean_gain | all               |         68620 |    0.522803 |  0.0157035 |   0.459615 |   0.484297 |   0.495912 |   0.5125   |   0.523312 |   0.53362  |   0.547813 |   0.557337 |   0.582244 |
| new4    | selected_status         | non_selected      |         14826 |    0.523079 |  0.0156072 |   0.465137 |   0.485072 |   0.496287 |   0.512836 |   0.523445 |   0.533805 |   0.547982 |   0.557791 |   0.579355 |
| new4    | selected_status         | selected          |         53794 |    0.522726 |  0.015729  |   0.459615 |   0.484075 |   0.495812 |   0.51243  |   0.523269 |   0.533573 |   0.547759 |   0.557164 |   0.582244 |
| new4    | selected_outcome        | non_selected      |         14826 |    0.523079 |  0.0156072 |   0.465137 |   0.485072 |   0.496287 |   0.512836 |   0.523445 |   0.533805 |   0.547982 |   0.557791 |   0.579355 |
| new4    | selected_outcome        | selected_negative |         27635 |    0.522808 |  0.0158615 |   0.459881 |   0.483982 |   0.495578 |   0.512462 |   0.523301 |   0.533715 |   0.548101 |   0.557627 |   0.581793 |
| new4    | selected_outcome        | selected_positive |         26159 |    0.52264  |  0.0155874 |   0.459615 |   0.484175 |   0.49598  |   0.512374 |   0.523233 |   0.533423 |   0.547386 |   0.556677 |   0.582244 |

## Session Paired Summary

| variant                 | split   |   sessions |   stp_winning_sessions |   total_delta_stp_minus_v1 |   mean_delta_stp_minus_v1 |
|:------------------------|:--------|-----------:|-----------------------:|---------------------------:|--------------------------:|
| stp_50epoch             | new3    |          6 |                      0 |                    -135095 |                 -22515.8  |
| stp_50epoch             | new4    |          4 |                      0 |                     -74975 |                 -18743.8  |
| stp_50epoch             | val     |          6 |                      2 |                     -18575 |                  -3095.83 |
| stp_extended80_selected | new3    |          6 |                      5 |                     102935 |                  17155.8  |
| stp_extended80_selected | new4    |          4 |                      3 |                      47905 |                  11976.2  |
| stp_extended80_selected | val     |          6 |                      5 |                     105170 |                  17528.3  |

## Claim Boundary

STP-like dynamic gain is not full biological STP. This phase does not overwrite PHC-lite v1, does not adopt STP, does not use NEW5 for selection, and does not claim real-world profitability, true exchange replay, or FPGA PnL improvement.
