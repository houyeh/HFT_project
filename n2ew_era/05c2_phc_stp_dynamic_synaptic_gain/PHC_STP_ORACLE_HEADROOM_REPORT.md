# PHC STP-like Oracle Headroom Report

Oracle capture is computed on val/NEW3/NEW4 sequence-window universes only. NEW5 is not used.
Oracle uses future labels and is diagnostic only.

| split   | policy_id                                     |   policy_selected_count |   policy_pnl |   matched_count_oracle_pnl |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | new5_used   |
|:--------|:----------------------------------------------|------------------------:|-------------:|---------------------------:|----------------:|------------------:|:----------------------------|:------------|
| val     | side_by_queue_imbalance_abs_0p1               |                   68981 |  1.23714e+06 |                7.64484e+06 |        0.161827 |       6.4077e+06  | True                        | False       |
| val     | ml_hgb_regressor_shallow_convergence_selected |                   61679 |  1.67168e+06 |                7.75438e+06 |        0.215578 |       6.0827e+06  | True                        | False       |
| val     | phc_lite                                      |                   63766 |  1.92717e+06 |                7.72307e+06 |        0.249534 |       5.7959e+06  | True                        | False       |
| val     | phc_stp                                       |                   64670 |  1.9086e+06  |                7.70951e+06 |        0.247564 |       5.80092e+06 | True                        | False       |
| new3    | side_by_queue_imbalance_abs_0p1               |                  144745 |  2.95477e+06 |                1.8744e+07  |        0.157638 |       1.57892e+07 | True                        | False       |
| new3    | ml_hgb_regressor_shallow_convergence_selected |                  134210 |  4.01058e+06 |                1.8902e+07  |        0.212178 |       1.48914e+07 | True                        | False       |
| new3    | phc_lite                                      |                  138261 |  5.02152e+06 |                1.88412e+07 |        0.266517 |       1.38197e+07 | True                        | False       |
| new3    | phc_stp                                       |                  140115 |  4.88642e+06 |                1.88134e+07 |        0.259731 |       1.3927e+07  | True                        | False       |
| new4    | side_by_queue_imbalance_abs_0p1               |                   56147 |  1.00391e+06 |                7.32802e+06 |        0.136996 |       6.32412e+06 | True                        | False       |
| new4    | ml_hgb_regressor_shallow_convergence_selected |                   51753 |  1.39542e+06 |                7.39394e+06 |        0.188725 |       5.99852e+06 | True                        | False       |
| new4    | phc_lite                                      |                   53169 |  1.7625e+06  |                7.3727e+06  |        0.239057 |       5.6102e+06  | True                        | False       |
| new4    | phc_stp                                       |                   53794 |  1.68752e+06 |                7.36332e+06 |        0.229179 |       5.6758e+06  | True                        | False       |
