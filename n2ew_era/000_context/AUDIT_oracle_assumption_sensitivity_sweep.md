status: resolved
# Audit: Oracle Assumption Sensitivity Sweep

Hard rules: NEW5 was not loaded. The script loads only `phc_lite_{train,val,new3,new4}_sequences.pt` and frozen policy/checkpoint artifacts. No threshold, checkpoint, or model was changed.

Execution assumptions swept: `inventory_cap in {1,2,3}` x `fill_model in {depth_haircut, queue_side_haircut, conservative_l2_stress}`. The `conservative_l2_stress` probabilities match the existing 05A/05C4 conservative L2 formula; the two other fill models match Phase 02F semantics.

Complete row-level table: `n2ew_era/000_context/results/oracle_assumption_sensitivity_sweep/oracle_assumption_sensitivity_sweep.csv`.

## Capture Ratio Range

| policy_id                                     | min_capture | max_capture | min_pnl    | max_pnl      |
| --------------------------------------------- | ----------- | ----------- | ---------- | ------------ |
| ml_hgb_regressor_shallow_convergence_selected | 0.0932729   | 0.28411     | 97,680.00  | 2,152,320.00 |
| phc_stp_v2_cosine_lr                          | 0.127714    | 0.287329    | 129,160.00 | 2,768,895.00 |
| side_by_queue_imbalance_abs_0p1               | 0.064696    | 0.203825    | 72,415.00  | 1,639,045.00 |

## 9-Combination Aggregate Summary

| policy_id                                     | fill_model             | inventory_cap | dev_pnl      | oos_pnl      | val_capture | new3_capture | new4_capture | total_fill_count | total_forced_no_quote |
| --------------------------------------------- | ---------------------- | ------------- | ------------ | ------------ | ----------- | ------------ | ------------ | ---------------- | --------------------- |
| ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 1             | 289,125.00   | 508,770.00   | 0.159067    | 0.155462     | 0.0961058    | 28150            | 6834                  |
| ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 2             | 346,410.00   | 631,690.00   | 0.159067    | 0.180991     | 0.150628     | 34984            | 0                     |
| ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 3             | 346,410.00   | 631,690.00   | 0.159067    | 0.180991     | 0.150628     | 34984            | 0                     |
| ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 1             | 1,375,035.00 | 2,294,815.00 | 0.19869     | 0.185172     | 0.141926     | 124554           | 33661                 |
| ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 2             | 1,628,785.00 | 2,853,180.00 | 0.2068      | 0.209969     | 0.184557     | 152716           | 5499                  |
| ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 3             | 1,757,230.00 | 2,872,665.00 | 0.2068      | 0.211887     | 0.184557     | 158215           | 0                     |
| ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 1             | 1,123,895.00 | 1,751,315.00 | 0.28411     | 0.174628     | 0.0932729    | 101529           | 24562                 |
| ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 2             | 1,349,460.00 | 2,192,050.00 | 0.171555    | 0.175816     | 0.15389      | 126091           | 0                     |
| ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 3             | 1,349,460.00 | 2,192,050.00 | 0.171555    | 0.175816     | 0.15389      | 126091           | 0                     |
| phc_stp_v2_cosine_lr                          | conservative_l2_stress | 1             | 453,770.00   | 740,255.00   | 0.192967    | 0.232417     | 0.127714     | 36567            | 2589                  |
| phc_stp_v2_cosine_lr                          | conservative_l2_stress | 2             | 453,770.00   | 830,520.00   | 0.192967    | 0.232417     | 0.216968     | 39156            | 0                     |
| phc_stp_v2_cosine_lr                          | conservative_l2_stress | 3             | 453,770.00   | 830,520.00   | 0.192967    | 0.232417     | 0.216968     | 39156            | 0                     |
| phc_stp_v2_cosine_lr                          | depth_haircut          | 1             | 1,949,970.00 | 3,532,120.00 | 0.238488    | 0.275411     | 0.256275     | 154060           | 13364                 |
| phc_stp_v2_cosine_lr                          | depth_haircut          | 2             | 2,210,940.00 | 3,721,405.00 | 0.24793     | 0.274053     | 0.245227     | 167424           | 0                     |
| phc_stp_v2_cosine_lr                          | depth_haircut          | 3             | 2,210,940.00 | 3,721,405.00 | 0.24793     | 0.274053     | 0.245227     | 167424           | 0                     |
| phc_stp_v2_cosine_lr                          | queue_side_haircut     | 1             | 1,631,355.00 | 2,820,315.00 | 0.287329    | 0.272452     | 0.174819     | 132527           | 8730                  |
| phc_stp_v2_cosine_lr                          | queue_side_haircut     | 2             | 1,770,770.00 | 2,959,350.00 | 0.214978    | 0.236947     | 0.213461     | 141257           | 0                     |
| phc_stp_v2_cosine_lr                          | queue_side_haircut     | 3             | 1,770,770.00 | 2,959,350.00 | 0.214978    | 0.236947     | 0.213461     | 141257           | 0                     |
| side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 1             | 264,280.00   | 376,115.00   | 0.116936    | 0.115847     | 0.0718995    | 30730            | 4329                  |
| side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 2             | 264,280.00   | 467,680.00   | 0.116936    | 0.136791     | 0.108298     | 35059            | 0                     |
| side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 3             | 264,280.00   | 467,680.00   | 0.116936    | 0.136791     | 0.108298     | 35059            | 0                     |
| side_by_queue_imbalance_abs_0p1               | depth_haircut          | 1             | 1,158,700.00 | 1,720,805.00 | 0.148969    | 0.141662     | 0.102182     | 139017           | 29035                 |
| side_by_queue_imbalance_abs_0p1               | depth_haircut          | 2             | 1,316,650.00 | 2,143,845.00 | 0.156543    | 0.161097     | 0.134588     | 167054           | 998                   |
| side_by_queue_imbalance_abs_0p1               | depth_haircut          | 3             | 1,316,650.00 | 2,159,775.00 | 0.156543    | 0.162678     | 0.134588     | 168052           | 0                     |
| side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 1             | 980,810.00   | 1,261,630.00 | 0.203825    | 0.128483     | 0.064696     | 113727           | 16090                 |
| side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 2             | 980,810.00   | 1,586,095.00 | 0.124818    | 0.130068     | 0.107961     | 129817           | 0                     |
| side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 3             | 980,810.00   | 1,586,095.00 | 0.124818    | 0.130068     | 0.107961     | 129817           | 0                     |

## Complete OOS 9-Combination Table

| split | policy_id                                                              | fill_model             | inventory_cap | sequential_excess_pnl | fill_count | forced_no_quote_count | capture_ratio_against_matched_count |
| ----- | ---------------------------------------------------------------------- | ---------------------- | ------------- | --------------------- | ---------- | --------------------- | ----------------------------------- |
| val   | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 1             | 517,475.00            | 29476      | 3486                  | 0.148969                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 1             | 700,205.00            | 27274      | 2995                  | 0.19869                             |
| val   | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 1             | 835,615.00            | 28380      | 3366                  | 0.238488                            |
| val   | oracle_dense_full_action                                               | depth_haircut          | 1             | 2,460,075.00          | 18036      | 10009                 |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 1             | 3,473,715.00          | 31361      | 2078                  | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 1             | 3,524,115.00          | 28001      | 2078                  | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 1             | 3,503,805.00          | 29355      | 2078                  | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 2             | 587,685.00            | 32962      | 0                     | 0.156543                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 2             | 786,780.00            | 30269      | 0                     | 0.2068                              |
| val   | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 2             | 938,225.00            | 31746      | 0                     | 0.24793                             |
| val   | oracle_dense_full_action                                               | depth_haircut          | 2             | 3,835,040.00          | 28045      | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 2             | 3,754,145.00          | 33439      | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 2             | 3,804,545.00          | 30079      | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 2             | 3,784,235.00          | 31433      | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 3             | 587,685.00            | 32962      | 0                     | 0.156543                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 3             | 786,780.00            | 30269      | 0                     | 0.2068                              |
| val   | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 3             | 938,225.00            | 31746      | 0                     | 0.24793                             |
| val   | oracle_dense_full_action                                               | depth_haircut          | 3             | 3,835,040.00          | 28045      | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 3             | 3,754,145.00          | 33439      | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 3             | 3,804,545.00          | 30079      | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 3             | 3,784,235.00          | 31433      | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 1             | 424,680.00            | 25128      | 0                     | 0.203825                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 1             | 592,525.00            | 23985      | 0                     | 0.28411                             |
| val   | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 1             | 598,665.00            | 21650      | 4979                  | 0.287329                            |
| val   | oracle_dense_full_action                                               | queue_side_haircut     | 1             | 2,244,300.00          | 16801      | 9218                  |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 1             | 2,083,550.00          | 17593      | 13890                 | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 1             | 2,085,545.00          | 17460      | 10593                 | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 1             | 2,083,550.00          | 17593      | 11831                 | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 2             | 424,680.00            | 25128      | 0                     | 0.124818                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 2             | 592,525.00            | 23985      | 0                     | 0.171555                            |
| val   | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 2             | 738,080.00            | 26629      | 0                     | 0.214978                            |
| val   | oracle_dense_full_action                                               | queue_side_haircut     | 2             | 3,484,330.00          | 26019      | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 2             | 3,402,400.00          | 31483      | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 2             | 3,453,850.00          | 28053      | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 2             | 3,433,285.00          | 29424      | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 3             | 424,680.00            | 25128      | 0                     | 0.124818                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 3             | 592,525.00            | 23985      | 0                     | 0.171555                            |
| val   | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 3             | 738,080.00            | 26629      | 0                     | 0.214978                            |
| val   | oracle_dense_full_action                                               | queue_side_haircut     | 3             | 3,484,330.00          | 26019      | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 3             | 3,402,400.00          | 31483      | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 3             | 3,453,850.00          | 28053      | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 3             | 3,433,285.00          | 29424      | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 1             | 103,260.00            | 6416       | 0                     | 0.116936                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 1             | 142,435.00            | 6231       | 0                     | 0.159067                            |
| val   | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 1             | 171,835.00            | 6891       | 0                     | 0.192967                            |
| val   | oracle_dense_full_action                                               | conservative_l2_stress | 1             | 903,690.00            | 6814       | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 1             | 883,050.00            | 8190       | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 1             | 895,440.00            | 7364       | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 1             | 890,490.00            | 7694       | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 2             | 103,260.00            | 6416       | 0                     | 0.116936                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 2             | 142,435.00            | 6231       | 0                     | 0.159067                            |
| val   | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 2             | 171,835.00            | 6891       | 0                     | 0.192967                            |
| val   | oracle_dense_full_action                                               | conservative_l2_stress | 2             | 903,690.00            | 6814       | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 2             | 883,050.00            | 8190       | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 2             | 895,440.00            | 7364       | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 2             | 890,490.00            | 7694       | 0                     | 1                                   |
| val   | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 3             | 103,260.00            | 6416       | 0                     | 0.116936                            |
| val   | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 3             | 142,435.00            | 6231       | 0                     | 0.159067                            |
| val   | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 3             | 171,835.00            | 6891       | 0                     | 0.192967                            |
| val   | oracle_dense_full_action                                               | conservative_l2_stress | 3             | 903,690.00            | 6814       | 0                     |                                     |
| val   | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 3             | 883,050.00            | 8190       | 0                     | 1                                   |
| val   | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 3             | 895,440.00            | 7364       | 0                     | 1                                   |
| val   | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 3             | 890,490.00            | 7694       | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 1             | 1,410,415.00          | 64500      | 10179                 | 0.141662                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 1             | 1,858,870.00          | 61823      | 9131                  | 0.185172                            |
| new3  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 1             | 2,749,780.00          | 74469      | 979                   | 0.275411                            |
| new3  | oracle_dense_full_action                                               | depth_haircut          | 1             | 10,072,705.00         | 68354      | 852                   |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 1             | 9,956,170.00          | 76123      | 852                   | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 1             | 10,038,625.00         | 70626      | 852                   | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 1             | 9,984,280.00          | 74249      | 852                   | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 2             | 1,623,115.00          | 73681      | 998                   | 0.161097                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 2             | 2,132,835.00          | 70093      | 861                   | 0.209969                            |
| new3  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 2             | 2,768,895.00          | 75448      | 0                     | 0.274053                            |
| new3  | oracle_dense_full_action                                               | depth_haircut          | 2             | 10,191,925.00         | 69206      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 2             | 10,075,390.00         | 76975      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 2             | 10,157,845.00         | 71478      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 2             | 10,103,500.00         | 75101      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 3             | 1,639,045.00          | 74679      | 0                     | 0.162678                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 3             | 2,152,320.00          | 70954      | 0                     | 0.211887                            |
| new3  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 3             | 2,768,895.00          | 75448      | 0                     | 0.274053                            |
| new3  | oracle_dense_full_action                                               | depth_haircut          | 3             | 10,191,925.00         | 69206      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 3             | 10,075,390.00         | 76975      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 3             | 10,157,845.00         | 71478      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 3             | 10,103,500.00         | 75101      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 1             | 1,029,835.00          | 50052      | 7883                  | 0.128483                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 1             | 1,413,980.00          | 49369      | 7213                  | 0.174628                            |
| new3  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 1             | 2,191,305.00          | 63633      | 0                     | 0.272452                            |
| new3  | oracle_dense_full_action                                               | queue_side_haircut     | 1             | 9,335,740.00          | 64504      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 1             | 8,015,355.00          | 63684      | 8503                  | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 1             | 8,097,090.00          | 58235      | 8503                  | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 1             | 8,042,910.00          | 61847      | 8503                  | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 2             | 1,199,290.00          | 57935      | 0                     | 0.130068                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 2             | 1,635,485.00          | 56582      | 0                     | 0.175816                            |
| new3  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 2             | 2,191,305.00          | 63633      | 0                     | 0.236947                            |
| new3  | oracle_dense_full_action                                               | queue_side_haircut     | 2             | 9,335,740.00          | 64504      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 2             | 9,220,510.00          | 72187      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 2             | 9,302,245.00          | 66738      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 2             | 9,248,065.00          | 70350      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 3             | 1,199,290.00          | 57935      | 0                     | 0.130068                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 3             | 1,635,485.00          | 56582      | 0                     | 0.175816                            |
| new3  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 3             | 2,191,305.00          | 63633      | 0                     | 0.236947                            |
| new3  | oracle_dense_full_action                                               | queue_side_haircut     | 3             | 9,335,740.00          | 64504      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 3             | 9,220,510.00          | 72187      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 3             | 9,302,245.00          | 66738      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 3             | 9,248,065.00          | 70350      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 1             | 303,700.00            | 13821      | 2113                  | 0.115847                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 1             | 411,090.00            | 13955      | 1993                  | 0.155462                            |
| new3  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 1             | 611,095.00            | 17907      | 0                     | 0.232417                            |
| new3  | oracle_dense_full_action                                               | conservative_l2_stress | 1             | 2,653,305.00          | 18173      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 1             | 2,621,550.00          | 20290      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 1             | 2,644,305.00          | 18773      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 1             | 2,629,305.00          | 19773      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 2             | 358,605.00            | 15934      | 0                     | 0.136791                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 2             | 478,595.00            | 15948      | 0                     | 0.180991                            |
| new3  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 2             | 611,095.00            | 17907      | 0                     | 0.232417                            |
| new3  | oracle_dense_full_action                                               | conservative_l2_stress | 2             | 2,653,305.00          | 18173      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 2             | 2,621,550.00          | 20290      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 2             | 2,644,305.00          | 18773      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 2             | 2,629,305.00          | 19773      | 0                     | 1                                   |
| new3  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 3             | 358,605.00            | 15934      | 0                     | 0.136791                            |
| new3  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 3             | 478,595.00            | 15948      | 0                     | 0.180991                            |
| new3  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 3             | 611,095.00            | 17907      | 0                     | 0.232417                            |
| new3  | oracle_dense_full_action                                               | conservative_l2_stress | 3             | 2,653,305.00          | 18173      | 0                     |                                     |
| new3  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 3             | 2,621,550.00          | 20290      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 3             | 2,644,305.00          | 18773      | 0                     | 1                                   |
| new3  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 3             | 2,629,305.00          | 19773      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 1             | 310,390.00            | 17935      | 10545                 | 0.102182                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 1             | 435,945.00            | 16178      | 10861                 | 0.141926                            |
| new4  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 1             | 782,340.00            | 24027      | 4422                  | 0.256275                            |
| new4  | oracle_dense_full_action                                               | depth_haircut          | 1             | 3,916,120.00          | 26352      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 1             | 3,037,605.00          | 23714      | 5777                  | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 1             | 3,071,640.00          | 21445      | 5777                  | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 1             | 3,052,740.00          | 22705      | 5777                  | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 2             | 520,730.00            | 28480      | 0                     | 0.134588                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 2             | 720,345.00            | 27039      | 0                     | 0.184557                            |
| new4  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 2             | 952,510.00            | 28449      | 0                     | 0.245227                            |
| new4  | oracle_dense_full_action                                               | depth_haircut          | 2             | 3,916,120.00          | 26352      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 2             | 3,869,065.00          | 29491      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 2             | 3,903,100.00          | 27222      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 2             | 3,884,200.00          | 28482      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | depth_haircut          | 3             | 520,730.00            | 28480      | 0                     | 0.134588                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | depth_haircut          | 3             | 720,345.00            | 27039      | 0                     | 0.184557                            |
| new4  | phc_stp_v2_cosine_lr                                                   | depth_haircut          | 3             | 952,510.00            | 28449      | 0                     | 0.245227                            |
| new4  | oracle_dense_full_action                                               | depth_haircut          | 3             | 3,916,120.00          | 26352      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | depth_haircut          | 3             | 3,869,065.00          | 29491      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | depth_haircut          | 3             | 3,903,100.00          | 27222      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | depth_haircut          | 3             | 3,884,200.00          | 28482      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 1             | 231,795.00            | 13888      | 8207                  | 0.064696                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 1             | 337,335.00            | 12832      | 8819                  | 0.0932729                           |
| new4  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 1             | 629,010.00            | 20429      | 3751                  | 0.174819                            |
| new4  | oracle_dense_full_action                                               | queue_side_haircut     | 1             | 3,629,335.00          | 24771      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 1             | 3,582,835.00          | 27872      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 1             | 3,616,645.00          | 25618      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 1             | 3,598,060.00          | 26857      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 2             | 386,805.00            | 22095      | 0                     | 0.107961                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 2             | 556,565.00            | 21651      | 0                     | 0.15389                             |
| new4  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 2             | 768,045.00            | 24180      | 0                     | 0.213461                            |
| new4  | oracle_dense_full_action                                               | queue_side_haircut     | 2             | 3,629,335.00          | 24771      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 2             | 3,582,835.00          | 27872      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 2             | 3,616,645.00          | 25618      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 2             | 3,598,060.00          | 26857      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | queue_side_haircut     | 3             | 386,805.00            | 22095      | 0                     | 0.107961                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | queue_side_haircut     | 3             | 556,565.00            | 21651      | 0                     | 0.15389                             |
| new4  | phc_stp_v2_cosine_lr                                                   | queue_side_haircut     | 3             | 768,045.00            | 24180      | 0                     | 0.213461                            |
| new4  | oracle_dense_full_action                                               | queue_side_haircut     | 3             | 3,629,335.00          | 24771      | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | queue_side_haircut     | 3             | 3,582,835.00          | 27872      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | queue_side_haircut     | 3             | 3,616,645.00          | 25618      | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | queue_side_haircut     | 3             | 3,598,060.00          | 26857      | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 1             | 72,415.00             | 3820       | 2216                  | 0.0718995                           |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 1             | 97,680.00             | 3629       | 2479                  | 0.0961058                           |
| new4  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 1             | 129,160.00            | 4237       | 2589                  | 0.127714                            |
| new4  | oracle_dense_full_action                                               | conservative_l2_stress | 1             | 1,019,845.00          | 6817       | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 1             | 1,007,170.00          | 7662       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 1             | 1,016,380.00          | 7048       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 1             | 1,011,325.00          | 7385       | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 2             | 109,075.00            | 6036       | 0                     | 0.108298                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 2             | 153,095.00            | 6108       | 0                     | 0.150628                            |
| new4  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 2             | 219,425.00            | 6826       | 0                     | 0.216968                            |
| new4  | oracle_dense_full_action                                               | conservative_l2_stress | 2             | 1,019,845.00          | 6817       | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 2             | 1,007,170.00          | 7662       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 2             | 1,016,380.00          | 7048       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 2             | 1,011,325.00          | 7385       | 0                     | 1                                   |
| new4  | side_by_queue_imbalance_abs_0p1                                        | conservative_l2_stress | 3             | 109,075.00            | 6036       | 0                     | 0.108298                            |
| new4  | ml_hgb_regressor_shallow_convergence_selected                          | conservative_l2_stress | 3             | 153,095.00            | 6108       | 0                     | 0.150628                            |
| new4  | phc_stp_v2_cosine_lr                                                   | conservative_l2_stress | 3             | 219,425.00            | 6826       | 0                     | 0.216968                            |
| new4  | oracle_dense_full_action                                               | conservative_l2_stress | 3             | 1,019,845.00          | 6817       | 0                     |                                     |
| new4  | oracle_matched_count_for_side_by_queue_imbalance_abs_0p1               | conservative_l2_stress | 3             | 1,007,170.00          | 7662       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected | conservative_l2_stress | 3             | 1,016,380.00          | 7048       | 0                     | 1                                   |
| new4  | oracle_matched_count_for_phc_stp_v2_cosine_lr                          | conservative_l2_stress | 3             | 1,011,325.00          | 7385       | 0                     | 1                                   |

## Policy Rank Consistency

| split | fill_model             | inventory_cap | rank_1               | rank_2                                        | rank_3                          | ml_beats_queue | phc_beats_ml |
| ----- | ---------------------- | ------------- | -------------------- | --------------------------------------------- | ------------------------------- | -------------- | ------------ |
| new3  | conservative_l2_stress | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | conservative_l2_stress | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | conservative_l2_stress | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | depth_haircut          | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | depth_haircut          | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | depth_haircut          | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | queue_side_haircut     | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | queue_side_haircut     | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new3  | queue_side_haircut     | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | conservative_l2_stress | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | conservative_l2_stress | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | conservative_l2_stress | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | depth_haircut          | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | depth_haircut          | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | depth_haircut          | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | queue_side_haircut     | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | queue_side_haircut     | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| new4  | queue_side_haircut     | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | conservative_l2_stress | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | conservative_l2_stress | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | conservative_l2_stress | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | depth_haircut          | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | depth_haircut          | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | depth_haircut          | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | queue_side_haircut     | 1             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | queue_side_haircut     | 2             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |
| val   | queue_side_haircut     | 3             | phc_stp_v2_cosine_lr | ml_hgb_regressor_shallow_convergence_selected | side_by_queue_imbalance_abs_0p1 | True           | True         |

## Conclusion

ML beats the queue rule in every OOS assumption cell: `True`.
PHC-STP-v2 beats HGB in every OOS assumption cell: `True`.
The absolute oracle gap is assumption-sensitive because fill_count and forced_no_quote_count both move with the execution model. The relative ordering among the three frozen policies is stable across the 9 OOS assumption cells in this sweep.

This is diagnostic only. The oracle rows use future labels and must not be interpreted as tradable policies.
