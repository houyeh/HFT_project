# PHC STP v2 Oracle Capture Report

Oracle capture is matched-count same-universe sequence-window oracle. Oracle uses future labels and is diagnostic only.

| split   | policy_id            |   policy_selected_count |   policy_pnl |   matched_count_oracle_pnl |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | new5_used   |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|:---------------------|------------------------:|-------------:|---------------------------:|----------------:|------------------:|:----------------------------|:------------|----------------------------:|-----------------------------:|---------------------------------:|
| val     | phc_stp_v2_cosine_lr |                   64568 |  2.03792e+06 |                7.71104e+06 |        0.264287 |       5.67312e+06 | True                        | False       |                    0.249534 |                    0.0147525 |                        0.05912   |
| new3    | phc_stp_v2_cosine_lr |                  141208 |  5.23741e+06 |                1.8797e+07  |        0.27863  |       1.35596e+07 | True                        | False       |                    0.266517 |                    0.0121124 |                        0.0454468 |
| new4    | phc_stp_v2_cosine_lr |                   54183 |  1.87168e+06 |                7.35748e+06 |        0.254392 |       5.4858e+06  | True                        | False       |                    0.239057 |                    0.0153349 |                        0.0641473 |
