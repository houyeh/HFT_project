# PHC Diagonal SSM Oracle Capture Report

Oracle capture is matched-count same-universe sequence-window oracle. Oracle uses future labels and is diagnostic only.

| split   | policy_id        |   policy_selected_count |   policy_pnl |   matched_count_oracle_pnl |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | new5_used   |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|:-----------------|------------------------:|-------------:|---------------------------:|----------------:|------------------:|:----------------------------|:------------|----------------------------:|-----------------------------:|---------------------------------:|
| val     | phc_diagonal_ssm |                   61828 |  1.77152e+06 |                7.75214e+06 |        0.228521 |       5.98062e+06 | True                        | False       |                    0.249534 |                   -0.0210134 |                       -0.0842106 |
| new3    | phc_diagonal_ssm |                  136027 |  4.72742e+06 |                1.88747e+07 |        0.250463 |       1.41473e+07 | True                        | False       |                    0.266517 |                   -0.0160543 |                       -0.0602374 |
| new4    | phc_diagonal_ssm |                   51950 |  1.64268e+06 |                7.39098e+06 |        0.222255 |       5.7483e+06  | True                        | False       |                    0.239057 |                   -0.0168024 |                       -0.0702861 |
