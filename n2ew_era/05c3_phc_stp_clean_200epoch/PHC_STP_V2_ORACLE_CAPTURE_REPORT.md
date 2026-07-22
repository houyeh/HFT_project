# PHC STP v2 Oracle Capture Report

Oracle capture is matched-count same-universe sequence-window oracle. Oracle uses future labels and is diagnostic only.

| split   | policy_id                 |   policy_selected_count |   policy_pnl |   matched_count_oracle_pnl |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | new5_used   |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|:--------------------------|------------------------:|-------------:|---------------------------:|----------------:|------------------:|:----------------------------|:------------|----------------------------:|-----------------------------:|---------------------------------:|
| val     | phc_stp_v2_clean_200epoch |                   62220 |  2.08874e+06 |                7.74626e+06 |        0.269646 |       5.65752e+06 | True                        | False       |                    0.249534 |                    0.0201114 |                        0.0805959 |
| new3    | phc_stp_v2_clean_200epoch |                  137913 |  5.33485e+06 |                1.88464e+07 |        0.283069 |       1.35116e+07 | True                        | False       |                    0.266517 |                    0.0165519 |                        0.0621042 |
| new4    | phc_stp_v2_clean_200epoch |                   52692 |  1.90854e+06 |                7.37985e+06 |        0.258614 |       5.47132e+06 | True                        | False       |                    0.239057 |                    0.0195572 |                        0.0818099 |
