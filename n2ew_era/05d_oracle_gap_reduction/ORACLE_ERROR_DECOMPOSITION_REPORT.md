# Oracle Error Decomposition Report

Universe: PHC sequence-window terminal states on val/NEW3/NEW4. NEW5 is not used.
Oracle uses future `excess_pnl_H20` labels and is diagnostic only; it is not a tradable strategy and is not used for model selection.

## Capture Ratios

| split   | model_id                                      |   total_model_pnl |   total_oracle_pnl |   capture_ratio |
|:--------|:----------------------------------------------|------------------:|-------------------:|----------------:|
| new3    | ml_hgb_regressor_shallow_convergence_selected |       4.01058e+06 |        1.89695e+07 |        0.211423 |
| new3    | phc_lite_v1                                   |       5.02152e+06 |        1.89695e+07 |        0.264715 |
| new3    | phc_stp_extended80                            |       5.12445e+06 |        1.89695e+07 |        0.270142 |
| new3    | side_by_queue_imbalance_abs_0p1               |       2.95477e+06 |        1.89695e+07 |        0.155764 |
| new4    | ml_hgb_regressor_shallow_convergence_selected |       1.39542e+06 |        7.41726e+06 |        0.188131 |
| new4    | phc_lite_v1                                   |       1.7625e+06  |        7.41726e+06 |        0.237621 |
| new4    | phc_stp_extended80                            |       1.8104e+06  |        7.41726e+06 |        0.244079 |
| new4    | side_by_queue_imbalance_abs_0p1               |       1.00391e+06 |        7.41726e+06 |        0.135348 |
| val     | ml_hgb_regressor_shallow_convergence_selected |       1.67168e+06 |        7.81614e+06 |        0.213875 |
| val     | phc_lite_v1                                   |       1.92717e+06 |        7.81614e+06 |        0.246563 |
| val     | phc_stp_extended80                            |       2.03234e+06 |        7.81614e+06 |        0.260018 |
| val     | side_by_queue_imbalance_abs_0p1               |       1.23714e+06 |        7.81614e+06 |        0.158281 |

## Error Counts by Split / Model

| split   | model_id                                      |   correct_no_quote |   correct_selected |   false_positive |   missed_opportunity |   side_error |
|:--------|:----------------------------------------------|-------------------:|-------------------:|-----------------:|---------------------:|-------------:|
| new3    | ml_hgb_regressor_shallow_convergence_selected |              11339 |              64282 |            35707 |                31205 |        34221 |
| new3    | phc_lite_v1                                   |              10532 |              69275 |            36514 |                27961 |        32472 |
| new3    | phc_stp_extended80                            |              10697 |              69580 |            36349 |                28045 |        32083 |
| new3    | side_by_queue_imbalance_abs_0p1               |               7648 |              64663 |            39398 |                24361 |        40684 |
| new4    | ml_hgb_regressor_shallow_convergence_selected |               4488 |              24304 |            13936 |                12379 |        13513 |
| new4    | phc_lite_v1                                   |               4253 |              26085 |            14171 |                11198 |        12913 |
| new4    | phc_stp_extended80                            |               4406 |              26167 |            14018 |                11293 |        12736 |
| new4    | side_by_queue_imbalance_abs_0p1               |               2981 |              24636 |            15443 |                 9492 |        16068 |
| val     | ml_hgb_regressor_shallow_convergence_selected |               6554 |              28361 |            18849 |                14727 |        14469 |
| val     | phc_lite_v1                                   |               6038 |              29935 |            19365 |                13156 |        14466 |
| val     | phc_stp_extended80                            |               6212 |              30021 |            19191 |                13500 |        14036 |
| val     | side_by_queue_imbalance_abs_0p1               |               3754 |              29417 |            21649 |                10225 |        17915 |

## PHC vs HGB Error Rate Comparison

| split   | error_type         |   phc_rate |   hgb_rate | phc_reduces_vs_hgb   |
|:--------|:-------------------|-----------:|-----------:|:---------------------|
| val     | missed_opportunity |   0.158582 |   0.177519 | True                 |
| val     | false_positive     |   0.233426 |   0.227206 | False                |
| val     | side_error         |   0.174373 |   0.174409 | True                 |
| new3    | missed_opportunity |   0.158192 |   0.176545 | True                 |
| new3    | false_positive     |   0.206581 |   0.202015 | False                |
| new3    | side_error         |   0.183713 |   0.193608 | True                 |
| new4    | missed_opportunity |   0.163189 |   0.180399 | True                 |
| new4    | false_positive     |   0.206514 |   0.203089 | False                |
| new4    | side_error         |   0.188181 |   0.196925 | True                 |

Required interpretation: dominant PHC gap types should be read from `oracle_error_by_split_model.csv`; session concentration is in `oracle_error_by_session.csv`.
