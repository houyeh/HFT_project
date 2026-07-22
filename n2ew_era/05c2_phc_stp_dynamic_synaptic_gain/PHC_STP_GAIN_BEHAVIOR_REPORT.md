# PHC STP-like Gain Behavior Report

Gain diagnostics use the validation-selected STP-like model and exclude NEW5.

Is gain_t non-trivial by state-level std > 0.005? `True`.
Does gain_t vary across regime buckets by mean range > 0.005? `True`.

## Overall Gain Statistics

| split   | group_type              | group_value   |   num_samples |   gain_mean |   gain_std |   gain_min |   gain_q01 |   gain_q05 |   gain_q25 |   gain_q50 |   gain_q75 |   gain_q95 |   gain_q99 |   gain_max |
|:--------|:------------------------|:--------------|--------------:|------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| val     | overall_state_mean_gain | all           |         82960 |    0.527751 |  0.0161895 |   0.457767 |   0.487394 |   0.499953 |   0.517081 |   0.528519 |   0.539157 |   0.55305  |   0.561967 |   0.587029 |
| new3    | overall_state_mean_gain | all           |        176754 |    0.522957 |  0.0161836 |   0.454306 |   0.483473 |   0.495504 |   0.512219 |   0.52346  |   0.534145 |   0.548711 |   0.558374 |   0.589219 |
| new4    | overall_state_mean_gain | all           |         68620 |    0.522803 |  0.0157035 |   0.459615 |   0.484297 |   0.495912 |   0.5125   |   0.523312 |   0.53362  |   0.547813 |   0.557337 |   0.582244 |

## Selected vs Non-selected

| split   | group_type      | group_value   |   num_samples |   gain_mean |   gain_std |   gain_min |   gain_q01 |   gain_q05 |   gain_q25 |   gain_q50 |   gain_q75 |   gain_q95 |   gain_q99 |   gain_max |
|:--------|:----------------|:--------------|--------------:|------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| val     | selected_status | non_selected  |         18290 |    0.528703 |  0.0160333 |   0.461223 |   0.489088 |   0.501246 |   0.518101 |   0.529531 |   0.540062 |   0.553544 |   0.562714 |   0.585239 |
| val     | selected_status | selected      |         64670 |    0.527481 |  0.0162232 |   0.457767 |   0.487045 |   0.499549 |   0.516793 |   0.528232 |   0.538896 |   0.552919 |   0.561778 |   0.587029 |
| new3    | selected_status | non_selected  |         36639 |    0.523261 |  0.01619   |   0.457628 |   0.483617 |   0.495844 |   0.512434 |   0.523715 |   0.534437 |   0.549094 |   0.558894 |   0.589219 |
| new3    | selected_status | selected      |        140115 |    0.522878 |  0.016181  |   0.454306 |   0.483431 |   0.49543  |   0.512157 |   0.523392 |   0.534064 |   0.54862  |   0.558217 |   0.584611 |
| new4    | selected_status | non_selected  |         14826 |    0.523079 |  0.0156072 |   0.465137 |   0.485072 |   0.496287 |   0.512836 |   0.523445 |   0.533805 |   0.547982 |   0.557791 |   0.579355 |
| new4    | selected_status | selected      |         53794 |    0.522726 |  0.015729  |   0.459615 |   0.484075 |   0.495812 |   0.51243  |   0.523269 |   0.533573 |   0.547759 |   0.557164 |   0.582244 |

## Regime Mean Ranges

| split   | group_type              |   gain_mean_range |
|:--------|:------------------------|------------------:|
| new3    | event_rate_bucket       |        0.00108457 |
| new3    | imbalance_change_bucket |        0.00244945 |
| new3    | volatility_bucket       |        0.0108069  |
| new4    | event_rate_bucket       |        0.00112826 |
| new4    | imbalance_change_bucket |        0.00375074 |
| new4    | volatility_bucket       |        0.0101463  |
| val     | event_rate_bucket       |        0.00371403 |
| val     | imbalance_change_bucket |        0.0025472  |
| val     | volatility_bucket       |        0.0125752  |

Interpretation: gain_t is diagnostic only. STP is not adopted unless it matches or beats frozen PHC-lite v1 on OOS, causal-fill, and oracle capture.
