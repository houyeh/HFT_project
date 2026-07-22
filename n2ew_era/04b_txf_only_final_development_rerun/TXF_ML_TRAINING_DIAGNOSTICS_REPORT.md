# TXF-only ML Training Diagnostics Report

Diagnostics are for simple Ridge/Logistic/shallow HGB only. No TCN/PHC/deep sequence training was run.

## Convergence Log

| model_id              | n_iter   | max_iter   | hit_max_iter   | convergence_note                                                |
|:----------------------|:---------|:-----------|:---------------|:----------------------------------------------------------------|
| ridge_l2              |          |            | False          | diagnostic only; model/threshold selection uses validation only |
| logistic_l2           | 29       | 200        | False          | diagnostic only; model/threshold selection uses validation only |
| hgb_regressor_shallow | 80       | 80         | True           | diagnostic only; model/threshold selection uses validation only |

## Confusion Matrices

| split   | model_id              |   threshold_selected_on_val |    tn |    fp |    fn |    tp |   precision |   recall | test_new3_new4_used_for_selection   |
|:--------|:----------------------|----------------------------:|------:|------:|------:|------:|------------:|---------:|:------------------------------------|
| val     | hgb_regressor_shallow |                           0 | 95226 | 19794 | 17521 | 17459 |    0.46866  | 0.499114 | False                               |
| new3    | hgb_regressor_shallow |                           0 | 94112 | 19384 | 18774 | 17730 |    0.477717 | 0.4857   | False                               |
| new4    | hgb_regressor_shallow |                           0 | 93478 | 19695 | 19565 | 17262 |    0.467083 | 0.468732 | False                               |

## Feature Importance

| feature                    |   importance |   signed_value |
|:---------------------------|-------------:|---------------:|
| side_queue_imbalance       |  0.174103    |            nan |
| side_imbalance_change_past |  0.0752897   |            nan |
| side_deep_imbalance        |  0.0245637   |            nan |
| spread_change_past         |  0.017745    |            nan |
| action_is_ask              |  0.0126839   |            nan |
| top_depth                  |  0.00816923  |            nan |
| volatility_past_50         |  0.00234029  |            nan |
| deep_imbalance             |  0.00132267  |            nan |
| spread_ticks               |  0.00115974  |            nan |
| hawkes_excite              |  0.000748211 |            nan |
| imbalance_change_past      |  0.000667361 |            nan |
| abs_return_past_50         |  0.000361543 |            nan |
| event_rate_past_50         |  0.000259547 |            nan |
| ask_sz_1                   |  5.54785e-05 |            nan |
| bid_sz_1                   |  4.49085e-05 |            nan |
| action_is_bid              |  0           |            nan |
| side_sign                  |  0           |            nan |
| queue_imbalance            | -9.9643e-05  |            nan |
| inter_arrival_ms           | -0.000459555 |            nan |
