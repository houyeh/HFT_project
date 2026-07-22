# PHC-lite Decision Diagnostics Report

The selected-action confusion matrix compares model selection to whether the chosen action has positive `excess_pnl_H20`. The oracle matrix compares the model action against an untradable future-information oracle `argmax(excess_bid, excess_ask, 0)`.

## Selected Action Confusion

| split   |    tn |    fp |   fn |    tp | selected_positive_definition           | actual_positive_definition          |
|:--------|------:|------:|-----:|------:|:---------------------------------------|:------------------------------------|
| val     | 19194 | 33831 |    0 | 29935 | model_score_above_validation_threshold | selected_action_excess_pnl_H20_gt_0 |
| new3    | 38493 | 68986 |    0 | 69275 | model_score_above_validation_threshold | selected_action_excess_pnl_H20_gt_0 |
| new4    | 15451 | 27084 |    0 | 26085 | model_score_above_validation_threshold | selected_action_excess_pnl_H20_gt_0 |

## Oracle Action Confusion

| split   | oracle_action   | model_action   |   count |
|:--------|:----------------|:---------------|--------:|
| val     | no_quote        | no_quote       |    6038 |
| val     | no_quote        | bid            |    9670 |
| val     | no_quote        | ask            |    9695 |
| val     | bid             | no_quote       |    6522 |
| val     | bid             | bid            |   15272 |
| val     | bid             | ask            |    7231 |
| val     | ask             | no_quote       |    6634 |
| val     | ask             | bid            |    7235 |
| val     | ask             | ask            |   14663 |
| new3    | no_quote        | no_quote       |   10532 |
| new3    | no_quote        | bid            |   18500 |
| new3    | no_quote        | ask            |   18014 |
| new3    | bid             | no_quote       |   13650 |
| new3    | bid             | bid            |   35274 |
| new3    | bid             | ask            |   15906 |
| new3    | ask             | no_quote       |   14311 |
| new3    | ask             | bid            |   16566 |
| new3    | ask             | ask            |   34001 |
| new4    | no_quote        | no_quote       |    4253 |
| new4    | no_quote        | bid            |    7123 |
| new4    | no_quote        | ask            |    7048 |
| new4    | bid             | no_quote       |    5486 |
| new4    | bid             | bid            |   13383 |
| new4    | bid             | ask            |    6264 |
| new4    | ask             | no_quote       |    5712 |
| new4    | ask             | bid            |    6649 |
| new4    | ask             | ask            |   12702 |

This is diagnostic only. It does not imply the model predicts the oracle perfectly.
