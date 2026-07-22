# PHC STP-like Extended Epoch Sanity Report

Extended sanity uses max_epoch=80, patience=10, seeds 0/1/2, same TXF-only data, same features, same target `excess_pnl_H20`, and validation-only seed/threshold selection. NEW3/NEW4/NEW5 are not used for selection.

## Seed Summary

|   seed |   best_epoch_t0 |   best_val_pnl_at_threshold0 |   best_val_pnl_epoch_le_50 |   best_val_pnl_epoch_gt_50 | improved_after_epoch50   | max_epoch_hit   |
|-------:|----------------:|-----------------------------:|---------------------------:|---------------------------:|:-------------------------|:----------------|
|      0 |              75 |                  1.94044e+06 |                1.86789e+06 |                1.94044e+06 | True                     | True            |
|      1 |              75 |                  1.9346e+06  |                1.87642e+06 |                1.9346e+06  | True                     | True            |
|      2 |              78 |                  2.03234e+06 |                1.9086e+06  |                2.03234e+06 | True                     | True            |

## Selected Extended Model Evaluation

| split   |   seed_selected_on_val |   threshold_selected_on_val |   best_epoch |   action_pnl |   v1_action_pnl |   action_delta_vs_v1 |   selected_action_count |   causal_fill_pnl |   v1_causal_fill_pnl |   causal_delta_vs_v1 |   causal_fill_count |   causal_max_drawdown |   oracle_capture_ratio |   v1_oracle_capture_ratio |   oracle_capture_delta_vs_v1 | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|:--------|-----------------------:|----------------------------:|-------------:|-------------:|----------------:|---------------------:|------------------------:|------------------:|---------------------:|---------------------:|--------------------:|----------------------:|-----------------------:|--------------------------:|-----------------------------:|:--------------------------|:--------------------------|:--------------------------|
| val     |                      2 |                           0 |           78 |  2.03234e+06 |     1.92717e+06 |               105170 |                   63248 |            174370 |               170310 |                 4060 |                6742 |                  2060 |               0.262887 |                  0.249534 |                   0.0133532  | False                     | False                     | False                     |
| new3    |                      2 |                           0 |           78 |  5.12445e+06 |     5.02152e+06 |               102935 |                  138012 |            582810 |               488870 |                93940 |               17486 |                  3935 |               0.271927 |                  0.266517 |                   0.00540938 | False                     | False                     | False                     |
| new4    |                      2 |                           0 |           78 |  1.8104e+06  |     1.7625e+06  |                47905 |                   52921 |            117470 |               121765 |                -4295 |                3943 |                  1940 |               0.245431 |                  0.239057 |                   0.00637379 | False                     | False                     | False                     |

## Answers

Does validation PnL continue improving after epoch 50? `True`.
Does STP exceed PHC-lite v1 on validation? `True`.
Does OOS NEW3/NEW4 improve versus v1? `True`.
Does causal-fill improve versus v1? `False`.
Evidence of overfitting by val-improves/OOS-fails rule: `False`.

This is still an STP-like dynamic gain ablation, not full biological STP, and it does not adopt STP unless it matches or beats frozen PHC-lite v1.
