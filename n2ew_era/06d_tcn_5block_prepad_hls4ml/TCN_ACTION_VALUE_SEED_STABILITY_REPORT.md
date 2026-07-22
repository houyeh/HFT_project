# TCN 5-Block PrePad Seed Stability Report

Each seed uses its own validation-selected threshold. NEW3/NEW4/NEW5 are not used for selection.

| model_variant   |   seed | split   |   best_epoch |   threshold_selected_on_val_for_seed |   action_pnl |   selected_action_count | action_pnl_positive   | phase   | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|:----------------|-------:|:--------|-------------:|-------------------------------------:|-------------:|------------------------:|:----------------------|:--------|:--------------------------|:--------------------------|:--------------------------|
| 18feat          |      0 | val     |          142 |                                    0 |  1.63561e+06 |                   60489 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      0 | new3    |          142 |                                    0 |  4.25602e+06 |                  133521 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      0 | new4    |          142 |                                    0 |  1.50862e+06 |                   51073 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      1 | val     |          156 |                                    0 |  1.64595e+06 |                   55813 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      1 | new3    |          156 |                                    0 |  4.30386e+06 |                  123871 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      1 | new4    |          156 |                                    0 |  1.53275e+06 |                   47111 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      2 | val     |          179 |                                    0 |  1.71809e+06 |                   61538 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      2 | new3    |          179 |                                    0 |  4.50013e+06 |                  134780 | True                  | 06d     | False                     | False                     | False                     |
| 18feat          |      2 | new4    |          179 |                                    0 |  1.6034e+06  |                   51521 | True                  | 06d     | False                     | False                     | False                     |
