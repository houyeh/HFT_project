# PHC STP v2 Seed Stability Report

Each seed uses its own validation-selected threshold. NEW3/NEW4/NEW5 are not used for selection.

|   seed | split   |   best_epoch |   threshold_selected_on_val_for_seed |   action_pnl |   selected_action_count | action_pnl_positive   | phase   | optimizer_restarted   | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|-------:|:--------|-------------:|-------------------------------------:|-------------:|------------------------:|:----------------------|:--------|:----------------------|:--------------------------|:--------------------------|:--------------------------|
|      0 | val     |          196 |                              0       |  2.06699e+06 |                   64917 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      0 | new3    |          196 |                              0       |  5.29976e+06 |                  142272 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      0 | new4    |          196 |                              0       |  1.88446e+06 |                   54705 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      1 | val     |          200 |                              1.7482  |  2.08554e+06 |                   62220 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      1 | new3    |          200 |                              1.7482  |  5.28866e+06 |                  137377 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      1 | new4    |          200 |                              1.7482  |  1.8871e+06  |                   52708 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      2 | val     |          190 |                              3.30199 |  2.08874e+06 |                   62220 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      2 | new3    |          190 |                              3.30199 |  5.33485e+06 |                  137913 | True                  | 05c3    | False                 | False                     | False                     | False                     |
|      2 | new4    |          190 |                              3.30199 |  1.90854e+06 |                   52692 | True                  | 05c3    | False                 | False                     | False                     | False                     |
