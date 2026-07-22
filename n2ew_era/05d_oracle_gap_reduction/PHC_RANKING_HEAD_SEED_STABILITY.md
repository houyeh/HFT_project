# PHC Ranking Head Seed Stability

Each seed uses the same validation-selected variant/lambda family and a per-seed threshold selected on validation only. NEW3/NEW4/NEW5 are not used for seed or threshold selection.

## Summary
| split   |   action_pnl_mean |   action_pnl_min |   action_pnl_max |   causal_fill_pnl_mean |   causal_fill_pnl_min |   oracle_capture_mean |   seed_count |
|:--------|------------------:|-----------------:|-----------------:|-----------------------:|----------------------:|----------------------:|-------------:|
| new3    |       5.02612e+06 |      4.92763e+06 |      5.1532e+06  |                 479930 |                468660 |              0.266725 |            3 |
| new4    |       1.75899e+06 |      1.70014e+06 |      1.81528e+06 |                 121838 |                114765 |              0.23836  |            3 |
| val     |       1.94052e+06 |      1.90446e+06 |      1.98794e+06 |                 169760 |                167120 |              0.25051  |            3 |

## Per-Seed Detail
|   seed | run_id                     | split   |   threshold_selected_on_val_for_seed |   best_epoch |   action_pnl |   selected_action_count |   causal_fill_pnl |   oracle_capture_ratio | new3_used_for_selection   | new4_used_for_selection   | new5_used_for_selection   |
|-------:|:---------------------------|:--------|-------------------------------------:|-------------:|-------------:|------------------------:|------------------:|-----------------------:|:--------------------------|:--------------------------|:--------------------------|
|      0 | phc_rank_head_lam0p3_seed0 | val     |                             0.936975 |           48 |  1.98794e+06 |                   62220 |            174780 |               0.256633 | False                     | False                     | False                     |
|      0 | phc_rank_head_lam0p3_seed0 | new3    |                             0.936975 |           48 |  5.1532e+06  |                  139116 |            494545 |               0.273693 | False                     | False                     | False                     |
|      0 | phc_rank_head_lam0p3_seed0 | new4    |                             0.936975 |           48 |  1.81528e+06 |                   53037 |            125720 |               0.24615  | False                     | False                     | False                     |
|      1 | phc_rank_head_lam0p3_seed1 | val     |                             0.258913 |           50 |  1.92914e+06 |                   62220 |            167380 |               0.249042 | False                     | False                     | False                     |
|      1 | phc_rank_head_lam0p3_seed1 | new3    |                             0.258913 |           50 |  4.99752e+06 |                  139214 |            476585 |               0.265445 | False                     | False                     | False                     |
|      1 | phc_rank_head_lam0p3_seed1 | new4    |                             0.258913 |           50 |  1.76156e+06 |                   53070 |            125030 |               0.238883 | False                     | False                     | False                     |
|      2 | phc_rank_head_lam0p3_seed2 | val     |                             0.385355 |            2 |  1.90446e+06 |                   62220 |            167120 |               0.245855 | False                     | False                     | False                     |
|      2 | phc_rank_head_lam0p3_seed2 | new3    |                             0.385355 |            2 |  4.92763e+06 |                  135861 |            468660 |               0.261036 | False                     | False                     | False                     |
|      2 | phc_rank_head_lam0p3_seed2 | new4    |                             0.385355 |            2 |  1.70014e+06 |                   51993 |            114765 |               0.230048 | False                     | False                     | False                     |
