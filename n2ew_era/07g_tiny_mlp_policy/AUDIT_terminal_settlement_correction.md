# Terminal Same-Index Settlement Audit

status: `resolved_positive`

## Scope

This audit does not modify `common_05a.py` or overwrite any prior metric. It locally corrects one scheduling defect: when a fill is opened at the event chosen as its own clipped close index, the legacy loop schedules the close after that index has already been consumed. The corrected diagnostic settles that fill immediately. Train/val/NEW3/NEW4 data and deterministic fill draws are otherwise unchanged; NEW5 was not loaded.

## Regression gate

Policies without a same-index terminal fill changed in neither PnL nor forced-count: `True`. Every corrected run ended with zero inventory: `True`.

## Affected policy/split rows

| split   | policy_id                          |   legacy_sequential_excess_pnl |   corrected_sequential_excess_pnl |   pnl_correction |   legacy_forced_no_quote_count |   corrected_forced_no_quote_count |   corrected_terminal_immediate_settlement_count |
|:--------|:-----------------------------------|-------------------------------:|----------------------------------:|-----------------:|-------------------------------:|----------------------------------:|------------------------------------------------:|
| new3    | queue_theta_val_tuned              |                  296225.000000 |                     350955.000000 |     54730.000000 |                           1658 |                                 0 |                                               1 |
| new3    | tiny_mlp_direct_action_h4_seed1    |                  377525.000000 |                     442410.000000 |     64885.000000 |                           2001 |                                 0 |                                               1 |
| new3    | tiny_mlp_direct_action_h8_seed0    |                  385380.000000 |                     449950.000000 |     64570.000000 |                           2042 |                                 0 |                                               1 |
| new3    | tiny_mlp_direct_action_h8_seed1    |                  406195.000000 |                     471655.000000 |     65460.000000 |                           2056 |                                 0 |                                               1 |
| new3    | tiny_mlp_direct_action_h8_seed2    |                  397775.000000 |                     462520.000000 |     64745.000000 |                           1937 |                                 0 |                                               1 |
| new3    | tiny_mlp_score_regression_h4_seed2 |                  349400.000000 |                     401945.000000 |     52545.000000 |                           2157 |                                 0 |                                               1 |
| new3    | tiny_mlp_score_regression_h8_seed0 |                  336255.000000 |                     396195.000000 |     59940.000000 |                           1744 |                                 0 |                                               1 |
| new3    | tiny_mlp_score_regression_h8_seed1 |                  343765.000000 |                     400385.000000 |     56620.000000 |                           1712 |                                 0 |                                               1 |
| new3    | tiny_mlp_score_regression_h8_seed2 |                  339735.000000 |                     397175.000000 |     57440.000000 |                           1724 |                                 0 |                                               1 |
| new4    | phc_eventwise_reference            |                  129160.000000 |                     219310.000000 |     90150.000000 |                           2589 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h4_seed0    |                   85550.000000 |                     139205.000000 |     53655.000000 |                           2022 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h4_seed1    |                   95960.000000 |                     156630.000000 |     60670.000000 |                           2101 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h4_seed2    |                   87465.000000 |                     148330.000000 |     60865.000000 |                           2088 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h8_seed0    |                   93335.000000 |                     151015.000000 |     57680.000000 |                           2167 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h8_seed1    |                   94770.000000 |                     153970.000000 |     59200.000000 |                           2199 |                                 0 |                                               1 |
| new4    | tiny_mlp_direct_action_h8_seed2    |                   84600.000000 |                     141165.000000 |     56565.000000 |                           2148 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h4_seed0 |                   82205.000000 |                     125530.000000 |     43325.000000 |                           1924 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h4_seed1 |                   79725.000000 |                     121350.000000 |     41625.000000 |                           1924 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h4_seed2 |                   86240.000000 |                     135545.000000 |     49305.000000 |                           2052 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h8_seed0 |                   89705.000000 |                     140535.000000 |     50830.000000 |                           1817 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h8_seed1 |                   84800.000000 |                     131550.000000 |     46750.000000 |                           1869 |                                 0 |                                               1 |
| new4    | tiny_mlp_score_regression_h8_seed2 |                   91605.000000 |                     136030.000000 |     44425.000000 |                           1864 |                                 0 |                                               1 |

## Val-selected tiny MLP under corrected semantics

Val selected `tiny_mlp_direct_action_h8_seed1`. NEW3/NEW4 were not used for selection.

| split   | policy_id                       | objective     |    width |     seed |   parameter_count |   legacy_sequential_excess_pnl |   corrected_sequential_excess_pnl |   pnl_correction |   legacy_forced_no_quote_count |   corrected_forced_no_quote_count |   corrected_terminal_immediate_settlement_count |   corrected_final_inventory |   phc_eventwise_reference |   queue_theta_val_tuned |   tiny_mlp_direct_action_h8_seed1 |   retention_vs_phc_pct |   delta_vs_static |
|:--------|:--------------------------------|:--------------|---------:|---------:|------------------:|-------------------------------:|----------------------------------:|-----------------:|-------------------------------:|----------------------------------:|------------------------------------------------:|----------------------------:|--------------------------:|------------------------:|----------------------------------:|-----------------------:|------------------:|
| val     | tiny_mlp_direct_action_h8_seed1 | direct_action | 8.000000 | 1.000000 |        163.000000 |                  139400.000000 |                     139400.000000 |         0.000000 |                              0 |                                 0 |                                               0 |                           0 |             171835.000000 |           113850.000000 |                     139400.000000 |              81.124334 |      25550.000000 |
| new3    | tiny_mlp_direct_action_h8_seed1 | direct_action | 8.000000 | 1.000000 |        163.000000 |                  406195.000000 |                     471655.000000 |     65460.000000 |                           2056 |                                 0 |                                               1 |                           0 |             611095.000000 |           350955.000000 |                     471655.000000 |              77.181944 |     120700.000000 |
| new4    | tiny_mlp_direct_action_h8_seed1 | direct_action | 8.000000 | 1.000000 |        163.000000 |                   94770.000000 |                     153970.000000 |     59200.000000 |                           2199 |                                 0 |                                               1 |                           0 |             219310.000000 |           115875.000000 |                     153970.000000 |              70.206557 |      38095.000000 |

## Paired session bootstrap

| split   | student_policy_id               | comparator              |   session_count |   corrected_total_difference |   mean_session_difference |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_probability_difference_le_zero |   n_bootstrap |
|:--------|:--------------------------------|:------------------------|----------------:|-----------------------------:|--------------------------:|-------------------:|--------------------:|-------------------------------------------:|--------------:|
| val     | tiny_mlp_direct_action_h8_seed1 | queue_theta_val_tuned   |               6 |                 25550.000000 |               4258.333333 |        2203.854167 |         6373.333333 |                                   0.000000 |         10000 |
| val     | tiny_mlp_direct_action_h8_seed1 | phc_eventwise_reference |               6 |                -32435.000000 |              -5405.833333 |       -7885.000000 |        -2813.333333 |                                   1.000000 |         10000 |
| new3    | tiny_mlp_direct_action_h8_seed1 | queue_theta_val_tuned   |               6 |                120700.000000 |              20116.666667 |       11196.666667 |        28625.000000 |                                   0.000000 |         10000 |
| new3    | tiny_mlp_direct_action_h8_seed1 | phc_eventwise_reference |               6 |               -139440.000000 |             -23240.000000 |      -31799.166667 |       -13552.500000 |                                   1.000000 |         10000 |
| new4    | tiny_mlp_direct_action_h8_seed1 | queue_theta_val_tuned   |               4 |                 38095.000000 |               9523.750000 |         840.000000 |        19683.750000 |                                   0.019500 |         10000 |
| new4    | tiny_mlp_direct_action_h8_seed1 | phc_eventwise_reference |               4 |                -65340.000000 |             -16335.000000 |      -30032.500000 |        -2637.500000 |                                   1.000000 |         10000 |

## Decision

Under the locally corrected settlement semantics, the val-selected tiny MLP clears the predeclared task-level gate and beats the tuned static rule on val, NEW3, and NEW4. This is still CPU-only diagnostic evidence. Before treating it as the official result, the shared simulator correction must be reviewed and all frozen policy baselines must be regenerated under one versioned protocol.
