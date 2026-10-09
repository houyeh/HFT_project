# Sparse Linear Policy Audit

status: `resolved_negative`

## Claim boundary

This is a CPU-only task-level feasibility experiment. It does not provide HLS, routed timing, power, or on-board evidence. The frozen 05C4 seed-2 PHC scores are the distillation target; PHC itself is not retrained. Train fits coefficients, val selects the model, and NEW3/NEW4 are post-selection diagnostics. NEW5 was not loaded.

## Pre-registered candidates

Three model families were evaluated with `[2, 4, 6, 8, 12, 16]` nonzero terms per output: base features with uniform score regression, base features with decision-boundary weighting, and base plus all pairwise interactions with boundary weighting. Ridge alpha is `1.0`. Boundary samples receive up to `5.0x` weight; the train-only distance scale is `6.1128798`. The frozen PHC action threshold is `0` and the val-tuned queue comparator threshold is `0.25`.

## Val-selected champion

Val selected `sparse_base_boundary_weighted_k4`: variant `base_boundary_weighted`, `4` terms per output, `8` total coefficients, and `5` unique terms. Selection used causal-fill sequential PnL, not R-squared.

| split   | variant                |   budget_per_output |   coefficient_count |   unique_term_count |   score_r2_uniform |   action_agreement_rate |   decision_flip_count |   sequential_excess_pnl |   fill_count |   avg_pnl_per_fill |   max_drawdown |   phc_eventwise_reference |   pnl_retention_vs_phc_pct |   queue_theta_val_tuned |   pnl_delta_vs_static |
|:--------|:-----------------------|--------------------:|--------------------:|--------------------:|-------------------:|------------------------:|----------------------:|------------------------:|-------------:|-------------------:|---------------:|--------------------------:|---------------------------:|------------------------:|----------------------:|
| val     | base_boundary_weighted |                   4 |                   8 |                   5 |             0.4023 |                  0.5469 |                 37589 |             116450.0000 |         4950 |            23.5253 |      2040.0000 |               171835.0000 |                    67.7685 |             113850.0000 |             2600.0000 |
| new3    | base_boundary_weighted |                   4 |                   8 |                   5 |             0.4236 |                  0.5609 |                 77605 |             292890.0000 |        10895 |            26.8830 |      3280.0000 |               611095.0000 |                    47.9287 |             296225.0000 |            -3335.0000 |
| new4    | base_boundary_weighted |                   4 |                   8 |                   5 |             0.4045 |                  0.5510 |                 30807 |              78050.0000 |         2931 |            26.6291 |      2645.0000 |               129160.0000 |                    60.4289 |             115875.0000 |           -37825.0000 |

## Full capacity sweep

| split   | variant                       |   budget_per_output |   coefficient_count |   unique_term_count |   score_r2_uniform |   action_agreement_rate |   sequential_excess_pnl |   fill_count |   avg_pnl_per_fill |   max_drawdown |
|:--------|:------------------------------|--------------------:|--------------------:|--------------------:|-------------------:|------------------------:|------------------------:|-------------:|-------------------:|---------------:|
| val     | base_uniform                  |                   2 |                   4 |                   2 |             0.4176 |                  0.5715 |             104915.0000 |         5859 |            17.9066 |      2470.0000 |
| val     | base_uniform                  |                   4 |                   8 |                   4 |             0.4314 |                  0.5696 |             106515.0000 |         5819 |            18.3047 |      2585.0000 |
| val     | base_uniform                  |                   6 |                  12 |                   6 |             0.4325 |                  0.5719 |             108965.0000 |         5809 |            18.7580 |      2470.0000 |
| val     | base_uniform                  |                   8 |                  16 |                   8 |             0.4325 |                  0.5720 |             109040.0000 |         5804 |            18.7870 |      2255.0000 |
| val     | base_uniform                  |                  12 |                  24 |                  14 |             0.4326 |                  0.5732 |             111050.0000 |         5790 |            19.1796 |      2255.0000 |
| val     | base_uniform                  |                  16 |                  32 |                  16 |             0.4329 |                  0.5732 |             109665.0000 |         5789 |            18.9437 |      2255.0000 |
| val     | base_boundary_weighted        |                   2 |                   4 |                   3 |             0.3870 |                  0.5442 |             113680.0000 |         4908 |            23.1622 |      2065.0000 |
| val     | base_boundary_weighted        |                   4 |                   8 |                   5 |             0.4023 |                  0.5469 |             116450.0000 |         4950 |            23.5253 |      2040.0000 |
| val     | base_boundary_weighted        |                   6 |                  12 |                   7 |             0.4037 |                  0.5469 |             116100.0000 |         4940 |            23.5020 |      2120.0000 |
| val     | base_boundary_weighted        |                   8 |                  16 |                   9 |             0.4038 |                  0.5474 |             116080.0000 |         4948 |            23.4600 |      2040.0000 |
| val     | base_boundary_weighted        |                  12 |                  24 |                  13 |             0.4050 |                  0.5489 |             115255.0000 |         5003 |            23.0372 |      2015.0000 |
| val     | base_boundary_weighted        |                  16 |                  32 |                  16 |             0.4052 |                  0.5488 |             114955.0000 |         5003 |            22.9772 |      2540.0000 |
| val     | interaction_boundary_weighted |                   2 |                   4 |                   2 |             0.1755 |                  0.4049 |              36825.0000 |         3285 |            11.2100 |      4150.0000 |
| val     | interaction_boundary_weighted |                   4 |                   8 |                   4 |             0.4012 |                  0.5416 |             113900.0000 |         4840 |            23.5331 |      2105.0000 |
| val     | interaction_boundary_weighted |                   6 |                  12 |                   7 |             0.4106 |                  0.5341 |             112125.0000 |         4745 |            23.6301 |      2060.0000 |
| val     | interaction_boundary_weighted |                   8 |                  16 |                  10 |             0.4128 |                  0.5333 |             108390.0000 |         4754 |            22.7997 |      1990.0000 |
| val     | interaction_boundary_weighted |                  12 |                  24 |                  14 |             0.4173 |                  0.5298 |             112000.0000 |         4620 |            24.2424 |      2020.0000 |
| val     | interaction_boundary_weighted |                  16 |                  32 |                  17 |             0.4194 |                  0.5253 |             112130.0000 |         4558 |            24.6007 |      1905.0000 |
| new3    | base_uniform                  |                   2 |                   4 |                   2 |             0.4588 |                  0.5911 |             304720.0000 |        12733 |            23.9315 |      3570.0000 |
| new3    | base_uniform                  |                   4 |                   8 |                   4 |             0.4734 |                  0.5907 |             301965.0000 |        12690 |            23.7955 |      3570.0000 |
| new3    | base_uniform                  |                   6 |                  12 |                   6 |             0.4739 |                  0.5915 |             301590.0000 |        12735 |            23.6820 |      3155.0000 |
| new3    | base_uniform                  |                   8 |                  16 |                   8 |             0.4739 |                  0.5916 |             302230.0000 |        12739 |            23.7248 |      3155.0000 |
| new3    | base_uniform                  |                  12 |                  24 |                  14 |             0.4752 |                  0.5930 |             300525.0000 |        12746 |            23.5780 |      3270.0000 |
| new3    | base_uniform                  |                  16 |                  32 |                  16 |             0.4756 |                  0.5925 |             298910.0000 |        12707 |            23.5233 |      3155.0000 |
| new3    | base_boundary_weighted        |                   2 |                   4 |                   3 |             0.4023 |                  0.5560 |             295610.0000 |        10687 |            27.6607 |      3280.0000 |
| new3    | base_boundary_weighted        |                   4 |                   8 |                   5 |             0.4236 |                  0.5609 |             292890.0000 |        10895 |            26.8830 |      3280.0000 |
| new3    | base_boundary_weighted        |                   6 |                  12 |                   7 |             0.4239 |                  0.5610 |             293390.0000 |        10895 |            26.9289 |      3280.0000 |
| new3    | base_boundary_weighted        |                   8 |                  16 |                   9 |             0.4248 |                  0.5612 |             293100.0000 |        10901 |            26.8874 |      3280.0000 |
| new3    | base_boundary_weighted        |                  12 |                  24 |                  13 |             0.4272 |                  0.5629 |             301325.0000 |        11066 |            27.2298 |      3280.0000 |
| new3    | base_boundary_weighted        |                  16 |                  32 |                  16 |             0.4274 |                  0.5631 |             300600.0000 |        11021 |            27.2752 |      3280.0000 |
| new3    | interaction_boundary_weighted |                   2 |                   4 |                   2 |             0.1693 |                  0.3823 |              95160.0000 |         6617 |            14.3811 |      4395.0000 |
| new3    | interaction_boundary_weighted |                   4 |                   8 |                   4 |             0.4134 |                  0.5554 |             296480.0000 |        10629 |            27.8935 |      3280.0000 |
| new3    | interaction_boundary_weighted |                   6 |                  12 |                   7 |             0.4355 |                  0.5537 |             294355.0000 |        10624 |            27.7066 |      3280.0000 |
| new3    | interaction_boundary_weighted |                   8 |                  16 |                  10 |             0.4381 |                  0.5532 |             296740.0000 |        10665 |            27.8237 |      3280.0000 |
| new3    | interaction_boundary_weighted |                  12 |                  24 |                  14 |             0.4422 |                  0.5496 |             303555.0000 |        10324 |            29.4028 |      3090.0000 |
| new3    | interaction_boundary_weighted |                  16 |                  32 |                  17 |             0.4445 |                  0.5465 |             299570.0000 |        10203 |            29.3610 |      3090.0000 |
| new4    | base_uniform                  |                   2 |                   4 |                   2 |             0.4198 |                  0.5800 |              77915.0000 |         3480 |            22.3894 |      2595.0000 |
| new4    | base_uniform                  |                   4 |                   8 |                   4 |             0.4373 |                  0.5787 |              77575.0000 |         3476 |            22.3173 |      2595.0000 |
| new4    | base_uniform                  |                   6 |                  12 |                   6 |             0.4362 |                  0.5793 |              79770.0000 |         3423 |            23.3041 |      2695.0000 |
| new4    | base_uniform                  |                   8 |                  16 |                   8 |             0.4362 |                  0.5794 |              79755.0000 |         3424 |            23.2929 |      2695.0000 |
| new4    | base_uniform                  |                  12 |                  24 |                  14 |             0.4377 |                  0.5804 |              81040.0000 |         3445 |            23.5239 |      2625.0000 |
| new4    | base_uniform                  |                  16 |                  32 |                  16 |             0.4382 |                  0.5806 |              82500.0000 |         3441 |            23.9756 |      2510.0000 |
| new4    | base_boundary_weighted        |                   2 |                   4 |                   3 |             0.3832 |                  0.5469 |             115550.0000 |         4650 |            24.8495 |      5045.0000 |
| new4    | base_boundary_weighted        |                   4 |                   8 |                   5 |             0.4045 |                  0.5510 |              78050.0000 |         2931 |            26.6291 |      2645.0000 |
| new4    | base_boundary_weighted        |                   6 |                  12 |                   7 |             0.4047 |                  0.5513 |              78295.0000 |         2928 |            26.7401 |      2645.0000 |
| new4    | base_boundary_weighted        |                   8 |                  16 |                   9 |             0.4048 |                  0.5515 |              79505.0000 |         2934 |            27.0978 |      2660.0000 |
| new4    | base_boundary_weighted        |                  12 |                  24 |                  13 |             0.4066 |                  0.5534 |              79905.0000 |         2974 |            26.8679 |      2660.0000 |
| new4    | base_boundary_weighted        |                  16 |                  32 |                  16 |             0.4070 |                  0.5533 |              79310.0000 |         2967 |            26.7307 |      2745.0000 |
| new4    | interaction_boundary_weighted |                   2 |                   4 |                   2 |             0.1633 |                  0.3844 |              40655.0000 |         3363 |            12.0889 |      3430.0000 |
| new4    | interaction_boundary_weighted |                   4 |                   8 |                   4 |             0.3936 |                  0.5463 |             115635.0000 |         4611 |            25.0781 |      5045.0000 |
| new4    | interaction_boundary_weighted |                   6 |                  12 |                   7 |             0.4163 |                  0.5448 |              77060.0000 |         2817 |            27.3553 |      2400.0000 |
| new4    | interaction_boundary_weighted |                   8 |                  16 |                  10 |             0.4193 |                  0.5443 |              77245.0000 |         2818 |            27.4113 |      2400.0000 |
| new4    | interaction_boundary_weighted |                  12 |                  24 |                  14 |             0.4241 |                  0.5409 |             116840.0000 |         4464 |            26.1738 |      5030.0000 |
| new4    | interaction_boundary_weighted |                  16 |                  32 |                  17 |             0.4262 |                  0.5370 |             116745.0000 |         4417 |            26.4308 |      5030.0000 |

## Champion formula

The following coefficients are in the original feature basis. The score is `intercept + sum(coefficient * term)`; a term containing `*` is one pairwise product. These are directly inspectable formulas, although fixed-point implementation has not yet been tested.

| variant                |   budget_per_output | output   | term            |   raw_coefficient | is_interaction   |
|:-----------------------|--------------------:|:---------|:----------------|------------------:|:-----------------|
| base_boundary_weighted |                   4 | bid      | <intercept>     |      -15.346562   | False            |
| base_boundary_weighted |                   4 | bid      | ask_sz_1        |       16.6871876  | False            |
| base_boundary_weighted |                   4 | bid      | top_depth       |      -14.9597886  | False            |
| base_boundary_weighted |                   4 | bid      | queue_imbalance |       27.8052264  | False            |
| base_boundary_weighted |                   4 | bid      | log1p_bid_sz_1  |        5.10481171 | False            |
| base_boundary_weighted |                   4 | ask      | <intercept>     |      -14.8576559  | False            |
| base_boundary_weighted |                   4 | ask      | bid_sz_1        |       -2.93412313 | False            |
| base_boundary_weighted |                   4 | ask      | ask_sz_1        |      -17.8303194  | False            |
| base_boundary_weighted |                   4 | ask      | top_depth       |       14.8324739  | False            |
| base_boundary_weighted |                   4 | ask      | queue_imbalance |      -30.3565353  | False            |

## Paired session bootstrap

The canonical simulator carries its inventory variable across session boundaries. Session bootstrap necessarily resets inventory for each sampled session, so both the canonical full-split difference and the session-reset estimand are shown. Four to six sessions per split provide low statistical power.

| split   | student_policy_id                | comparator              |   session_count |   full_split_total_difference |   session_reset_total_difference |   mean_session_difference |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_probability_difference_le_zero |   n_bootstrap |
|:--------|:---------------------------------|:------------------------|----------------:|------------------------------:|---------------------------------:|--------------------------:|-------------------:|--------------------:|-------------------------------------------:|--------------:|
| val     | sparse_base_boundary_weighted_k4 | queue_theta_val_tuned   |               6 |                   2600.000000 |                      2600.000000 |                433.333333 |         -65.000000 |         1193.333333 |                                   0.078900 |         10000 |
| val     | sparse_base_boundary_weighted_k4 | phc_eventwise_reference |               6 |                 -55385.000000 |                    -55385.000000 |              -9230.833333 |      -11445.833333 |        -6810.000000 |                                   1.000000 |         10000 |
| new3    | sparse_base_boundary_weighted_k4 | queue_theta_val_tuned   |               6 |                  -3335.000000 |                     -3840.000000 |               -640.000000 |       -1297.500000 |          205.833333 |                                   0.934500 |         10000 |
| new3    | sparse_base_boundary_weighted_k4 | phc_eventwise_reference |               6 |                -318205.000000 |                   -263980.000000 |             -43996.666667 |      -58332.500000 |       -26160.833333 |                                   1.000000 |         10000 |
| new4    | sparse_base_boundary_weighted_k4 | queue_theta_val_tuned   |               4 |                 -37825.000000 |                     -1910.000000 |               -477.500000 |       -1127.500000 |          172.500000 |                                   0.922900 |         10000 |
| new4    | sparse_base_boundary_weighted_k4 | phc_eventwise_reference |               4 |                 -51110.000000 |                   -105460.000000 |             -26365.000000 |      -49367.500000 |        -3362.500000 |                                   1.000000 |         10000 |

## Decision

The sparse policy does not simultaneously retain at least 80% of PHC val PnL and beat the tuned static queue rule on both NEW3 and NEW4. This experiment therefore does not yet justify an HLS implementation.

## Direct-action follow-up

The companion direct-action experiment in `AUDIT_sparse_action_policy.md` improves the task-level result with only five unique terms. It is the stronger sparse candidate, but its status is still `mixed`: canonical PnL is 117,380 / 301,415 / 81,885 on val / NEW3 / NEW4, corresponding to 68.31% / 49.32% / 63.40% retention versus eventwise PHC. It beats the tuned static rule on val and NEW3, not NEW4; with per-session inventory reset it is above static on all three splits. No OOS result was used to choose the model.

## Outputs

- `results/sparse_policy_metrics.csv`
- `results/sparse_policy_coefficients.csv`
- `results/reference_results.csv`
- `results/selected_champion.json`
- `results/paired_session_bootstrap.csv`
- `figures/sparse_policy_pnl_frontier.png`
