# Final Non-ML Feasibility Decision Report

Final classification: `signal_found_but_strategy_not_proven`.

The current non-ML results should be classified as signal discovery, not a final strategy. The proxy environment was too weak.

No ML was trained, and no parameters were tuned on `test`, `NEW3`, or `NEW4`. The selected L2 queue rule remains the fixed Phase 02D rule: `side_by_queue_imbalance_abs_0p1` at `H20`.

## 1. NEW3 Frozen OOS Validity

NEW3 validity audit pass: `True`.

| new3_session_id   |   new3_quote_eligible_rows | session_id_overlaps_existing   | timestamp_hash_overlaps_existing   | classification   |
|:------------------|---------------------------:|:-------------------------------|:-----------------------------------|:-----------------|
| 20260113          |                    1277836 | False                          | False                              | valid_frozen_oos |
| 20260114          |                     910661 | False                          | False                              | valid_frozen_oos |
| 20260115          |                     976150 | False                          | False                              | valid_frozen_oos |
| 20260116          |                     976422 | False                          | False                              | valid_frozen_oos |
| 20260119Day       |                    1365489 | False                          | False                              | valid_frozen_oos |
| 20260330          |                     150804 | False                          | False                              | valid_frozen_oos |

## 2. NEW3 Frozen Results

| split   | policy_id                       |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_pnl |   win_rate |   loss_rate |   max_drawdown |   matched_random_mean |   matched_random_95p |   matched_random_99p | empirical_p_value_random_ge_model   |   random_trials | beats_matched_random_95p   | selection_split   | new3_used_for_selection   |
|:--------|:--------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|------------:|-----------:|------------:|---------------:|----------------------:|---------------------:|---------------------:|:------------------------------------|----------------:|:---------------------------|:------------------|:--------------------------|
| new3    | no_quote                        |        20 |       5657362 |                       0 |         0        |           0 |           0 | 0           |   0        |   0         |              0 |         nan           |        nan           |        nan           | nan                                 |             nan | False                      | val               | False                     |
| new3    | old_bid_only_spread_ge_1        |        20 |       5657362 |                 5657362 |         1        |     5657362 |           0 | 1.43247e+09 |   0.903431 |   0.0965482 |          60230 |           1.43247e+09 |          1.43247e+09 |          1.43247e+09 | 1.0                                 |            1000 | False                      | val               | False                     |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 |       5657362 |                 4631249 |         0.818623 |     2291586 |     2339663 | 1.33371e+09 |   0.940022 |   0.0599627 |          52125 |           1.17734e+09 |          1.17775e+09 |          1.17794e+09 | <0.001                              |            1000 | True                       | val               | False                     |

## 3. Old Bid-Only PnL Decomposition

| split   | session_id   |   horizon |   selected_action_count |   total_pnl |   half_spread_advantage_pnl |   mid_price_drift_pnl |   maker_fee_total |   avg_total_pnl |   avg_half_spread_advantage_pnl |   avg_mid_price_drift_pnl |   positive_rate |   spread_component_share_of_positive_components |   drift_component_share_of_positive_components |
|:--------|:-------------|----------:|------------------------:|------------:|----------------------------:|----------------------:|------------------:|----------------:|--------------------------------:|--------------------------:|----------------:|------------------------------------------------:|-----------------------------------------------:|
| new3    | __ALL__      |        20 |                 5657242 | 1.43247e+09 |                 1.51532e+09 |            2.004e+06  |       8.48586e+07 |         253.209 |                         267.855 |                0.354236   |        0.90345  |                                        0.998679 |                                    0.00132075  |
| new4    | __ALL__      |        20 |                 2196560 | 5.34754e+08 |                 5.6664e+08  |            1.0625e+06 |       3.29484e+07 |         243.451 |                         257.967 |                0.483711   |        0.892344 |                                        0.998128 |                                    0.00187158  |
| test    | __ALL__      |        20 |                 1166281 | 2.45096e+08 |                 2.61056e+08 |            1.5334e+06 |       1.74942e+07 |         210.151 |                         223.837 |                1.31478    |        0.909079 |                                        0.99416  |                                    0.00583952  |
| val     | __ALL__      |        20 |                  563457 | 1.1671e+08  |                 1.2516e+08  |         1800          |       8.45186e+06 |         207.132 |                         222.129 |                0.00319456 |        0.881075 |                                        0.999986 |                                    1.43814e-05 |

Interpretation: if half-spread advantage dominates, old bid-only is likely benefiting from the full-action markout simulator assuming quote opportunities without true fill probability or queue priority.

## 4. Sequential Inventory Proxy

Conservative view: `fill_mode=conservative_adverse_fill`, `inventory_cap=1`, `exit_rule=fixed_horizon_H20`.

The strategy-candidate hurdle requires the policy to stay positive **and** beat the `matched_random_95p` sequential proxy comparator under this conservative view.

| split   | policy_id                       |   horizon | fill_mode                 |   inventory_cap | exit_rule         |   sequential_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   avg_trade_pnl |   random_trials_for_95p_policy |
|:--------|:--------------------------------|----------:|:--------------------------|----------------:|:------------------|-----------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|----------------:|-------------------------------:|
| test    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| test    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      1.16368e+07 |        55806 |                  579486 |                   1 |           4800 |   0.899957 |         208.523 |                              0 |
| test    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      3.43407e+07 |       141464 |                  368641 |                   1 |           4200 |   0.938684 |         242.753 |                              0 |
| test    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      4.88139e+07 |       245125 |                  174199 |                   1 |           3200 |   0.867362 |         199.139 |                            250 |
| new4    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| new4    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.52911e+07 |       105240 |                 1112274 |                   1 |          13015 |   0.878145 |         240.318 |                              0 |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      8.49369e+07 |       306263 |                  656184 |                   1 |           5515 |   0.916709 |         277.333 |                              0 |
| new4    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      1.06115e+08 |       462343 |                  328660 |                   1 |           7715 |   0.842848 |         229.516 |                            250 |
| new3    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| new3    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      6.78404e+07 |       270886 |                 2832102 |                   1 |           4700 |   0.889769 |         250.439 |                              0 |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.25636e+08 |       781297 |                 1674581 |                   1 |          20600 |   0.926919 |         288.797 |                              0 |
| new3    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.84631e+08 |      1184932 |                  829880 |                   1 |          14790 |   0.858365 |         240.208 |                            250 |

## 5. Strict Execution Stress

Moderate stress for this phase is `fee_multiplier=1`, `positive_fill_keep_rate=0.5`, `adverse_multiplier=1.5`, `delay_events=0`. Fee stress also checks `fee_multiplier=5`.

| split   | policy_id                       |   horizon |   fee_multiplier |   positive_fill_keep_rate |   adverse_multiplier |   delay_events |   state_count |   selected_action_count |   selection_rate |   stress_total_pnl |   win_rate |   loss_rate |   max_drawdown | positive_under_stress   | moderate_stress   | strict_fee5_adverse3_keep01_delay10   |
|:--------|:--------------------------------|----------:|-----------------:|--------------------------:|---------------------:|---------------:|--------------:|------------------------:|-----------------:|-------------------:|-----------:|------------:|---------------:|:------------------------|:------------------|:--------------------------------------|
| val     | old_bid_only_spread_ge_1        |        20 |                1 |                       0.5 |                  1.5 |              0 |        563477 |                  563477 |         1        |        5.31344e+07 |   0.881044 |   0.118921  |        95422.5 | True                    | True              | False                                 |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |                1 |                       0.5 |                  1.5 |              0 |        563477 |                  468627 |         0.83167  |        5.27034e+07 |   0.924183 |   0.0757767 |        95422.5 | True                    | True              | False                                 |
| test    | old_bid_only_spread_ge_1        |        20 |                1 |                       0.5 |                  1.5 |              0 |       1166321 |                 1166321 |         1        |        1.15426e+08 |   0.909047 |   0.0909184 |         7072.5 | True                    | True              | False                                 |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |                1 |                       0.5 |                  1.5 |              0 |       1166321 |                  971600 |         0.833047 |        1.13652e+08 |   0.948985 |   0.0509757 |         2422.5 | True                    | True              | False                                 |
| new4    | old_bid_only_spread_ge_1        |        20 |                1 |                       0.5 |                  1.5 |              0 |       2196640 |                 2196640 |         1        |        2.49521e+08 |   0.892312 |   0.107652  |        16672.5 | True                    | True              | False                                 |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |                1 |                       0.5 |                  1.5 |              0 |       2196640 |                 1799304 |         0.819116 |        2.38079e+08 |   0.931123 |   0.0688394 |         9472.5 | True                    | True              | False                                 |
| new3    | old_bid_only_spread_ge_1        |        20 |                1 |                       0.5 |                  1.5 |              0 |       5657362 |                 5657362 |         1        |        6.75169e+08 |   0.903431 |   0.0965482 |         5572.5 | True                    | True              | False                                 |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 |                1 |                       0.5 |                  1.5 |              0 |       5657362 |                 4631249 |         0.818623 |        6.44863e+08 |   0.940022 |   0.0599627 |        12622.5 | True                    | True              | False                                 |

## 6. Feasibility Scorecard

| policy_id                       |   beats_matched_random_test_new4 | beats_matched_random_new3   | beats_sequential_matched_random_95p_cap1_H20   | survives_fee_5x_all_frozen   | survives_moderate_adverse_fill_all_frozen   | inventory_cap_1_fixed_H20_positive_all_frozen   |   strict_sequential_min_pnl_cap1_H20 |   min_gap_vs_sequential_random_cap1_H20 | strategy_candidate_pass   |
|:--------------------------------|---------------------------------:|:----------------------------|:-----------------------------------------------|:-----------------------------|:--------------------------------------------|:------------------------------------------------|-------------------------------------:|----------------------------------------:|:--------------------------|
| old_bid_only_spread_ge_1        |                              nan | False                       | False                                          | True                         | True                                        | True                                            |                          1.16368e+07 |                            -2.1679e+08  | False                     |
| side_by_queue_imbalance_abs_0p1 |                                1 | True                        | False                                          | True                         | True                                        | True                                            |                          3.43407e+07 |                            -5.89943e+07 | False                     |

## 7. Final Decision

The current non-ML results should be classified as signal discovery, not a final strategy. The proxy environment was too weak.

Allowed wording: `non-trivial L2 signal discovery`, `proxy quote-control primitive`, and `candidate for stricter execution modeling`.

Forbidden wording: do not claim true queue replay, real-world profitability, or FPGA-driven PnL improvement.
