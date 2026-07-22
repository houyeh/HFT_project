# Excess Markout and Causal Fill Proxy Report

Final classification: `signal_only_after_spread_adjustment`.

After removing half-spread capture, some residual signal remains, but it does not satisfy the full strategy-candidate hurdle.

No ML was trained. No policy thresholds or parameters were tuned on `test`, `NEW3`, or `NEW4`. The only evaluated non-ML policies are frozen controls from earlier phases.

## 1. Excess Markout Definition

The raw Phase 01 markout included mechanical half-spread capture. Phase 02F removes that component:

```text
excess_bid_pnl_H = (future_mid_H - mid_t) * point_value - fee
excess_ask_pnl_H = (mid_t - future_mid_H) * point_value - fee
no_quote excess_pnl_H = 0
```

Future mid is used only as a label. It is not used for policy decisions or fill-proxy decisions.

## 2. Excess Label Summary

| split   | action_side   |   horizon |   row_count |   finite_label_count |   mean_excess_pnl |   median_excess_pnl |   positive_rate |   negative_rate |
|:--------|:--------------|----------:|------------:|---------------------:|------------------:|--------------------:|----------------:|----------------:|
| new3    | ask           |        20 |     5657362 |              5657242 |          -15.435  |                 -15 |        0.354914 |        0.645086 |
| new3    | bid           |        20 |     5657362 |              5657242 |          -14.565  |                 -15 |        0.354535 |        0.645465 |
| new3    | no_quote      |        20 |     5657362 |              5657362 |            0      |                   0 |        0        |        0        |
| new4    | ask           |        20 |     2196640 |              2196560 |          -15.8372 |                 -15 |        0.357472 |        0.642528 |
| new4    | bid           |        20 |     2196640 |              2196560 |          -14.1628 |                 -15 |        0.357368 |        0.642632 |
| new4    | no_quote      |        20 |     2196640 |              2196640 |            0      |                   0 |        0        |        0        |
| test    | ask           |        20 |     1166321 |              1166281 |          -16.6478 |                 -15 |        0.30709  |        0.69291  |
| test    | bid           |        20 |     1166321 |              1166281 |          -13.3522 |                 -15 |        0.318269 |        0.681731 |
| test    | no_quote      |        20 |     1166321 |              1166321 |            0      |                   0 |        0        |        0        |
| train   | ask           |        20 |     3592764 |              3592644 |          -15.1784 |                 -15 |        0.350983 |        0.649017 |
| train   | bid           |        20 |     3592764 |              3592644 |          -14.8216 |                 -15 |        0.355113 |        0.644887 |
| train   | no_quote      |        20 |     3592764 |              3592764 |            0      |                   0 |        0        |        0        |
| val     | ask           |        20 |      563477 |               563457 |          -15.0032 |                 -15 |        0.347457 |        0.652543 |
| val     | bid           |        20 |      563477 |               563457 |          -14.9968 |                 -15 |        0.352863 |        0.647137 |
| val     | no_quote      |        20 |      563477 |               563477 |            0      |                   0 |        0        |        0        |

## 3. Excess Policy Evaluation

| split   | policy_id                       | base_policy_id                  |   horizon |   state_count |   selected_action_count |   selection_rate |        bid_count |        ask_count |   total_excess_pnl |   win_rate |   loss_rate |     max_drawdown |   matched_random_mean |   matched_random_95p |   matched_random_std_normal_approx | random_method                                              | beats_matched_random_95p   | selection_split   | test_new3_new4_used_for_selection   |
|:--------|:--------------------------------|:--------------------------------|----------:|--------------:|------------------------:|-----------------:|-----------------:|-----------------:|-------------------:|-----------:|------------:|-----------------:|----------------------:|---------------------:|-----------------------------------:|:-----------------------------------------------------------|:---------------------------|:------------------|:------------------------------------|
| test    | old_bid_only_spread_ge_1        | old_bid_only_spread_ge_1        |        20 |   1.16632e+06 |                 1166321 |         1        |      1.16632e+06 |      0           |       -1.59608e+07 |   0.307402 |    0.692564 |      1.5966e+07  |          -1.59608e+07 |         -4.28868e+06 |                        7.09616e+06 | finite_population_normal_approx_side_session_spread_bucket | False                      | val               | False                               |
| test    | side_by_queue_imbalance_abs_0p1 | side_by_queue_imbalance_abs_0p1 |        20 |   1.16632e+06 |                  971600 |         0.833047 | 480075           | 491525           |        1.63572e+07 |   0.392513 |    0.607448 |  44890           |          -1.45915e+07 |         -1.44354e+07 |                    94902.9         | finite_population_normal_approx_side_session_spread_bucket | True                       | val               | False                               |
| new4    | old_bid_only_spread_ge_1        | old_bid_only_spread_ge_1        |        20 |   2.19664e+06 |                 2196640 |         1        |      2.19664e+06 |      0           |       -3.18859e+07 |   0.356891 |    0.643073 |      3.20564e+07 |          -3.18859e+07 |         -1.32413e+07 |                        1.13351e+07 | finite_population_normal_approx_side_session_spread_bucket | False                      | val               | False                               |
| new4    | side_by_queue_imbalance_abs_0p1 | side_by_queue_imbalance_abs_0p1 |        20 |   2.19664e+06 |                 1799304 |         0.819116 | 887511           | 911793           |        2.98959e+07 |   0.427337 |    0.572626 |  85265           |          -2.69926e+07 |         -2.67097e+07 |                   171949           | finite_population_normal_approx_side_session_spread_bucket | True                       | val               | False                               |
| new3    | old_bid_only_spread_ge_1        | old_bid_only_spread_ge_1        |        20 |   5.65736e+06 |                 5657362 |         1        |      5.65736e+06 |      0           |       -8.28546e+07 |   0.357061 |    0.642917 |      8.30586e+07 |          -8.28546e+07 |         -4.70337e+07 |                        2.17776e+07 | finite_population_normal_approx_side_session_spread_bucket | False                      | val               | False                               |
| new3    | side_by_queue_imbalance_abs_0p1 | side_by_queue_imbalance_abs_0p1 |        20 |   5.65736e+06 |                 4631249 |         0.818623 |      2.29159e+06 |      2.33966e+06 |        8.67477e+07 |   0.434742 |    0.565242 | 115340           |          -6.94709e+07 |         -6.90301e+07 |                   267956           | finite_population_normal_approx_side_session_spread_bucket | True                       | val               | False                               |

Matched random is a finite-population normal approximation matching side, session, spread bucket, and selected count. It is used as a strict non-ML hurdle, not for tuning.

## 4. Causal Fill Proxy

Fill decisions use current L2 state only. They do not inspect `pnl_H`, `excess_pnl_H`, future mid, or whether the quote was profitable.

| split   | policy_id                       | fill_model             |   state_count |   selected_action_count |   mean_fill_probability_selected |   deterministic_fill_count |   deterministic_fill_rate_selected |   bid_fill_count |   ask_fill_count | uses_future_pnl_for_fill   | normalization_source           |
|:--------|:--------------------------------|:-----------------------|--------------:|------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-----------------:|-----------------:|:---------------------------|:-------------------------------|
| val     | old_bid_only_spread_ge_1        | depth_haircut          |        563477 |                  563477 |                        0.471963  |                     266160 |                          0.472353  |           266160 |                0 | False                      | train_split_current_l2_medians |
| val     | old_bid_only_spread_ge_1        | queue_side_haircut     |        563477 |                  563477 |                        0.469162  |                     264245 |                          0.468954  |           264245 |                0 | False                      | train_split_current_l2_medians |
| val     | old_bid_only_spread_ge_1        | conservative_l2_stress |        563477 |                  563477 |                        0.118995  |                      67011 |                          0.118924  |            67011 |                0 | False                      | train_split_current_l2_medians |
| val     | side_by_queue_imbalance_abs_0p1 | depth_haircut          |        563477 |                  468627 |                        0.465581  |                     217995 |                          0.465178  |           107444 |           110551 | False                      | train_split_current_l2_medians |
| val     | side_by_queue_imbalance_abs_0p1 | queue_side_haircut     |        563477 |                  468627 |                        0.352568  |                     165273 |                          0.352675  |            81229 |            84044 | False                      | train_split_current_l2_medians |
| val     | side_by_queue_imbalance_abs_0p1 | conservative_l2_stress |        563477 |                  468627 |                        0.0894391 |                      42072 |                          0.0897772 |            20836 |            21236 | False                      | train_split_current_l2_medians |
| test    | old_bid_only_spread_ge_1        | depth_haircut          |       1166321 |                 1166321 |                        0.488733  |                     570047 |                          0.488757  |           570047 |                0 | False                      | train_split_current_l2_medians |
| test    | old_bid_only_spread_ge_1        | queue_side_haircut     |       1166321 |                 1166321 |                        0.486188  |                     567328 |                          0.486425  |           567328 |                0 | False                      | train_split_current_l2_medians |
| test    | old_bid_only_spread_ge_1        | conservative_l2_stress |       1166321 |                 1166321 |                        0.126995  |                     148326 |                          0.127174  |           148326 |                0 | False                      | train_split_current_l2_medians |
| test    | side_by_queue_imbalance_abs_0p1 | depth_haircut          |       1166321 |                  971600 |                        0.479268  |                     465684 |                          0.479296  |           231007 |           234677 | False                      | train_split_current_l2_medians |
| test    | side_by_queue_imbalance_abs_0p1 | queue_side_haircut     |       1166321 |                  971600 |                        0.364203  |                     354330 |                          0.364687  |           176067 |           178263 | False                      | train_split_current_l2_medians |
| test    | side_by_queue_imbalance_abs_0p1 | conservative_l2_stress |       1166321 |                  971600 |                        0.0947086 |                      92292 |                          0.0949897 |            45815 |            46477 | False                      | train_split_current_l2_medians |
| new4    | old_bid_only_spread_ge_1        | depth_haircut          |       2196640 |                 2196640 |                        0.52143   |                    1145787 |                          0.521609  |          1145787 |                0 | False                      | train_split_current_l2_medians |
| new4    | old_bid_only_spread_ge_1        | queue_side_haircut     |       2196640 |                 2196640 |                        0.515708  |                    1132740 |                          0.515669  |          1132740 |                0 | False                      | train_split_current_l2_medians |
| new4    | old_bid_only_spread_ge_1        | conservative_l2_stress |       2196640 |                 2196640 |                        0.14211   |                     312375 |                          0.142206  |           312375 |                0 | False                      | train_split_current_l2_medians |
| new4    | side_by_queue_imbalance_abs_0p1 | depth_haircut          |       2196640 |                 1799304 |                        0.507532  |                     913535 |                          0.507716  |           453441 |           460094 | False                      | train_split_current_l2_medians |
| new4    | side_by_queue_imbalance_abs_0p1 | queue_side_haircut     |       2196640 |                 1799304 |                        0.391561  |                     704267 |                          0.391411  |           349319 |           354948 | False                      | train_split_current_l2_medians |
| new4    | side_by_queue_imbalance_abs_0p1 | conservative_l2_stress |       2196640 |                 1799304 |                        0.106462  |                     191203 |                          0.106265  |            95327 |            95876 | False                      | train_split_current_l2_medians |
| new3    | old_bid_only_spread_ge_1        | depth_haircut          |       5657362 |                 5657362 |                        0.529205  |                    2994244 |                          0.529265  |          2994244 |                0 | False                      | train_split_current_l2_medians |
| new3    | old_bid_only_spread_ge_1        | queue_side_haircut     |       5657362 |                 5657362 |                        0.521804  |                    2952179 |                          0.52183   |          2952179 |                0 | False                      | train_split_current_l2_medians |
| new3    | old_bid_only_spread_ge_1        | conservative_l2_stress |       5657362 |                 5657362 |                        0.145281  |                     822627 |                          0.145408  |           822627 |                0 | False                      | train_split_current_l2_medians |
| new3    | side_by_queue_imbalance_abs_0p1 | depth_haircut          |       5657362 |                 4631249 |                        0.515683  |                    2388618 |                          0.515761  |          1185454 |          1203164 | False                      | train_split_current_l2_medians |
| new3    | side_by_queue_imbalance_abs_0p1 | queue_side_haircut     |       5657362 |                 4631249 |                        0.399104  |                    1846827 |                          0.398775  |           917882 |           928945 | False                      | train_split_current_l2_medians |
| new3    | side_by_queue_imbalance_abs_0p1 | conservative_l2_stress |       5657362 |                 4631249 |                        0.109454  |                     506638 |                          0.109396  |           252950 |           253688 | False                      | train_split_current_l2_medians |

## 5. Sequential Excess Inventory

Focus view: `conservative_l2_stress`, `inventory_cap=1`, `fee_multiplier=1`, `H20`.

| split   | policy_id                       |   horizon | fill_model             |   inventory_cap |   fee_multiplier |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   |
|:--------|:--------------------------------|----------:|:-----------------------|----------------:|-----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|
| test    | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |                1 |       -776330           |        41363 |                  106963 |                   1 |         519585 |   0.2964   |    0.703576 |               -18.7687 | False                  |
| test    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                1 |        818715           |        51440 |                   40852 |                   1 |           3930 |   0.3986   |    0.60138  |                15.9159 | False                  |
| new4    | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |                1 |            -1.59508e+06 |        81563 |                  230812 |                   1 |         721470 |   0.347081 |    0.65287  |               -19.5565 | False                  |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                1 |             1.85773e+06 |       109382 |                   81821 |                   1 |           6245 |   0.436214 |    0.563749 |                16.9839 | False                  |
| new3    | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |                1 |            -4.03201e+06 |       211797 |                  610830 |                   1 |         972850 |   0.347418 |    0.652568 |               -19.0371 | False                  |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                1 |             5.46686e+06 |       287034 |                  219604 |                   1 |          18845 |   0.440561 |    0.559422 |                19.0461 | False                  |

Fee-5 strict view for the selected queue rule:

| split   | policy_id                       |   horizon | fill_model             |   inventory_cap |   fee_multiplier |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   |
|:--------|:--------------------------------|----------:|:-----------------------|----------------:|-----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|
| test    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                5 |            -2.26768e+06 |        51440 |                   40852 |                   1 |    1.49535e+06 |   0.3986   |    0.6014   |               -44.0841 | False                  |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                5 |            -4.70513e+06 |       109382 |                   81821 |                   1 |    2.06621e+06 |   0.436214 |    0.563786 |               -43.0156 | False                  |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |                5 |            -1.17552e+07 |       287034 |                  219604 |                   1 |    2.72052e+06 |   0.440561 |    0.559439 |               -40.9539 | False                  |

Equity figure: `figures/sequential_excess_equity_curves.png`.

## 6. Final Environment Adequacy Decision

After removing half-spread capture, some residual signal remains, but it does not satisfy the full strategy-candidate hurdle.

If policies fail to beat matched random under `excess_pnl`, the previous high raw PnL must be described as a proxy artifact. Any remaining value is limited to signal discovery and an HLS-friendly quote gate primitive, not a final strategy.

Allowed wording: `spread-adjusted signal discovery`, `causal L2 quote-gate proxy`, `environment adequacy failure for final strategy claims`.

Forbidden wording: real-world profitability, true queue replay, or FPGA-proven PnL improvement.
