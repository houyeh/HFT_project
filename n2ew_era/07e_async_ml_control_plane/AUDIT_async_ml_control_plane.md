# Async PHC Control-Plane Audit

status: `resolved_negative`

## Claim boundary

This experiment does **not** demonstrate that the recurrent PHC model was physically executed only once per update interval. It replays cached eventwise PHC scores from Phase 07D, aggregates the previous completed time block, and applies that lagged control output to a fast queue rule. The supported claim is therefore: **PHC can be sampled as an asynchronous off-critical-path controller**. True low-frequency recurrent execution with state evolution only at update times remains a separate experiment. NEW3 and NEW4 are post-selection diagnostics; NEW5 was neither loaded nor used.

## Pre-registered mechanism

- Sessions are blocked independently; no block crosses a session boundary.
- The main result uses the **mean score from the previous completed block**. The first block of every session falls back to the val-tuned static queue rule.
- Update intervals are 1, 2, 5, 10, 30, and 60 seconds.
- `enable` gates quoting, `threshold` switches between 0.75x/1.25x the static threshold, `direction` allows BID-only/ASK-only outside a train-calibrated dead band, and `combined` applies all three.
- Hyperparameter quantiles are calibrated on train and selected on val only. NEW3/NEW4 do not select intervals or settings.
- Non-ML controls use lagged volatility plus event rate for activity. Because those variables have no side, the direction control necessarily uses lagged mean queue imbalance; this is explicitly not a volatility-only directional signal.

## Static threshold control

The val-selected static queue threshold is `0.25` with causal-fill PnL `113850.000`. This is the comparator; improvements over the historical theta=0.1 rule alone are not credited to ML.

## Reference endpoints

| split   | policy_id               |   sequential_excess_pnl |   fill_count |   avg_pnl_per_fill |   max_drawdown |
|:--------|:------------------------|------------------------:|-------------:|-------------------:|---------------:|
| val     | queue_theta_0p1         |             102915.0000 |         6419 |            16.0329 |      2875.0000 |
| val     | queue_theta_val_tuned   |             113850.0000 |         4930 |            23.0933 |      2085.0000 |
| val     | phc_eventwise_reference |             171835.0000 |         6891 |            24.9361 |      3240.0000 |
| new3    | queue_theta_0p1         |             303585.0000 |        13822 |            21.9639 |      3615.0000 |
| new3    | queue_theta_val_tuned   |             296225.0000 |        10706 |            27.6691 |      3280.0000 |
| new3    | phc_eventwise_reference |             611095.0000 |        17907 |            34.1260 |      5370.0000 |
| new4    | queue_theta_0p1         |              72415.0000 |         3820 |            18.9568 |      3560.0000 |
| new4    | queue_theta_val_tuned   |             115875.0000 |         4675 |            24.7861 |      5045.0000 |
| new4    | phc_eventwise_reference |             129160.0000 |         4237 |            30.4838 |      2695.0000 |

`phc_eventwise_reference` is a reference endpoint, not a guaranteed upper bound: a controller may outperform it by trading less often.

## Val-selected controller champions

| source   | mode      | policy_id                |   interval_s |   selected_q_on_val |   sequential_excess_pnl |   fill_count |   avg_pnl_per_fill |   max_drawdown |
|:---------|:----------|:-------------------------|-------------:|--------------------:|------------------------:|-------------:|-------------------:|---------------:|
| phc      | threshold | phc_threshold_5s_mean    |            5 |              0.7500 |             114645.0000 |         5077 |            22.5812 |      2445.0000 |
| nonml    | threshold | nonml_threshold_10s_mean |           10 |              0.5000 |             114610.0000 |         5266 |            21.7641 |      2420.0000 |
| phc      | direction | phc_direction_1s_mean    |            1 |              0.7500 |             102810.0000 |         4386 |            23.4405 |      2130.0000 |
| nonml    | direction | nonml_direction_2s_mean  |            2 |              0.7500 |             100900.0000 |         4360 |            23.1422 |      2290.0000 |
| phc      | enable    | phc_enable_5s_mean       |            5 |              0.2500 |              88990.0000 |         3694 |            24.0904 |      1580.0000 |
| nonml    | enable    | nonml_enable_1s_mean     |            1 |              0.2500 |              86010.0000 |         3426 |            25.1051 |      3030.0000 |
| phc      | combined  | phc_combined_5s_mean     |            5 |              0.2500 |              47400.0000 |         2360 |            20.0847 |      2850.0000 |
| nonml    | combined  | nonml_combined_5s_mean   |            5 |              0.2500 |              46365.0000 |         2609 |            17.7712 |      3630.0000 |

## Champion evaluation

| split   | source   | mode      | policy_id                |   interval_s |   sequential_excess_pnl |   fill_count |   avg_pnl_per_fill |   max_drawdown |
|:--------|:---------|:----------|:-------------------------|-------------:|------------------------:|-------------:|-------------------:|---------------:|
| new3    | nonml    | combined  | nonml_combined_5s_mean   |            5 |             153360.0000 |         6736 |            22.7672 |      4260.0000 |
| new4    | nonml    | combined  | nonml_combined_5s_mean   |            5 |              43030.0000 |         1779 |            24.1877 |      1795.0000 |
| val     | nonml    | combined  | nonml_combined_5s_mean   |            5 |              46365.0000 |         2609 |            17.7712 |      3630.0000 |
| new3    | nonml    | direction | nonml_direction_2s_mean  |            2 |             272380.0000 |         9729 |            27.9967 |      3280.0000 |
| new4    | nonml    | direction | nonml_direction_2s_mean  |            2 |             103575.0000 |         4255 |            24.3420 |      5015.0000 |
| val     | nonml    | direction | nonml_direction_2s_mean  |            2 |             100900.0000 |         4360 |            23.1422 |      2290.0000 |
| new3    | nonml    | enable    | nonml_enable_1s_mean     |            1 |             212070.0000 |         7883 |            26.9022 |      3555.0000 |
| new4    | nonml    | enable    | nonml_enable_1s_mean     |            1 |              86020.0000 |         3512 |            24.4932 |      4975.0000 |
| val     | nonml    | enable    | nonml_enable_1s_mean     |            1 |              86010.0000 |         3426 |            25.1051 |      3030.0000 |
| new3    | nonml    | threshold | nonml_threshold_10s_mean |           10 |             294740.0000 |        11505 |            25.6184 |      3225.0000 |
| new4    | nonml    | threshold | nonml_threshold_10s_mean |           10 |             110100.0000 |         5040 |            21.8452 |      5630.0000 |
| val     | nonml    | threshold | nonml_threshold_10s_mean |           10 |             114610.0000 |         5266 |            21.7641 |      2420.0000 |
| new3    | phc      | combined  | phc_combined_5s_mean     |            5 |             134675.0000 |         5555 |            24.2439 |      2760.0000 |
| new4    | phc      | combined  | phc_combined_5s_mean     |            5 |              28575.0000 |         1276 |            22.3942 |      2010.0000 |
| val     | phc      | combined  | phc_combined_5s_mean     |            5 |              47400.0000 |         2360 |            20.0847 |      2850.0000 |
| new3    | phc      | direction | phc_direction_1s_mean    |            1 |             264210.0000 |         9647 |            27.3878 |      3280.0000 |
| new4    | phc      | direction | phc_direction_1s_mean    |            1 |             101585.0000 |         4241 |            23.9531 |      5155.0000 |
| val     | phc      | direction | phc_direction_1s_mean    |            1 |             102810.0000 |         4386 |            23.4405 |      2130.0000 |
| new3    | phc      | enable    | phc_enable_5s_mean       |            5 |             214550.0000 |         7611 |            28.1895 |      3995.0000 |
| new4    | phc      | enable    | phc_enable_5s_mean       |            5 |              71030.0000 |         3178 |            22.3505 |      4435.0000 |
| val     | phc      | enable    | phc_enable_5s_mean       |            5 |              88990.0000 |         3694 |            24.0904 |      1580.0000 |
| new3    | phc      | threshold | phc_threshold_5s_mean    |            5 |             295270.0000 |        11023 |            26.7867 |      3280.0000 |
| new4    | phc      | threshold | phc_threshold_5s_mean    |            5 |             114245.0000 |         4797 |            23.8159 |      5360.0000 |
| val     | phc      | threshold | phc_threshold_5s_mean    |            5 |             114645.0000 |         5077 |            22.5812 |      2445.0000 |

## Combined-controller decision test

| split   |   ml_combined |   nonml_combined |   static_tuned |   eventwise_phc |
|:--------|--------------:|-----------------:|---------------:|----------------:|
| val     |    47400.0000 |       46365.0000 |    113850.0000 |     171835.0000 |
| new3    |   134675.0000 |      153360.0000 |    296225.0000 |     611095.0000 |
| new4    |    28575.0000 |       43030.0000 |    115875.0000 |     129160.0000 |

The combined PHC controller does not beat the tuned static rule on val, beats its same-mechanism non-ML control on val, and does not beat the tuned static rule on both OOS diagnostics. This yields status `resolved_negative`.

## Statistical checks

Matched-random tests below are action-level tests inherited from Phase 05A. This exploratory run uses `100` trials, so its minimum empirical p-value resolution is `0.01`. They test each selected policy against a count/session/side/spread-matched random policy; they do **not** test PHC against the non-ML controller. Pairwise claims use a session-paired bootstrap on causal-fill PnL. Each bootstrap session is simulated independently with inventory reset, whereas the canonical full-split simulator carries its inventory variable across session boundaries. The table therefore reports both `full_split_total_difference` and the session-reset estimand; they need not be identical. With only four to six sessions per split, intervals are wide and must not be over-interpreted.

| split   | policy_id                |   model_total_excess_pnl_H20 |     random_mean |   random_std |      random_95p |      random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   |   n_trials | matching                                                    | matched_random_metric             |
|:--------|:-------------------------|-----------------------------:|----------------:|-------------:|----------------:|----------------:|---------------------------------------------:|:----------------------------|:-------------------|-----------:|:------------------------------------------------------------|:----------------------------------|
| val     | phc_threshold_5s_mean    |               1312150.000000 |  -797532.650000 | 27570.302832 |  -750249.250000 |  -745681.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | phc_threshold_5s_mean    |               3006395.000000 | -1678275.950000 | 43783.093134 | -1614180.750000 | -1588017.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | phc_threshold_5s_mean    |               1053835.000000 |  -647251.900000 | 24571.479969 |  -602665.000000 |  -590265.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | nonml_threshold_10s_mean |               1307555.000000 |  -819010.400000 | 31110.717331 |  -764120.000000 |  -746127.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | nonml_threshold_10s_mean |               3022015.000000 | -1746566.850000 | 44232.852563 | -1672365.750000 | -1655022.700000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | nonml_threshold_10s_mean |               1032575.000000 |  -681638.800000 | 27764.804881 |  -637615.000000 |  -607012.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | phc_direction_1s_mean    |               1165380.000000 |  -693414.000000 | 23277.278245 |  -656860.000000 |  -646454.000000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | phc_direction_1s_mean    |               2719400.000000 | -1493792.250000 | 38254.562763 | -1426940.750000 | -1415412.000000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | phc_direction_1s_mean    |                959835.000000 |  -578900.650000 | 26265.825978 |  -541106.500000 |  -526768.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | nonml_direction_2s_mean  |               1124320.000000 |  -691972.000000 | 26009.669048 |  -643743.500000 |  -630932.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | nonml_direction_2s_mean  |               2705695.000000 | -1494556.950000 | 44483.249796 | -1420540.000000 | -1408966.300000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | nonml_direction_2s_mean  |                948335.000000 |  -581121.650000 | 23488.346196 |  -546215.750000 |  -519640.000000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | phc_enable_5s_mean       |                959570.000000 |  -586295.600000 | 27226.166479 |  -542028.500000 |  -532697.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | phc_enable_5s_mean       |               2146470.000000 | -1176610.800000 | 35565.035848 | -1123436.500000 | -1108214.700000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | phc_enable_5s_mean       |                721160.000000 |  -439565.950000 | 23264.953940 |  -396285.000000 |  -388458.300000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | nonml_enable_1s_mean     |                914045.000000 |  -544865.500000 | 27199.867127 |  -495770.000000 |  -476130.300000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | nonml_enable_1s_mean     |               2145965.000000 | -1191165.350000 | 42797.494969 | -1130475.000000 | -1096018.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | nonml_enable_1s_mean     |                765625.000000 |  -475592.700000 | 24509.432627 |  -437464.250000 |  -427605.300000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | phc_combined_5s_mean     |                539915.000000 |  -367885.250000 | 19324.373099 |  -343170.750000 |  -325719.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | phc_combined_5s_mean     |               1135940.000000 |  -735541.500000 | 31710.950518 |  -682659.250000 |  -672384.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | phc_combined_5s_mean     |                376000.000000 |  -275884.400000 | 19625.621701 |  -242884.250000 |  -233253.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | nonml_combined_5s_mean   |                544640.000000 |  -426740.700000 | 21816.966311 |  -389899.250000 |  -384333.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | nonml_combined_5s_mean   |               1264810.000000 |  -894011.950000 | 35336.687620 |  -835840.000000 |  -820256.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | nonml_combined_5s_mean   |                465985.000000 |  -359554.250000 | 22743.621571 |  -326021.500000 |  -318699.000000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | queue_theta_val_tuned    |               1298370.000000 |  -778716.150000 | 26881.552298 |  -735795.000000 |  -716211.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| val     | phc_eventwise_reference  |               2037925.000000 |  -969762.850000 | 31881.602056 |  -918531.500000 |  -893798.700000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | queue_theta_val_tuned    |               3010075.000000 | -1648227.650000 | 43134.210097 | -1573760.000000 | -1565692.850000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new3    | phc_eventwise_reference  |               5237410.000000 | -2113426.050000 | 44583.619437 | -2041604.250000 | -1986078.450000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | queue_theta_val_tuned    |               1043970.000000 |  -638738.400000 | 25621.996603 |  -597677.250000 |  -590607.150000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |
| new4    | phc_eventwise_reference  |               1871685.000000 |  -815690.900000 | 26215.593092 |  -779165.000000 |  -767815.000000 |                                     0.000000 | < 0.01                      | True               |        100 | count_session_side_spread_bucket_with_session_side_fallback | action_level_total_excess_pnl_H20 |

| split   | ml_policy_id          | comparator            |   session_count |   full_split_total_difference |   session_reset_total_difference |   mean_session_difference |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_probability_difference_le_zero |   n_bootstrap |
|:--------|:----------------------|:----------------------|----------------:|------------------------------:|---------------------------------:|--------------------------:|-------------------:|--------------------:|-------------------------------------------:|--------------:|
| val     | phc_threshold_5s_mean | queue_theta_val_tuned |               6 |                    795.000000 |                       795.000000 |                132.500000 |        -283.333333 |          535.833333 |                                   0.260900 |         10000 |
| val     | phc_threshold_5s_mean | nonml_threshold       |               6 |                     35.000000 |                        35.000000 |                  5.833333 |        -231.666667 |          236.666667 |                                   0.488900 |         10000 |
| new3    | phc_threshold_5s_mean | queue_theta_val_tuned |               6 |                   -955.000000 |                     -1790.000000 |               -298.333333 |       -1891.666667 |         1163.604167 |                                   0.642300 |         10000 |
| new3    | phc_threshold_5s_mean | nonml_threshold       |               6 |                    530.000000 |                       120.000000 |                 20.000000 |       -2683.333333 |         2980.000000 |                                   0.509000 |         10000 |
| new4    | phc_threshold_5s_mean | queue_theta_val_tuned |               4 |                  -1630.000000 |                     -1630.000000 |               -407.500000 |       -1053.750000 |          351.250000 |                                   0.879200 |         10000 |
| new4    | phc_threshold_5s_mean | nonml_threshold       |               4 |                   4145.000000 |                      4145.000000 |               1036.250000 |       -1197.500000 |         3270.000000 |                                   0.189900 |         10000 |
| val     | phc_direction_1s_mean | queue_theta_val_tuned |               6 |                 -11040.000000 |                    -11040.000000 |              -1840.000000 |       -2557.500000 |        -1089.166667 |                                   1.000000 |         10000 |
| val     | phc_direction_1s_mean | nonml_direction       |               6 |                   1910.000000 |                      1910.000000 |                318.333333 |        -799.166667 |         1415.000000 |                                   0.280000 |         10000 |
| new3    | phc_direction_1s_mean | queue_theta_val_tuned |               6 |                 -32015.000000 |                    -38240.000000 |              -6373.333333 |       -9556.666667 |        -3435.000000 |                                   1.000000 |         10000 |
| new3    | phc_direction_1s_mean | nonml_direction       |               6 |                  -8170.000000 |                    -10905.000000 |              -1817.500000 |       -4250.833333 |          644.166667 |                                   0.925800 |         10000 |
| new4    | phc_direction_1s_mean | queue_theta_val_tuned |               4 |                 -14290.000000 |                    -14290.000000 |              -3572.500000 |       -5947.500000 |        -1195.000000 |                                   1.000000 |         10000 |
| new4    | phc_direction_1s_mean | nonml_direction       |               4 |                  -1990.000000 |                     -1990.000000 |               -497.500000 |       -1777.500000 |          782.500000 |                                   0.767200 |         10000 |
| val     | phc_enable_5s_mean    | queue_theta_val_tuned |               6 |                 -24860.000000 |                    -24860.000000 |              -4143.333333 |       -5912.500000 |        -2374.166667 |                                   1.000000 |         10000 |
| val     | phc_enable_5s_mean    | nonml_enable          |               6 |                   2980.000000 |                      2980.000000 |                496.666667 |       -1670.000000 |         3055.000000 |                                   0.369800 |         10000 |
| new3    | phc_enable_5s_mean    | queue_theta_val_tuned |               6 |                 -81675.000000 |                   -103085.000000 |             -17180.833333 |      -26661.666667 |        -8298.333333 |                                   1.000000 |         10000 |
| new3    | phc_enable_5s_mean    | nonml_enable          |               6 |                   2480.000000 |                     -2450.000000 |               -408.333333 |       -3135.833333 |         1391.666667 |                                   0.616200 |         10000 |
| new4    | phc_enable_5s_mean    | queue_theta_val_tuned |               4 |                 -44845.000000 |                    -44845.000000 |             -11211.250000 |      -17205.000000 |        -5217.500000 |                                   1.000000 |         10000 |
| new4    | phc_enable_5s_mean    | nonml_enable          |               4 |                 -14990.000000 |                    -14990.000000 |              -3747.500000 |       -5808.750000 |        -2057.500000 |                                   1.000000 |         10000 |
| val     | phc_combined_5s_mean  | queue_theta_val_tuned |               6 |                 -66450.000000 |                    -66450.000000 |             -11075.000000 |      -12610.395833 |        -9159.166667 |                                   1.000000 |         10000 |
| val     | phc_combined_5s_mean  | nonml_combined        |               6 |                   1035.000000 |                      1035.000000 |                172.500000 |       -1791.666667 |         2332.500000 |                                   0.457500 |         10000 |
| new3    | phc_combined_5s_mean  | queue_theta_val_tuned |               6 |                -161550.000000 |                   -216280.000000 |             -36046.666667 |      -49615.833333 |       -20607.500000 |                                   1.000000 |         10000 |
| new3    | phc_combined_5s_mean  | nonml_combined        |               6 |                 -18685.000000 |                    -18685.000000 |              -3114.166667 |       -6287.500000 |          286.020833 |                                   0.957900 |         10000 |
| new4    | phc_combined_5s_mean  | queue_theta_val_tuned |               4 |                 -87300.000000 |                    -74670.000000 |             -18667.500000 |      -29346.250000 |        -7753.750000 |                                   1.000000 |         10000 |
| new4    | phc_combined_5s_mean  | nonml_combined        |               4 |                 -14455.000000 |                    -15005.000000 |              -3751.250000 |       -4945.000000 |        -2557.500000 |                                   1.000000 |         10000 |

## Supplementary aggregation check

`results/async_control_plane_all_results.csv` includes the supplementary `last` aggregation. It is not used for the headline result because it would make the control depend on one potentially noisy event.

## Outputs

- `results/static_threshold_scan.csv`
- `results/val_hyperparameter_selection.csv`
- `results/async_control_plane_all_results.csv`
- `results/controller_champions.csv`
- `results/matched_random_champions.csv`
- `results/paired_session_bootstrap.csv`
- `figures/async_control_plane_pnl.png`
