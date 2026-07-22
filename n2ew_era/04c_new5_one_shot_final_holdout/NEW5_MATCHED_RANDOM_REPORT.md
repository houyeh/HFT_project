# NEW5 Matched Random Report

Matched random controls were run once for the frozen ML candidate and queue rule on NEW5 20260128. Matching dimensions: count, side, and session. No threshold or policy selection used NEW5.

| split              | session_id          | policy_id                                     |   model_total_excess_pnl_H20 |   model_selected_action_count |   bid_count |   ask_count |   random_trials |   random_mean |   random_std |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   | matching                 | new5_used_for_tuning   |
|:-------------------|:--------------------|:----------------------------------------------|-----------------------------:|------------------------------:|------------:|------------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------------------------------------:|:----------------------------|:-------------------|:-------------------------|:-----------------------|
| new5_final_holdout | NEW5_20260128_TXFB6 | ml_hgb_regressor_shallow_convergence_selected |                  1.56481e+07 |                        617738 |      318294 |      299444 |            1000 |  -9.24726e+06 |      98761.9 | -9.08599e+06 | -9.02187e+06 |                                            0 | < 0.001                     | True               | exact_count_side_session | False                  |
| new5_final_holdout | NEW5_20260128_TXFB6 | side_by_queue_imbalance_abs_0p1               |                  1.14398e+07 |                        672049 |      325153 |      346896 |            1000 |  -1.00975e+07 |     102906   | -9.92494e+06 | -9.84803e+06 |                                            0 | < 0.001                     | True               | exact_count_side_session | False                  |

If no random trial beats the policy, the report uses empirical p-value `< 0.001`; it does not claim exact zero probability.
