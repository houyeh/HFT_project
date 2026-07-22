# PHC-lite Matched Random Report

Matched random trials match selected count, session, side, and spread bucket when feasible, with session-side fallback for sparse buckets.

| split   | policy_id   |   model_total_excess_pnl_H20 |      random_mean |   random_std |        random_95p |        random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   |   n_trials | matching                                                    |
|:--------|:------------|-----------------------------:|-----------------:|-------------:|------------------:|------------------:|---------------------------------------------:|:----------------------------|:-------------------|-----------:|:------------------------------------------------------------|
| val     | phc_lite    |                  1.92717e+06 | -959503          |      29021.5 | -911549           | -897899           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |
| new3    | phc_lite    |                  5.02152e+06 |      -2.0716e+06 |      45888   |      -1.99325e+06 |      -1.96327e+06 |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |
| new4    | phc_lite    |                  1.7625e+06  | -796094          |      27906.1 | -750466           | -732424           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |

Empirical p-value is defined as the fraction of matched random trials whose PnL is greater than or equal to PHC-lite PnL. If no random trial beats the policy in 1000 trials, wording is `empirical p-value < 0.001`, not exactly zero.
