# Exact Random OOS Report

Selected policy: `side_by_queue_imbalance_abs_0p1`, horizon `H20`.
Rule: If queue_imbalance > +0.1 quote bid; if queue_imbalance < -0.1 quote ask; otherwise no_quote.

The OOS exact random check was attempted on `test` and `NEW4` with 1000 actual without-replacement trials. The requested match was selected count, side ratio, and session. The implementation also matched spread bucket, making the control stricter.

In 1000 exact random trials, no random trial exceeded the selected policy. Therefore empirical p-value < 0.001.

| split   | policy_id                       |   horizon |   trials |   model_pnl |   selected_action_count |   exact_random_mean |   exact_random_std |   exact_random_90p |   exact_random_95p |   exact_random_99p | empirical_p_value_exact_random_ge_model   | beats_exact_random_mean   | beats_exact_random_95p   | matching_keys                                            | requested_match                                                                  | computational_status                |
|:--------|:--------------------------------|----------:|---------:|------------:|------------------------:|--------------------:|-------------------:|-------------------:|-------------------:|-------------------:|:------------------------------------------|:--------------------------|:-------------------------|:---------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------|
| test    | side_by_queue_imbalance_abs_0p1 |        20 |     1000 | 2.3387e+08  |                  971600 |         2.02888e+08 |            95366.2 |        2.03013e+08 |        2.03047e+08 |        2.03094e+08 | <0.001                                    | True                      | True                     | side + session_id + spread_bucket + exact selected count | matched count + side ratio + session; spread bucket included as stricter control | completed_exact_without_replacement |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |     1000 | 4.95871e+08 |                 1799304 |         4.38869e+08 |           170410   |        4.3909e+08  |        4.39142e+08 |        4.39251e+08 | <0.001                                    | True                      | True                     | side + session_id + spread_bucket + exact selected count | matched count + side ratio + session; spread bucket included as stricter control | completed_exact_without_replacement |
