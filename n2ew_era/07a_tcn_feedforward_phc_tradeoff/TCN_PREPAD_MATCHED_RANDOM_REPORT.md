# TCN PrePad Matched Random Report

Matched random uses the Phase 05A matched-count/session/side/spread-bucket method. Empirical p-value is the fraction of random trials with PnL >= model PnL.

| split | policy_id | model_total_excess_pnl_H20 | random_mean | random_std | random_95p | random_99p | empirical_p_value_fraction_random_ge_model | empirical_p_value_wording | beats_random_95p | n_trials | matching | variant | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | tcn_prepad_scratch_16feat | 1.92089e+06 | -926128 | 30117.2 | -875187 | -854625 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | scratch | 2 |
| new3 | tcn_prepad_scratch_16feat | 5.00592e+06 | -2.06534e+06 | 43762.7 | -1.99296e+06 | -1.96997e+06 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | scratch | 2 |
| new4 | tcn_prepad_scratch_16feat | 1.79528e+06 | -790983 | 28617.3 | -744085 | -729081 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | scratch | 2 |
| val | tcn_prepad_distilled_16feat | 1.97155e+06 | -918868 | 27853.4 | -874200 | -855365 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | distilled | 1 |
| new3 | tcn_prepad_distilled_16feat | 5.09123e+06 | -2.02214e+06 | 43383.8 | -1.94771e+06 | -1.91724e+06 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | distilled | 1 |
| new4 | tcn_prepad_distilled_16feat | 1.81446e+06 | -774641 | 27723.8 | -729612 | -713336 | 0 | < 0.001 | True | 1000 | count_session_side_spread_bucket_with_session_side_fallback | distilled | 1 |
