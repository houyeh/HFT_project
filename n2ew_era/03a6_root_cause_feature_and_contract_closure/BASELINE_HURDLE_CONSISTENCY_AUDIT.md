# Baseline Hurdle Consistency Audit

This audit verifies apples-to-apples baseline comparisons: same split universe, same H20 excess objective, same fee/fill contract for sequential proxy, and matched-random count/side/session/spread controls.

| split   | audit_item                              | pass   | details                                                                            |
|:--------|:----------------------------------------|:-------|:-----------------------------------------------------------------------------------|
| val     | excess_policy_same_universe             | True   | state_count_values=[np.float64(563477.0)]                                          |
| val     | matched_random_same_excess_objective    | True   | matched random rows use finite_population_normal_approx_side_session_spread_bucket |
| val     | sequential_proxy_same_fee_fill_contract | True   | conservative_l2_stress, inventory_cap=1, fee_multiplier=1.0, H20                   |
| test    | excess_policy_same_universe             | True   | state_count_values=[np.float64(1166321.0)]                                         |
| test    | matched_random_same_excess_objective    | True   | matched random rows use finite_population_normal_approx_side_session_spread_bucket |
| test    | sequential_proxy_same_fee_fill_contract | True   | conservative_l2_stress, inventory_cap=1, fee_multiplier=1.0, H20                   |
| new4    | excess_policy_same_universe             | True   | state_count_values=[np.float64(2196640.0)]                                         |
| new4    | matched_random_same_excess_objective    | True   | matched random rows use finite_population_normal_approx_side_session_spread_bucket |
| new4    | sequential_proxy_same_fee_fill_contract | True   | conservative_l2_stress, inventory_cap=1, fee_multiplier=1.0, H20                   |
| new3    | excess_policy_same_universe             | True   | state_count_values=[np.float64(5657362.0)]                                         |
| new3    | matched_random_same_excess_objective    | True   | matched random rows use finite_population_normal_approx_side_session_spread_bucket |
| new3    | sequential_proxy_same_fee_fill_contract | True   | conservative_l2_stress, inventory_cap=1, fee_multiplier=1.0, H20                   |
