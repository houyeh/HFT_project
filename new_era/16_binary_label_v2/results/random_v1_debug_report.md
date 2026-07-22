# random_v1 Debug Report

## Symptom

The Round 2 `strict_random_baseline_distribution.csv` reported `random_v1` with zero PnL variance.

## Root Cause

The `random_v1` grouping has no conditioning keys. In the prior implementation, the no-key bucket was represented as an empty tuple in one helper and as a nested tuple in another helper. That mismatch produced empty `action_ids` for `random_v1`, so no fill actions were ever cancelled in v1 trials. The result collapsed to the no-cancel PnL.

## Fix

`15_strict_random_baseline.py` now uses an explicit `ALL` bucket for no-key `random_v1` grouping, so candidate rows and fill action ids share the same key.

## Prior v1 Row

- prior_mean: `63785.0`
- prior_std: `0.0`
- prior_p95: `63785.0`

## Post-Fix Smoke Test

- common rows: `1166321`
- fill actions total: `934`
- fill actions by side: `{'ask': 515, 'bid': 419}`
- debug 25-trial net PnL std: `29876.1521`
- mean cancelled fill actions in smoke test: `132.92`

Spec rows:

- side=bid, required_model_cancels=172130, candidate_pool_size=1166321, action_ids_available_after_fix=419
- side=ask, required_model_cancels=161295, candidate_pool_size=1166321, action_ids_available_after_fix=515

## Interpretation

`random_v1` should be treated as the loosest sanity baseline. The main statistical hurdles remain the stricter `random_v2` and `random_v3` rows because they match session and spread structure.