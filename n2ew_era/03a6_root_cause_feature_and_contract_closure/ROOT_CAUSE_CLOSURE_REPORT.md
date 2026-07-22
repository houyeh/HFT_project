# Root-Cause Feature and Contract Closure Report

Final closure status: `phase03b_permission_granted_with_constraints`.

No final ML model was trained and no strategy was tuned. Diagnostic models were used only to decide whether absolute price features should be allowed, forbidden, or transformed.

## Closure Gates

| gate                                   | pass   | decision                 |
|:---------------------------------------|:-------|:-------------------------|
| absolute_price_feature_decision_closed | True   | forbid                   |
| dynamic_features_rebuilt_causally      | True   | allow_rebuilt            |
| rolling_feature_no_leak_proof          | True   | allow_volatility_past_50 |
| cost_unit_contract_closed              | True   | proxy_units_documented   |
| split_regime_drift_quantified          | True   | oos_harshness_documented |
| baseline_hurdles_apples_to_apples      | True   | hurdles_closed           |

## Absolute Price Decision

Decision: `forbid`.

Price-only diagnostic has weak frozen-OOS rank correlation; raw absolute price levels are not reliable standalone causal features.

## Dynamic Feature Rebuild

Rebuilt dynamic features are allowed only from `data/rebuilt_dynamic_features.parquet`, with session-boundary resets.

## Rolling Causality

| feature            |   max_abs_error | recompute_pass   |
|:-------------------|----------------:|:-----------------|
| volatility_past_50 |               0 | True             |

## Cost / Unit

Cost units remain proxy assumptions, not exchange-certified values. See `COST_UNIT_CONTRACT.md`.

## Split Drift

OOS drift is quantified in `results/split_regime_drift_closure.csv`; this supports harsh OOS validity rather than invalidating the split.

## Baseline Hurdles

Baseline hurdle consistency passed across validation and frozen OOS splits.

## Phase 03B Permission

`Phase 03B may proceed only under these constraints:`

- Do not use raw future labels or raw PnL as features.
- Use excess markout targets only.
- Treat raw absolute price levels according to `absolute_price_feature_decision.json`.
- Use rebuilt dynamic features only from the causal rebuild artifact.
- Select models/thresholds only on validation.
- Keep test, NEW3, and NEW4 frozen.
