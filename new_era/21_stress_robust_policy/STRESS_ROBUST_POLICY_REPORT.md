# 021 Stress-Robust Policy Report

## 1. Why 020 Was Insufficient

The 020 selected policy was base-positive but failed absolute moderate-stress profitability. In the locked test stress setting, 020 moderate PnL was `-74,805` with spread capture `50,100`, adverse loss `120,450`, fees `4,455`, and `297` trades.

The failure mode is insufficient surviving spread capture under the `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, `delay=0` setting. Defensive cancels reduce damage versus no-cancel/random baselines, but the remaining fills still do not generate enough net edge.

## 2. Stress PnL Decomposition

Diagnostics are in `diagnostics/stress_pnl_decomposition.csv` and `diagnostics/stress_pnl_decomposition.md`.

The decomposition identity is `net_pnl = spread_capture - adverse_loss - fees`. The adverse-loss term is a reconciled residual containing adverse markout, stress toxic penalty, and terminal inventory effects.

## 3. Oracle Bounds

Oracle outputs are in `oracle/oracle_perfect_upper_bounds.csv`, `oracle/oracle_best_validation_score_threshold.csv`, and `oracle/oracle_best_validation_spread_score_grid.csv`.

- Best perfect oracle moderate test PnL: `217,660` from `perfect_quote_pnl_filter`.
- Worst perfect oracle moderate test PnL: `-87,355` from `perfect_toxic_cancel`.

Because at least one perfect oracle is positive under moderate stress, the locked simulator is not mathematically infeasible. The bottleneck is model/policy selectivity under validation-only tradable rules.

## 4. Validation Stress-Aware Search

Search output: `search/validation_stress_aware_policy_search.csv`.

- Selected policy: `EV_s2_lam5_thr0`
- Family: `EV_based_quote_selection`
- Validation moderate PnL: `40,405`
- Validation objective: `21,555.546`
- Constraints passed on validation: `True`
- Test used for selection: `False`

## 5. Frozen Test Performance

Frozen test output: `evaluation/frozen_stress_robust_policy_test.csv`.

- 021 base test PnL: `79,265`
- 021 moderate test PnL: `-92,650`
- Moderate gap vs no-cancel: `4,620`
- Moderate gap vs random_v3_95p: `-55,186.665`
- Decision Status: `Fail`

The validation-selected 021 policy did not generalize under frozen test stress. It is worse than 020_selected, 016_v2, and 015_common on moderate test PnL, so it should not replace those frozen defensive baselines.

## 6. HLS Mapping

The HLS policy spec is in `hls_stress_robust_policy_spec.md`. The policy is fixed threshold/LUT/comparator logic and does not require online optimization or dynamic retraining.

## 7. Limiting Factor

If moderate-positive PnL is not achieved, the limiting factor is not the locked stress simulator itself, because perfect oracles can be positive. The limiting factor is the gap between future-informed quote selection and validation-selected, hardware-friendly risk signals. The next research step is improving causal signal quality or execution realism, not tuning on test.

## 8. Conservative Conclusion

021 diagnoses whether a stress-robust FPGA-friendly controller is feasible under locked proxy stress. It must be interpreted as latency risk-control research, not proof of real-world HFT profitability or proof of FPGA production necessity.
