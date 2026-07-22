# 023 Root-Cause Analysis for Stress-Robust Profitability

The current 20260127 test set remains burned and diagnostic. This phase explains the gap between the prior non-tradable oracle and causal policies; it does not create a new deployable policy claim.

## 1. Locked Oracle Definition

- `moderate stress = neutral_fill_haircut=0.5, toxic_multiplier=1.5, delay=0`.
- `oracle_positive_quote = realized stress-adjusted quote PnL > 0`.
- `oracle_selected_quote = perfect_quote_pnl_filter selected quote`.
- Oracle is non-tradable, uses future information, and is only an upper bound.

Prior 021 inventory-constrained simulator reference:
- Perfect quote PnL filter oracle moderate-stress PnL: `217660.000`.
- 021 selected causal policy moderate-stress PnL: `-92650.000`.

The 023 tables below use action-level quote opportunity PnL to diagnose selection direction, false positives, and missed positives. These action-level sums are not a replacement for the inventory-constrained simulator PnL.
`spread_only` in 023 is a validation-selected raw `spread_ticks` gate because the serialized 022 sklearn model is version-incompatible in this runtime; the threshold manifest records `test_used_for_selection=false`.

## 2. Oracle Opportunity Map

- Top `1%` quote actions capture `0.195` of positive oracle PnL.
- Top `5%` quote actions capture `0.502` of positive oracle PnL.
- Top `10%` quote actions capture `0.748` of positive oracle PnL.

## 3. Model-Oracle Alignment

- `spread_only`: selected quote PnL `-577930.000`, recall `0.557`, precision `0.236`, missed oracle PnL `901610.000`.
- `016_v2`: selected quote PnL `-631450.000`, recall `0.789`, precision `0.221`, missed oracle PnL `529685.000`.
- `015`: selected quote PnL `-678520.000`, recall `0.907`, precision `0.219`, missed oracle PnL `225780.000`.
- `020`: selected quote PnL `-749665.000`, recall `0.871`, precision `0.212`, missed oracle PnL `268025.000`.
- `021`: selected quote PnL `-832550.000`, recall `0.866`, precision `0.206`, missed oracle PnL `402010.000`.

## 4. Within-Spread ML Lift

- `016_v2` incremental PnL vs matched same-bucket random: `84525.020`.
- `ensemble_015_016` incremental PnL vs matched same-bucket random: `43792.100`.
- `015` incremental PnL vs matched same-bucket random: `35685.960`.
- `spread_only` incremental PnL vs matched same-bucket random: `0.000`.
- `021_ev_score` incremental PnL vs matched same-bucket random: `-187452.300`.

## 5. Calibration Transfer and Rolling Quantile

Validation-selected rolling policy diagnostic test PnL: `-43560.000`; test sweep rows are marked `DIAGNOSTIC_ONLY` and `used_for_model_selection=False`.
Best test-diagnostic-only rolling row is `015_w200_q0.1` at `275855.000`. This is evidence of calibration transfer failure, not a selectable result.

## 6. Spread-First, ML-Second Policy

- `spread_first_plus_rolling_ML`: quote PnL `-43560.000`, incremental over spread-first `534370.000`.
- `spread_first_plus_ensemble`: quote PnL `-229715.000`, incremental over spread-first `348215.000`.
- `spread_first_plus_015`: quote PnL `-352075.000`, incremental over spread-first `225855.000`.
- `spread_first_plus_016_v2`: quote PnL `-367325.000`, incremental over spread-first `210605.000`.
- `spread_first_only`: quote PnL `-577930.000`, incremental over spread-first `0.000`.

## 7. Bottleneck Decision

Dominant bottleneck: **calibration shift**.

- Top 5% quote actions capture `0.502` of positive oracle PnL; oracle-positive rate is `0.2077`.
- Spread-first-only action PnL is `-577930.000`; best spread-first ablation is `spread_first_plus_rolling_ML` at `-43560.000`.
- Best within-spread ML incremental PnL vs matched random is `016_v2` at `84525.020`.
- Best causal oracle recall is `015` at `0.907` with precision `0.219`.
- Missing true queue data remains a simulator-limit caveat, not the primary data-selected bottleneck unless the measured diagnostics are inconclusive.

## 8. Final Interpretation

The oracle's positive moderate-stress PnL is concentrated enough that selection quality matters, but not so sparse that opportunity alone explains the failure. The causal ML signals show some within-spread lift, yet validation-selected calibration admits too much false-positive loss and misses too much oracle PnL when transferred to the burned diagnostic test. The result explains why the non-tradable oracle can reach positive PnL while validation-selected ML policies fail to generalize. This is a root-cause diagnosis, not a new profitability claim.
