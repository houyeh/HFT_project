# 024 Calibration-Robust Policy Report

## 1. Purpose

023 found evidence that ranking signals have value, but static absolute thresholds suffer calibration shift. 024 tests whether causal past-only rolling thresholds can convert ranking into a more stable action policy.

## 2. Evidence Audit

- `Oracle opportunity exists under moderate stress`: evidence_of_upper_bound. Safe wording: Evidence that positive moderate-stress PnL is feasible for a non-tradable future-information oracle, not proof of a causal policy.
- `Oracle opportunity is learnable`: evidence_of_structure_not_proof. Safe wording: Evidence of concentrated structure in oracle-positive quotes; not absolute proof that causal features can learn it.
- `ML ranking has incremental value inside spread buckets`: evidence_of_incremental_value. Safe wording: Evidence that ML ranking can beat matched same-bucket random controls; not a profitability guarantee.
- `Static absolute thresholds fail across regimes`: supported_diagnostic. Safe wording: Evidence of calibration shift: validation-selected static/EV thresholds did not transfer to the burned diagnostic test.
- `+275k rolling threshold test sweep proves profitability`: diagnostic_only_not_selectable. Safe wording: The +275k row is diagnostic-only. It may motivate rolling calibration but cannot be used as a selected-policy claim.

## 3. Past-Only Calibration Design

For every candidate, `threshold_t = quantile(score[t-N:t-1], q)`. The current quote action and all future actions are excluded from the threshold calculation.
Candidate families: global rolling quantile, spread-bucket conditional rolling quantile, and side-specific rolling quantile.

## 4. Validation Selection

Selected policy: `side_specific__016_v2__w50__bid0.50__ask0.10`.
Score: `016_v2` / `side_score_016`; window `50`; groups `['session_id', 'side']`.
Selection split: `validation`; `test_used_for_selection=false`.

Top validation rows by selected quote PnL, including non-main spread-only diagnostic baselines:
- `side_specific__spread_only__w100__bid0.50__ask0.10`: validation quote PnL `802790.000`, precision `0.389`.
- `side_specific__spread_only__w100__bid0.30__ask0.10`: validation quote PnL `705315.000`, precision `0.381`.
- `side_specific__spread_only__w50__bid0.50__ask0.10`: validation quote PnL `689505.000`, precision `0.322`.
- `side_specific__spread_only__w200__bid0.50__ask0.10`: validation quote PnL `679700.000`, precision `0.351`.
- `side_specific__spread_only__w200__bid0.30__ask0.10`: validation quote PnL `679700.000`, precision `0.361`.

Top main-eligible ML rows:
- `side_specific__016_v2__w50__bid0.50__ask0.10`: validation quote PnL `644490.000`, precision `0.377`.
- `side_specific__021_ev_score__w50__bid0.50__ask0.10`: validation quote PnL `633755.000`, precision `0.379`.
- `side_specific__021_ev_score__w100__bid0.50__ask0.10`: validation quote PnL `626820.000`, precision `0.381`.
- `side_specific__021_ev_score__w25__bid0.50__ask0.10`: validation quote PnL `626190.000`, precision `0.338`.
- `side_specific__ensemble_015_016__w25__bid0.50__ask0.10`: validation quote PnL `605520.000`, precision `0.345`.

## 5. Frozen Diagnostic Test

Frozen test selected quote PnL: `637510.000`.
Oracle recall `0.474`, precision `0.374`, false-positive loss `331260.000`, missed oracle PnL `1361320.000`.
Selected quote count `246`, selection rate `0.263`.
Session/side/spread groups with positive selected quote PnL: `7` / `14`.

## 6. Did Rolling Thresholds Bridge Calibration Shift?

Yes under the action-level moderate-stress proxy: the validation-selected rolling calibration produced positive frozen diagnostic test quote PnL.

## 7. Conservative Interpretation

This phase tests calibration robustness under the current simulator-derived action-level proxy. It does not prove real-world profitability, true exchange fill toxicity, or FPGA production necessity. Any positive result is evidence that rolling/rank-based calibration may reduce regime sensitivity; it remains subject to the missing true queue and aggressor-side limitations documented in prior phases.
