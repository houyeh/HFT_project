# Latest Project State - 020 Through 023

Updated: 2026-05-25

This file is the current source of truth for the post-020 diagnostic sequence. Do not let older `project_control` notes overwrite these findings.

## 020 FPGA-Friendly Policy Search

- Selected policy: validation-selected 015/016 weighted ensemble.
- Frozen diagnostic test Base PnL: `+77,195`.
- Moderate stress setting: `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, `delay=0`.
- Moderate stress PnL: `-74,805`.
- Interpretation: defensive value survived versus no-cancel/random baselines, but absolute profitability failed under moderate queue/fill stress.

## 021 Stress-Robust Policy Optimization

- Selected stress-aware policy: `EV_s2_lam5_thr0`.
- Validation moderate stress PnL: `+40,405`.
- Frozen diagnostic test moderate stress PnL: `-92,650`.
- Perfect quote PnL filter oracle moderate stress PnL: `+217,660`.
- Interpretation: positive moderate-stress PnL is mathematically possible under the simulator, but causal policies did not capture it.

## 022 Signal Generalization & Quote-EV Modeling

- Current 20260127 test set is treated as burned for diagnosis.
- Quote-EV labels are simulator-derived proxy labels, not true exchange-level queue-aware PnL.
- Best validation Quote-EV model was `logistic_sgd/spread_only`, indicating execution regime and spread dominate the current proxy EV signal.
- Walk-forward pre-burned folds: `3`; positive folds: `2`; mean fold PnL: `112,478.333`.
- Interpretation: evidence is mixed and diagnostic. It does not justify a production profitability claim.

## 023 Root-Cause Stress Profitability Analysis

- Purpose: explain why the non-tradable oracle is positive under moderate stress while causal ML policies fail to generalize.
- Required oracle definition:
  - `moderate stress = neutral_fill_haircut=0.5, toxic_multiplier=1.5, delay=0`
  - `oracle_positive_quote = realized stress-adjusted quote PnL > 0`
  - `oracle_selected_quote = perfect_quote_pnl_filter selected quote`
  - Oracle uses future information and is only an upper bound.
- Required diagnostics: split by Side, Session, and Spread Bucket; report oracle recall, oracle precision, captured/missed oracle PnL, false positive loss, avoided bad PnL, and missed good PnL.
