# 026 Robust Baseline & ML Incremental Diagnosis

## 1. Robust Non-ML Baseline

Validation-selected non-ML policy: `bid_only_spread_ge_1` with validation quote PnL `1218095.000`.
Frozen diagnostic test non-ML PnL: `2312590.000`, selected quotes `419`, precision `0.449`.

Walk-forward non-ML summary:
- `fixed_026_selected`: mean PnL `594731.667`, positive folds `2` / `3`.
- `fold_val_selected_non_ml`: mean PnL `-235151.667`, positive folds `1` / `3`.

## 2. ML Incremental Veto

Baseline PnL `2312590.000`; baseline+ML veto PnL `2288065.000`; incremental `-24525.000`.
Matched random veto 95p `2272117.250`; ML beats random 95p: `True`.
Interpretation: ML chooses less damaging vetoes than matched random, but the vetoed quotes are still net positive, so ML does not add economic value beyond the robust non-ML baseline.
- `bid` ML incremental vs baseline: `-24525.000`, vetoed quote PnL `24525.000`.
- `ask` ML incremental vs baseline: `0.000`, vetoed quote PnL `0.000`.

## 3. Decision Overlap

- ML-only blocked `bid` quotes: count `20`, total PnL `24525.000`, toxic rate `1.000`.

## 4. Calibration and Label Alignment

Test label/PnL matrix counts: toxic_bad `514`, toxic_good `143`, non_toxic_bad `226`, non_toxic_good `51`.
The toxic label only partially aligns with the quote-PnL objective: many bid-side toxic quotes are still economically positive under the current proxy, while ask-side toxic quotes are mostly bad.

## 5. NEW4 External OOS Check

NEW4 external OOS applies the frozen non-ML policy `bid_only_spread_ge_1` with no NEW4 tuning.
NEW4 selected quote PnL `6348770.000`, selected quotes `714`, precision `0.363`, symbol `TXFA6`.
NEW4 positive session/side groups: `3` / `4`.

## 6. Final Diagnosis

Is spread/side the dominant signal? `True`.
Does ML have incremental value beyond the robust baseline? `False`.
ML incremental value on `bid`: `False`.
ML incremental value on `ask`: `False`.
No further ML training is recommended unless ML veto beats matched random controls and shows stable side-specific value.
