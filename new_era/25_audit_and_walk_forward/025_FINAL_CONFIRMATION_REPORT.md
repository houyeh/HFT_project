# 025 Final Confirmation Report

## 1. Leakage Audit

Leakage detected: `False`.

## 2. PnL Concentration

Selected quote count `246`, total PnL `637510.000`, mean `2591.504`, median `0.000`, max win `32335.000`, max loss `-49365.000`.
- Top `1%` positive quotes contribute `0.150` of total selected PnL.
- Top `5%` positive quotes contribute `0.439` of total selected PnL.
- Top `10%` positive quotes contribute `0.720` of total selected PnL.

## 3. Side and Group Breakdown

- `ask`: selected PnL `-322185.000`, selected count `56`, selection rate `0.109`.
- `bid`: selected PnL `959695.000`, selected count `190`, selection rate `0.453`.
Positive session/side/spread groups: `7` / `14`.

## 4. Spread-Only vs ML Frozen Test

ML beats spread-only frozen baseline: `False`. ML `637510.000` vs spread-only `762510.000`.

## 5. Fresh Walk-Forward Confirmation

Fresh folds exclude 20260127. Because original out-of-sample 016 scores are unavailable for pre-burned training sessions, the walk-forward check trains a fold-local 016-style causal linear risk ranker on earlier sessions; this confirms the rolling-calibration architecture, not the exact original 016 TCN weights.
- `fixed_024_architecture`: folds `3`, mean test PnL `212901.667`, positive folds `2`.
- `val_selected_ml_architecture`: folds `3`, mean test PnL `-66380.000`, positive folds `1`.
- `val_selected_spread_only`: folds `3`, mean test PnL `-228923.333`, positive folds `0`.

## 6. Final Decision

Pending/Partial: 024 remains a breakthrough candidate, but at least one audit gate requires caution or more fresh holdout evidence.

No real-world profitability, true queue priority, or FPGA production necessity claim is made.
