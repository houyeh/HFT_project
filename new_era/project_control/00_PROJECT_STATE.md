# HFT FPGA Trading Project — Current State

## Project Goal
- FPGA-friendly passive market making
- ML-based toxic-flow cancel filter
- Not active directional trading

## Main Baseline Model
- model path: `new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot/`
- dataset: `NEW_ALL` (`TXFB6`)
- sessions: 9 sessions (`TXFB6__20260121Day` to `TXFB6__20260127Night`)
- split: Session-ordered 70/15/15 split
- train / val / test samples: 112,698 val / 233,268 test
- feature count: 8 (`obi`, `queue_imbalance`, `weighted_mid_deviation`, `trade_intensity_short`, `deep_obi`, `queue_imbalance_change`, `normalized_spread`, `cancel_pressure_short`)
- zscore window: 2000
- window size: 100 ticks
- horizon: 5 ticks
- label: `smoothed_mid_move_3class`
- model architecture: TCN (channels: 16, 16, 16, kernel_size: 3, dropout: 0.1)

## Training Result
- best epoch: 30
- min val loss: 0.7814
- final train loss: 0.8561
- Down / Up precision and recall: Down Precision ~2.83%, Recall ~69.41%; Up Precision ~2.93%, Recall ~64.19%
- interpretation: High recall but low absolute precision due to class imbalance; not suitable for active directional trading, but suitable as a high-sensitivity cancel filter candidate.

## Backtest Result
- Model 0.55: Net PnL = 84,980.0, trades = 468
- Model 0.60: Net PnL = 83,460.0, trades = 516
- Validation-selected 0.70: Net PnL = 63,440.0, trades = 584
- random+1std comparison: Model 0.55 beats Random_0.55 + 1std (80,473.19). Model 0.60 beats Random_0.60 + 1std (79,049.33). Model 0.70 does **not** beat Random_0.70 + 1std (72,778.7).
- conclusion: Model exhibits local predictive edge in the NEW test split at lower thresholds, but the validation-selected threshold (0.70) fails to outperform a matched random cancel baseline by 1 standard deviation.

## Regime Findings
- NEW no-gate: PnL = 84,980.0
- NEW spread_filter: PnL = 101,455.0 (`spread_ticks_min=1.5`)
- NEW combined_filter: PnL = 65,535.0 (suboptimal)
- NEW3 no-gate: PnL = -379,275.0 (heavily negative baseline -557,165.0)
- NEW3 spread_filter: PnL = -321,405.0
- NEW3 combined_filter: PnL = 72,520.0 (`trend_ticks=20.0, vol=0.2097, spread=2.0`)
- interpretation: Strong regime dependency. Different markets (NEW vs NEW3) require entirely different static quote control policies (spread vs. combined) to remain profitable.

## 016 Binary Label
- toxic rate: train=3.59%, val=3.23%, test=2.77% (bid)
- PR-AUC: test bid=0.0724, test ask=0.0803
- best PnL: 266,045.0 (bid=0.40, ask=0.20 on NEW test)
- trade count: 77
- random+1std: 272,035.83
- verdict: C (New binary label converges but no trading edge, does not beat random+1std).

## 017 Auto-Gate
- static gate results: NEW best = 101,455.0 (spread_filter); NEW3 best = 72,520.0 (combined_filter)
- auto-gate result: NEW test PnL = 10,000.0; NEW3 PnL = -630,960.0
- EWMA result: NEW PnL = 8,845.0; NEW3 PnL = -132,325.0
- verdict: C (Auto-gate cannot beat static combined_filter on either NEW or NEW3).

## 018 Queue-aware
- status: queue_validation_pending
- result if available: MISSING
- pending if not available: Pending. Queue data audit confirmed lack of `aggressor_side`, preventing true queue replay; proxies must be used.

## Current Main Claim
TCN has local cancel signal on NEW, but robust deployment is limited by regime dependency and simplified fill assumptions.

## Current Project Direction
- main baseline: `smoothed_mid_move_3class` acts as risk score
- regime diagnosis: proven macro dependency
- 016 as label PoC: validates fill-conditional quote risk, but model not ready
- 017 as negative result: causal auto-gating prototype failed to outperform static oracle
- 018 queue-aware as next realism validation: required before stronger trading claims can be made
