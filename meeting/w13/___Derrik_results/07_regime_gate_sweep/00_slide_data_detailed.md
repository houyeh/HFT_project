# Regime Gate Results — Verified Numbers

來源檔案: `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`

## NEW Test — Best Gate Configuration
- Gate: **spread_filter** (spread_ticks_min=1.5 OR 2.0, identical result)
- TCN threshold: **0.55**
- net_pnl: **101,455 TWD** (baseline no-gate: 42,925; improvement: +58,530)
- active_event_rate: **82.35%**  trade_count: 223
- (Note: spread_min=1.5 and 2.0 give same result because test data spread always ≥1.5 ticks)

## NEW3 Full — Best Gate Configuration
- Gate: **combined_filter** (trend_ticks=20, vol_threshold=0.2097, spread_min=2.0)
- TCN threshold: **0.55**
- net_pnl: **72,520 TWD** (baseline no-gate: -191,835; improvement: +264,355)
- active_event_rate: **82.57%**  trade_count: 992
- vol_threshold=0.2097 is the 80th percentile of rolling_vol computed on NEW3 train-like data

## Regime Dependency Summary

| Dataset | No-Gate PnL | Best Gate | Gate PnL | Gate Improvement |
|---------|------------|-----------|---------|-----------------|
| NEW test | 42,925 | spread_filter thr=0.55 | **101,455** | +58,530 |
| NEW3 full | -191,835 | combined_filter thr=0.55 | **72,520** | +264,355 |

→ Different datasets require different gates → regime dependency proven
→ But gate selection is RETROSPECTIVE — not yet causal auto-gating

## What Combined Filter Does (NEW3)
- trend_ticks=20: if |rolling_return| > 20 ticks → risk-off
- vol_threshold=0.2097: if realized_vol > 80th pctile → risk-off
- spread_min=2.0: only quote if spread ≥ 2 ticks
- All three conditions must be within range to remain active

## Source
- `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`
