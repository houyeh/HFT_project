# backtest_explanation

## Purpose

This folder turns model outputs into tradable decisions and replay-based PnL.

## Files

- `baseline.py`
  - vectorized signal backtest for quick net-pnl and hit-rate checks
- `event_driven.py`
  - more realistic single-position replay with market/passive entry logic
- `hybrid_market_maker.py`
  - queue-aware market-making simulator with toxicity, inventory, and latency controls
- `signals.py`
  - converts raw class probabilities into gated trade signals
- `symbol_calibration.py`
  - searches per-symbol thresholds, time cutoffs, and regime rules

## Why It Matters For HLS

This folder is where the project proves that prediction quality alone is not enough.
The later hardware story depends on whether lower latency improves execution and realized PnL in these backtests.
