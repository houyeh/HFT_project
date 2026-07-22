# Project Reset

## Why Reset?

The old baseline produced strong terminal quote EV, but the sequential short-horizon proxy backtest showed weak or negative short-horizon edge. The apparent positive PnL was largely explained by session-final directional markout rather than immediately executable market-making performance.

The residual ML veto experiment was also too narrow. It trained only inside the `bid_only_spread_ge_1` selected universe, leaving just 820 train samples. That makes it a valid incremental-veto audit, but not a valid test of all L2-based ML potential.

## What We Keep

- L2 data loading and cleaning patterns.
- Quote candidate construction as a useful abstraction.
- L2 feature ideas such as queue imbalance, deep imbalance, top depth, event rate, spread change, and volatility.
- Random baseline, OOS, and frozen-test discipline.
- Validation-only selection discipline.
- HLS design mindset: small feature sets, fixed thresholds, simple LUTs, deterministic control paths.

## What We Downgrade

- `quote_ev_moderate` as a final objective. It remains diagnostic only.
- Residual ML veto as a full ML conclusion. It only tested incremental value after a narrow non-ML baseline.
- `bid_only_spread_ge_1` as a complete strategy. It remains a baseline and reference controller.
- Terminal/session-final markout as evidence of market-making profitability.

## New Target

The new target is a sequential proxy quote-control system:

- Optimize and evaluate short-horizon sequential proxy PnL.
- Discover L2 imbalance regimes that improve short-horizon quote quality.
- Use full-action quote selection, not only residual veto inside a pre-filtered universe.
- Preserve strict Train/Val/Test/OOS separation.
- Produce an HLS-compatible quote-control primitive rather than a speculative production trading claim.
