# n2ew_era Project Reset

`n2ew_era` is a clean research line for short-horizon L2 quote-control experiments. It is intentionally separate from `new_era`, which contains the historical experiments, audits, stress tests, and falsification work. Old results remain preserved as evidence and context, but they are not treated as final strategy proof.

## Why This Exists

The previous research line found several important failure modes:

- Positive quote EV was dominated by terminal or session-final markout rather than short-horizon executable market-making edge.
- Sequential proxy PnL was negative at short horizons such as H=1, H=5, H=10, and H=20.
- The residual ML veto experiment used a very narrow baseline-selected universe, with only 820 train samples, so it was not a full test of L2-based ML potential.
- The `bid_only_spread_ge_1` controller is useful as a baseline and regime filter, but it is not a complete strategy.

## New Objective

The new objective is short-horizon sequential proxy PnL using causal L2 order book features. The main research question is whether L2 imbalance, depth, event-rate, and volatility regimes can improve quote-control decisions at short horizons.

Terminal quote EV is diagnostic only. It may help explain directional drift or oracle limits, but it is not the main optimization target and must not be described as trading PnL.

## Scope Boundary

This phase is documentation and project reset only. It does not train models, run backtests, tune strategies, or modify old `new_era` results.
