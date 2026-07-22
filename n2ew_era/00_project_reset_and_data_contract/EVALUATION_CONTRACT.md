# Evaluation Contract

## Main Objective

The main objective is sequential proxy PnL, not terminal quote EV.

Terminal quote EV may be reported only as a diagnostic to explain directional drift, oracle limits, or differences between markout definitions. It must not be optimized as the final objective.

## Horizons

Evaluate the following horizons:

- H = 1
- H = 5
- H = 10
- H = 20
- H = 50
- H = 100

## Metrics

Required metrics:

- sequential_proxy_pnl_H
- trade_count_H
- win_rate_H
- avg_trade_pnl_H
- max_drawdown_H
- max_inventory_H
- session_pnl_H
- LOSO robustness

Recommended diagnostics:

- PnL by side.
- PnL by spread bucket.
- PnL by event-rate bucket.
- PnL by imbalance regime.
- Gap versus terminal quote EV.

## Splits

- Train: model fitting only.
- Val: parameter selection, threshold selection, rule selection, and model selection.
- Test: frozen evaluation only.
- NEW4/OOS: external frozen evaluation only.

Test, NEW4, and unused OOS data must not be used to tune thresholds, select model families, choose horizons, or revise policy rules.

## Baseline Hierarchy

Evaluate in the following hierarchy:

1. No quote.
2. Old baseline.
3. Simple L2 rule.
4. Best Val non-ML rule.
5. ML full-action model.

Every more complex method must be compared against simpler baselines using the same universe, same split definitions, same fee assumptions, and same sequential proxy horizon.
