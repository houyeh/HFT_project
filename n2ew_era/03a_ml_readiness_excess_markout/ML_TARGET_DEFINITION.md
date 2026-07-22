# ML Target Definition

Phase 03A audits ML readiness for the Phase 02F spread-adjusted objective. It does not train final ML models and does not tune on test, NEW3, or NEW4.

## Main Target Family

1. Regression: `target_excess_pnl_H20`

`target_excess_pnl_H20 = excess_pnl_H20`, where mechanical half-spread capture has been removed.

2. Binary classification: `target_positive_excess_H20`

`target_positive_excess_H20 = 1 if excess_pnl_H20 > 0 else 0`.

3. Action-value scoring

Input is current state plus `action_side` (`bid`, `ask`, or `no_quote`). Output is expected `excess_pnl_H20` for that state-action pair.

## Formula

```text
excess_bid_pnl_H = (future_mid_H - mid_t) * point_value - fee
excess_ask_pnl_H = (mid_t - future_mid_H) * point_value - fee
no_quote excess_pnl_H = 0
```

## Explicit Exclusions

- Raw markout PnL is not the main ML target because it contains mechanical half-spread capture.
- Future labels, including `future_mid_H*`, `excess_pnl_H*`, `positive_excess_pnl_H*`, and raw `pnl_H*`, are never allowed as features.
- Test, NEW3, and NEW4 are frozen evaluation or OOS diagnostic splits and must not be used for target selection, feature selection, threshold selection, or hyperparameter tuning.

## Candidate Causal Features

Allowed feature candidates are current or past-only L2 state variables such as `spread_ticks`, `queue_imbalance`, `deep_imbalance`, `top_depth`, `bid_sz_1`, `ask_sz_1`, and `volatility_past_50`. `event_rate_past_50` should be restored in the next dataset build if available from the source, because it is requested by the research contract but absent from the Phase 02F parquet schema.
