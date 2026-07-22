# Sequential Proxy Inventory Backtest Report

This is an evaluation framework upgrade only. It does not tune the strategy, change thresholds, or alter the final controller.

## Method

- Policy: frozen `bid_only_spread_ge_1`.
- Universe: filled candidate actions from the Phase 031 full action universe for frozen `test` and external OOS `NEW4` only.
- Chronological state variables: `cash`, `inventory`, `realized_pnl`, `unrealized_pnl`, and drawdown from proxy equity.
- Point value: `200.0`.
- Maker fee per fill: `15.0`.
- Close rule: each fill is closed at the H-event future mid proxy with `H=10`.
- This is still a proxy. It does not model true queue priority, exchange order acknowledgements, partial fills, latency races, or production inventory controls.

## Split Summary

| split   |   fill_count |   quote_level_ev_sum_moderate |   quote_level_ev_sum_base |   sequential_proxy_pnl_H_mid |   max_abs_inventory |   max_drawdown |
|:--------|-------------:|------------------------------:|--------------------------:|-----------------------------:|--------------------:|---------------:|
| NEW4    |          714 |                     6,348,770 |                13,061,790 |                     -123,110 |                   4 |         61,235 |
| test    |          419 |                     2,312,590 |                 5,275,115 |                      -22,885 |                   2 |         22,495 |

## Interpretation

`quote_ev_moderate` and the sequential proxy are intentionally different measurements. `quote_ev_moderate` is a quote-level session-final markout with a queue haircut and adverse H-horizon stress penalty. The sequential proxy opens inventory on each fill and closes each lot at the H-event future mid. The comparison is useful for intuition, but neither number is real-world exchange PnL.

Maximum inventory and drawdown should be read as proxy exposure diagnostics. They help explain whether quote opportunities cluster in time, but they are not a deployable risk model.

## Outputs

- `results/sequential_proxy_backtest_by_session.csv`
- `results/sequential_proxy_equity_curve.csv`
- `figures/equity_curve_by_session.png`
- `figures/inventory_curve_by_session.png`
- `figures/drawdown_curve.png`
