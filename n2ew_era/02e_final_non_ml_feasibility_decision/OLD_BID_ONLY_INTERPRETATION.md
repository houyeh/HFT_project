# Old Bid-Only Interpretation

For `old_bid_only_spread_ge_1`, the Phase 01 bid markout formula can be decomposed as:

```text
bid_pnl_H = (future_mid_H - best_bid) * POINT_VALUE - MAKER_FEE
          = (mid_t - best_bid) * POINT_VALUE + (future_mid_H - mid_t) * POINT_VALUE - MAKER_FEE
```

`(mid_t - best_bid)` is the half-spread advantage. `(future_mid_H - mid_t)` is directional mid-price drift. This audit checks whether the apparent PnL is mostly mechanical spread capture rather than a directional forecasting edge.

## H20 Split-Level Decomposition

| split   | session_id   |   horizon |   selected_action_count |   total_pnl |   half_spread_advantage_pnl |   mid_price_drift_pnl |   maker_fee_total |   avg_total_pnl |   avg_half_spread_advantage_pnl |   avg_mid_price_drift_pnl |   positive_rate |   spread_component_share_of_positive_components |   drift_component_share_of_positive_components | spread_dominated   |
|:--------|:-------------|----------:|------------------------:|------------:|----------------------------:|----------------------:|------------------:|----------------:|--------------------------------:|--------------------------:|----------------:|------------------------------------------------:|-----------------------------------------------:|:-------------------|
| new3    | __ALL__      |        20 |                 5657242 | 1.43247e+09 |                 1.51532e+09 |            2.004e+06  |       8.48586e+07 |         253.209 |                         267.855 |                0.354236   |        0.90345  |                                        0.998679 |                                    0.00132075  | True               |
| new4    | __ALL__      |        20 |                 2196560 | 5.34754e+08 |                 5.6664e+08  |            1.0625e+06 |       3.29484e+07 |         243.451 |                         257.967 |                0.483711   |        0.892344 |                                        0.998128 |                                    0.00187158  | True               |
| test    | __ALL__      |        20 |                 1166281 | 2.45096e+08 |                 2.61056e+08 |            1.5334e+06 |       1.74942e+07 |         210.151 |                         223.837 |                1.31478    |        0.909079 |                                        0.99416  |                                    0.00583952  | True               |
| val     | __ALL__      |        20 |                  563457 | 1.1671e+08  |                 1.2516e+08  |         1800          |       8.45186e+06 |         207.132 |                         222.129 |                0.00319456 |        0.881075 |                                        0.999986 |                                    1.43814e-05 | True               |

## Interpretation

At H20, spread component absolute magnitude exceeds drift component in 4/4 evaluated split-level rows.

Advisor-safe conclusion: a large half-spread component indicates proxy artifact risk because Phase 01 assumes candidate quotes receive markout labels without true queue priority or fill probability. Strong old-bid-only PnL should therefore not be promoted to a final trading strategy without stricter execution simulation.
