# Non-ML Stress Robustness Report

- Policy: `bid_only_spread_ge_1`.
- NEW4 symbol: `TXFA6`.
- NEW4 tick size: `1.0`.
- Selection was frozen before NEW4 evaluation; NEW4 is not used for tuning.

## Stress PnL

- `base`: selected quote PnL `13061790`, positive selected quote rate `0.710`.
- `mild`: selected quote PnL `9915630`, positive selected quote rate `0.535`.
- `moderate`: selected quote PnL `6348770`, positive selected quote rate `0.363`.

## Concentration

- Moderate selected quote PnL: `6348770`.
- Top 1/5/10 percent quote contribution shares: `0.099`, `0.404`, `0.721`.
- Detailed NEW4 breakdown is in `new4_robustness_breakdown.csv` by session, spread bucket, and time bucket.

The NEW4 stress evidence supports the spread/side controller as an external OOS proxy result, not as real exchange profitability.
