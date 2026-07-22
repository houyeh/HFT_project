# Cost / Unit / Product Contract

`point_value = 200.0` and `maker_fee = 15.0` are inherited proxy assumptions used by Phase 01/02F formulae.

`maker_fee=15` is treated as a one-way proxy maker fee subtracted once per quote-level markout/fill. The current files do not certify whether this includes exchange fees, tax, broker fees, or round-trip costs.

Therefore these are proxy PnL units, not exchange-certified values.

## Symbol / Tick Size Audit

| split   | symbol   |   sessions |   tick_size_min |   tick_size_max |
|:--------|:---------|-----------:|----------------:|----------------:|
| new3    | TXFA6    |          6 |               1 |               1 |
| new4    | TXFA6    |          4 |               1 |               1 |
| test    | TXFB6    |          2 |               1 |               1 |
| train   | TXFB6    |          6 |               1 |               1 |
| val     | TXFB6    |          1 |               1 |               1 |

Historical train/val/test use `TXFB6`; NEW3/NEW4 use `TXFA6`. Tick size is consistent at 1.0 in the generated parquet, but contract month differs, so OOS comparison is valid as a harsh proxy regime check, not as certified same-contract trading replay.
