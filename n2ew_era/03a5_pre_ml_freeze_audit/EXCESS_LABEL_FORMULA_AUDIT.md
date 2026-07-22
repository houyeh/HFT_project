# Excess Label Formula Audit

Hard rule: for every horizon, raw bid/ask markout must reconstruct exactly from half-spread plus excess markout.

```text
raw_bid_pnl_H = (mid_t - best_bid) * point_value + excess_bid_pnl_H
raw_ask_pnl_H = (best_ask - mid_t) * point_value + excess_ask_pnl_H
```

Numerical tolerance: `1e-09`.

|   horizon |   max_abs_bid_reconstruction_error |   max_abs_ask_reconstruction_error | formula_pass   |
|----------:|-----------------------------------:|-----------------------------------:|:---------------|
|         1 |                                  0 |                                  0 | True           |
|         5 |                                  0 |                                  0 | True           |
|        10 |                                  0 |                                  0 | True           |
|        20 |                                  0 |                                  0 | True           |
|        50 |                                  0 |                                  0 | True           |
|       100 |                                  0 |                                  0 | True           |
