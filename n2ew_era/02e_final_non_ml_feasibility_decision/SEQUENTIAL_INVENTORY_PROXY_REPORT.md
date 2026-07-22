# Sequential Inventory Proxy Report

This is a stricter chronological proxy simulator for frozen policies. It tracks cash, signed inventory, forced no-quote from inventory caps, and drawdown over fill/close events. It remains a proxy: it still lacks true queue priority, exchange acknowledgements, partial fills, and aggressor-side replay.

Fill modes:

- `markout_fill_all`: every selected quote fills; optimistic.
- `conservative_adverse_fill`: losing fills always occur, winning fills are kept at probability 0.5.
- `random_fill_rate`: all selected quotes fill at probability 0.5.

Exit rules:

- `fixed_horizon_H20`: close each fill at the H20 future mid.
- `session_end_mtm`: mark all open inventory to the session-final mid.

## Conservative H20 / Cap=1 View

| split   | policy_id                       |   horizon | fill_mode                 |   inventory_cap | exit_rule         |   sequential_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   avg_trade_pnl |   random_trials_for_95p_policy |
|:--------|:--------------------------------|----------:|:--------------------------|----------------:|:------------------|-----------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|----------------:|-------------------------------:|
| test    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| test    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      1.16368e+07 |        55806 |                  579486 |                   1 |           4800 |   0.899957 |         208.523 |                              0 |
| test    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      3.43407e+07 |       141464 |                  368641 |                   1 |           4200 |   0.938684 |         242.753 |                              0 |
| test    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      4.88139e+07 |       245125 |                  174199 |                   1 |           3200 |   0.867362 |         199.139 |                            250 |
| new4    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| new4    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.52911e+07 |       105240 |                 1112274 |                   1 |          13015 |   0.878145 |         240.318 |                              0 |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      8.49369e+07 |       306263 |                  656184 |                   1 |           5515 |   0.916709 |         277.333 |                              0 |
| new4    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      1.06115e+08 |       462343 |                  328660 |                   1 |           7715 |   0.842848 |         229.516 |                            250 |
| new3    | no_quote                        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      0           |            0 |                       0 |                   0 |              0 |   0        |           0     |                              0 |
| new3    | old_bid_only_spread_ge_1        |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      6.78404e+07 |       270886 |                 2832102 |                   1 |           4700 |   0.889769 |         250.439 |                              0 |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.25636e+08 |       781297 |                 1674581 |                   1 |          20600 |   0.926919 |         288.797 |                              0 |
| new3    | matched_random_95p              |        20 | conservative_adverse_fill |               1 | fixed_horizon_H20 |      2.84631e+08 |      1184932 |                  829880 |                   1 |          14790 |   0.858365 |         240.208 |                            250 |

## Decision Use

If policies collapse under conservative adverse fills and small inventory caps, they should be classified as signal discovery rather than final strategy candidates.
