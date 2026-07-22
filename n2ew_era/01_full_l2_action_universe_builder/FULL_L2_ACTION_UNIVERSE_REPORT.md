# Full L2 Action Universe Report

This phase builds a broad full-action L2 universe for short-horizon sequential proxy labels. It does not train ML, tune strategies, use terminal quote EV as a target, or filter actions through future fill logic.

## 1. Raw Rows vs Candidate Actions

Total raw rows loaded from the historical quote-eligible source lineage: `7,520,666`.
Total candidate actions created: `22,557,606`.

| split   | dataset   | symbol   |   raw_rows |   valid_book_rows |   quote_eligible_rows |   candidate_action_rows |   bid_actions |   ask_actions |   no_quote_actions | candidate_action_match   |
|:--------|:----------|:---------|-----------:|------------------:|----------------------:|------------------------:|--------------:|--------------:|-------------------:|:-------------------------|
| new4    | NEW4      | TXFA6    |    2197106 |           2197080 |               2196640 |                 6589920 |       2196640 |       2196640 |            2196640 | True                     |
| test    | NEW_ALL   | TXFB6    |    1166541 |           1166540 |               1166321 |                 3498963 |       1166321 |       1166321 |            1166321 | True                     |
| train   | NEW_ALL   | TXFB6    |    3593426 |           3593424 |               3592764 |                10778292 |       3592764 |       3592764 |            3592764 | True                     |
| val     | NEW_ALL   | TXFB6    |     563593 |            563587 |                563477 |                 1690431 |        563477 |        563477 |             563477 | True                     |

Every quote-eligible row emits exactly three candidate actions: `bid`, `ask`, and `no_quote`.

## NEW4 Status

NEW4 was included in Phase 01 as split `new4` from `C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW4`. The compatible raw L2 loader detected symbol `TXFA6` with `2,197,106` raw rows and `2,196,640` quote-eligible rows.

The NEW4 inclusion check used only raw L2 compatibility: valid 5-level book columns, finite positive bid/ask depth, positive spread, and the same first/last tick boundary removal. It did not use future fill logic or any strategy threshold.

## 2. Bottleneck Check

The old residual ML train universe had `820` rows. The new train full-action universe has `10,778,292` candidate actions.
This successfully avoids the old 820-row bottleneck and keeps the full quote-eligible decision surface available for later non-ML and ML research.

## 3. Causal vs Future Columns

Strictly causal/action-context columns are allowed as future feature candidates only if they are known at the quote decision time.

Causal/action columns include:

`action_side`, `action_slot`, `quote_price`, `best_bid`, `best_ask`, `mid_price`, `tick_size`, `spread_ticks`, `quote_eligible`, `bid_px_1`, `bid_sz_1`, `ask_px_1`, `ask_sz_1`, `bid_px_2`, `bid_sz_2`, `ask_px_2`, `ask_sz_2`, `bid_px_3`, `bid_sz_3`, `ask_px_3`, `ask_sz_3`, `bid_px_4`, `bid_sz_4`, `ask_px_4`, `ask_sz_4`, `bid_px_5`, `bid_sz_5`, `ask_px_5`, `ask_sz_5`, `queue_imbalance`, `deep_imbalance`, `top_depth`, `depth_ratio`, `event_rate_past_50`, `inter_arrival_ms`, `volatility_past_50`, `abs_return_past_50`, `imbalance_change_past`, `spread_change_past`, `hawkes_excite`, `normalized_spread`

Future label columns are not allowed as features:

`future_mid_H1`, `markout_ticks_H1`, `pnl_H1`, `positive_pnl_H1`, `negative_pnl_H1`, `severe_negative_pnl_H1`, `future_mid_H5`, `markout_ticks_H5`, `pnl_H5`, `positive_pnl_H5`, `negative_pnl_H5`, `severe_negative_pnl_H5`, `future_mid_H10`, `markout_ticks_H10`, `pnl_H10`, `positive_pnl_H10`, `negative_pnl_H10`, `severe_negative_pnl_H10`, `future_mid_H20`, `markout_ticks_H20`, `pnl_H20`, `positive_pnl_H20`, `negative_pnl_H20`, `severe_negative_pnl_H20`, `future_mid_H50`, `markout_ticks_H50`, `pnl_H50`, `positive_pnl_H50`, `negative_pnl_H50`, `severe_negative_pnl_H50`, `future_mid_H100`, `markout_ticks_H100`, `pnl_H100`, `positive_pnl_H100`, `negative_pnl_H100`, `severe_negative_pnl_H100`

The generated future label set includes `negative_pnl_H* = pnl_H* < 0` for each horizon.

The full column-level audit is in `results/causality_audit.csv`.

## 4. Sequential Proxy PnL Formula

Point value: `200.0`. Maker fee: `15.0`.

PnL unit note: `POINT_VALUE = 200.0` and `MAKER_FEE = 15.0` are inherited from prior project constants in `new_era/models_playaround_15/config.py` and `new_era/18_queue_aware/queue_stress_core.py`. The data symbols are Taiwan futures-style `TXFB6`/`TXFA6`; a 200 TWD point value corresponds to the project assumption for TXF-like contracts, not mini-TXF. The maker fee is treated as a one-way per-contract fee per quote action. The current project source does not verify whether this fee includes tax or the exact exchange/broker schedule. These are proxy PnL units inherited from project assumptions, not exchange-certified values.

For horizon `H`:

```text
future_mid_H = mid_price[t + H]
bid pnl_H = (future_mid_H - best_bid[t]) * point_value - maker_fee
ask pnl_H = (best_ask[t] - future_mid_H) * point_value - maker_fee
no_quote pnl_H = 0
markout_ticks_H = side-specific markout divided by tick_size
```

Horizons generated: `[1, 5, 10, 20, 50, 100]`.

Rows near the end of a session can have missing `future_mid_H` for bid/ask labels when `t + H` does not exist. `no_quote` remains zero by definition.

Phase 01 evaluates all possible bid/ask/no_quote actions using future-mid markout labels. It does not model fill probability or queue priority. This is intentional: Phase 01 builds a broad decision universe and avoids the old future-fill-conditioned residual bottleneck. Fill/execution realism will be handled in later phases.

## 5. Feature Drift Snapshot

Before Phase 02 policy search, the causal feature distributions were summarized across splits. The full table is in `results/feature_drift_train_val_test.csv`; if NEW4 is available it is included as external OOS context, not as a tuning split.

| split   | feature            |   count |         mean |        std |        p05 |         p50 |        p95 |
|:--------|:-------------------|--------:|-------------:|-----------:|-----------:|------------:|-----------:|
| new4    | spread_ticks       | 2196640 |   2.57966    |  0.893914  |  1         |   3         |   4        |
| new4    | queue_imbalance    | 2196640 |  -0.0066339  |  0.432443  | -0.714286  |   0         |   0.68     |
| new4    | deep_imbalance     | 2196640 |  -0.00514598 |  0.210834  | -0.353535  |   0         |   0.341463 |
| new4    | event_rate_past_50 | 2196640 | 143.013      | 47.8437    | 68.1378    | 141.144     | 225.015    |
| new4    | volatility_past_50 | 2196640 |   0.221558   |  0.112669  |  0.0989743 |   0.202031  |   0.389138 |
| new4    | top_depth          | 2196640 |   7.62566    |  5.71115   |  2         |   6         |  16        |
| test    | spread_ticks       | 1166321 |   2.23838    |  0.779321  |  1         |   2         |   3        |
| test    | queue_imbalance    | 1166321 |  -0.00704098 |  0.4446    | -0.714286  |   0         |   0.714286 |
| test    | deep_imbalance     | 1166321 |  -0.0227105  |  0.200624  | -0.365854  |  -0.0196078 |   0.301587 |
| test    | event_rate_past_50 | 1166321 | 114.73       | 56.0209    | 40.1325    | 105.907     | 214.778    |
| test    | volatility_past_50 | 1166321 |   0.146394   |  0.0704802 |  0         |   0.142857  |   0.24661  |
| test    | top_depth          | 1166321 |   8.78417    |  6.37057   |  3         |   7         |  18        |
| train   | spread_ticks       | 3592764 |   2.46548    |  0.889234  |  1         |   2         |   4        |
| train   | queue_imbalance    | 3592764 |  -0.008666   |  0.428991  | -0.714286  |   0         |   0.666667 |
| train   | deep_imbalance     | 3592764 |  -0.0184635  |  0.204948  | -0.36      |  -0.0149254 |   0.309524 |
| train   | event_rate_past_50 | 3592764 | 154.217      | 48.744     | 80.7819    | 151.14      | 237.99     |
| train   | volatility_past_50 | 3592764 |   0.15709    |  0.186098  |  0.0707107 |   0.142857  |   0.247642 |
| train   | top_depth          | 3592764 |   8.08496    |  5.69767   |  3         |   7         |  17        |
| val     | spread_ticks       |  563477 |   2.2213     |  0.979104  |  1         |   2         |   3        |
| val     | queue_imbalance    |  563477 |  -0.0108241  |  0.436339  | -0.714286  |   0         |   0.714286 |
| val     | deep_imbalance     |  563477 |  -0.0222583  |  0.185313  | -0.348837  |  -0.0136986 |   0.263158 |
| val     | event_rate_past_50 |  563477 | 154.091      | 46.2059    | 82.3812    | 151.428     | 233.764    |
| val     | volatility_past_50 |  563477 |   0.158915   |  0.423071  |  0.0989743 |   0.142857  |   0.234955 |
| val     | top_depth          |  563477 |   9.32969    |  6.06785   |  3         |   8         |  18        |

## Label Snapshot

| split   | action_side   |   horizon |   row_count |   pnl_mean |   pnl_median |   pnl_positive_rate |   pnl_negative_rate |   markout_ticks_mean |   markout_ticks_median |
|:--------|:--------------|----------:|------------:|-----------:|-------------:|--------------------:|--------------------:|---------------------:|-----------------------:|
| new4    | ask           |         1 |     2196640 |    242.942 |          285 |            0.989772 |          0.0102261  |              1.28971 |                    1.5 |
| new4    | bid           |         1 |     2196640 |    242.99  |          285 |            0.989832 |          0.010166   |              1.28995 |                    1.5 |
| new4    | ask           |        10 |     2196640 |    242.729 |          285 |            0.938654 |          0.0613278  |              1.28865 |                    1.5 |
| new4    | bid           |        10 |     2196640 |    243.204 |          285 |            0.939228 |          0.0607542  |              1.29102 |                    1.5 |
| new4    | ask           |        50 |     2196640 |    241.587 |          285 |            0.773949 |          0.22596    |              1.28293 |                    1.5 |
| new4    | bid           |        50 |     2196640 |    244.353 |          285 |            0.776986 |          0.222923   |              1.29676 |                    1.5 |
| new4    | ask           |       100 |     2196640 |    240.275 |          285 |            0.703972 |          0.295846   |              1.27638 |                    1.5 |
| new4    | bid           |       100 |     2196640 |    245.669 |          285 |            0.707064 |          0.292754   |              1.30335 |                    1.5 |
| test    | ask           |         1 |     1166321 |    208.77  |          185 |            0.992947 |          0.00705123 |              1.11885 |                    1   |
| test    | bid           |         1 |     1166321 |    208.905 |          185 |            0.992999 |          0.00699893 |              1.11953 |                    1   |
| test    | ask           |        10 |     1166321 |    208.177 |          185 |            0.948387 |          0.0515956  |              1.11588 |                    1   |
| test    | bid           |        10 |     1166321 |    209.498 |          185 |            0.949518 |          0.0504647  |              1.12249 |                    1   |
| test    | ask           |        50 |     1166321 |    205.551 |          185 |            0.804716 |          0.195198   |              1.10276 |                    1   |
| test    | bid           |        50 |     1166321 |    212.121 |          185 |            0.814898 |          0.185016   |              1.1356  |                    1   |
| test    | ask           |       100 |     1166321 |    202.287 |          185 |            0.725997 |          0.273831   |              1.08644 |                    1   |
| test    | bid           |       100 |     1166321 |    215.385 |          185 |            0.74413  |          0.255698   |              1.15193 |                    1   |
| train   | ask           |         1 |     3592764 |    231.545 |          185 |            0.994489 |          0.00550885 |              1.23273 |                    1   |
| train   | bid           |         1 |     3592764 |    231.55  |          185 |            0.994321 |          0.00567724 |              1.23275 |                    1   |
| train   | ask           |        10 |     3592764 |    231.521 |          185 |            0.947126 |          0.0528574  |              1.2326  |                    1   |
| train   | bid           |        10 |     3592764 |    231.572 |          185 |            0.946734 |          0.0532495  |              1.23286 |                    1   |
| train   | ask           |        50 |     3592764 |    231.439 |          185 |            0.779398 |          0.220519   |              1.2322  |                    1   |
| train   | bid           |        50 |     3592764 |    231.654 |          285 |            0.782194 |          0.217723   |              1.23327 |                    1.5 |
| train   | ask           |       100 |     3592764 |    231.414 |          185 |            0.706664 |          0.293169   |              1.23207 |                    1   |
| train   | bid           |       100 |     3592764 |    231.677 |          285 |            0.712899 |          0.286934   |              1.23338 |                    1.5 |
| val     | ask           |         1 |      563477 |    207.129 |          185 |            0.993125 |          0.0068734  |              1.11065 |                    1   |
| val     | bid           |         1 |      563477 |    207.13  |          185 |            0.993675 |          0.00632324 |              1.11065 |                    1   |
| val     | ask           |        10 |      563477 |    207.127 |          185 |            0.935932 |          0.0640505  |              1.11064 |                    1   |
| val     | bid           |        10 |      563477 |    207.133 |          185 |            0.939643 |          0.0603396  |              1.11067 |                    1   |
| val     | ask           |        50 |      563477 |    207.138 |          185 |            0.765907 |          0.234004   |              1.11069 |                    1   |
| val     | bid           |        50 |      563477 |    207.117 |          185 |            0.772095 |          0.227816   |              1.11059 |                    1   |
| val     | ask           |       100 |      563477 |    207.152 |          185 |            0.700861 |          0.298962   |              1.11076 |                    1   |
| val     | bid           |       100 |      563477 |    207.091 |          185 |            0.708579 |          0.291243   |              1.11045 |                    1   |

## Figures

- `figures/action_count_waterfall.png`
- `figures/action_side_distribution.png`
- `figures/horizon_label_distribution.png`
- `figures/feature_distribution_snapshot.png`
- `figures/feature_distribution_by_split.png`
- `figures/old_residual_vs_new_universe.png`
