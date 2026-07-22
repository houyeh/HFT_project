# PHC-lite Sequence Dataset Report

Phase 05A uses the frozen Phase 04B TXF-only action universe. `NEW5` is not loaded. Development is `NEW2 -> train` and `NEW1 -> val`; diagnostic OOS is `NEW3` and `NEW4`.

Sequence setting: `seq_len=128`, `stride=32`, target is the final timestep, target horizon is `H20`.

Allowed PHC-lite input features:

1. spread_ticks, 
2. bid_sz_1, 
3. ask_sz_1, 
4. top_depth, 
5. queue_imbalance, 
6. deep_imbalance, 
7. volatility_past_50, 
8. event_rate_past_50, 
9. inter_arrival_ms, 
10. imbalance_change_past, 
11. spread_change_past, 
12. log1p_bid_sz_1, 
13. log1p_ask_sz_1, 
14. log1p_top_depth, 
15. clipped_inter_arrival_ms, 
16. zscore_volatility_past_50_by_train_stats

## Feature Formula / Meaning

The table below documents the pre-scaler feature values. In the final saved sequence tensor, all 16 features are then standardized by the Phase 05A train-only feature scaler:

`x_feature = (feature_value - train_feature_mean) / max(train_feature_std, 1e-6)`

The only exception in wording is `zscore_volatility_past_50_by_train_stats`: it is itself a derived z-score feature before the final 16-feature scaler is applied. This means the model receives a consistent train-only normalized tensor, while the table remains the source-level definition of each feature.

| # | feature | 中文名稱 | formula / calculation | 中文定義 / meaning | code source |
|---:|---|---|---|---|---|
| 1 | `spread_ticks` | 買賣價差 tick 數 | `best_ask - best_bid` in Phase 04B TXF-only pipeline because `tick_size = 1.0`; equivalently `(ask_px_1 - bid_px_1) / tick_size`. | 目前最佳賣價與最佳買價之間的價差，以 tick 表示。數值越大代表 L1 報價越寬，通常表示交易成本、等待成本或流動性風險較高。 | `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:264-267`; carried into 05A by `common_05a.py:206-218, 240-248`. |
| 2 | `bid_sz_1` | 第一檔買方掛單量 | Raw L1 bid size from source column `1th_buy_qty`, renamed to `bid_sz_1`. | 最佳買價上的可見掛單量，代表買方隊列厚度。它是目前這一列可觀測的 L1 bid-side liquidity。 | `run_phase04b_pipeline.py:66-78, 269`; carried into action rows at `388-404`. |
| 3 | `ask_sz_1` | 第一檔賣方掛單量 | Raw L1 ask size from source column `1th_sell_qty`, renamed to `ask_sz_1`. | 最佳賣價上的可見掛單量，代表賣方隊列厚度。它是目前這一列可觀測的 L1 ask-side liquidity。 | `run_phase04b_pipeline.py:66-78, 270`; carried into action rows at `388-404`. |
| 4 | `top_depth` | 第一檔總深度 | `bid_sz_1 + ask_sz_1`, with missing values filled to `0.0` when built. | 最佳買價與最佳賣價的總掛單量。數值越大表示 top-of-book 越厚，通常代表被動單排隊更深、成交機率可能更低。 | `run_phase04b_pipeline.py:269-275`. |
| 5 | `queue_imbalance` | 第一檔隊列不平衡 | `safe_ratio(bid_sz_1 - ask_sz_1, bid_sz_1 + ask_sz_1)`. `safe_ratio` returns a finite ratio and avoids invalid divide-by-zero cases. | 衡量 L1 買賣兩側掛單量偏向哪一邊。正值代表買方掛單量相對較厚，負值代表賣方掛單量相對較厚；這也是手寫 queue rule 的核心訊號。 | `run_phase04b_pipeline.py:269-274`. |
| 6 | `deep_imbalance` | 五檔深度不平衡 | Let `bid_depth = sum(bid_sz_1 ... bid_sz_5)` and `ask_depth = sum(ask_sz_1 ... ask_sz_5)`. Then `safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)`. | 衡量 L2 五檔買賣深度的相對偏斜。它把單純第一檔不平衡擴展到更深的 order book liquidity。 | `run_phase04b_pipeline.py:271-274`. |
| 7 | `volatility_past_50` | 過去 50 事件中價波動 | Within each session, compute `mid_ret = mid_price.diff()`, then `mid_ret.rolling(50, min_periods=3).std().fillna(0.0)`. | 使用當前與過去資料計算的 rolling mid-price 變動標準差。它描述最近短窗內價格跳動程度，不使用 centered window 或 future rows。 | `run_phase04b_pipeline.py:288-289`; recompute proof in `n2ew_era/03a6_root_cause_feature_and_contract_closure/run_root_cause_closure.py:285-300`. |
| 8 | `event_rate_past_50` | 過去 50 事件到達率 | Within each session, compute `dt_us = timestamp_us.diff()` with day-wrap correction, then `event_rate = clip(1_000_000 / dt_us, 0, 1_000_000)`, then `rolling(50, min_periods=3).mean().fillna(0.0)`. | 最近事件流速，單位近似為 events/second。數值越高代表市場更新越快、microstructure regime 越活躍。 | `run_phase04b_pipeline.py:279-287`. |
| 9 | `inter_arrival_ms` | 事件間隔毫秒 | Within each session, `dt_us = timestamp_us_t - timestamp_us_{t-1}` with negative differences adjusted by one day; then `clip(dt_us / 1000.0, lower=0.0)`. First row or invalid diff becomes `0.0`. | 目前事件距離上一個事件的時間間隔，以毫秒表示。它是 session-local 的 causal timing feature。 | `run_phase04b_pipeline.py:279-283`. |
| 10 | `imbalance_change_past` | 隊列不平衡變化量 | Within each session, `queue_imbalance_t - queue_imbalance_{t-1}`, with first row filled as `0.0`. | 第一檔隊列不平衡相對上一事件的變化。正值表示買方相對壓力增加，負值表示賣方相對壓力增加。 | `run_phase04b_pipeline.py:291`; root-cause closure wording in `run_root_cause_closure.py:259-262`. |
| 11 | `spread_change_past` | 價差變化量 | Within each session, `spread_ticks_t - spread_ticks_{t-1}`, with first row filled as `0.0`. | 目前價差相對上一事件的變化。正值代表價差擴大，負值代表價差收斂。 | `run_phase04b_pipeline.py:292`; root-cause closure wording in `run_root_cause_closure.py:259-262`. |
| 12 | `log1p_bid_sz_1` | 第一檔買量對數轉換 | `log1p(max(bid_sz_1, 0.0))`. | 將最佳買價掛單量做 log 壓縮，保留深度大小順序，同時降低極端大掛單量對模型與 scaler 的影響。 | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:240-245`. |
| 13 | `log1p_ask_sz_1` | 第一檔賣量對數轉換 | `log1p(max(ask_sz_1, 0.0))`. | 將最佳賣價掛單量做 log 壓縮，與 `log1p_bid_sz_1` 對稱，用來提供較平滑的 ask-side liquidity magnitude。 | `common_05a.py:240-246`. |
| 14 | `log1p_top_depth` | 第一檔總深度對數轉換 | `log1p(max(top_depth, 0.0))`. | 將 L1 買賣總深度做 log 壓縮，讓模型看到 liquidity magnitude，同時降低深度 outlier 的支配性。 | `common_05a.py:240-246`. |
| 15 | `clipped_inter_arrival_ms` | 截斷後事件間隔 | `clip(inter_arrival_ms, 0.0, 5000.0)`. | 把事件間隔限制在 0 到 5000 毫秒之間，避免極端停頓時間主導 train-only scaler 或模型輸入。 | `common_05a.py:240-247`. |
| 16 | `zscore_volatility_past_50_by_train_stats` | 訓練集標準化波動特徵 | `(volatility_past_50 - volatility_mean_train) / max(volatility_std_train, 1e-6)`, where `volatility_mean_train` and `volatility_std_train` are fit on train only. Current stored values: mean `5729.84619140625`, std `4632.89111328125`. | 使用訓練集統計量標準化的短窗波動 regime 指標。它讓模型直接取得相對於 train regime 的波動位置，且不使用 val/NEW3/NEW4/NEW5 的統計量。 | `common_05a.py:240-248, 252-297, 308-318`; audit summary in `AUDIT_phc_lite_transform_features.md`. |

All base dynamic features are computed within session order and reset at session boundaries. Future labels such as `future_mid`, `raw_pnl`, `excess_pnl`, and positive/negative labels are not used as inputs.

Forbidden columns are not used as inputs: best bid/ask, mid/quote price, product id, future_mid, raw_pnl, excess_pnl, positive labels, terminal/session-final EV.

## Split Summary

| split   |   event_rows |   window_rows |   sessions |   bid_positive_rate |   ask_positive_rate |   target_mean_bid |   target_mean_ask |
|:--------|-------------:|--------------:|-----------:|--------------------:|--------------------:|------------------:|------------------:|
| new3    |      5656822 |        176754 |          6 |            0.363712 |            0.36333  |          -14.3211 |          -15.6785 |
| new4    |      2196280 |         68620 |          4 |            0.366219 |            0.365125 |          -14.9221 |          -15.0771 |
| train   |      2470559 |         77194 |          3 |            0.358769 |            0.351701 |          -14.6048 |          -15.3949 |
| val     |      2655369 |         82960 |          6 |            0.351095 |            0.346406 |          -14.322  |          -15.676  |

## Normalization

Feature scaler is fit on train only and applied unchanged to val/NEW3/NEW4.
Train rows used for scaler: `2470559`.
Target scale: `201.1750030517578`. The target is divided by this positive scale during optimization only; no mean shift is applied, so raw zero remains the no-quote reference.

No NEW5 data is used for scaler fitting, architecture, early stopping, thresholding, or diagnostics in this phase.
