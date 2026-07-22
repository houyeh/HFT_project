# Dataset & Model Identity — Verified Data

## Data Source
- Path: `data/NEW_ALL`  Symbol: TXFB6 (台指期)
- File: one CSV per session subfolder; loaded via `load_new_tw_data()`
- Normalization: rolling z-score window=2000 events per feature

## Session Split (time-ordered, 70/15/15)
Total sessions: 9  (2026-01-21 to 2026-01-27)

| Role | Sessions (date) | Samples |
|------|----------------|---------|
| Train (70%) | 20260121Day, 20260122Day, 20260122Night, 20260123Day, 20260123Night, 20260126Day | 718,562 |
| Val (15%) | 20260126Night | 112,698 |
| Test (15%) | 20260127Day, 20260127Night | 233,268 |

## Session-level Sample & Class Distribution
See `session_class_distribution.csv` (same folder).

Key test sessions:
- 20260127Day:   82,119 samples  (Down=1,204 / Stat=79,656 / Up=1,259)
- 20260127Night: 151,149 samples (Down=1,558 / Stat=148,055 / Up=1,536)
- Test total: 233,268 samples; Down+Up ≈ 2.36% (severe class imbalance)

## Model Config
- label: smoothed_mid_move_3class, H=5, threshold=0.5 ticks
- features: 8 (obi, queue_imbalance, weighted_mid_deviation, trade_intensity_short,
              deep_obi, queue_imbalance_change, normalized_spread, cancel_pressure_short)
- TCN: channels=[16,16,16], kernel=3, dropout=0.1, window=100, seed=42

## 資料來源
- `lt=.../dataset_manifest.json`
- `lt=.../experiment_config.json`

## Label Definition (smoothed_mid_move_3class)
- Horizon: `H=5`
- Threshold: `0.5 ticks`
- For each event `i`:
  - `future_mid_smooth(i) = mean(mid_{i+1}, ..., mid_{i+H})`
  - `delta_ticks(i) = (future_mid_smooth(i) - mid_i) / tick_size`
- Class mapping:
  - `Down (0)`: `delta_ticks < -0.5`
  - `Stationary (1)`: `-0.5 <= delta_ticks <= 0.5`
  - `Up (2)`: `delta_ticks > 0.5`

## Feature Definitions (8 features)
All features are converted to rolling z-score using window `2000` events:
`x_z = (x - rolling_mean_2000) / rolling_std_2000`.

1. `obi`  
   `obi = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)`

2. `queue_imbalance`  
   Let  
   `weighted_bid = sum_{l=1..5}(bid_vol_l / l)`,  
   `weighted_ask = sum_{l=1..5}(ask_vol_l / l)`.  
   Then  
   `queue_imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)`.

3. `weighted_mid_deviation`  
   `weighted_mid = (ask_px_1 * bid_vol_1 + bid_px_1 * ask_vol_1) / (bid_vol_1 + ask_vol_1)`  
   `mid = (ask_px_1 + bid_px_1) / 2`  
   `weighted_mid_deviation = (weighted_mid - mid) / tick_size`.

4. `trade_intensity_short`  
   `trade_intensity_short = rolling_sum_20(trade_volume_interval) / rolling_mean_20(bid_vol_1 + ask_vol_1 + bid_vol_2 + ask_vol_2)`.

5. `deep_obi`  
   `deep_obi = (sum_{l=1..5} bid_vol_l - sum_{l=1..5} ask_vol_l) / (sum_{l=1..5} bid_vol_l + sum_{l=1..5} ask_vol_l)`.

6. `queue_imbalance_change`  
   `queue_imbalance_change = queue_imbalance_t - queue_imbalance_{t-1}`.

7. `normalized_spread`  
   `normalized_spread = (ask_px_1 - bid_px_1) / tick_size`.

8. `cancel_pressure_short`  
   First define:
   - `bid_cancel_proxy = max((bid_vol_1(t-1) - bid_vol_1(t)) - sell_aggressor_volume_interval(t), 0)` when `bid_px_1` unchanged
   - `ask_cancel_proxy = max((ask_vol_1(t-1) - ask_vol_1(t)) - buy_aggressor_volume_interval(t), 0)` when `ask_px_1` unchanged  
   Then:
   `cancel_pressure_short = rolling_sum_20(ask_cancel_proxy - bid_cancel_proxy) / rolling_sum_20(ask_cancel_proxy + bid_cancel_proxy)`.
