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

## Sources
- `lt=.../dataset_manifest.json`
- `lt=.../experiment_config.json`
