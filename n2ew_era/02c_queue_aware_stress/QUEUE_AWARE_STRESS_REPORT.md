# Queue-Aware Stress Report

Phase 02C applies proxy queue/fill stress only to policies that passed Phase 02B. It does not implement true queue replay, aggressor-side replay, or order-ID level matching.

Main stressed candidate: `side_by_queue_imbalance_abs_0p1` at horizon `H20`.

Stress formula:

```text
if base_pnl > 0: stressed_pnl = base_pnl * fill_keep_rate
if base_pnl < 0: stressed_pnl = base_pnl * adverse_multiplier
action_delay_events shifts the policy decision forward within the session; initial delayed rows are no_quote.
```

## Moderate Stress

Moderate stress is `fill_keep_rate = 0.5`, `adverse_multiplier = 1.5`, `action_delay_events = 0`.

| policy_id                       |   horizon | split   |   fill_keep_rate |   adverse_multiplier |   action_delay_events |   state_count |   selected_action_count |   selection_rate |   stress_total_pnl |   win_rate |   loss_rate |    max_drawdown | positive_under_stress   | moderate_stress   |
|:--------------------------------|----------:|:--------|-----------------:|---------------------:|----------------------:|--------------:|------------------------:|-----------------:|-------------------:|-----------:|------------:|----------------:|:------------------------|:------------------|
| side_by_queue_imbalance_abs_0p1 |        20 | val     |              0.5 |                  1.5 |                     0 |        563477 |                  468627 |         0.83167  |        5.27034e+07 |   0.924183 |   0.0757767 |     1.03545e+06 | True                    | True              |
| side_by_queue_imbalance_abs_0p1 |        20 | test    |              0.5 |                  1.5 |                     0 |       1166321 |                  971600 |         0.833047 |        1.13652e+08 |   0.948985 |   0.0509757 | 47425           | True                    | True              |
| side_by_queue_imbalance_abs_0p1 |        20 | new4    |              0.5 |                  1.5 |                     0 |       2196640 |                 1799304 |         0.819116 |        2.38079e+08 |   0.931123 |   0.0688394 | 90417.5         | True                    | True              |

## Fill Keep Break-Even

| split   |   adverse_multiplier |   min_fill_keep_rate_positive |
|:--------|---------------------:|------------------------------:|
| test    |                 1    |                           0.1 |
| test    |                 1.25 |                           0.1 |
| test    |                 1.5  |                           0.1 |
| test    |                 2    |                           0.1 |
| new4    |                 1    |                           0.1 |
| new4    |                 1.25 |                           0.1 |
| new4    |                 1.5  |                           0.1 |
| new4    |                 2    |                           0.1 |

## Outputs

- `results/queue_stress_grid.csv`
- `results/delay_sensitivity.csv`
- `results/fill_keep_break_even.csv`
- `figures/queue_stress_heatmap.png`
- `figures/delay_sensitivity.png`
