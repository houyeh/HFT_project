# Dynamic Feature Rebuild Report

Missing dynamic features were rebuilt strictly within each session using current and past rows only. Session starts reset to zero for inter-arrival, event-rate rolling mean, imbalance change, and spread change.

Rebuilt features:

- `inter_arrival_ms = timestamp_t - timestamp_{t-1}` within the same session.
- `event_rate_past_50 = rolling_mean(1e9 / inter_arrival_ns, window=50, min_periods=3)` within the same session.
- `imbalance_change_past = queue_imbalance_t - queue_imbalance_{t-1}` within the same session.
- `spread_change_past = spread_ticks_t - spread_ticks_{t-1}` within the same session.

| split   | session_id    |   row_count |   first_inter_arrival_ms |   first_event_rate_past_50 |   first_imbalance_change_past |   first_spread_change_past |   max_event_rate_past_50 | causal_session_reset_pass   | uses_future_data   |
|:--------|:--------------|------------:|-------------------------:|---------------------------:|------------------------------:|---------------------------:|-------------------------:|:----------------------------|:-------------------|
| train   | 20260121Day   |      705812 |                        0 |                          0 |                             0 |                          0 |                  470.205 | True                        | False              |
| train   | 20260122Day   |      944608 |                        0 |                          0 |                             0 |                          0 |                  447.226 | True                        | False              |
| train   | 20260122Night |      820409 |                        0 |                          0 |                             0 |                          0 |                  571.903 | True                        | False              |
| train   | 20260123Day   |      182203 |                        0 |                          0 |                             0 |                          0 |                  423.111 | True                        | False              |
| train   | 20260123Night |      565974 |                        0 |                          0 |                             0 |                          0 |                  464.224 | True                        | False              |
| train   | 20260126Day   |      373758 |                        0 |                          0 |                             0 |                          0 |                  380.145 | True                        | False              |
| val     | 20260126Night |      563477 |                        0 |                          0 |                             0 |                          0 |                  412.303 | True                        | False              |
| test    | 20260127Day   |      410586 |                        0 |                          0 |                             0 |                          0 |                  423.333 | True                        | False              |
| test    | 20260127Night |      755735 |                        0 |                          0 |                             0 |                          0 |                  372     | True                        | False              |
| new4    | 20260102      |      502106 |                        0 |                          0 |                             0 |                          0 |                  460     | True                        | False              |
| new4    | 20260105      |       65520 |                        0 |                          0 |                             0 |                          0 |                  434.346 | True                        | False              |
| new4    | 20260109      |      644384 |                        0 |                          0 |                             0 |                          0 |                  412.495 | True                        | False              |
| new4    | 20260112      |      984630 |                        0 |                          0 |                             0 |                          0 |                  453.222 | True                        | False              |
| new3    | 20260113      |     1277836 |                        0 |                          0 |                             0 |                          0 |                  404.833 | True                        | False              |
| new3    | 20260114      |      910661 |                        0 |                          0 |                             0 |                          0 |                  445.389 | True                        | False              |
| new3    | 20260115      |      976150 |                        0 |                          0 |                             0 |                          0 |                  450.868 | True                        | False              |
| new3    | 20260116      |      976422 |                        0 |                          0 |                             0 |                          0 |                  443.259 | True                        | False              |
| new3    | 20260119Day   |     1365489 |                        0 |                          0 |                             0 |                          0 |                  451.667 | True                        | False              |
| new3    | 20260330      |      150804 |                        0 |                          0 |                             0 |                          0 |                  384     | True                        | False              |

Output data: `data/rebuilt_dynamic_features.parquet`.
