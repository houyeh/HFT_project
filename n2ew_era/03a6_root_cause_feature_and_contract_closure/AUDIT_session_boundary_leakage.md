status: resolved
# Audit: Session Boundary Leakage in Rebuilt Dynamic Features

The dynamic feature rebuild explicitly resets at session starts. The first row of every audited session has zero inter-arrival, event-rate, imbalance-change, and spread-change values.

## Boundary samples, at least three sessions per split
| split   | session_id    |   row_count |   first_inter_arrival_ms |   first_event_rate_past_50 |   first_imbalance_change_past |   first_spread_change_past |   max_event_rate_past_50 | causal_session_reset_pass   | uses_future_data   |
|:--------|:--------------|------------:|-------------------------:|---------------------------:|------------------------------:|---------------------------:|-------------------------:|:----------------------------|:-------------------|
| new3    | 20260113      |     1277836 |                        0 |                          0 |                             0 |                          0 |                  404.833 | True                        | False              |
| new3    | 20260114      |      910661 |                        0 |                          0 |                             0 |                          0 |                  445.389 | True                        | False              |
| new3    | 20260115      |      976150 |                        0 |                          0 |                             0 |                          0 |                  450.868 | True                        | False              |
| new4    | 20260102      |      502106 |                        0 |                          0 |                             0 |                          0 |                  460     | True                        | False              |
| new4    | 20260105      |       65520 |                        0 |                          0 |                             0 |                          0 |                  434.346 | True                        | False              |
| new4    | 20260109      |      644384 |                        0 |                          0 |                             0 |                          0 |                  412.495 | True                        | False              |
| test    | 20260127Day   |      410586 |                        0 |                          0 |                             0 |                          0 |                  423.333 | True                        | False              |
| test    | 20260127Night |      755735 |                        0 |                          0 |                             0 |                          0 |                  372     | True                        | False              |
| train   | 20260121Day   |      705812 |                        0 |                          0 |                             0 |                          0 |                  470.205 | True                        | False              |
| train   | 20260122Day   |      944608 |                        0 |                          0 |                             0 |                          0 |                  447.226 | True                        | False              |
| train   | 20260122Night |      820409 |                        0 |                          0 |                             0 |                          0 |                  571.903 | True                        | False              |
| val     | 20260126Night |      563477 |                        0 |                          0 |                             0 |                          0 |                  412.303 | True                        | False              |

## Rolling volatility recompute samples
| split   | session_id    |   row_count | feature            | formula                                                     |   max_abs_error |   mean_abs_error |   tolerance | recompute_pass   | uses_future_data   |
|:--------|:--------------|------------:|:-------------------|:------------------------------------------------------------|----------------:|-----------------:|------------:|:-----------------|:-------------------|
| new3    | 20260113      |     1277836 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| new3    | 20260114      |      910661 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| new3    | 20260115      |      976150 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| new4    | 20260102      |      502106 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| new4    | 20260105      |       65520 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| new4    | 20260109      |      644384 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| test    | 20260127Day   |      410586 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| test    | 20260127Night |      755735 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| train   | 20260121Day   |      705812 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| train   | 20260122Day   |      944608 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| train   | 20260122Night |      820409 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |
| val     | 20260126Night |      563477 | volatility_past_50 | mid_price.diff().rolling(50, min_periods=3).std().fillna(0) |               0 |                0 |       1e-12 | True             | False              |

Conclusion: existing 03A.6 artifacts show `causal_session_reset_pass=True`, `uses_future_data=False`, and rolling recompute max error 0.0 for the audited sessions. No session-boundary leakage was detected in these artifacts.
