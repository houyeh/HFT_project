status: resolved
# Audit: PHC-lite Transform Feature Definitions

| feature                                  | formula                                                                        | source                                                                                        | causality                                               |
|:-----------------------------------------|:-------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|:--------------------------------------------------------|
| log1p_bid_sz_1                           | log1p(max(bid_sz_1, 0.0))                                                      | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:244                   | current-row L2 size only                                |
| log1p_ask_sz_1                           | log1p(max(ask_sz_1, 0.0))                                                      | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:245                   | current-row L2 size only                                |
| log1p_top_depth                          | log1p(max(top_depth, 0.0))                                                     | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:246                   | current-row L2 depth only                               |
| clipped_inter_arrival_ms                 | clip(inter_arrival_ms, 0.0, 5000.0)                                            | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:247                   | rebuilt as past/current session-local timing feature    |
| zscore_volatility_past_50_by_train_stats | (volatility_past_50 - volatility_mean_train) / max(volatility_std_train, 1e-6) | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:248, 252-297, 308-318 | train-only mean/std; applied unchanged to val/NEW3/NEW4 |

## Train-only Normalization Evidence

- `normalization_fit_split`: `train`
- `train_feature_rows`: `2470559`
- `volatility_mean_train`: `5729.84619140625`
- `volatility_std_train`: `4632.89111328125`
- `new3_used_for_scaler`: `False`
- `new4_used_for_scaler`: `False`
- `new5_used_for_scaler`: `False`

## Source Excerpt

```python
240: def base_feature_frame(frame: pd.DataFrame, vol_mean: float = 0.0, vol_std: float = 1.0) -> pd.DataFrame:
241:     out: dict[str, np.ndarray] = {}
242:     for col in BASE_EVENT_COLUMNS:
243:         out[col] = numeric_series(frame, col)
244:     out["log1p_bid_sz_1"] = np.log1p(np.maximum(out["bid_sz_1"], 0.0)).astype(np.float32)
245:     out["log1p_ask_sz_1"] = np.log1p(np.maximum(out["ask_sz_1"], 0.0)).astype(np.float32)
246:     out["log1p_top_depth"] = np.log1p(np.maximum(out["top_depth"], 0.0)).astype(np.float32)
247:     out["clipped_inter_arrival_ms"] = np.clip(out["inter_arrival_ms"], 0.0, 5000.0).astype(np.float32)
248:     out["zscore_volatility_past_50_by_train_stats"] = ((out["volatility_past_50"] - vol_mean) / max(vol_std, 1e-6)).astype(np.float32)
```

Conclusion: all five requested transform features are explicitly implemented. No future label enters these transformations; z-score constants are fit on train only.
