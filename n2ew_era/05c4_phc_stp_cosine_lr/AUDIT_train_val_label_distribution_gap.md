# AUDIT: Train/Val Label Distribution Gap

status: `resolved_distributional_explanation`

## Scope Boundary

- Splits used: `train` and `val` only.
- NEW5 was not loaded or touched; loaded sequence payloads were checked for `new5_used=False`.
- No model was trained, no hyperparameter was changed, and no frozen checkpoint, scaler, or training curve file was modified.
- Payload session grouping uses `session_code + session_labels`; there is no per-row `session_id` field in the `.pt` payload.
- Scaler source: `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/data/phc_lite_feature_scaler.json`.

## Split Summary

| split | row_count | session_count | day_count | y_bid_mean_raw | y_bid_std_raw_population | y_ask_mean_raw | y_ask_std_raw_population | avg_std_normalized_population_bid_ask | avg_zero_prediction_huber_normalized_bid_ask |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 77194 | 3 | 2 | -14.587857 | 205.718161 | -15.411755 | 205.718161 | 1.022583 | 0.266489 |
| val | 82960 | 6 | 3 | -14.212150 | 149.123096 | -15.786403 | 149.123104 | 0.741261 | 0.228291 |

## Day Summary

| split | day_id | row_count | session_count | y_bid_std_raw_population | y_ask_std_raw_population | avg_std_normalized_population_bid_ask | position_vs_train_session_std_range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 20260121 | 22050 | 1 | 293.006928 | 293.006928 | 1.456478 | within_train_range |
| train | 20260122 | 55144 | 2 | 157.837394 | 157.837396 | 0.784578 | within_train_range |
| val | 20260123 | 23368 | 2 | 139.419139 | 139.419138 | 0.693024 | below_train_min |
| val | 20260126 | 29276 | 2 | 164.195947 | 164.195948 | 0.816185 | within_train_range |
| val | 20260127 | 30316 | 2 | 140.827979 | 140.828009 | 0.700027 | below_train_min |

## Session Summary

| split | session_id | day_id | row_count | y_bid_std_raw_population | y_ask_std_raw_population | avg_std_normalized_population_bid_ask | rank_if_inserted_in_train_sessions_by_avg_norm_std_1_lowest | position_vs_train_session_std_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | NEW2_20260121Day_TXFB6 | 20260121 | 22050 | 293.006928 | 293.006928 | 1.456478 | 3 | within_train_range |
| train | NEW2_20260122Day_TXFB6 | 20260122 | 29513 | 150.396250 | 150.396253 | 0.747589 | 1 | within_train_range |
| train | NEW2_20260122Night_TXFB6 | 20260122 | 25631 | 165.992688 | 165.992688 | 0.825116 | 2 | within_train_range |
| val | NEW1_20260123Day_TXFB6 | 20260123 | 5688 | 153.509477 | 153.509495 | 0.763064 | 2 | within_train_range |
| val | NEW1_20260123Night_TXFB6 | 20260123 | 17680 | 134.572117 | 134.572117 | 0.668931 | 1 | below_train_min |
| val | NEW1_20260126Day_TXFB6 | 20260126 | 11674 | 137.000199 | 137.000165 | 0.681000 | 1 | below_train_min |
| val | NEW1_20260126Night_TXFB6 | 20260126 | 17602 | 179.968740 | 179.968740 | 0.894588 | 3 | within_train_range |
| val | NEW1_20260127Day_TXFB6 | 20260127 | 12825 | 143.925997 | 143.926033 | 0.715427 | 1 | below_train_min |
| val | NEW1_20260127Night_TXFB6 | 20260127 | 17491 | 138.512331 | 138.512356 | 0.688517 | 1 | below_train_min |

## Health Check

| split | row_count | y_bid_nan_count | y_ask_nan_count | y_bid_unique_count | y_ask_unique_count | y_bid_all_same | y_ask_all_same | health_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 77194 | 0 | 0 | 36 | 36 | False | False | True |
| val | 82960 | 0 | 0 | 28 | 28 | False | False | True |

## Loss-Scale Proxy

| epoch_reference | observed_train_huber_loss | observed_val_huber_loss | observed_train_minus_val_huber_loss | train_avg_norm_std | val_avg_norm_std | train_over_val_avg_norm_std_ratio | train_over_val_avg_norm_variance_ratio | zero_prediction_huber_gap_train_minus_val | zero_prediction_gap_over_observed_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 171 | 0.233591 | 0.201340 | 0.032251 | 1.022583 | 0.741261 | 1.379519 | 1.903073 | 0.038198 | 1.184410 |

## Honest Conclusion

The distributional hypothesis is supported. The validation labels have much lower normalized standard deviation than train (`0.741261` vs `1.022583` averaged across bid/ask). The observed epoch-171 train-val Huber gap is `0.032251`, while the zero-prediction normalized-label Huber gap is `0.038198`. That target-distribution-only gap is larger than the observed model-loss gap, so the label variance difference is large enough in magnitude to explain why val loss is persistently lower. The finer table is slightly more nuanced: `4` of `6` validation sessions are below the train-session std minimum and `2` are within the train-session range; by day, `2` validation days are below the train-session minimum and `1` are within the train-session range. So the effect is not that every validation session/day is lower than every train session/day; rather, the aggregate train split is pulled up by a high-volatility train session/day that is absent from validation. This is a sample composition explanation, not evidence of leakage.

CSV output: `results/train_val_label_distribution_check.csv`.
