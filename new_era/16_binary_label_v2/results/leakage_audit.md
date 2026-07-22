# Leakage Audit

Advisor 10-question checklist:

## 1. Are z-scores causal?
Yes. Past-only z-score uses `shift(1)` before rolling in `02_preprocess_quote_eligible.py:78`.

## 2. Are centered windows used?
No. The only normalization window is `shift(1).rolling(...)`, documented in `02_preprocess_quote_eligible.py:351`.

## 3. Are labels allowed to use future data?
Yes, labels use future data only as targets. `03_make_binary_labels.py` defines `future_mid_return_H_ticks` and fill proxies, not features.

## 4. Is train balancing isolated from val/test?
Yes. Split construction is explicit in `04_train_binary_model.py:104`, and downsampling only branches inside train data at `04_train_binary_model.py:387`.

## 5. Was threshold selection validation-only?
Yes. `05_select_thresholds.py:274-275` writes `selection_split='val'` and `test_used=False`.

## 6. Does threshold selection read test predictions?
No. `05_select_thresholds.py` reads only `val_predictions_bid.parquet` and `val_predictions_ask.parquet` from the model directory.

## 7. Does the backtest use frozen thresholds?
Yes. `06_backtest_binary_cancel.py:293` reads `selected_thresholds`, and line `295` rejects non-validation threshold files.

## 8. Is test evaluated once?
Yes. `06_backtest_binary_cancel.py:325` stamps `test_evaluation_count=1` into the summary.

## 9. Are raw data or 015 baselines overwritten?
No. All 016-v2 artifacts write under `new_era/16_binary_label_v2/outputs` or `results`.

## 10. Is aggressor-side ground truth used?
No. The pipeline explicitly uses a proxy because true `aggressor_side` and queue position are unavailable.
