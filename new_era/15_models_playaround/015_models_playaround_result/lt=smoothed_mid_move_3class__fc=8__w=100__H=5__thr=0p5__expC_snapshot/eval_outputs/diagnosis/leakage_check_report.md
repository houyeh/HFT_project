# Data Leakage & Label Risk Check Report

| 檢查項目 | 程式碼位置 | 判斷 | 原因 | 建議修改 |
| :--- | :--- | :--- | :--- | :--- |
| 1. train/val/test 是否嚴格按時間切分？ | `tcn_pipeline_utils.py` 的 `split_ordered_session_ids` | **PASS** | 程式碼依照順序將 session ids 切分，保證了訓練集在測試集之前，沒有使用 random split。 | 無需修改。 |
| 2. normalization / scaler 是否只用 train fit？ | `tcn_pipeline_utils.py` 的 `apply_rolling_zscore` | **PASS** | 使用 rolling window 計算移動平均與標準差進行 Z-score，不依賴全局（或包含未來）的 mean/std，避免了傳統 scaler 的 leakage。 | 無需修改。 |
| 3. rolling features 是否只使用過去資料？ | `tcn_pipeline_utils.py` 的 `apply_rolling_zscore` | **PASS** | pandas 的 `rolling()` 預設為 backward-looking，僅包含當前與過去時間步的資料。 | 無需修改。 |
| 4. smoothed_mid_move_3class label 是否使用未來資料？ | `src/data/tw_train_helpers.py` 等 | **PASS (合理)** | Label 本身就是對未來 `H` 步的價格變化分類，包含未來資料是正常的。關鍵是特徵 (features) 沒有使用到計算 Label 的未來價格，這點已確認。 | 無需修改。 |
| 5. feature window 和 label horizon 是否有重疊？ | `src/training/datasets.py` 的 `WindowedLOBSequenceDataset` | **PASS** | 序列擷取時，`x` 取 `[i - window : i]`，而 `y` 的計算對應未來的視窗，兩者在時間軸上完美區隔。 | 無需修改。 |
| 6. 是否同一天資料被切進 train 和 test？ | `split_ordered_session_ids` | **WARNING** | 如果依照 session 數量做 70/15/15 切分，可能剛好切在某一天中間。這會造成上午是 train，下午是 test，存在輕微的 regime leakage 風險。 | 建議改為以 `date` 進行切分，確保 test set 的日子完全未在 train 出現過。 |
| 7. test set 是否與 model selection 重複使用？ | `15_models_playaround/...` | **WARNING** | 雖然原本腳本有使用 val 挑選 strategy，但我們目前在 test set 上進行了廣泛的 threshold sweep 分析。如果根據這個分析回頭挑選最佳 threshold，就會對 test set overfit。 | 確保此報告僅用於「診斷模型」，最終上線參數應以 Validation set 決定，Test set 只看一次。 |
