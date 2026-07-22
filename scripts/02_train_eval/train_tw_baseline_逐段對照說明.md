# `train_tw_baseline.py` 逐段對照說明

本文件依 `scripts/02_train_eval/train_tw_baseline.py` 的程式順序，逐段解釋：
- 這段在做什麼
- 為什麼要這樣寫
- 和前後段落的關係

---

## A. 檔頭與環境準備

### A1. 模組說明字串
- 開頭 docstring 說明：這是台灣期交所歷史重播資料的主訓練腳本，用於 execution-aware TCN baseline。

### A2. import 區塊
- 先載入標準庫（`argparse/json/os/sys/datetime/pathlib`）
- 再載入第三方（`pandas/torch/DataLoader`）
- `sys.path.insert(...)` 把專案根目錄加入 Python path，讓腳本可以直接 `from src... import ...`

### A3. 專案內部模組引入
- `src.data.tw_replay`：讀 replay 資料
- `src.features.*`：特徵工程
- `src.models.tcn`：模型建立
- `src.training.*`：資料集切分、訓練、推論、指標
- `src.backtest.*`：gating、回測、校準

---

## B. 常數與特徵集合定義

### B1. `DEFAULT_SYMBOLS`
- 預設商品：`TXFJ5`, `MXFJ5`, `CDFJ5`
- 若 CLI 沒給 `--symbols`，就使用這組。

### B2. 特徵欄位清單
- `BASE_MODEL_FEATURE_COLS`：基礎微結構特徵
- `STRATEGY_AWARE_FEATURE_COLS`：在 base 上加策略判斷常用特徵
- `EXECUTION_AWARE_FEATURE_COLS`：在 base 上加執行層特徵
- `FULL_MICROSTRUCTURE_FEATURE_COLS`：strategy + execution 全部合併

設計意圖：用 `--feature-set` 快速切換實驗配置，不需手動改程式。

---

## C. 參數解析函式

### C1. `parse_args()`
- 集中定義全部 CLI 參數，回傳 `Namespace`
- 內容涵蓋：
  - 資料路徑 / 商品 / 抽樣
  - 視窗長度 / horizon / label threshold
  - 訓練超參數（epochs, lr, weight decay, patience）
  - gating 參數
  - event-driven 回測參數
  - 校準搜尋網格參數
  - 輸出目錄

設計意圖：把整個 pipeline 做成可重現、可批次執行的命令列工具。

---

## D. 特徵工程核心函式

### D1. `build_tw_feature_frame(lob_df, trade_df, symbol, tick_size)`

這是資料前處理的核心，流程如下：

1. 先用 `build_feature_dataframe(...)` 產生基礎無量綱特徵  
2. 推導 `weighted_mid_deviation`  
3. 以 rolling std 產生 `rolling_volatility`  
4. 建立 `feature_df` 主表，保留時間戳、價格、深度一檔量價  
5. 額外計算：
   - `obi_change`
   - `near_depth_obi`（近端深度不平衡）
   - `book_slope_imbalance`（深度斜率差）
   - `queue_imbalance` 與其變化
6. 用 `align_trade_features_to_quotes(...)` 將成交特徵對齊到 quote 時間軸
7. 回傳最終可訓練特徵表

設計意圖：把「報價訊息 + 成交訊息」融合成同一時間軸的模型輸入。

---

## E. 輔助解析函式群

### E1. `compute_class_weights(dataset)`
- 計算三類別樣本數，轉成反比權重
- 避免某類樣本太多時，模型偏向該類。

### E2. `feature_columns_from_name(feature_set)`
- 根據字串回傳對應特徵欄位清單
- 若給未知名稱，直接 `raise ValueError`，避免默默用錯配置。

### E3. `parse_time_to_ns(text)`
- 把 `HH:MM:SS` 轉奈秒，供 event 回測時間窗過濾。

### E4. 文字轉型工具
- `parse_symbol_weight_map`
- `parse_float_list`
- `parse_int_list`
- `parse_text_list`

用途：把 CLI 的逗號字串轉成結構化資料。

---

## F. `main()` 主流程逐段

### F1. 初始化 run 與輸出目錄
- `args = parse_args()`
- 用目前時間建立 `run_name`
- 建立 `output_dir`
- 寫出 `pipeline_args.json`

目的：完整保留這次實驗配置。

### F2. 決定交易商品與權重
- 解析 `selected_symbols`
- `trade_symbols` 若沒指定就等於 `selected_symbols`
- 解析 `train_symbol_weights` 後，補齊沒有指定的 symbol 權重

目的：可分離「有資料訓練」與「允許下單交易」的商品集合。

### F3. 掃描資料可用商品
- `scan_replay_symbols(...)`
- 匯出 `symbol_scan.csv`

目的：先盤點 replay 裡到底有哪些 symbol，便於除錯。

### F4. 載入 LOB 與成交資料
- `load_tw_replay_as_lob(...)`
- `load_tw_replay_trades(...)`
- 若沒載到 LOB，直接報錯終止

目的：避免後續流程在空資料上執行。

### F5. 逐商品建特徵與摘要
- 每個 symbol：
  - 推斷 tick size
  - 取對應 trade tape
  - 呼叫 `build_tw_feature_frame(...)`
  - 收集資料摘要（筆數、時間範圍）
- 匯出 `data_summary.csv`

目的：每商品獨立處理，再組成 multi-symbol 訓練資料。

### F6. 建立時間序列資料集切分
- 依 `feature_set` 選特徵欄位
- 呼叫 `build_symbol_datasets(...)`
- 輸出 `split_sizes.csv`

目的：嚴守時間序切分，確保 val/test 發生在 train 之後。

### F7. DataLoader、模型與訓練設定
- 建立 train/val/test DataLoader
- `build_default_model(num_inputs=...)`
- `TrainerConfig(...)`（epochs/lr/device/patience）

目的：準備可訓練的 PyTorch pipeline。

### F8. 類別權重（可選）
- 若未停用 class weighting：
  - 計算 `class_weights`
  - 匯出 `class_weights.csv`

目的：降低分類不平衡造成的偏差。

### F9. 訓練模型
- `train_with_early_stopping(...)`
- 回傳訓練後模型與歷程 `history_df`

目的：在驗證集不再進步時提前停止，減少 overfitting。

### F10. 產生 val/test 預測
- `predict_model(...)` 各跑一次
- 得到 `val_predictions` 與 `predictions`（test）

### F11. 套用 gating（或略過）
- 若 `--disable-probability-gating`：
  - 直接把 `y_pred` 當 `gated_signal`
- 否則：
  - 用 `apply_probability_gating(...)`
  - 可加機率閾值、margin、spread/volatility 懲罰
  - 限制只在 `trade_symbols` 內交易
- 輸出 `val_predictions.csv`、`test_predictions.csv`

目的：把「分類結果」轉成更可交易的訊號。

### F12. 分類評估
- `classification_metrics(...)`
- 輸出 `test_metrics.json` 與 `confusion_matrix.csv`

目的：先看純 ML 分類品質。

### F13. 向量化訊號回測
- `run_signal_backtest(...)`
- 輸出 trade log / by_symbol / summary

目的：快速估計策略層面的報酬與風險。

### F14. 事件驅動回測
- `run_event_driven_backtest(...)`
- 使用更貼近真實的進場、等待、持有、平倉機制
- 輸出 event trade log / by_symbol / summary

目的：驗證訊號在更真實執行條件下是否仍有效。

### F15. 驗證集校準（可選）
- 條件：`--calibrate-symbol-rules-on-val`
- 在驗證集搜尋最佳參數組（symbol thresholds、end times、cooldowns、regime guards）
- 用最佳配置套用到測試集，再跑事件回測
- 儲存 `val_*` 與 `test_calibrated_from_val_*` 成果

目的：把「統一門檻」升級為「按商品差異化規則」。

### F16. 產生整體總結
- 建立 `run_summary`（訓練設定 + 分類結果 + 回測績效）
- 若有校準，再附加校準後績效
- 輸出 `run_summary.json`
- 最後 `print(json.dumps(run_summary, indent=2))`

目的：讓每次 run 有單一總覽入口。

---

## G. 程式結尾

### G1. 進入點
- `if __name__ == "__main__":`
  - `torch.manual_seed(42)`
  - `main()`

作用：固定隨機種子以提升可重現性，並啟動整體流程。

---

## H. 一句話理解整個腳本

這份程式是「**從重播資料生成微結構特徵 -> 訓練 TCN 三分類模型 -> 轉換為可交易訊號 -> 用兩種回測框架驗證 -> （可選）做商品級校準**」的完整研究/實驗 pipeline。
