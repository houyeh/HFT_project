# `train_tw_baseline.py` 程式說明

本文件解釋 `scripts/02_train_eval/train_tw_baseline.py` 的：
- 各指令（命令列參數與主要函式）
- 指令來源（標準庫 / 第三方套件 / 專案內模組）
- 程式撰寫邏輯（由資料到訓練再到回測）

---

## 1) 程式用途（一句話版）

此腳本用於訓練台灣期交所歷史重播資料的 TCN 基線模型，並輸出分類指標、訊號回測、事件驅動回測，以及（可選）驗證集校準結果。

---

## 2) Import 與來源

## 2.1 Python 標準庫

- `argparse`：解析命令列參數（`--epochs`, `--feature-set` 等）
- `json`：儲存設定與評估結果（如 `run_summary.json`）
- `os`, `sys`：調整模組路徑，讓腳本可匯入 `src/...`
- `datetime`：建立執行時間戳（run_name）
- `pathlib.Path`：跨平台路徑與資料夾建立

## 2.2 第三方套件

- `pandas as pd`：資料表操作、CSV 讀寫
- `torch`：深度學習框架
- `torch.utils.data.DataLoader`：批次載入資料

## 2.3 專案內模組（`src/...`）

- `src.data.tw_replay`：重播資料讀取、symbol 掃描、tick size 推斷
  - `scan_replay_symbols`, `load_tw_replay_as_lob`, `load_tw_replay_trades`, `infer_tick_size_from_lob`
- `src.features.*`：特徵工程
  - `build_feature_dataframe`, `compute_queue_imbalance`, `align_trade_features_to_quotes`
- `src.training.*`：資料集切分、訓練、預測、指標
  - `build_symbol_datasets`, `TrainerConfig`, `train_with_early_stopping`, `predict_model`, `classification_metrics`
- `src.models.tcn`：建立 TCN 模型 (`build_default_model`)
- `src.backtest.*`：訊號回測、事件驅動回測、機率閘門與校準
  - `run_signal_backtest`, `run_event_driven_backtest`, `apply_probability_gating`
  - `fit_symbol_rule_configuration`, `save_symbol_calibration_artifacts`

---

## 3) 各主要函式（可視為程式內指令）

## `parse_args()`

建立所有命令列參數，回傳 `argparse.Namespace`。

## `build_tw_feature_frame(lob_df, trade_df, symbol, tick_size)`

把 LOB 與成交資料整合成訓練特徵表，包含：
- 基礎微結構特徵（spread、micro_return、obi）
- 策略感知特徵（rolling volatility、obi 變化、深度斜率）
- 執行感知特徵（queue imbalance、成交流向、cancel pressure）

## `compute_class_weights(dataset)`

統計三分類標籤數量（Down/Stationary/Up），計算反比權重，降低類別不平衡影響。

## `feature_columns_from_name(feature_set)`

依字串選擇特徵集合：
- `base`
- `strategy_aware`
- `execution_aware`
- `full_microstructure`

## `parse_time_to_ns(text)`

把 `HH:MM:SS` 轉成奈秒 timestamp，給事件回測做交易時段過濾。

## `parse_symbol_weight_map / parse_float_list / parse_int_list / parse_text_list`

把命令列字串轉成對應型別，用於權重設定與校準搜尋網格。

## `main()`

主流程：讀參數 -> 讀資料 -> 建特徵 -> 建資料集 -> 訓練 -> 預測 -> 回測 -> 校準(可選) -> 輸出報告。

---

## 4) 命令列參數（CLI 指令）總覽

以下為常用分組（完整參數以程式為準）：

## 4.1 資料與取樣

- `--replay-path`：重播 CSV 檔
- `--symbols`：要處理的商品（逗號分隔）
- `--trade-symbols`：允許交易的商品子集
- `--depth`：LOB 深度層數
- `--quote-stride`：報價抽樣步長
- `--sample-stride`：訓練樣本抽樣步長
- `--window-size`：模型輸入視窗長度
- `--horizon`：標籤預測期

## 4.2 特徵與標籤門檻

- `--feature-set`：特徵組合名稱（base/strategy_aware/execution_aware/full_microstructure）
- `--threshold-ticks`：分類門檻（tick）
- `--threshold-mode`：門檻模式（如 static）
- `--threshold-spread-multiplier`
- `--threshold-volatility-multiplier`

## 4.3 訓練設定

- `--train-frac`, `--val-frac`：時間序列切分比例
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--patience`
- `--disable-class-weighting`：停用 class weighting
- `--train-symbol-weights`：指定商品權重（格式 `SYM:1.2,SYM2:0.8`）
- `--non-trade-symbol-weight`：非交易商品訓練權重
- `--train-nonstationary-weight`：非盤整樣本權重
- `--train-return-weight-alpha`：報酬權重混合係數

## 4.4 訊號閘門（Probability Gating）

- `--disable-probability-gating`
- `--gating-prob-threshold`
- `--gating-margin-threshold`
- `--gating-spread-penalty`
- `--gating-volatility-penalty`

## 4.5 事件驅動回測

- `--event-entry-mode`：進場模式（market/passive）
- `--event-spread-scale`
- `--event-max-holding-steps`
- `--event-passive-confidence-threshold`
- `--event-max-passive-wait-steps`
- `--event-cooldown-steps`
- `--event-size-mode`, `--event-max-position-size`
- `--event-confidence-floor`, `--event-confidence-ceiling`
- `--event-start-time`, `--event-end-time`

## 4.6 驗證集校準（可選）

- `--calibrate-symbol-rules-on-val`：啟用校準
- `--calibration-prob-offsets`
- `--calibration-margin-thresholds`
- `--calibration-end-times`
- `--calibration-cooldowns`
- `--calibration-vol-quantiles`
- `--calibration-trade-quantiles`
- `--calibration-cancel-quantiles`
- `--calibration-liq-quantiles`

## 4.7 輸出

- `--output-dir`：本次 run 的根輸出資料夾

---

## 5) 程式撰寫邏輯（流程圖式）

1. **初始化與記錄設定**
   - 解析參數、建立 `run_name`、輸出 `pipeline_args.json`

2. **資料盤點與載入**
   - 掃描 replay 內可用商品（`symbol_scan.csv`）
   - 載入 LOB 與 trade tape

3. **特徵工程**
   - 每個商品推斷 tick size
   - 產生基礎 + 策略感知 + 執行感知特徵
   - 匯出 `data_summary.csv`

4. **資料集切分**
   - 依時間切 train/val/test（避免未來資料洩漏）
   - 產生 `split_sizes.csv`

5. **模型訓練**
   - 建立 TCN 模型
   - （可選）計算 class weights
   - Early stopping 訓練並保存歷程

6. **推論與分類評估**
   - 產生 val/test 預測
   - 套用或不套用 probability gating
   - 匯出 `test_metrics.json`、`confusion_matrix.csv`

7. **交易層評估（回測）**
   - 訊號回測（向量化）
   - 事件驅動回測（較貼近真實進出場機制）

8. **可選校準**
   - 在驗證集搜尋每商品最佳規則
   - 把規則套到測試集，輸出校準後績效

9. **最終彙總**
   - 寫出 `run_summary.json`，並 `print` 到終端

---

## 6) 你實際會看到的輸出檔（每次 run）

在 `output-dir/<timestamp>/` 內，常見包含：
- `pipeline_args.json`
- `symbol_scan.csv`, `data_summary.csv`, `split_sizes.csv`
- `class_weights.csv`（若啟用）
- `val_predictions.csv`, `test_predictions.csv`
- `test_metrics.json`, `confusion_matrix.csv`
- `backtest_trade_log.csv`, `backtest_by_symbol.csv`, `backtest_summary.json`
- `event_backtest_trade_log.csv`, `event_backtest_by_symbol.csv`, `event_backtest_summary.json`
- `run_summary.json`
- 若啟用校準：`val_*` 與 `test_calibrated_from_val_*` 系列檔案

---

## 7) 設計重點（為何這樣寫）

- **訓練與交易邏輯分離**：先做分類品質，再檢驗交易可行性
- **時間序切分**：避免 look-ahead bias
- **多層級特徵集**：可從 base 漸進到 execution-aware
- **可控風險機制**：gating 與 event-driven 規則讓訊號更可交易
- **校準機制**：允許針對不同商品做門檻與時段微調

如果你要，我可以再幫你補一份「參數調整建議表」（例如想提高 Sharpe-like / 降低過度交易時，優先調哪些參數）。
