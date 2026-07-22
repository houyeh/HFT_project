# HFT 專案：從模型訓練到硬體交付（簡報母稿）

副標題：給 ML 與硬體初學者的端到端實作說明（可直接貼到 Gamma / Canva）

---

## Slide 1｜專案目標與一句話摘要

### 要解決的問題

- 在高頻交易場景下，將台灣期交所重播資料訓練出的 TCN 模型，轉成可硬體部署（INT8/HLS）流程。

### 一句話摘要

- 我們先建立可驗證的 ML pipeline（資料 -> 特徵 -> 訓練 -> 回測），再做量化與 hardware handoff，讓模型從研究走向可落地部署。

### 視覺建議

- 放一張「左到右流程圖」：`Replay Data -> Features -> Train TCN -> Backtest -> INT8 Export -> HLS/FPGA`

---

## Slide 2｜為什麼要做這個專案

### 業務與工程動機

- **低延遲需求**：交易訊號產生越快越有價值。
- **可重現性需求**：每次模型更新要能穩定驗證。
- **可部署性需求**：不能只停在 Python，必須進入 C++/HLS/FPGA。

### 技術目標

- 建立 execution-aware 特徵與三分類模型（Down/Stationary/Up）。
- 建立可交易的訊號評估，而非只看分類準確率。
- 建立 FP32 -> INT8 -> HLS 的 golden 驗證鏈。

---

## Slide 3｜整體 Dataflow（建議重點圖）

### 端到端資料流

1. `replay1_TW.csv` 載入 quote/trade 資料  
2. 轉成 LOB + 成交對齊特徵  
3. 建立時間窗資料集（train/val/test）  
4. 訓練 TCN（FP32）  
5. 產生預測與分類指標  
6. 套用 gating 產生可交易訊號  
7. 跑 signal/eveㄇnt-driven 回測  
8. 匯出 INT8 權重與 HLS 測試向量  

### 圖表位置建議

- 此頁放「總覽 dataflow 圖」(全寬 16:9)。
- 每個方塊底下可標註對應腳本路徑（見附錄）。

---

## Slide 4｜資料來源與前處理邏輯

### 資料來源

- 委託簿快照（LOB）
- 逐筆成交（trade tape）

### 前處理設計原因

- **市場微結構訊號在短時間窗內變化快**，需同時觀察 quote 與 trade。
- 需用 `tick_size` 將特徵標準化，讓不同商品可比較。
- 採時間序資料切分，避免未來資訊洩漏。

### 對應腳本

- `scripts/02_train_eval/train_tw_baseline.py`

### 圖表位置建議

- 左側：資料來源示意（quote vs trade）
- 右側：前處理流程（clean -> align -> feature frame）

---

## Slide 5｜特徵工程：模型看到的是什麼

### 特徵族群（建議用表格）

- **Base**：`normalized_spread`, `micro_return`, `obi`, `weighted_mid_deviation`, `deep_obi`
- **Strategy-aware**：加上波動與深度結構（`rolling_volatility`, `obi_change`, `near_depth_obi`, `book_slope_imbalance`）
- **Execution-aware**：加上排隊與成交/撤單壓力（`queue_imbalance`, `rolling_signed_flow_ratio`, `cancel_pressure_short`...）

### 為什麼這樣設計

- Base 對應價格微結構的核心狀態。
- Strategy-aware 增加方向判斷穩定性。
- Execution-aware 增加可成交性與實際執行訊號。

### 圖表位置建議

- 放「特徵分群表」+ 一張「feature importance 或概念圖」（若暫無 importance，可用概念示意）。

---

## Slide 6｜標籤定義與任務設定

### 任務類型

- 三分類：`Down / Stationary / Up`

### 標籤形成

- 由 horizon 之後的價格變化與 threshold（tick-based）決定類別。

### 參數與含義

- `window_size`：模型一次看多長的歷史序列。
- `horizon`：預測多遠的未來。
- `threshold_ticks`：把微小波動歸為盤整，降低雜訊標籤。

### 為什麼重要

- 標籤門檻太小：噪音多、訊號不穩。
- 標籤門檻太大：樣本變少、學習困難。

---

## Slide 7｜模型架構：為什麼選 TCN

### TCN 優勢

- 適合序列資料，支援 dilation 擴大感受野。
- 相較 RNN，平行化友善，較適合後續硬體化。
- 結構規律，有利固定 shape 的 HLS 設計。

### 本專案固定規格（硬體友善）

- Inputs: `12`
- Sequence length: `100`
- Channels: `16`
- Kernel size: `3`
- Classes: `3`

### 圖表位置建議

- 放「TCN block 結構圖」（Conv1 -> Conv2 -> Residual，加 dilation 1/2/4）。

---

## Slide 8｜訓練流程（Training Loop）

### 流程

1. 準備 `DataLoader(train/val/test)`  
2. 建立模型與 `TrainerConfig`  
3. 訓練 + early stopping  
4. 在 val/test 產生預測  
5. 儲存模型歷程與指標  

### 關鍵訓練參數（可做一張表）

- `epochs`（預設 6）
- `batch_size`（預設 256）
- `learning_rate`（預設 1e-3）
- `weight_decay`（預設 1e-4）
- `patience`（預設 2）
- `train_frac`（0.7）, `val_frac`（0.15）

### 為什麼這樣設計

- 先用保守參數快速收斂與驗證 pipeline 可用性。
- early stopping 防止過擬合，縮短迭代時間。

### 圖表位置建議

- 放「train/val loss 曲線圖」。
- 資料來源：每次 run 的 training history 輸出（從 run artifacts 抽圖）。

---

## Slide 9｜類別不平衡與權重設計

### 問題

- 金融時間序列常出現類別不平衡（Stationary 比例高）。

### 解法

- 啟用 class weighting（可關閉）
- 額外支援 symbol 權重、non-stationary 權重、return 權重

### 參數

- `--disable-class-weighting`
- `--train-symbol-weights`
- `--train-nonstationary-weight`
- `--train-return-weight-alpha`

### 圖表位置建議

- 放「label 分佈長條圖（train/val/test）」。
- 放「class_weights.csv 的可視化」。

---

## Slide 10｜模型評估：ML 指標 vs 交易指標

### ML 指標

- Accuracy
- Macro-F1
- Confusion Matrix

### 交易指標

- Net PnL (ticks)
- Sharpe-like
- 按商品拆分績效（by-symbol）

### 核心觀念

- 分類好不等於可交易；必須進入回測層驗證成本與執行條件。

### 圖表位置建議

- 左：confusion matrix heatmap
- 右：回測累積 PnL 曲線

---

## Slide 11｜Probability Gating：把分類變成可交易訊號

### 做法

- 對預測機率設門檻與 margin，過濾低信心訊號。
- 可對 spread/volatility 加懲罰，減少高風險交易。

### 參數

- `--gating-prob-threshold`
- `--gating-margin-threshold`
- `--gating-spread-penalty`
- `--gating-volatility-penalty`
- `--disable-probability-gating`

### 設計原因

- 把「分類器」變成「交易決策器」。
- 控制過度交易與噪音訊號。

### 圖表位置建議

- 放「gating 前後交易次數與 PnL 對比條圖」。

---

## Slide 12｜雙回測框架：Signal vs Event-Driven

### Signal Backtest（向量化）

- 快速、便於大量參數掃描。

### Event-Driven Backtest（貼近真實）

- 模擬市價/限價、等待時間、持有時間、cooldown、部位大小。

### 為什麼要兩種

- Signal：快，適合研究早期。
- Event-driven：真，適合部署前驗證。

### 圖表位置建議

- 放「兩種回測結果對照表」：PnL、Sharpe-like、交易次數、勝率。

---

## Slide 13｜校準（Calibration）與商品差異化規則

### 核心想法

- 不同商品微結構不同，用同一組門檻不一定最佳。

### 做法

- 在 validation 上搜尋：
  - 商品別概率/邊際門檻
  - 結束交易時間
  - cooldown
  - regime guard（波動、交易強度、可見流動性、撤單壓力）

### 參數

- `--calibrate-symbol-rules-on-val`
- `--calibration-prob-offsets`
- `--calibration-margin-thresholds`
- `--calibration-end-times`
- `--calibration-cooldowns`

### 圖表位置建議

- 放「calibration 前後績效對照圖（test）」。

---

## Slide 14｜Hardware Handoff：從 FP32 到 INT8 交付資產

### 交付內容

- `tcn_int8_weights.wts`
- `tensor_manifest.json`
- `reference_input.txt`
- `reference_output.txt`
- `fixed_tcn_weights.hpp`
- `fixed_tcn_test_vectors.hpp`

### 為什麼重要

- 讓 C++ / HLS / testbench 共用同一份權重與 golden。
- 降低跨團隊對接成本與 debug 歧義。

### 圖表位置建議

- 放「artifact 關係圖」：誰產生、誰使用、驗證在哪一層。

---

## Slide 15｜hls4ml 嘗試結果與工程決策

### 目前遇到的障礙

- 原始模型在轉換時卡 `permute`
- channel-first 版本卡 `pad`

### 工程解讀

- 問題不在模型大小，而在 graph 的工具可轉換性。

### 主線決策

- 保留 hls4ml 嘗試
- 主線採 fixed-shape manual C++/HLS rewrite

### 圖表位置建議

- 放「決策樹」：Auto-convert 可行？否 -> 手刻路線。

---

## Slide 16｜延遲估算與硬體價值

### 目前估算（範例）

- PyTorch CPU p50：`0.4465 ms`
- PyTorch CPU p99：`1.7499 ms`
- C++ INT8 估算：`57.6 us`
- FPGA 估算：`9.2 us`

### 正確解讀方式

- 這不是最終上板值（尚缺 post-route/board measurement）
- 但可支持「是否值得硬體化」的早期決策

### 圖表位置建議

- 放「Latency 梯度條圖（ms -> us）」。

---

## Slide 17｜風險、限制與下一步

### 目前限制

- hls4ml 對特定 graph 操作支援有限
- 估算值仍需板上驗證
- 回測與實盤仍有 domain gap

### 下一步

1. 固化 HLS testbench regression  
2. 推進 kernel 化與 pragma 優化  
3. 做 post-synth / post-route timing  
4. 板上量測端到端 latency（含 I/O）  

---

## Slide 18｜結論（可做收尾頁）

### 三個關鍵結論

1. 這個專案已建立從 ML 訓練到硬體交付的完整可驗證鏈。  
2. 真正瓶頸在「硬體友善的模型圖與資料流」，不只在模型大小。  
3. 以目前結果看，INT8 + HLS 路線具備明顯延遲潛力，值得持續投入。  

---

## 附錄 A｜報告中建議必放圖表清單（含來源）

1. **端到端 Dataflow 圖**  
   - 來源：手動畫（建議）+ 腳本路徑標註

2. **資料切分與樣本量表**  
   - 來源：`split_sizes.csv`（每次 run output）

3. **Label 分佈圖**  
   - 來源：train/val/test dataset 統計（可由 predictions 或 dataset 匯出）

4. **訓練曲線（train/val loss）**  
   - 來源：training history 輸出

5. **Confusion Matrix**  
   - 來源：`confusion_matrix.csv`

6. **回測績效圖（signal vs event-driven）**  
   - 來源：`backtest_summary.json`, `event_backtest_summary.json`, trade log

7. **Gating 前後對照圖**  
   - 來源：`test_predictions.csv` + 回測摘要

8. **Calibration 前後對照圖**  
   - 來源：`test_calibrated_from_val_*` artifact

9. **Latency 對照條圖**  
   - 來源：`latency_estimate.json` + CPU benchmark

10. **Hardware artifact 關係圖**  

- 來源：`export_tcn_int8_artifacts.py` 輸出檔關係

---

## 附錄 B｜主要腳本對照（簡報可直接貼）

- 訓練主腳本：`scripts/02_train_eval/train_tw_baseline.py`
- 硬體交付匯出：`scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`
- hls4ml 轉換嘗試：`scripts/05_hardware_handoff/attempt_hls4ml_conversion.py`

---

## 附錄 C｜給 Gamma / Canva 的貼上建議

### 貼法

- 直接以本 MD 每個 `Slide X` 區塊作為一頁。
- 標題維持短句，內文 3-5 個 bullet 即可。
- 每頁最多 1 張主圖，避免資訊過載。

### 版型建議

- 技術頁：左文右圖
- 結果頁：上結論下圖表
- Dataflow 頁：全寬流程圖 + 底部註解

### 字級建議

- 標題 32-40
- 內文 20-24
- 圖表註解 14-16

---

## 一句話收尾（可放最後一頁）

這份專案的價值，不只是訓練出一個模型，而是把模型變成「可量測、可驗證、可部署」的工程系統。
