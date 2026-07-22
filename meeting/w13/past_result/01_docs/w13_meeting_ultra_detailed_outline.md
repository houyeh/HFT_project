# Week 13 Meeting 簡報大綱 (Ultra-Detailed Standard)

這是一份嚴格、量化、具備高度視覺化要求的簡報大綱指引。未來的內容生成 (Agent) 必須**嚴格遵守**此處列出的具體數據、表格欄位、與圖表規範，絕不允許含糊其辭。所有結論必須建立在具體數值之上。

---

## 1. 嘗試模型與定調 (Model Experimentation & Identity)

### 1.1 Dataset, 參數, Feature 與 Label 詳細清單
**生成要求**：不可只說「用了各種參數」，必須給出具體規格表。
*   **Dataset 規格**：
    *   **來源檔案**：`C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW` (主力訓練)、後續驗證使用 `NEW1`, `NEW2`, `NEW3` (涵蓋高低波動與趨勢等不同 Regime)。
    *   **商品與資料量**：台指期 (TXFB6) LOB 數據。Test set 獨立樣本數高達 `233,268` 筆，Validation set 為 `112,698` 筆。
    *   **切分方法 (Data Splitting)**：嚴格依照 Session 時間順序進行 `70/15/15` 切分 (`train_session_frac: 0.7`, `val_session_frac: 0.15`)，保證零未來數據洩漏 (Zero Look-ahead bias)。
*   **模型與特徵參數**：
    *   **Features**：`feature_count = 8` (Z-score 滾動標準化，`zscore_window = 2000`)。包含微觀 LOB 的訂單簿失衡、價差等。
    *   **Window & Horizon**：`window_size = 100` ticks (擷取歷史特徵), `horizon = 5` ticks (標籤預測視距)。
*   **Label 定義**：
    *   `smoothed_mid_move_3class`：未來 5 步的 Mid price 變化，分為 Down(0)、Stationary(1)、Up(2)。判定閾值為 `threshold_ticks = 0.5`。

### 1.2 訓練結果與收斂 (Training Results)
**生成要求**：必須展示雙子圖，並標註極值。
*   **圖表要求**：展示 `train_history_plot.png`。
    *   左圖：Training vs Validation Loss 隨 Epoch 下降曲線。必須標出 Min Val Loss 的具體 Epoch 與數值 (如 0.8704)。
    *   右圖：Macro F1 曲線。必須標出 Max Val F1 (約 0.29)。
*   **論述重點**：證明模型有穩定收斂，無嚴重 Overfitting。

### 1.3 分類指標深度剖析 (Confusion Matrix & Decision)
**生成要求**：不僅要放圖，還要放表格，並解釋 Precision 與 Recall 的非對稱性。
*   **圖表要求**：放入 `confusion_matrix_percent_test.png`。
*   **數據表格要求**：列出 classification report 中的關鍵 metrics：
    *   Down: Precision ~2.8%, Recall ~69.4%
    *   Stationary: Precision ~98.7%, Recall ~45.2%
    *   Up: Precision ~2.9%, Recall ~64.2%
*   **專業定調 (The Verdict)**：
    *   由於市場 Stationary 佔比壓倒性多數，模型預測 Up/Down 的 Precision 極低。
    *   **不作為主動交易訊號**，轉而利用其高 Recall 特性，將模型定位為 **「高敏感度撤單過濾器 (High-Sensitivity Cancel Filter)」**。

---

## 2. 系統架構與市場狀態 (Flow Architecture & Regime)

### 2.1 完整系統 Flow (System Architecture)
**生成要求**：文字必須呈現具備層次感的數據流向。
*   **Flow Chart 描述**：`LOB Data ➔ 8-Feature Z-score ➔ FPGA-friendly TCN (Micro Cancel Filter) ➔ Macro Regime Guard (Trend Filter) ➔ Maker-only Strategy`。

### 2.2 為何需要 Regime Guard？ (The Why)
**生成要求**：結合量化金融理論與模型限制進行專業解讀。
*   **核心痛點**：TCN 的 Window 只有 100 Ticks，專精於捕捉微觀結構的 Adverse Selection (毒性訂單)。但在單邊趨勢強烈的市場 (Trending Market) 中，被動造市會累積大量單邊庫存 (Inventory Risk)。若沒有宏觀保護，造市商將成為趨勢的犧牲品。
*   **不同 Dataset 的 Regime 取向**：
    *   `NEW1`, `NEW2`, `NEW3` 分別代表不同波動度與趨勢強度的市場狀態 (Regime)。我們需要確保策略在所有 Regime 下生存。

### 2.3 Regime 的判斷方式 (The How)
**生成要求**：具體說明指標與機制。
*   **實作方法**：透過計算長週期的指數移動平均 (EWMA) 或價格動量 (Momentum) 判斷趨勢方向。
*   **Gating 機制**：當 Regime 判定為強烈上漲時，完全禁止 Ask (賣出) 側的掛單；反之亦然。這被稱為 Macro Trend Gate。

---

## 3. 交易策略與深度回測驗證 (Trading Strategy & Backtest Rigor)

### 3.1 嚴謹的回測腳本與環境 (Backtest Environment)
**生成要求**：列出清晰的假設條件與限制。
*   **交易邏輯**：Maker-only (被動造市)。預設掛單在最佳 Bid/Ask，當 TCN 輸出 `P(Down) > Threshold`，立刻撤銷 Bid；`P(Up) > Threshold` 立刻撤銷 Ask。
*   **考慮到的環境因素 (Conservative Assumptions)**：
    *   精準的手續費計算 (`maker_fee = 15 TWD`, `taker_fee = 50 TWD` 用於強制平倉)。
    *   `max_inventory = 3`，嚴格的庫存風險控管。
    *   **FIFO Realized Side PnL**：不使用簡單的買賣流水帳，而是嚴謹地對每一筆一買一賣 (Round-trip) 進行 FIFO 搓合，分別算出真正的 Bid-side 與 Ask-side 淨利 (`realized_side_pnl_from_trade_log.csv`)。
*   **未考慮的因素 (Limitations)**：
    *   無交易所真實 Queue Position (排隊位置) 模擬，觸價即視為成交，這可能對 Fill Rate 過度樂觀。

### 3.2 數據說話：Threshold Sweep 與 Baseline 對比
**生成要求**：必須包含嚴謹的 CSV 數據對比表。
*   **最佳參數選擇表 (`threshold_sweep_enhanced.csv`)**：
    *   列出 Threshold = `0.55` 與 `0.60` 時的 Net PnL, Signal Rate, Trade Count, Max Drawdown。
*   **Baseline Edge 證明表 (`baseline_comparison.csv`)**：
    *   加入 `No-Model Baseline` 以及 `Random Cancel Baseline` (控制 Signal Rate 相同的情況下隨機撤單，跑 20 個 Seed 取平均)。
    *   **專業結論**：Model 撤單避開的 Toxic Fills 與 Net PnL **顯著擊敗 Random Cancel**，證明訊號具備真實的 Alpha (Edge)，而非單純減少掛單導致。

### 3.3 機會成本深度分析 (Opportunity Cost Analysis)
**生成要求**：量化 False Positive 的代價。
*   **分析表格 (`missed_opportunity_diagnostics.csv`)**：
    *   **Missed Profitable PnL** (錯失的潛在獲利) vs **Missed Toxic PnL Avoided** (成功避開的虧損)。
    *   定義 `Net Cancel Value = Avoided Loss - Missed Profit`。
    *   **專業結論**：Net Cancel Value 為巨大的正值，證明「寧可錯殺一百」在造市防護中是完全正確且高度獲利的策略。

### 3.4 硬體延遲敏感度 (Latency Sensitivity - Bonus for FPGA)
**生成要求**：圖表化展示延遲對 PnL 的破壞力。
*   **實驗設計**：人工在回測中加入 `Delay = 0, 1, 2, 3, 5, 10` Ticks。
*   **專業結論 (`latency_sensitivity.csv`)**：當延遲超過 3 個 Ticks 時，策略淨利斷崖式崩盤。這不僅是風險，更是本專題採用 FPGA 進行超低延遲加速的最強合理性證明 (Perfect Justification)！

---
**💡 給後續 Agent 的執行指令：**
在生成最終簡報內容時，必須去 `c:\Users\holbo\00VS_projjects\00_HFT_project\new_era\15_models_playaround\015_models_playaround_result\lt=...\eval_outputs\diagnosis\` 讀取對應的 `.csv` 與 `.png`，並將真實數據填入上述的大綱中，不可憑空捏造數值。
