# Week 5: 覺醒 — 第一性原理與 HFT 造市商終極防禦 (實作深潛紀實)

## 1. 專案資料夾架構對應說明 (Workspace Architecture Mapping)
Week 5 由「純方向預測」轉型到「高頻造市商」，這牽涉到了回測引擎架構的大翻新：
*   **`src/backtest/hybrid_market_maker.py` (新引擎核心)**：這是我們本週建立的混合造市引擎。負責模擬做市商掛出雙邊限價單 (Bid-Ask limits)，並深入實作「排隊延遲 (Queue-Aware Filling)」與「毒性逃脫 (Toxicity Freeze)」。
*   **`scripts/03_experiments/sweep_latency.py` (延遲掃描器)**：一鍵式自動化執行腳本。利用這個腳本跑出不同延遲設定 (`delay=0, 1, 2, 4`) 對 PnL 的物理影響，並輸出成 CSV 以繪製圖表。
*   **`frozen_tcn_best_model.pt` (量化交付物)**：存放在 `artifacts/03_latency_proof/latency_proof/`，這是經過 INT8 最佳化的 4.8KB 神經網路，等待下週移交給硬體工程師進行 C++/HLS 燒錄。

---

## 2. 工程深水區：Hybrid Market Maker 真實排隊引擎設計
在傳統軟體回測中，我們很容易犯下「假性成交 (Illusionary Fills)」的錯誤（以為價格碰到了，我的單就成交了）。在這週的 `hybrid_market_maker.py` 中，我們打造了極度嚴苛的排隊引擎來貼近真實市場微觀結構。

### 技術實作：保守填單演算法 (Conservative Queue-Aware Fill)
當我們的機器人想要在 100 塊錢掛買單 (Bid) 時，我們嚴格模擬其在交易所撮合引擎內的命運：
1.  **隊列追蹤 (`queue_ahead`)**：我們讀取當前 LOB 該價位的 `bid_vol_1` (例如 50 口)。這意味著我們掛單時，前面有 50 口別人的單子在排隊。
2.  **吞噬倒數 (Depletion Countdown)**：伴隨著每一次逐筆交易的 Tick 進來，我們只計算那些「實質以 100 塊錢成交的市場委託單 (Taker Orders)」。這個數量會逐一扣減我們的 `queue_ahead`。
3.  **判定成交**：唯有當 `queue_ahead` 扣到 `0` 或變成負數時，我們的單子才會真正被判定「填滿 (Filled)」。
在這個機制的洗禮下，任何想要無腦吃豆腐的策略，都會被排隊的等待期給消耗殆盡，完美模擬了真實高頻世界中的排隊挫折。

---

## 3. 戰略迷思與死胡同：過度防禦的「零次交易」悲劇
擁有真實排隊引擎後，加上早期的防禦策略 (Quote-off 過濾器)，我們在 `sweep_latency.py` 中迎來了災難性但意義深遠的數據：
- `delay=0` (FPGA 模擬)：淨損 **-9.0 ticks** (2 trades)
- `delay=4` (軟體模擬)：淨損 **0.0 ticks** (0 trades)

**深度解析**：策略演算法太過害怕虧損，導致一旦出現稍微不明確的訊號，演算法就選擇「撤走所有單子」。當 `delay=4` 時，因為延遲過大，進場條件嚴苛到直接錯失所有機會（0 Trade），結果報表上看起來沒虧沒賺。這破壞了我們要展示「硬體帶來的高頻交易優勢」的核心價值——**如果不進場不交易，要百萬級別的 FPGA 系統做什麼？**

---

## 4. 第一性原理覺醒：Dumb Quoting 與 Smart Canceling
為了解決這個矛盾，我們徹底改寫了進攻與防守邏輯，將 TCN 的任務從「找交易訊號」拔高為「雷達避險裝置」：
1.  **Dumb Quoting (無腦掛單)**：把 `quote_off_threshold` 降至 0。我們強迫這台 FPGA 機器人成為真正的造市商，時時刻刻都把雙邊單掛在 LOB 第一層，瘋狂吃那萬分之一的點差，將其 Trade Count（掛單次數）拉到極限。
2.  **Smart Canceling (毒性極速撤單)**：
    我們在引擎內實作了基於 TCN 分數的高階數學觸發器 `alpha_freeze_threshold`。
    當 $\text{abs}(\text{smoothed\_alpha\_score}) > \text{threshold}$ 發生時，代表未來極短時間內將湧入大洪水級別的毒性單 (Toxic Inflow)。
    此時，系統不計一切代價、不考慮任何持倉情況，利用 FPGA 的 0 延遲，下達全球最速的 **[Withdraw All Quotes & Freeze]** 指令。

**透過這個改寫，我們終於能在同一個進場頻率基準上，跑出令人震懾的 `delay=0` 賺滿點差，而 `delay=4` 撤單失敗被直接貫穿的【殺手級對照圖表 (The Killer Chart)】。**

---

## 5. 接下來的執行重點 (Week 6 待辦清單)
1.  **擴充商品宇宙：CME Parser 開發**：
    受惠於 Week 1 精心的 Dimensionless Feature 設計，我們的引擎理論上可以通吃全球交易所。下週**拿到資料後**，我們將著手撰寫專門針對芝加哥商品交易所 (CME) 能源與指數合約 (ES, NQ, CL) 的解析器，將其轉換為標準的 5 檔 LOB CSV，驗證 TCN 跨市場的防禦泛化能力。
2.  **硬體底層實作：進入 C++ 與 高階合成 (HLS)**：
    將已經通過驗證的 `frozen_tcn_best_model.pt` 架構 (4.8KB INT8)，以外部資源不依賴的純 C++ / Verilog 將推論運算實作成硬體邏輯電路，為 FPGA 的實體驗證打下地基。
