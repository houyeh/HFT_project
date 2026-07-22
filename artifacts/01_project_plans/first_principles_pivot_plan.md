# HFT 第一性原理轉向與 Meeting 準備計畫

## 核心目標 (Goal)
將專案從「追求精準軟體預測模型」死胡同中解放出來，回歸「高頻造市商的第一性原理」：**預設雙邊報價吃價差 (Dumb Quoting)，依賴 TCN 作為毒性警報器觸發極速撤單 (Smart Canceling)。** 
此轉向能真正把 FPGA **低延遲 (Latency=0)** 的商業價值給數學化地證明出來。同時需要將這段「彎路到覺醒」的歷程，梳理成教授 Meeting 的簡報大綱與初學者教學文件。

---

## 執行方案 (Proposed Steps)

### 1. 策略引擎重構 (Engine Refactoring)
*   **檔案**：`src/backtest/hybrid_market_maker.py`
*   **修改邏輯**：
    *   移除或大幅放寬 `quote_off_score_threshold`，讓 Bot 不再追求「有把握才作」，而是「沒危險就無腦掛單賺價差」。
    *   將 TCN 出來的 `alpha_score` (prob_up - prob_down) 主要與 `toxicity_freeze_threshold` 對接。當分數極度偏轉時（預示大單來砸），啟動 Withdraw All Quotes。

### 2. 教學與專案紀錄 (Documentation Updates)
*   **檔案**：`week5.md` (與 `week4.md` 的尾聲更新)
*   **內容**：記錄 Week 4 迷失於「方向性預測 PnL」與「減少交易避開危險」的過程。再帶出 Week 5 的頓悟：HFT 需要 Turnover，防禦靠速度而非躲避。
*   **檔案**：`teaching/16_HFT_First_Principles.md`
*   **內容**：給學弟妹的白話文教學。解釋「為什麼延遲為 0 不見得賺錢，如果你的模型本來就錯了」、「造市商賺死人的錢，躲聰明人的單」。

### 3. 指導教授 Meeting 大綱與圖表規劃 (Meeting Prep)
*   **檔案**：`meeting/01_Professor_Update_First_Principles.md`
*   **內容**：簡報的骨架 (Storyline)。從當前困境、數據發現、第一性原理頓悟、到新 Benchmark 的設計。
*   **檔案**：`charts/README.md`
*   **內容**：列出簡報中需要放上的圖表 (例如：軟硬體延遲 vs 獲利敏感度對照圖)，讓之後跑完腳本直接貼上去。

---

## 驗證計畫 (Verification)
我們需要新的 `sweep_latency.py` 產出預期結果：**`delay=0` 時擁有極高的成交量且 PnL 轉正，隨著 `delay` 增加，被逆向選擇擊穿導致 PnL 急速轉負。** 這就是我們帶去給教授看的最強圖表。
