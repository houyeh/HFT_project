# Week 4: FP32 訓練、策略評估與事件驅動回測

## 1. 專案資料夾架構對應說明 (Workspace Architecture Mapping)
Week 4 進入了「訓練模型」與「回測驗證」的深水區，對應到龐大的執行管線：
*   **`src/training/` (訓練管線)**：包含 `trainer.py` (控制 epochs, optimizer) 與 `datasets.py` (產生 PyTorch DataLoader)。在這裡，我們把 Week 3 的 TCN 真正跑出權重。
*   **`src/backtest/baseline.py` (基礎回測)**：將 TCN 當作簡單的買賣方向預測器，跑最基礎的無延遲 PnL 驗證。
*   **`src/backtest/event_driven.py` (高階回測核心)**：導入極度嚴格的 Tick-by-tick 事件驅動回測，實作「保守排隊填單 (Conservative queue-aware fill)」，真實模擬限價單 (Limit Order) 被擊穿的心酸血淚。
*   **`src/backtest/symbol_calibration.py` (校準器)**：負責根據不同商品 (TXFJ5, MXFJ5)，校準最適當的進場機率門檻與波動率條件。

## 2. 發展軌跡：從單向 Alpha 尋找到 Execution-Aware
*   **W4.1 ~ W4.3 (FP32 Baseline)**：利用 CrossEntropyLoss 練出基礎 TCN。
*   **W4.4 ~ W4.5 (Signal Backtest & Baseline Sweep)**：引入 class weighting，測試不同 horizon 下的預測準確率 (Accuracy)。
*   **W4.6 ( Event-Driven Backtest )**：這是一個**分水嶺**。將時間從 Vectorized (向量化一秒算完) 改成真實的 Tick-by-tick 事件驅動，瞬間把紙上富貴打回原形。排隊與滑價把原本的利潤吃光。
*   **W4.7 ~ W4.9 (進階特徵與 Symbol-aware)**：為了彌補 Event-Driven 的虧損，我們開始瘋狂增加過濾器：加入 `probability gating` (夠確定才出手)、`execution-aware features` (觀察 Queue imbalance 及 Cancel proxy 做防禦判斷)，以及區別對待不用商品的 `symbol-aware`。
*   **W4.10 (Data Universe)**：發現資料過少的標的（如 CDFJ5）會嚴重破壞訓練品質，從 Trading Universe 中剔除。

## 3. 從 FP32 訓練到 INT8 量化 (Post-Training Quantization, PTQ)
在 Week 4 的核心工程中，我們不僅把 TCN 訓練出來，更是為了下一階段的 FPGA 硬體佈署完成了「極限瘦身」。詳細流程如下：

### Step A: FP32 全精度訓練
這是深度學習的標準流程。我們使用 PyTorch，在 GPU 上以 FP32 (32位元單精度浮點數) 進行訓練。
我們使用 **CrossEntropyLoss** 來計算漲/跌分類誤差，並使用 **AdamW Optimizer** 配合 Learning Rate Scheduler 來找出最佳權重。
這時候產出的模型非常精準，但體積較大 (約 20 KB)，且需要消耗浮點數乘法器。而 FPGA 跑浮點數非常消耗邏輯閘與運算週期，絕對不是硬體層面上的最佳解。

### Step B: 校準數據集收集 (Calibration Setup)
為了把 FP32 降級為整數運算 (INT8)，我們不能直接把小數點砍掉。
我們從 Validation Set 中隨機抽出千筆數據作為「校準集 (Calibration Dataset)」。讓這批數據跑過已訓練好的 FP32 模型，紀錄每一層神經網路的輸入 (Activations) 與權重 (Weights) 產生的最大值與最小值 (Min-Max Range)。

### Step C: 靜態整數量化 (Static INT8 PTQ)
我們採用了 **對稱量化 (Symmetric Quantization)** 演算法。
這是一個純數學映射的過程。假設校準後某層權重的真實範圍是 `[-0.8, +0.8]`，我們運用映射公式：
`Scale = 0.8 / 127 = 0.0063`
將這層所有的浮點數除以 Scale 並四捨五入，使其嚴格限縮到 `-128 到 +127` 之間的 8-Bit 整數空間。
當未來放到 FPGA 硬體時，晶片只需將兩個 INT8 數字進行極速整數相乘，最後再透過 bit-shift 或簡單乘回 Scale，就能還原精準度。

### 量化總結與硬體指標 (The Result)
*   **記憶體佔用 (Footprint)**：從近 20 KB 暴縮至原本的 1/4，僅需 **4.8 KB**。這讓 FPGA 能夠直接把整個模型塞進最靠近運算核心的 SRAM 中，免去了外部記憶體傳輸的巨大延遲。
*   **運算力需求 (Compute)**：推論一次僅需 **460K 級別的整數 MACs (乘加運算)**。這對現代 FPGA 來說，是能在一微秒內輕鬆吞吐的微量運算。
*   **精度損傷最小化 (No Accuracy Drop)**：驚人的是，我們發現 TCN 在轉換為 INT8 後，其方向預測準確率與 FP32 相比誤差極小，完全無損高頻造市引擎的毒性判定能力！

## 4. 面臨的巨大迷思
在 Week 4 的最後，我們的 TCN 方向預測率變得很高，我們的過濾器很嚴密，但我們的 PnL 卻始終無法有效突破。因為模型為了解決「可能虧損的風險」，選擇了「盡可能不要交易」。這為我們後續 Week 5 的**「混合造市與第一性原理覺醒」**埋下了伏筆。
