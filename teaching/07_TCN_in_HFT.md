# 07. 為何在高頻交易選擇 TCN？(TCN in HFT)

當我們捨棄了傳統 ML，進入深度學習處理時間序列時，初學者常會立刻想到 **RNN (遞歸神經網路)** 或 **LSTM (長短期記憶網路)**。
但這項專題我們選擇了 **TCN (時間卷積網路, Temporal Convolutional Network)**。為什麼？

## 1. 什麼是 TCN？
TCN 是一種利用「卷積 (Convolution)」來處理時間序列的模型。它結合了兩個核心概念：
1.  **因果卷積 (Causal Convolution)**：確保模型推論時，「未來的資訊」絕對不會洩漏到「現在的預測」中。
2.  **膨脹卷積 (Dilated Convolution)**：允許模型在不增加過多參數或層數的情況下，擁有極大的「感受野 (Receptive Field)」去看到很久以前的歷史資料。

## 2. 為何 TCN 在高頻交易中完勝 LSTM/RNN？
在 HFT 以及最終要佈署到 FPGA (硬體晶片) 的情境下，TCN 擁有壓倒性的幾項優勢：

### A. 全平行化運算 (Massive Parallelism)
*   **RNN/LSTM**：必須「一步一步來」。它必須算完 t=1，才能算 t=2，才能算 t=3。因為它的隱藏狀態 (Hidden State) 依賴上一步的結果。這種**序列依賴性 (Sequential Dependency)** 讓它無法利用現代 GPU 或 FPGA 的大規模平行運算能力。
*   **TCN**：可以一次性處理整個時間序列！在硬體中，TCN 的矩陣乘法是可以做到 100% 平行展開的 (Fully pipelined & unrolled)，速度遠不是 RNN 能比擬的。

### B. 穩定的硬體記憶體佔用 (Deterministic Memory)
LSTM 擁有複雜的「閘門機制 (Gates)」，這需要頻繁地讀取和寫入內部狀態。對 FPGA 的 SRAM (靜態隨機存取記憶體) 來說，不斷的狀態讀寫會拖慢速度並消耗頻寬。
而 TCN 的數學運算就只是一堆簡單的 **MAC (Multiply-Accumulate, 乘加運算)**，也就是「權重乘以輸入再相加」。這非常適合用 FPGA 的 DSP (數位訊號處理單元) 進行最高速的硬體佈線。

### C. 梯度的穩定度 (Stable Gradients)
因為 TCN 不會像 RNN 那樣把誤差「無限次回傳」到過去（容易產生梯度消失），它可以很穩定地訓練長週期的微觀特徵 (如過去 100~200 個 ticks 的 LOB 變化)。

**總結**：**TCN 快、好訓練、且 100% 契合 FPGA 硬體的數學架構**，這就是它是當代高頻交易與量化團隊新寵兒的原因。
