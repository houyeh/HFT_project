# Week 3: 硬體友善型 TCN 模型架構設計 (實作紀錄)

## 1. 專案資料夾架構對應說明 (Workspace Architecture Mapping)
Week 3 是本專案的「大腦」建構期，對應到以下幾個核心檔案：
*   **`src/models/tcn.py` (實作核心)**：包含最底層的神經網路架構 (`TemporalBlock`, `CausalConv1d`, `TCNClassifier`)。
*   **`src/utils/constants.py` (全域設定)**：寫死 TCN 的超參數（如通道數 `TCN_CHANNELS`、核心大小 `TCN_KERNEL_SIZE`），確保所有模組叫用到的是同一套網路標準。
*   **`tests/test_model.py` (驗證)**：對 TCN 模型進行 10 項嚴格的矩陣與因果性驗證。

## 2. TCN 如何被設計與訓練？
在 `tcn.py` 中，我們運用了三個重要技術來打造這顆大腦：
1.  **因果卷積 (Causal Convolution)**：強制 Padding 只加在序列左邊 (過去)，確保運算 `T` 時刻絕對拿不到 `T+1` 的資料，徹底斷絕未來洩漏 (Look-ahead bias)。
2.  **膨脹卷積 (Dilated Convolution)**：為了讓神經網路「看得很遠」卻又「保持極小的硬體資源」，我們讓卷積層的跳躍間距以 $2^n$ (1, 2, 4...) 增長。僅需幾層薄薄的網路，感受野 (Receptive Field) 就能擴展到 29 ticks 以上。
3.  **全域平均池化 (GAP) 與 分類器 (Classifier)**：把擷取到的極長特徵平滑化，最後通過線性層轉換成三分類 (Up, Down, Stationary) 的發生機率。
*在訓練層面上 (後續 Week 4 實作)，我們使用 `CrossEntropyLoss` 搭配 `AdamW`，將這些轉化好的 3D LOB 張量餵給模型，持續更新其權重。*

## 3. 靈魂拷問：我們真的在 Python 裡做到「奈秒級 (Nanosecond)」了嗎？
**答案是：還沒有！Python 物理上不可能做到奈秒級。**
*   **現在的軟體階段 (Week 3~5)**：我們是在 Pytorch 中 **「模擬」** TCN。每次 Forward Pass 可能需要數毫秒 (Milliseconds)。但在這套回測系統中，我們假定 `delay=0` 時，就是在「數學上模擬」它擁有奈秒級處理速度時的投資報酬率 (PnL)。
*   **未來的硬體階段 (Week 6~7)**：因為我們在 `tcn.py` 的設計上，嚴格限制了參數數量 (< 100K)、全平行卷積運算 (無 LSTM 的遞迴分支)、以及 INT8 量化相容。因此，只要我們把權重凍結並放入 FPGA 的 DSP 乘加器中 (HLS / RTL C++)，它在未來實體晶片上的推論，將會真正達到 **百奈秒 (sub-microsecond) 甚至低於奈秒的物理極速**！

## 4. 模型架構設計細節與超參數調整 (Hyperparameter Tuning)
在高頻交易中，模型的超參數 (Hyperparameters) 不僅關乎準確率，更直接決定了「能不能塞進 FPGA」。以下是我們的設計與調參原理：

- **時間步長 (Sequence Length = 100)**：我們設定模型每次看過去 100 個 ticks。這大約涵蓋了數秒到數十秒的微觀結構變化，足夠捕捉毒性大單壓境的前兆，而不會因為時間太長導致運算延遲。
- **通道設計 (TCN Channels = [16, 16, 16])**：一開始我們嘗試過更大的通道數 (例如 32 或 64)，雖然準確率高了一點點，但在計算 MAC (乘加運算) 數量時發現超出了目標 FPGA 的極速 DSP 資源限制。我們最終收斂到 16 通道，因為 16 是 2 的冪次 ($2^4$)，FPGA 在實作乘法時可以使用 `Bit-Shift` (二進位位移) 輕易加速，將運算成本降到最低。
- **卷積核與膨脹率 (Kernel = 3, Dilation = [1, 2, 4])**：
  這是 TCN 的靈魂。膨脹率 (Dilation) 會隨著層數指數成長。
  $RF = 1 + 2 \times (3 - 1) \times (1 + 2 + 4) = 29 \text{ ticks}$。
  這表示我們只用了極少量的卷積核，就擁有了 29 ticks 的絕對感受野 (Receptive Field)。這比使用 LSTM 需要遞迴幾十次的效率高出百倍以上。
- **丟棄率 (Dropout = 0.1)**：LOB 數據的雜訊極大。我們在每一個卷積層後加上 10% 的 Dropout，強迫神經網路不要死背特定的掛單數字，而是學習「失衡的趨勢」，有效防止了過度擬合 (Overfitting)。

## 5. 檢視成果 (`tests/test_model.py`)
執行 `python -m pytest tests/test_model.py -v` 確保：
- Forward Pass 形狀正確 (Batch, 3)
- 因果性測試 (Causality) 完美通過
- 參數量足夠小適合 FPGA 佈署
