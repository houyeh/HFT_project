# TCN (HLS-Friendly) 模型訓練細節

## 1. 資料處理與分割 (Data Split)
- **總資料量**：`1,489,205` 筆高頻 Ticks 快照。
- **特徵數 (Features)**：8 個精選核心特徵 (經過 Rolling Z-score 標準化)。
- **時間步長 (Time Steps)**：100 (模型每次看過去 100 個 Ticks)。
- **資料切割**：
  - **Training Set (80%)**：1,191,364 筆
  - **Validation Set (20%)**：297,841 筆
  - *採用 Shuffle 切割以增加每個 Batch 的多樣性。*
- **標籤分佈與權重 (Class Weights)**：
  - 由於 `Stationary` (盤整) 資料遠多於漲跌，訓練時動態計算了類別權重：
    - `Down (0)`: 權重約 2.08
    - `Stationary (1)`: 權重約 0.49
    - `Up (2)`: 權重約 2.08
  - *這迫使模型更認真學習上漲與下跌的特徵，而不是無腦猜測盤整。*

## 2. 網路架構 (Hardware-Friendly Architecture)
本模型為 `src/models/tcn.py` 中的 `HLSFriendlyTCNClassifier`，特別針對 hls4ml 移除了 `F.pad` 與維度轉換，改用純粹的 Padding 與 `AdaptiveAvgPool1d`。

- **輸入層 (Input)**：`(Batch, 8 Features, 100 Time Steps)`
- **TCN Backbone (3 層 Temporal Blocks)**：
  - **Block 0**: CausalConv1d(in=8, out=16, kernel=3, dilation=1)
  - **Block 1**: CausalConv1d(in=16, out=16, kernel=3, dilation=2)
  - **Block 2**: CausalConv1d(in=16, out=16, kernel=3, dilation=4)
  - *每層皆包含 Residual Connection (殘差連接) 以穩定梯度。*
- **全局平均池化 (Global Average Pooling)**：
  - 將 100 個時間步的輸出壓平為單一向量 `(Batch, 16)`，不僅消除雜訊，也大幅節省 FPGA 的硬體資源。
- **輸出層 (Output)**：
  - 全連接層 (Dense): 16 -> 3
  - 輸出 3 個類別 (Down, Stationary, Up) 的 Logits。

## 3. 模型參數 (Model Size)
- **總參數量**：極致輕量的 **4,515** 個參數！
- **對比**：比之前的 Baseline MLP (26,405 參數) 還要小將近 6 倍！
- **硬體優勢**：極小的權重矩陣可以完全塞進 FPGA 的 BRAM (內建靜態記憶體) 中，不需要去讀取外部的 DDR 記憶體，這能讓推理延遲輕易壓縮到個位數微秒 (< 10 μs) 級別。

## 4. 訓練超參數 (Hyperparameters)
- **Loss Function**：`CrossEntropyLoss` (搭配上述的 Class Weights)
- **Optimizer**：`Adam`
- **Learning Rate**：0.001
- **Batch Size**：1024
- **Epochs**：20 (依據 Validation Accuracy 保存最佳模型)
