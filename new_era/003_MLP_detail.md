## 訓練/測試資料量 (Train/Test Split)

- 總資料量：合併 TXFB6 與 MXFB6 後，總計 1,489,205 筆時序快照 (經過 Rolling Z-score 正規化)。
- 切分比例：80% 訓練集 (約 119 萬筆)、20% 驗證集 (約 29 萬筆)。
- 模型架構 (Layers & Connections)：
這是一個三層的極簡網路。
  - 輸入層 (Input)：Flatten。將 (100 ticks, 8 features) 的二維陣列攤平成 800 維的一維向量。
  - 隱藏層 1 (Hidden 1)：Dense(32) + BatchNorm + ReLU。這一步將 800 維特徵壓縮萃取，設定為 32 顆神經元以避免消耗過多 FPGA 的 DSP (乘加單元)。
  - 隱藏層 2 (Hidden 2)：Dense(16) + BatchNorm + ReLU。
  - 輸出層 (Output)：Dense(3) + Softmax。輸出三分類機率 (Down, Stationary, Up)。
- 參數數量 (Parameters)：
總參數僅有 26,405 個。在 FPGA 領域，低於 10 萬個參數的模型都是極具延遲優勢的，絕對能在 50 微秒甚至 5 微秒內算完！
- 訓練配置與防禦機制：
為了對抗原本高達 80% 的「盤整 (Stationary)」極端不平衡，我們實作了動態類別權重 (Class Weights) 來強迫模型重視大漲與大跌的樣本。
- 驗證準確率 (Val Accuracy)：目前最佳收斂在 4% 左右（對比完全瞎猜的 33.3%，這在雜訊極大的 10 ticks 高頻預測中，已經具備了真實賺錢的邊際優勢了！）。
