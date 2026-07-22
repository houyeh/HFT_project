# 01 最終選擇的模型 (Final Model Identity)

## 架構與資料 (Architecture & Data)
- **資料集 (Dataset)**: NEW_ALL / TXFB6
- **資料切分 (Split)**: **Session-level Chronological Split** (以「交易盤別」為單位依時間排序，前 70% Sessions 僅供訓練，接下來 15% 僅供 Val，最後 15% 僅供 Test，確保彼此完全獨立不重疊)
- **特徵 (Features)**: 8 個微觀特徵 (Z-score window: 2000)
- **時間視窗 (Window)**: 100 ticks | **預測視距 (Horizon)**: 5 ticks
- **目標標籤 (Label)**: `smoothed_mid_move_3class` (Threshold: 0.5 ticks)
- **模型結構 (Model)**: TCN (channel_list=[16,16,16], kernel_size=3, dropout=0.1)

## 分類指標洞察 (Classification Insights)
- 在預測 Down/Up 類別上，模型呈現**極低的 Precision 但極高的 Recall**，這是因為 Stationary (盤整) 樣本佔據壓倒性多數。
- **結論**：這不是一個主動預測方向的交易模型 (Directional Trading Model)，而是極高敏感度的 **toxic-flow 撤單過濾器 (Cancel Filter)**。
