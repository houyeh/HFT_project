# 資料集與模型基本資訊

- **資料集來源**：見 00_data_source_details.md
- **交易對 (Symbol)**：TXF
- **資料切分 (Train/Val/Test)**：採時間序列切分，詳細筆數見 00_data_source_details.md
- **特徵與視窗**：特徵數 (feature_count)=8，正規化視窗 (zscore_window)=2000
- **模型視窗**：輸入視窗 (window_size)=100，預測區間 (horizon)=5
- **目標標籤**：`smoothed_mid_move_3class`
