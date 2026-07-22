# 06 下週行動事項 (Next Week Todos)

## 1. New Binary Quote-Risk Model
- **不再重試舊版 toxic labels**，而是建立 fill-conditional bid/ask quote-risk label。
- 訓練小型 Binary TCN 模型。
- 保留完整的 Loss curve, PR-AUC, Confusion Matrix 與 Threshold diagnostics。
- 與現有的 `smoothed_mid_move_3class` baseline 進行最公平的比較。

## 2. Regime Auto-Gating
- 驗證能否使用 causal rolling features 自動選擇最佳的 Filter (`spread_filter`, `combined_filter`, 或是 `risk-off`)。
- **不使用** full-session future trend，並確保**不在 NEW3 上調參**。
- 將 no-gate, always-spread, always-combined 與 auto-gate 的表現進行橫向對比。

## 3. FPGA MVP Path
- 若模型的 Edge (優勢) 不夠，將優先實作 rolling/EWMA gate 搭配 tiny TCN cancel filter 作為 MVP (最小可行性產品)。
- FPGA 的終極目標在於：降低 Cancel Latency 並消除 Delayed Toxic Loss。
