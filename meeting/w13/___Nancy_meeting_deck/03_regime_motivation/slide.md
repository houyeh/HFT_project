# 03 為何需要 Regime Gate? (Why Regime Gate Is Needed)

## 原始微觀模型的極限 (Micro vs Macro)
- TCN (window=100) 只能處理微觀的 adverse selection (毒性訂單)。
- **NEW3 壓力測試原始結果**：Model PnL (-551,705) vs Baseline (-557,165)，改善僅微幅 +5,460。
- **結論**：微觀的 cancel filter 無法拯救宏觀的 regime failure。Maker-only 策略在強烈趨勢下會如同接刀，累積致命的 inventory risk。

**重點提醒**：NEW3 是一個絕佳的壓力測試市場狀態 (Stress Regime)，不能錯誤地將其排除，而是需要事前辨識的 causal gate 來防禦。
