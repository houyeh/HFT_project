# 02 回測策略與結果 (Maker-only Cancel Filter Backtest)

## 策略邏輯 (Strategy Logic)
- **被動造市 (Maker-only)**：預設掛單在最佳 Bid/Ask。
- **撤單邏輯**：`P(Down) > threshold` → 撤銷 Bid | `P(Up) > threshold` → 撤銷 Ask。
- **參數設定**：Maker Fee = 15 TWD, Taker Fee = 50 TWD, Point Value = 200, Max Inventory = 3, allow_active_alpha = false。

## 關鍵結果 (Results)
- 評估的 Threshold：0.55 / 0.60 (Test-sweep candidates), 0.70 (Validation-selected)。
- **重點洞察**：0.55 與 0.60 在 NEW test split 中相較於無模型與隨機撤單 baseline 有著明顯的 local edge。
- **注意事項**：0.70 尚未顯著擊敗 `Random+1std`，因此 0.55 不能宣稱為 production threshold，策略尚不可稱為 production-ready。
