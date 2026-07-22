# 04 加入 Regime Gate 的結果 (Regime Gate Added)

## 實作與成效 (Implementation & Results)
- **目前實作**：Rolling trend + Rolling volatility + Spread filter (註：Kalman/EWMA 為 Future Work)。
- **NEW 最佳 Gate**：`spread_filter + TCN 0.55` (Net PnL 約 101,455，Active Event Rate 約 82.4%)。
- **NEW3 最佳 Gate**：`combined_filter + TCN 0.55` (Net PnL 約 72,520，Active Event Rate 約 82.6%)。

## 關鍵洞察 (Key Takeaways)
- **並非因減少交易次數而獲利**：Active rate 仍維持在 82% 左右的高水準。
- **Regime 的獨立性**：NEW 和 NEW3 分別需要不同的最佳 Gate，這是 Regime Dependency 的直接證據！
- **下一步**：目前尚未做到 production-ready 的 auto-gating，接下來需要開發 Causal Auto-gate Selector。
