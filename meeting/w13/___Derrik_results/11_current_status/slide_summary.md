# 目前狀態與限制 (Current Status and Limitations)

目前進度大約在 C+ / B- 的階段。

**已達成的里程碑：**
- 證明 TCN 在 NEW 測試集上有 Local edge。
- 模型 (Threshold 0.55 / 0.60) 成功擊敗 Random+1std。
- Regime Gate 成功在 NEW3 壓力測試中發揮救援作用。
- Fill-conditional 審查發現了毒性率的提升方向。
- 具體量化了 Cancel Delay 帶來的虧損風險。

**尚未達成的目標：**
- Validation 所選出的 Threshold 尚未能在測試集穩健擊敗 Random+1std。
- 尚未完成 `bid_toxic` / `ask_toxic` 的獨立重新訓練。
- 尚缺基於真實 Queue-aware 的回測驗證 (HftBacktest-style)。
- 尚未開發出動態適應的 Auto-Gating 機制。
- 尚未實作 FPGA 的硬體獲利證明。
