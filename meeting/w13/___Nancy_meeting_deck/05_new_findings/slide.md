# 05 新發現：Label 與 Latency 敏感度 (New Findings)

## A. Label 的發現
- **舊版 Label**：已經測試過 `toxic_bid/ask`，其 Macro F1 約為 0.5，但 PnL 卻是 0 或負數，這代表不需要再重試舊版 Label。
- **新方向 (Fill-conditional)**：建立 fill-conditional quote-risk label。根據 Proxy C 的審查，conditional toxic rate 高達數個 percent，明顯高於 unconditional 的 0.3%，展現出強烈潛力。
- **限制聲明**：這是基於 proxy 的分析，並非來自真實的 exchange fill（因缺乏 aggressor-side 數據）。

## B. Latency 的發現
- **延遲與虧損**：Cancel delay 會導致 delayed toxic loss 增加，但整體 Net PnL 的衰退是非單調的 (non-monotonic)。
- **重大衰退點**：當 Delay=10 時，表現明顯變差。
- **FPGA 加速的定位**：我們採用 FPGA 的合理性，是基於「Risk-control-oriented (風險控制)」，防堵延遲造成的巨額虧損，而非純粹為了證明 PnL 的絕對必要性。
