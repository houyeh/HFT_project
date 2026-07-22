# 1. Executive Summary
目前模型不適合方向交易；它可能作為 maker-only cancel filter，但是否有真實 edge 取決於 random cancel / OOS / true cancel delay 結果。

# 2. Confirmed Findings
- 模型是 risk score，不是 calibrated probability。
- 在 test split 上，Model_0.55/0.60 高於 no-model baseline。
- Validation-selected threshold (v2) = 0.70，test once net_pnl = 63440.00。
- NEW3 OOS 仍為負損益：model_net_pnl = -551705.00，僅小幅改善 5460.00。

# 3. Critical Checks
- Random cancel baseline：已修正並產出 baseline_comparison_fixed.csv / baseline_summary_fixed.csv。
- Missed opportunity：已修正並產出 missed_opportunity_diagnostics_fixed.csv。
- True cancel delay：已拆分 prediction / cancel / combined 三種延遲。
- OOS / NEW3：策略在不利 regime 下仍失效（虧損）。
- Regime diagnostics：已產出 regime_diagnostics.csv 與 regime_gate_sweep.csv。

# 4. Random Baseline Verdict
- Model 0.55 net_pnl=84980.00 vs Random_0.55 mean+1std=80473.19
- Model 0.60 net_pnl=83460.00 vs Random_0.60 mean+1std=79049.33
- Model 0.70 net_pnl=63440.00 vs Random_0.70 mean+1std=72778.70
- 若上述未穩定勝出，模型訊號尚未證明有 trading edge。

# 5. Latency Verdict
- prediction delay 3/5 事件表現變好：True（較可能是減少 false-positive cancel，不是低延遲優勢）。
- true cancel delay 增加時 PnL 是否下降：False。
- 結論：不可把 prediction delay 當作 cancel delay；FPGA必要性應以 true cancel delay 敏感度判定。

# 6. Regime Verdict
- NEW3 強趨勢情境下，maker-only + cancel filter 改善有限，策略仍虧損。
- 趨勢/高波動 regime 需要 macro regime gate。
- 最佳 gate（目前掃描）: combined_filter {"trend_ticks": 20.0, "vol_threshold": 0.20974979111878053, "spread_ticks_min": 2.0} @thr=0.55: net_pnl=72520.00, improvement=264355.00, active_rate=0.826

# 7. Final Strategy Recommendation
Regime Gate
→ Maker-only Quote Engine
→ TCN Risk Score
→ Cancel / Skew Decision
→ Latency-aware Backtest

# 8. Final Grade
C

# 9. Next Actions
- 強化 random baseline 顯著性檢定（bootstrap / block bootstrap）。
- 修正 label 與 horizon，分離 directional vs toxicity objective。
- 擴充 regime gate（session-level + volatility-aware）。
- 強化 backtest queue/cancel realism（true cancel delay / queue position decay）。
- 擴充 feature set（order-flow imbalance, cancel pressure, volatility state）。
