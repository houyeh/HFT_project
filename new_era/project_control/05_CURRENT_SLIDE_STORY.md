# Slide Story

1. Baseline model selected from label sweep.
- claim: smoothed_mid_move_3class was selected based on optimal PnL improvement.
- supporting evidence: label type sweep experiments showed +17,979 TWD avg improvement.
- caveat: mid-move label is a forecasting proxy, not a direct quote-risk label.

2. TCN classification converges but not a directional trader.
- claim: Model is highly sensitive but has low precision (~2.8%).
- supporting evidence: classification_report_test.csv.
- caveat: Not suitable for active directional trading, only cancel filter.

3. Backtest shows local edge under matched random cancel baseline.
- claim: Lower thresholds (0.55, 0.60) outperform matched random cancel + 1std.
- supporting evidence: baseline_summary_fixed.csv (Model 0.55 = 84k, Random+1std = 80k).
- caveat: Validation-selected threshold (0.70) fails this test.

4. Regime is the main bottleneck.
- claim: Macro regime overwhelms micro cancel filter.
- supporting evidence: NEW3 baseline is deeply negative (-557k), TCN improvement alone is minor (+5.4k).
- caveat: Micro models cannot save bad macro flow.

5. Static gates prove regime dependency.
- claim: Different markets require radically different gating policies.
- supporting evidence: NEW prefers spread_filter, NEW3 requires combined_filter (trend+vol+spread).
- caveat: These are static oracles, not deployable auto-selectors.

6. 017 simple auto-gate failed, showing regime calibration is harder.
- claim: Causal auto-gating underperformed static oracles.
- supporting evidence: NEW3 test dropped to -630k PnL.
- caveat: Failed valuable negative result.

7. 016 binary label validates quote-risk label pipeline but model is not ready.
- claim: Fill-conditional proxy exhibits ~10x toxic rate lift over unconditional.
- supporting evidence: label distribution CSVs (3% vs 0.3%).
- caveat: Model does not beat random+1std backtest (266k vs 272k).

8. Latency matters, but FPGA claim must be conservative.
- claim: Risk increases with latency, FPGA reduces worst-case cancel delay.
- supporting evidence: PnL noticeably drops at delay=10.
- caveat: Do not claim delay > 3 ticks collapses PnL (non-monotonicity exists).

9. Queue-aware validation is required before stronger trading claims.
- claim: Current backtest simplified queue rules.
- supporting evidence: queue_data_audit.md confirms lack of aggressor_side data.
- caveat: True queue replay impossible; rely on sensitivity proxies.

10. Next step is queue-aware validation + conservative FPGA MVP framing.
- claim: Future efforts focus on rigorous validation over model expansion.
- supporting evidence: 018 is next critical path.
- caveat: Production readiness is strictly not claimed.
