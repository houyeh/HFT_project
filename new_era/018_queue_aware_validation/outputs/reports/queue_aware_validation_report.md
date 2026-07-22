# 1. Executive Summary
Queue-aware sensitivity verdict: **C** (At least one key policy loses edge under conservative queue assumptions.)

# 2. Why Queue Matters
Current maker-only backtest is useful for first-order signal comparison, but price touch does not guarantee our queue position fills.

# 3. Available Data and Limitations
- LOB depth: available (L1-L5 with volume).
- Trade tape: available (timestamp, price, volume).
- aggressor_side: not available.
- True exchange queue replay: not possible from this data.

# 4. Sensitivity Scenarios
- original_fill (reference)
- uniform_fill_haircut p in {0.75, 0.50, 0.25}
- good_fill_haircut (mild / medium / severe)
- queue_ahead_best_level (queue_fraction {0.25,0.50,1.00}, fill_window {1,3,5,10})
- delayed_cancel_plus_queue (cancel_delay in {0,1,2,3,5,10})

# 5. NEW Results
- no_gate_tcn_055: original=84980.00, conservative_mean=13245.24, p10=-15725.00, verdict=fragile_but_positive.
- spread_filter_tcn_055: original=101455.00, conservative_mean=41494.63, p10=80.00, verdict=fragile_but_positive.
- combined_filter_tcn_055: original=71735.00, conservative_mean=26103.05, p10=-23530.00, verdict=fragile_but_positive.
- no_gate_tcn_070: original=63440.00, conservative_mean=23426.95, p10=-20975.00, verdict=fragile_but_positive.
- val_selected_tcn_070: original=63440.00, conservative_mean=23426.95, p10=-20975.00, verdict=fragile_but_positive.

# 6. NEW3 Results
- combined_filter_tcn_055: original=38010.00, conservative_mean=-22149.63, p10=-243460.00, verdict=not_robust.
- no_gate_tcn_055: original=-379275.00, conservative_mean=-384221.46, p10=-626655.00, verdict=not_robust.

# 7. Latency Under Queue Assumptions
- combined_filter_tcn_055 / delay_good_medium: delay 10 vs 0 delta = -133910.50.
- combined_filter_tcn_055 / delay_original_like: delay 10 vs 0 delta = -98307.50.
- combined_filter_tcn_055 / delay_queue_q50: delay 10 vs 0 delta = -98307.50.
- no_gate_tcn_055 / delay_good_medium: delay 10 vs 0 delta = -10069.50.
- no_gate_tcn_055 / delay_original_like: delay 10 vs 0 delta = -23330.00.
- no_gate_tcn_055 / delay_queue_q50: delay 10 vs 0 delta = -23330.00.
- spread_filter_tcn_055 / delay_good_medium: delay 10 vs 0 delta = -12890.50.
- spread_filter_tcn_055 / delay_original_like: delay 10 vs 0 delta = -68480.00.
- spread_filter_tcn_055 / delay_queue_q50: delay 10 vs 0 delta = -68480.00.

# 8. Verdict
C. At least one key policy loses edge under conservative queue assumptions.

# 9. Meeting-ready Claims
- Current backtest is first-order signal comparison, not full queue replay.
- Queue-aware sensitivity is necessary before production profitability claims.
- Under conservative assumptions, the result is verdict C.

# 10. Next Steps
- Do not claim production trading profitability from current assumptions.
- Prioritize control-system reliability, risk management, and queue data enrichment.