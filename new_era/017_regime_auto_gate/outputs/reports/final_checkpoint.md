# Final Checkpoint

1. Best static gate on NEW:
- policy: always_spread_filter ({"spread_ticks_min": 1.5})
- net_pnl: 101455.00
- active_event_rate: 0.8235

2. Best static gate on NEW3:
- policy: always_combined_filter ({"trend_ticks": 20.0, "vol_threshold": 0.20974979111878053, "spread_ticks_min": 2.0})
- net_pnl: 72520.00
- active_event_rate: 0.8257

3. Selected auto-gate policy:
- policy_name: volatility_risk_off
- params: {"spread_min": 1.5, "vol_quantile": 0.7, "vol_threshold": 0.1729132438508252, "vol_w": 100}
- selected on validation data: NEW validation split only
- model_threshold: 0.70
- validation caveat: Validation set contains only one session; parameter selection may have high variance.

4. Auto-gate result on NEW test:
- net_pnl: 10000.00
- active_event_rate: 0.5841
- trade_count: 200
- max_drawdown: 111395.00

5. Auto-gate result on NEW3:
- net_pnl: -630960.00
- active_event_rate: 0.6626
- trade_count: 944
- max_drawdown: 671225.00

6. EWMA result if run:
- best EWMA policy: ewma_trend_vol_auto alpha=0.1 velocity_q=0.8 vol_q=0.9
- net_pnl NEW: 8845.00
- net_pnl NEW3: -132325.00

7. Verdict:
- C
- reason: Auto-gate cannot beat static combined_filter on either NEW or NEW3.