# 1. Executive Summary
Causal auto-gating verdict: **C**. Selected policy `volatility_risk_off` @ threshold `0.70` (Auto-gate cannot beat static combined_filter on either NEW or NEW3.)

# 2. Motivation
Known static results: NEW best spread_filter net_pnl=101455.00; NEW3 best combined_filter net_pnl=72520.00.
This confirms regime dependency but does not by itself prove deployable auto-gating.

# 3. Causal Regime Features
Features use rolling/EWMA calculations only (backward-looking). No future/full-session leakage features are used.

# 4. Static Baselines and Oracle
- Baselines: no_gate_tcn, always_spread_filter, always_combined_filter.
- Oracle: dataset_oracle_best_gate (non-deployable).
- Oracle mapping from static summary: NEW -> spread_filter, NEW3 -> combined_filter.
- Oracle net_pnl: NEW=101455.00, NEW3=72520.00. Oracle is an upper bound, not a live policy.

# 5. Validation Selection
- Selection done on NEW validation only.
- val_session_count=1, val_sample_count=563489.
- Caveat: Validation set contains only one session; parameter selection may have high variance.
- Selected params: {"spread_min": 1.5, "vol_quantile": 0.7, "vol_threshold": 0.1729132438508252, "vol_w": 100}

# 6. NEW Test Results
NEW test auto-gate net_pnl=10000.00, improvement=-1150.00, active_event_rate=0.5841, max_drawdown=111395.00.

# 7. NEW3 Stress Results
NEW3 full auto-gate net_pnl=-630960.00, improvement=-183055.00, active_event_rate=0.6626, max_drawdown=671225.00.

# 8. Auto-Gate vs Static Gates
- NEW: beats_no_gate=False, beats_spread=False, beats_combined=False, beats_oracle=False.
- NEW3: beats_no_gate=False, beats_spread=False, beats_combined=False, beats_oracle=False.

# 9. EWMA Ablation
Best EWMA combo avg(net_pnl NEW/NEW3)=-61740.00 at alpha=0.1, velocity_q=0.8, vol_q=0.9.
If EWMA and rolling results are close, rolling gate is preferred for simplicity.

# 10. Verdict
C. Auto-gate cannot beat static combined_filter on either NEW or NEW3.

# 11. Meeting-ready Claims
- We proved regime dependency (NEW vs NEW3 static gate preference diverges).
- We tested a causal auto-gating prototype with event-level masks.
- Auto-gate verdict: C.
- Production readiness is not claimed unless robustness is clearly dominant.

# 12. Next Steps
- Integrate gating hooks with 016_new_binary once binary model evidence is finalized.
- Improve regime selector stability with more independent sessions.
- Extend ablation (EWMA/Kalman) after stronger validation coverage.
- Prepare FPGA MVP path for causal gate + cancel decision path.