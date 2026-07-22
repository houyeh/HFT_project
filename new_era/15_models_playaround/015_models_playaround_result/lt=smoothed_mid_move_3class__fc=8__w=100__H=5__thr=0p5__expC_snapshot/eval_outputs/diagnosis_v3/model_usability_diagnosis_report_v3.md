# 1. Executive Summary
TCN toxic-flow cancel filter has a local edge on NEW at threshold 0.55/0.60, but the validation-selected threshold 0.70 does not beat random+1std; NEW3 requires stronger regime gating, and current gate-choice evidence is still dataset-specific, not production-ready.

# 2. Existing Evidence
- Model_0.55 = 84,980.00 > Random_0.55 mean+1std = 80,473.19 (pass)
- Model_0.60 = 83,460.00 > Random_0.60 mean+1std = 79,049.33 (pass)
- Val-selected threshold 0.70: test_net_pnl = 63,440.00 vs Random_0.70 mean+1std = 72,778.70 (not significant)
- spread_filter + 0.55 on NEW: net_pnl = 101,455.00, active_event_rate = 0.8235
- combined_filter + 0.55 on NEW3: net_pnl = 72,520.00, active_event_rate = 0.8257
- threshold split remains unresolved for production: test-sweep best=0.55 vs val-selected=0.70

# 3. NEW3 Regime Diagnosis
- NEW3 session trend_strength values: 620.953, 459.430, 1413.233, 1323.538, 1076.491, 135.592
- NEW3 suitability distribution (by current hard rule): UNSUITABLE=6, MARGINAL=0, SUITABLE=0
- NEW test suitability distribution (same rule): UNSUITABLE=2, MARGINAL=0, SUITABLE=0
- Important bug/definition warning: this trend_strength rule is too strict in current scaling and classifies even NEW test sessions as UNSUITABLE, so suitability labels are not yet a reliable gate selector.
- Failure mode still aligns with regime story: strong trend sessions plus maker inventory carry lead to adverse outcomes.
- Causal pre-trade gate candidate remains valid in principle: rolling trend/return threshold computed from past data only.

# 4. NEW Positive Control
- NEW test split is mixed at session level (one positive, one negative baseline session), not uniformly favorable.
- So the positive control is statistical (overall improvement) rather than a clean "all sessions suitable" profile.

# 5. Gate Cross-Dataset Analysis
- Best NEW3 spread_filter row: net_pnl = -321,405.00, active_event_rate = 0.9161
- Best NEW combined_filter row: net_pnl = 65,535.00, active_event_rate = 0.7374
- We do observe shared improving gates across NEW and NEW3 in `spread_gate_cross_dataset_analysis.csv`, but:
  1. many rows are still negative on NEW3 absolute PnL,
  2. gate ranking changes by dataset,
  3. ex-ante rule for choosing spread_filter vs combined_filter is not walk-forward validated.
- Verdict: current evidence supports dataset-specific efficacy, not a production-ready universal gate policy.

# 6. Side-Specific Markout Labels
- H=10 rates:
  - NEW_train_val: bid_toxic=0.0038, ask_toxic=0.0036
  - NEW_test: bid_toxic=0.0028, ask_toxic=0.0029
  - NEW3_full: bid_toxic=0.0032, ask_toxic=0.0033
- These toxic rates are very low and close to each other, and are lower than original Down/Up base rates (~1.39% each).
- This indicates a label-definition issue (not fill-conditional, spread-cushion bias), so this specific side-markout statistic is not yet a good training target by itself.

# 7. FPGA Relevance
- Existing latency diagnostics (read-only) show:
  - cancel_delay=0 events: net_pnl=63,440, delayed_toxic_loss=0
  - cancel_delay=10 events: net_pnl=42,650, delayed_toxic_loss=14,000
- Small delays are non-monotonic, but large delay clearly hurts. FPGA relevance is conditional and should not be overstated yet.

# 8. Final Grade
- Grade: C+
- Reason: random-baseline and OOS diagnostics are now more complete, but val-selected threshold still lacks random-significance and regime-gate selection is not proven as a stable ex-ante policy across unseen regimes.

# 9. Next Actions
1. Recalibrate regime features (especially trend_strength scaling) and validate gate-selection logic by walk-forward, not dataset lookup.
2. Build fill-conditional side-toxic labels (bid/ask adverse excursion with fees), then retrain.
3. Keep threshold selection validation-driven; do not promote 0.55 from test sweep to production.
4. Re-test shared gate candidates on additional unseen contracts/months.

# 10. What NOT to claim
- Do not claim direction-trading capability.
- Do not call 0.55 a production threshold.
- Do not claim NEW3 improvement is model-only; gate effects are large.
- Do not claim FPGA is always required without monotonic cancel-delay evidence.
- Do not treat current dataset-specific gate winners as production-ready strategy logic.
