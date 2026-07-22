# NEW3 Regime Diagnosis

## 1) NEW3 per-session trend_strength
- 20260113: trend_strength=620.9528, cum_return_ticks=-121.00, baseline_net_pnl=-352235.00, trend_regime=strong-downtrend
- 20260114: trend_strength=459.4297, cum_return_ticks=82.50, baseline_net_pnl=-62285.00, trend_regime=strong-uptrend
- 20260115: trend_strength=1413.2329, cum_return_ticks=256.00, baseline_net_pnl=201630.00, trend_regime=strong-uptrend
- 20260116: trend_strength=1323.5384, cum_return_ticks=282.50, baseline_net_pnl=-125345.00, trend_regime=strong-uptrend
- 20260119Day: trend_strength=1076.4914, cum_return_ticks=178.50, baseline_net_pnl=-21295.00, trend_regime=strong-uptrend
- 20260330: trend_strength=135.5922, cum_return_ticks=24.00, baseline_net_pnl=-14805.00, trend_regime=strong-uptrend

## 2) Suitability distribution
- NEW3: UNSUITABLE: 6
- NEW test split: UNSUITABLE: 2

## 3) Main failure mode
- NEW3 failure distribution: strong_uptrend_adverse_selection: 4, strong_downtrend_adverse_selection: 1, none_or_minor: 1
- Dominant pattern is strong trend + negative baseline PnL, which is maker-unfriendly.

## 4) Why spread_filter is not enough
- Best NEW3 spread_filter row: threshold=0.55, net_pnl=-321405.00, active_event_rate=0.9161, gate_params={"spread_ticks_min": 1.5}
- NEW3 still stays negative under spread-only gating, so spread width alone cannot block trend toxicity.

## 5) Why combined_filter works better
- Best NEW3 combined_filter row: threshold=0.55, net_pnl=72520.00, active_event_rate=0.8257, gate_params={"trend_ticks": 20.0, "vol_threshold": 0.20974979111878053, "spread_ticks_min": 2.0}
- Combined filter explicitly includes trend and volatility terms, so it suppresses strong-trend windows before quoting.

## 6) Causal pre-trade identification rule (candidate)
- Compute rolling return in ticks over N events (example N=2000).
- If abs(rolling_return_ticks) > threshold, set DO_NOT_TRADE or ONE_SIDED_ONLY.
- This is a causal gate because it only uses past and current book states.

## 7) Limit of current evidence
- Current conclusion is based on fixed datasets (NEW vs NEW3).
- We still need walk-forward across additional months/contracts to validate if this gate-selection logic generalizes.
- Current suitability hard thresholds are over-strict in this scaling (both NEW3 and NEW test become UNSUITABLE), so that label should be treated as diagnostic only.
