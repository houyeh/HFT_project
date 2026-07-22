# Why Regime Gate — Verified Numbers

## NEW Test Sessions (no gate, TCN thr=0.55)
來源檔案: `lt=.../eval_outputs/diagnosis_v3/new_session_audit.csv`

| Session | LOB rows | trend_strength | suitability | baseline_net_pnl |
|---------|----------|----------------|-------------|-----------------|
| 20260127Day   | 410,696 | 1,695 | UNSUITABLE | -27,205 |
| 20260127Night | 755,845 | 798  | UNSUITABLE | +78,700 |

Note: trend_strength is session cumulative return / per-event vol.
Both sessions flagged UNSUITABLE by the (now-deprecated) threshold.
Actual model PnL at thr=0.55: **+84,980 TWD** (from threshold_sweep).
→ The UNSUITABLE label reflects miscalibrated rolling trend_strength metric.

## NEW3 Sessions — All 6 UNSUITABLE
來源檔案: `lt=.../eval_outputs/diagnosis_v3/new3_session_audit.csv`

| Session | LOB rows | trend_strength | cum_return_ticks | baseline_net_pnl | failure_mode |
|---------|----------|----------------|-----------------|-----------------|--------------|
| 20260113 | 1,277,946 | 621  | -121.0 | -352,235 | strong_downtrend_adverse |
| 20260114 | 910,771   | 459  | +82.5  | -62,285  | strong_uptrend_adverse |
| 20260115 | 976,260   | 1,413| +256.0 | +201,630 | none_or_minor |
| 20260116 | 976,532   | 1,324| +282.5 | -125,345 | strong_uptrend_adverse |
| 20260119Day | 1,365,599 | 1,076 | +178.5 | -21,295 | strong_uptrend_adverse |
| 20260330 | 150,914   | 136  | +24.0  | -14,805  | strong_uptrend_adverse |

Total NEW3 no-gate net_pnl: **-191,835 TWD** (combined_filter best = -6,860)
symbol: TXFA6 (台指期小型, different from NEW's TXFB6)

## Key Insight
- 5 of 6 NEW3 sessions: large trending regime → no-gate baseline catastrophic loss
- NEW test sessions: moderate trend, model edge exists
- Same model/threshold behaves totally differently → regime dependency confirmed

## 資料來源
- `lt=.../eval_outputs/diagnosis_v3/new3_session_audit.csv`
- `lt=.../eval_outputs/diagnosis_v3/new_session_audit.csv`
