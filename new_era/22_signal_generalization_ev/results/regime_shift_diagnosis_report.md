# Regime Shift Diagnosis

The current test set is treated as a burned diagnostic set, not as a pristine holdout for tuning.

## Largest PSI Variables

- volatility: PSI `0.1117`, val mean `0.163476`, test mean `0.152242`.
- score_016: PSI `0.0811`, val mean `0.00971438`, test mean `0.0117628`.
- ev_score_021: PSI `0.0474`, val mean `197.411`, test mean `197.075`.
- event_rate_hz: PSI `0.0341`, val mean `6.26251e+08`, test mean `7.32296e+08`.
- score_015: PSI `0.0145`, val mean `0.226695`, test mean `0.244838`.

## Failed 021 Region

The EV_s2_lam5_thr0 quoted region has validation avg stress quote PnL `-0.318021` and test avg stress quote PnL `-0.101912`.
Total proxy stress quote PnL changes from `-227455` on validation quotes to `-145045` on diagnostic test quotes.