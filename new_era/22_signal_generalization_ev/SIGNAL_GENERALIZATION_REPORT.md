# 022 Signal Generalization & Quote-EV Modeling Report

The current 20260127 test set is treated as burned for diagnosis. It is not used to tune thresholds, model families, feature sets, or final claims.

## 1. Regime Shift

- volatility: PSI `0.1117`, val mean `0.163476`, diagnostic test mean `0.152242`.
- score_016: PSI `0.0811`, val mean `0.00971438`, diagnostic test mean `0.0117628`.
- ev_score_021: PSI `0.0474`, val mean `197.411`, diagnostic test mean `197.075`.
- event_rate_hz: PSI `0.0341`, val mean `6.26251e+08`, diagnostic test mean `7.32296e+08`.
- score_015: PSI `0.0145`, val mean `0.226695`, diagnostic test mean `0.244838`.

The 021 selected region had validation avg stress quote PnL `-0.318021` and diagnostic test avg `-0.101912`.
Answer to the key diagnostic question: no, this specific 021 region was not profitable in validation and then unprofitable only on the burned diagnostic test. It was already negative under the moderate stress proxy on validation, then remained negative on diagnostic test with a different magnitude.

## 2. Score Bucket Transfer

- 015 bucket 10 stress avg PnL: val `-1.745048`, diagnostic test `-0.973786`.
- 016_v2 bucket 10 stress avg PnL: val `-2.331508`, diagnostic test `-1.068672`.
- EV_s2_lam5_thr0 bucket 10 stress avg PnL: val `-2.429311`, diagnostic test `-1.259441`.

High-risk buckets preserve their broad economic meaning as loss-heavy quote regions, but the toxic rates and stress-adjusted PnL magnitudes do not transfer exactly from validation to the burned diagnostic test. This is partial transfer, not a clean generalization proof.

## 3. Quote-EV Labels

Quote-EV labels are simulator-derived proxy labels, not true exchange-level queue-aware PnL. They use deterministic moderate-stress proxy fills and future mid-price information.

## 4. Quote-EV Models

Best validation-selected Quote-EV model policy: `logistic_sgd` / `spread_only` with validation proxy PnL `395195.000`.
Simple HLS-friendly baselines are included: spread_only, spread+queue_imbalance, spread+event_rate, and spread+cancel_pressure.
The best validation policy coming from `spread_only` is an important warning sign: the richer signal set did not yet demonstrate a robust EV advantage over a very simple HLS-friendly baseline.

## 5. Walk-Forward

Walk-forward frozen test folds: `3`; positive folds `2`; mean PnL `112478.333`.
This is encouraging but mixed: one pre-burned fold is negative, so the evidence supports continued diagnosis rather than a production-style profitability claim.

## 6. Limiting Factor Diagnosis

- Regime shift: present but moderate. The largest shifts are volatility, 016 score, 021 EV score, event rate, and 015 score.
- Score bucket non-transfer: partial. High-risk buckets remain economically bad, but toxic rates and loss magnitudes change across validation and burned diagnostic test.
- Label instability: unresolved. Quote-EV labels are deterministic simulator proxies with sparse positive-EV events, not exchange-confirmed queue outcomes.
- Feature insufficiency: likely. The validation winner is `spread_only`, while richer feature sets do not yet show a stable incremental EV edge.
- Policy overfit: still a risk. The current test is burned, and walk-forward consistency is positive in 2 of 3 pre-burned folds, not unanimous.
- Missing true queue data: the central unresolved limitation. The proxy EV label improves objective alignment, but it still lacks exchange-confirmed queue position, true aggressor side, and true fill priority.

## 7. Conservative Conclusion

022 diagnoses signal generalization rather than declaring a production-ready policy. Final claims should rely on walk-forward consistency across fresh pre-test splits, not current burned test performance.
