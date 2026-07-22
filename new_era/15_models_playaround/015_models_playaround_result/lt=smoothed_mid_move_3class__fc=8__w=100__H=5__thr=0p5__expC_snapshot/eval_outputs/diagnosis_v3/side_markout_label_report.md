# Side Markout Label Report

## H=10 core rates
- NEW_train_val: bid_toxic_rate=0.0038, ask_toxic_rate=0.0036, bid_mean=1.2191, ask_mean=1.2147
- NEW_test: bid_toxic_rate=0.0028, ask_toxic_rate=0.0029, bid_mean=1.1230, ask_mean=1.1162
- NEW3_full: bid_toxic_rate=0.0032, ask_toxic_rate=0.0033, bid_mean=1.3413, ask_mean=1.3397

- NEW3 bid/ask toxicity is relatively balanced (skew=-0.0001).

## Horizon suitability (target toxic rate 5%-20%)
- NEW_train_val: bid-best horizon=20 (rate=0.0116), ask-best horizon=20 (rate=0.0112)
- NEW_test: bid-best horizon=20 (rate=0.0081), ask-best horizon=20 (rate=0.0080)
- NEW3_full: bid-best horizon=20 (rate=0.0098), ask-best horizon=20 (rate=0.0102)

## Label redesign implication
- Side-specific toxic labels (bid_toxic / ask_toxic) are conceptually aligned with maker risk control.
- But this current side-markout definition is too optimistic: toxic rates are only about 0.3% to 1.2%, lower than the original Down/Up base rates (~1.39% each).
- Root issue: this stat is not fill-conditional and includes spread cushion, so it under-represents true toxic fills.
- Recommendation: next phase should use fill-conditional adverse labels (or fee-adjusted adverse excursion labels), not this raw side-markout threshold directly.
