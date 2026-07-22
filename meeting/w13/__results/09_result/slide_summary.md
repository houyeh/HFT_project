# Fill-Conditional Label Audit

- Using proxy C (`price_level_removed_proxy`) at horizon=10, fill_window=1.
- Conditional toxic rate lift vs unconditional is substantial (avg bid lift=10.23x, avg ask lift=10.27x).
- Conditional toxic rate remains in low single-digit percent for proxy C rows.
- No aggressor-side data: this is proxy evidence, not true exchange fill truth.
- Result motivates bid_toxic / ask_toxic retraining research.
