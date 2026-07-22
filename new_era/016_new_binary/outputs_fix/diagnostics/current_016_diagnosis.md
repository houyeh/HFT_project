# Current 016 Diagnosis

## 1) Why trade_count is only 77
- Best setup used `threshold_bid=0.40`, `threshold_ask=0.20` with `signal_rate_bid=0.255`, `signal_rate_ask=0.875`.
- This is a heavily asymmetric pull policy (ask side pulled most of the time), so the strategy participates in fewer complete round trips.

## 2) Bid/Ask signal rates at selected threshold
- bid_signal_rate=0.255478, ask_signal_rate=0.874696

## 3) Threshold selection source
- Original `binary_best` was selected from NEW test backtest sweep (not val-only), so it is not acceptable as final trading evidence.

## 4) Does PnL come from selectivity vs mostly not trading?
- NEW net_pnl=266,045, trade_count=77. Result is selective and sparse; cannot be interpreted as robust HFT operating point yet.

## 5) Random+1std check
- random_mean=251,104.75, random_std=20,931.08, random_mean+1std=272,035.83, beats=False.

## 6) PR-AUC vs base rate
- bid: val_pr_auc=0.128511, test_pr_auc=0.072409; ask: val_pr_auc=0.108589, test_pr_auc=0.080254.
- Both sides are above base toxic rate (~3%), so classifier quality is non-trivial.

## 7) Val-to-test degradation
- bid drop: +0.056102; ask drop: +0.028334.
- Degradation exists; threshold governance must be val-only.

## 8) Current status
- Verdict: **not_ready** (label pipeline PoC, but trading edge not significant vs random+1std).