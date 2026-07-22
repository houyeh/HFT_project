# New Binary Model Report

## 1. Motivation
- Old toxic/adverse labels were either non-conditional or too sparse.
- New labels are side-specific and fill-conditional using Proxy C.

## 2. Label Definition
- bid_quote_toxic_binary / ask_quote_toxic_binary
- fill proxy: price_level_removed (LOB proxy, not true fill)
- toxic threshold: markout_ticks < -0.5

## 3. Label Distribution
### bid
- train: samples=15643, toxic_rate=0.035863, fill_proxy_rate=0.021770
- val: samples=12603, toxic_rate=0.032294, fill_proxy_rate=0.022366
- test: samples=24329, toxic_rate=0.027704, fill_proxy_rate=0.020860
### ask
- train: samples=15856, toxic_rate=0.030777, fill_proxy_rate=0.022066
- val: samples=13314, toxic_rate=0.026889, fill_proxy_rate=0.023628
- test: samples=25037, toxic_rate=0.026641, fill_proxy_rate=0.021467

## 4. Training Convergence
- bid: best_epoch=28, best_val_loss=1.003543, val_pr_auc=0.126072, test_pr_auc=0.072409
- ask: best_epoch=30, best_val_loss=0.983688, val_pr_auc=0.108589, test_pr_auc=0.080254

## 5. Classification Results
- See outputs/models/*/classification_report_test.csv and curve PNGs.

## 6. Threshold Diagnostics
- See outputs/models/*/threshold_diagnostics_test.csv.

## 7. Trading Backtest
- Best NEW test binary setup: bid=0.40,ask=0.20, net_pnl=266045.00, random+1std=272035.83, smoothed_ref=84980.00

## 8. OOS / NEW3 Stress Test
- NEW3 net_pnl at best NEW setup: 229385.00

## 9. Verdict
- C. New binary label converges but no trading edge.

## 10. Next Steps
- keep smoothed_mid_move_3class as main baseline
- focus on regime gate / rule-based FPGA MVP