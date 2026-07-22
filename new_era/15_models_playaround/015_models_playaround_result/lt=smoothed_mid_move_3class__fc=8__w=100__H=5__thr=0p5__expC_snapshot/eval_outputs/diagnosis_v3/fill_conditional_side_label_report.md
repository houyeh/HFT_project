# 1. Executive Summary

Original unconditional markout toxic rates were too low because they were not fill-conditional; this audit uses LOB-based fill proxies to estimate conditional toxicity.

# 2. Fill Evidence Audit

- Trade tape exists.
- aggressor_side is missing.
- true exchange-level fill direction is unavailable.
- LOB 5-level depth is available.
- all labels in this report are proxy-based, not true fills.

# 3. Proxy Definitions and Limitations

## queue_depletion_proxy
- Advantage: uses volume changes and reflects queue dynamics.
- Limitation: volume drop can be fill or cancel.

## price_level_removed_proxy
- Advantage: simple and stable from LOB only.
- Limitation: removed level can be fill or cancel.
- Note: treated as primary candidate proxy, not true exchange fill.

## price_through_proxy
- Advantage: conservative and less likely to over-call fills.
- Limitation: can severely under-count fills (rare proxy).

# 4. Conditional Toxic Rate Results

| dataset | split | proxy_type | fill_window | bid_fill_count | bid_toxic_rate_conditional | ask_fill_count | ask_toxic_rate_conditional | bid_toxic_rate_lift | ask_toxic_rate_lift |
|---|---|---|---|---|---|---|---|---|---|
| NEW | NEW_train_val | queue_depletion_proxy | 1 | 6729 | 0.003418 | 7086 | 0.003810 | 0.898724 | 1.055831 |
| NEW | NEW_train_val | price_level_removed_proxy | 1 | 9164 | 0.032519 | 9349 | 0.033266 | 8.550279 | 9.217795 |
| NEW | NEW_train_val | price_through_proxy | 1 | 86 | 0.127907 | 93 | 0.150538 | 33.631275 | 41.713525 |
| NEW | NEW_train_val | queue_depletion_proxy | 3 | 32182 | 0.002206 | 33468 | 0.002540 | 0.580089 | 0.703754 |
| NEW | NEW_train_val | price_level_removed_proxy | 3 | 25903 | 0.031657 | 26487 | 0.028882 | 8.323632 | 8.003141 |
| NEW | NEW_train_val | price_through_proxy | 3 | 660 | 0.296970 | 697 | 0.268293 | 78.083852 | 74.343094 |
| NEW | NEW_train_val | queue_depletion_proxy | 5 | 52826 | 0.001912 | 54993 | 0.002055 | 0.502716 | 0.569381 |
| NEW | NEW_train_val | price_level_removed_proxy | 5 | 41283 | 0.027929 | 42441 | 0.025400 | 7.343569 | 7.038254 |
| NEW | NEW_train_val | price_through_proxy | 5 | 1463 | 0.313055 | 1527 | 0.289456 | 82.313344 | 80.207510 |
| NEW | NEW_test | queue_depletion_proxy | 1 | 2195 | 0.000911 | 2329 | 0.002576 | 0.322286 | 0.894947 |
| NEW | NEW_test | price_level_removed_proxy | 1 | 2252 | 0.034192 | 2303 | 0.031264 | 12.093945 | 10.860608 |
| NEW | NEW_test | price_through_proxy | 1 | 38 | 0.157895 | 52 | 0.038462 | 55.848728 | 13.361101 |
| NEW | NEW_test | queue_depletion_proxy | 3 | 8992 | 0.001446 | 9363 | 0.001922 | 0.511367 | 0.667841 |
| NEW | NEW_test | price_level_removed_proxy | 3 | 6411 | 0.026049 | 6738 | 0.027753 | 9.213748 | 9.641091 |
| NEW | NEW_test | price_through_proxy | 3 | 226 | 0.212389 | 248 | 0.161290 | 75.123953 | 56.030423 |
| NEW | NEW_test | queue_depletion_proxy | 5 | 14424 | 0.001387 | 14800 | 0.001419 | 0.490445 | 0.492916 |
| NEW | NEW_test | price_level_removed_proxy | 5 | 10091 | 0.023883 | 10428 | 0.025316 | 8.447505 | 8.794649 |
| NEW | NEW_test | price_through_proxy | 5 | 488 | 0.231557 | 525 | 0.177143 | 81.903838 | 61.537413 |
| NEW3 | NEW3_full | queue_depletion_proxy | 1 | 5676 | 0.004052 | 6101 | 0.003606 | 1.260147 | 1.092918 |
| NEW3 | NEW3_full | price_level_removed_proxy | 1 | 11825 | 0.032304 | 11971 | 0.035419 | 10.046109 | 10.734986 |
| NEW3 | NEW3_full | price_through_proxy | 1 | 154 | 0.259740 | 166 | 0.283133 | 80.774624 | 85.813545 |
| NEW3 | NEW3_full | queue_depletion_proxy | 3 | 28966 | 0.002520 | 30912 | 0.002232 | 0.783736 | 0.676532 |
| NEW3 | NEW3_full | price_level_removed_proxy | 3 | 34794 | 0.027304 | 34382 | 0.027893 | 8.490923 | 8.453831 |
| NEW3 | NEW3_full | price_through_proxy | 3 | 680 | 0.313235 | 729 | 0.294925 | 97.410633 | 89.387544 |
| NEW3 | NEW3_full | queue_depletion_proxy | 5 | 51557 | 0.001998 | 54535 | 0.001852 | 0.621277 | 0.561322 |
| NEW3 | NEW3_full | price_level_removed_proxy | 5 | 55708 | 0.023677 | 55206 | 0.024128 | 7.363137 | 7.312805 |
| NEW3 | NEW3_full | price_through_proxy | 5 | 1452 | 0.326446 | 1521 | 0.303748 | 101.519016 | 92.061667 |

# 5. NEW vs NEW3 Comparison

- Mean H=10 conditional bid_toxic_rate: NEW=0.0848, NEW3=0.1101
- Mean H=10 conditional ask_toxic_rate: NEW=0.0706, NEW3=0.1085
- NEW3 shows higher bid-side toxicity than NEW in this proxy audit.
- NEW3 also shows elevated ask-side toxicity in this proxy audit.
- These comparisons are proxy-level evidence and should be interpreted with missing aggressor-side limitations.

# 6. Best Proxy / Horizon Recommendation

- Selected: proxy_type=price_level_removed_proxy, fill_window=1, horizon=10
- fill_count(bid/ask)=(9164/9349), toxic_rate(bid/ask)=(0.0325/0.0333)
- Reason: Selected with proxy_priority=1, min_toxic=0.0325, lift_score=17.7681, fill_counts=(9164,9349).
- Limitation: still proxy-based, not true exchange-level fill labels.

# 7. Label Recommendation

- Selected: A. Use fill-conditional bid_toxic / ask_toxic labels for next model.
- Why: Proxy-conditional toxicity is meaningfully lifted vs unconditional and sample counts are usable. Still keep explicit proxy uncertainty in evaluation.

# 8. Next Coding Task

- Build bid-side and ask-side binary datasets separately.
- Fit/select on train+val only, then test once.
- Keep model architecture small and compare against current TCN risk score.