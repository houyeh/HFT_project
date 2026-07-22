# TCN PrePad Causal-Fill Report

Same proxy family as PHC-lite: H20 exit, conservative L2 fill, inventory cap 1. This is not true exchange replay. NEW5 is not used.

| split | policy_id | variant | seed | horizon | fill_model | inventory_cap | sequential_excess_pnl | fill_count | forced_no_quote_count | max_abs_inventory | max_drawdown | win_rate | loss_rate | avg_trade_excess_pnl | fill_uses_future_pnl | new5_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | tcn_prepad_scratch_16feat | scratch | 2 | 20 | conservative_l2_stress | 1 | 172195 | 6547 | 0 | 1 | 2965 | 0.463877 | 0.536123 | 26.3014 | False | False |
| new3 | tcn_prepad_scratch_16feat | scratch | 2 | 20 | conservative_l2_stress | 1 | 565260 | 17476 | 0 | 1 | 4635 | 0.498856 | 0.501144 | 32.3449 | False | False |
| new4 | tcn_prepad_scratch_16feat | scratch | 2 | 20 | conservative_l2_stress | 1 | 126915 | 4080 | 2547 | 1 | 2080 | 0.485784 | 0.514216 | 31.1066 | False | False |
| val | tcn_prepad_distilled_16feat | distilled | 1 | 20 | conservative_l2_stress | 1 | 173650 | 6470 | 0 | 1 | 2290 | 0.464606 | 0.535394 | 26.8393 | False | False |
| new3 | tcn_prepad_distilled_16feat | distilled | 1 | 20 | conservative_l2_stress | 1 | 587795 | 16847 | 0 | 1 | 3220 | 0.505075 | 0.494925 | 34.8902 | False | False |
| new4 | tcn_prepad_distilled_16feat | distilled | 1 | 20 | conservative_l2_stress | 1 | 133720 | 4033 | 2370 | 1 | 1730 | 0.492437 | 0.507563 | 33.1565 | False | False |
