# TCN PrePad OOS Evaluation Report

Evaluation uses the Phase 05A sequence-window universe and the same action-value decision rule as PHC-lite. NEW5 is not loaded.

| split | policy_id | horizon | state_count | selected_action_count | selection_rate | bid_count | ask_count | no_quote_count | total_excess_pnl_H20 | avg_selected_excess | win_rate | negative_rate | max_drawdown | positive_session_count | session_count | variant | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | no_quote | 20 | 82960 | 0 | 0 | 0 | 0 | 82960 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |  |  |
| val | side_by_queue_imbalance_abs_0p1 | 20 | 82960 | 68981 | 0.831497 | 34027 | 34954 | 13979 | 1.23714e+06 | 17.9346 | 0.426451 | 0.573491 | 18000 | 6 | 6 |  |  |
| val | phc_lite_v1_teacher | 20 | 82960 | 63766 | 0.768635 | 32177 | 31589 | 19194 | 1.92717e+06 | 30.2225 | 0.469451 | 0.530486 | 15585 | 6 | 6 |  |  |
| val | tcn_prepad_scratch_16feat | 20 | 82960 | 61837 | 0.745383 | 31561 | 30276 | 21123 | 1.92089e+06 | 31.0638 | 0.470333 | 0.529618 | 16490 | 6 | 6 | scratch | 2 |
| val | tcn_prepad_distilled_16feat | 20 | 82960 | 61114 | 0.736668 | 29955 | 31159 | 21846 | 1.97155e+06 | 32.2602 | 0.474752 | 0.525182 | 16405 | 6 | 6 | distilled | 1 |
| new3 | no_quote | 20 | 176754 | 0 | 0 | 0 | 0 | 176754 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |  |  |
| new3 | side_by_queue_imbalance_abs_0p1 | 20 | 176754 | 144745 | 0.818907 | 71697 | 73048 | 32009 | 2.95477e+06 | 20.4136 | 0.446737 | 0.553242 | 6385 | 6 | 6 |  |  |
| new3 | phc_lite_v1_teacher | 20 | 176754 | 138261 | 0.782223 | 70340 | 67921 | 38493 | 5.02152e+06 | 36.3191 | 0.501045 | 0.49894 | 4855 | 6 | 6 |  |  |
| new3 | tcn_prepad_scratch_16feat | 20 | 176754 | 137866 | 0.779988 | 69288 | 68578 | 38888 | 5.00592e+06 | 36.3101 | 0.502814 | 0.497178 | 8030 | 6 | 6 | scratch | 2 |
| new3 | tcn_prepad_distilled_16feat | 20 | 176754 | 134699 | 0.76207 | 66151 | 68548 | 42055 | 5.09123e+06 | 37.7971 | 0.506967 | 0.493025 | 5125 | 6 | 6 | distilled | 1 |
| new4 | no_quote | 20 | 68620 | 0 | 0 | 0 | 0 | 68620 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |  |  |
| new4 | side_by_queue_imbalance_abs_0p1 | 20 | 68620 | 56147 | 0.818231 | 27616 | 28531 | 12473 | 1.00391e+06 | 17.88 | 0.438777 | 0.561205 | 6835 | 4 | 4 |  |  |
| new4 | phc_lite_v1_teacher | 20 | 68620 | 53169 | 0.774832 | 27155 | 26014 | 15451 | 1.7625e+06 | 33.1489 | 0.490605 | 0.509357 | 5020 | 4 | 4 |  |  |
| new4 | tcn_prepad_scratch_16feat | 20 | 68620 | 52789 | 0.769295 | 26729 | 26060 | 15831 | 1.79528e+06 | 34.0086 | 0.494345 | 0.505636 | 6225 | 4 | 4 | scratch | 2 |
| new4 | tcn_prepad_distilled_16feat | 20 | 68620 | 51604 | 0.752026 | 25325 | 26279 | 17016 | 1.81446e+06 | 35.1611 | 0.497306 | 0.502674 | 4000 | 4 | 4 | distilled | 1 |
