# TCN PrePad Seed Stability Report

Each seed selects its own threshold on validation only, then evaluates val/NEW3/NEW4. NEW5 is not loaded.

| variant | policy_id | seed | split | best_epoch | threshold | action_pnl | selected_action_count | matched_random_95p | matched_random_pass | empirical_p_value_fraction_random_ge_model | causal_fill_pnl | causal_fill_count | causal_fill_max_drawdown | new3_used_for_selection | new4_used_for_selection | new5_used_for_selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | tcn_prepad_scratch_16feat | 0 | val | 114 | 0 | 1.885e+06 | 60624 | -866198 | True | 0 | 162745 | 6437 | 2935 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 0 | new3 | 114 | 0 | 4.90224e+06 | 135819 | -1.96957e+06 | True | 0 | 553180 | 16988 | 3240 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 0 | new4 | 114 | 0 | 1.76446e+06 | 52157 | -739334 | True | 0 | 133650 | 4211 | 2750 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 1 | val | 169 | 0 | 1.90311e+06 | 61609 | -877081 | True | 0 | 163675 | 6595 | 2295 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 1 | new3 | 169 | 0 | 4.90883e+06 | 136560 | -1.97258e+06 | True | 0 | 570605 | 17313 | 3690 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 1 | new4 | 169 | 0 | 1.73158e+06 | 52262 | -737270 | True | 0 | 127405 | 4174 | 3215 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 2 | val | 196 | 0 | 1.92089e+06 | 61837 | -877334 | True | 0 | 172195 | 6547 | 2965 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 2 | new3 | 196 | 0 | 5.00592e+06 | 137866 | -1.99283e+06 | True | 0 | 565260 | 17476 | 4635 | False | False | False |
| scratch | tcn_prepad_scratch_16feat | 2 | new4 | 196 | 0 | 1.79528e+06 | 52789 | -744470 | True | 0 | 126915 | 4080 | 2080 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 0 | val | 108 | 0 | 1.92456e+06 | 59346 | -843664 | True | 0 | 177295 | 6287 | 2425 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 0 | new3 | 108 | 0 | 5.00597e+06 | 131104 | -1.90219e+06 | True | 0 | 571050 | 16370 | 2625 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 0 | new4 | 108 | 0 | 1.79618e+06 | 49949 | -705582 | True | 0 | 131685 | 3982 | 1815 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 1 | val | 186 | 0 | 1.97155e+06 | 61114 | -872359 | True | 0 | 173650 | 6470 | 2290 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 1 | new3 | 186 | 0 | 5.09123e+06 | 134699 | -1.94658e+06 | True | 0 | 587795 | 16847 | 3220 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 1 | new4 | 186 | 0 | 1.81446e+06 | 51604 | -731411 | True | 0 | 133720 | 4033 | 1730 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 2 | val | 103 | 0 | 1.93711e+06 | 59730 | -848571 | True | 0 | 174875 | 6295 | 2720 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 2 | new3 | 103 | 0 | 5.0689e+06 | 132221 | -1.91089e+06 | True | 0 | 578075 | 16435 | 2675 | False | False | False |
| distilled | tcn_prepad_distilled_16feat | 2 | new4 | 103 | 0 | 1.79414e+06 | 50625 | -711064 | True | 0 | 125740 | 3885 | 1630 | False | False | False |
