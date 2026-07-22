# Val Latency Fill Probability Equivalence Check

Scope: val split only, selected decisions only. This does not run PnL, matched random, or NEW5.

| metric | value |
|---|---:|
| selected_action_count | 468627 |
| different_probability_count | 427 |
| different_probability_rate | 0.09111724% |
| max_abs_probability_diff | 0.257575757576 |
| mean_abs_probability_diff | 4.48098424718e-05 |
| diff_epsilon | 1.0e-12 |

Conclusion: The current proxy cannot meaningfully distinguish CPU and FPGA latency at this latency/data-resolution scale. Running n=1000 matched-random trials will not change that conclusion unless the latency scale or data resolution changes.

If the probability vectors are effectively identical, a full n=1000 matched-random run would mainly confirm the same result at much higher compute cost.
