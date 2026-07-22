# ML Veto Rejection Report

The residual Quote-EV veto is not accepted as part of the final architecture.

Rejection rule: the optional ML veto must improve the frozen non-ML baseline and beat the matched random veto 95th-percentile hurdle on both frozen test and NEW4 external OOS.

- `test`: baseline `2,312,590`, ML after-veto `2,312,590`, incremental `0`, random 95p `2,312,590`, beats random 95p `False`, improves baseline `False`.
- `new4`: baseline `6,348,770`, ML after-veto `6,348,770`, incremental `0`, random 95p `6,348,770`, beats random 95p `False`, improves baseline `False`.

Interpretation: the non-ML spread/side controller remains the core architecture. ML can be revisited only after it demonstrates incremental value over that robust baseline, not merely over naive random controls.
