# Multi-Seed Stability

Seeds evaluated: `0,1,2`.
Epochs per seed: `3`. Train stride: `10`.

This is a compute-constrained stability pass. Test was not evaluated in this script.

## Validation PR-AUC Summary

- ask: mean PR-AUC `0.009536`, std `0.000107`, min `0.009408`, max `0.009669`.
- bid: mean PR-AUC `0.008534`, std `0.000465`, min `0.007907`, max `0.009019`.

The signal is considered seed-stable only if validation PR-AUC remains consistently above the split positive rate across seeds.