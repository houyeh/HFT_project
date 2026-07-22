# Readout Dynamic Range Report

Input file: `n2ew_era/06f_phc_lite_manual_hls/results/phc_step_test_vectors.csv`.

Scope: reference PyTorch float32 `ref_readout_0` and `ref_readout_1` only. This report does not use C-Sim/Cosim DUT outputs and does not rerun simulation.

Comparison error reference: `max_abs_diff = 0.00725275` from the existing HLS C-Sim/Cosim evidence.

| channel       |      min |      max |      mean |      std |        p1 |        p50 |      p99 |   max_abs_diff_over_std_pct |   max_abs_diff_over_range_pct |
|:--------------|---------:|---------:|----------:|---------:|----------:|-----------:|---------:|----------------------------:|------------------------------:|
| ref_readout_0 | -1.27031 | 1.2164   | -0.02517  | 0.278577 | -0.654102 | -0.0214849 | 0.579323 |                      2.6035 |                      0.29166  |
| ref_readout_1 | -1.34891 | 0.921237 | -0.128431 | 0.27639  | -0.747142 | -0.130738  | 0.473843 |                      2.6241 |                      0.319483 |

Interpretation:
- `ref_readout_0`: max_abs_diff is 2.603% of readout std and 0.292% of the observed [min,max] range.
- `ref_readout_1`: max_abs_diff is 2.624% of readout std and 0.319% of the observed [min,max] range.

This supports treating the observed fixed-point approximation error as small relative to the reference readout dynamic range. It does not by itself prove policy-level PnL preservation; that remains a separate fixed-point policy-equivalence question.
