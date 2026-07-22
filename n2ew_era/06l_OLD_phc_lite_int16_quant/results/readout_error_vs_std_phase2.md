# Phase 2 Q16.8 Readout Error vs Dynamic Range

Scope: PHC-lite HLS Phase 2 quantization experiment in `n2ew_era/06l_phc_lite_int16_quant/`.

This report uses the copied 25,600-row reference vector file:

`n2ew_era/06l_phc_lite_int16_quant/results/phc_step_test_vectors.csv`

The input vector file is byte-identical to the Phase 06F vector file:

`MD5 = F7130075A7479F2099B6FD2857135F92`

## Reference Readout Dynamic Range

Only the float32 reference readout columns are summarized here: `ref_readout_0` and `ref_readout_1`. State outputs are excluded from this dynamic-range table.

| channel | min | max | mean | std | p1 | p50 | p99 | range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_readout_0 | -1.270310 | 1.216404 | -0.025170 | 0.278582 | -0.654102 | -0.021485 | 0.579323 | 2.486714 |
| ref_readout_1 | -1.348915 | 0.921237 | -0.128431 | 0.276395 | -0.747142 | -0.130738 | 0.473843 | 2.270152 |

## Error Comparison

Phase 06F baseline used Q24.12 and had existing C-Sim/Cosim evidence:

`total=25600, mismatches=0, max_abs_diff=0.00725275, mean_abs_diff=0.000372553`

Phase 06L uses Q16.8 for external/storage fixed-point values while keeping the wider accumulator types unchanged. The Phase 06L C-Sim result is:

`total=25600, mismatches=0, max_abs_diff=0.11749, mean_abs_diff=0.00589889`

The Phase 06L testbench summary `mean_abs_diff` is averaged over all reported outputs. For readout-only columns, the observed means are `0.0148517` on channel 0 and `0.00574521` on channel 1.

| metric | 06F Q24.12 baseline | 06L Q16.8 phase2 | phase2 / baseline |
|---|---:|---:|---:|
| total rows | 25,600 | 25,600 | 1.00x |
| mismatches | 0 | 0 | n/a |
| global max_abs_diff | 0.00725275 | 0.11749 | 16.20x |
| global mean_abs_diff | 0.000372553 | 0.00589889 | 15.83x |

## Error Relative To Readout Scale

| channel | 06F max_abs_diff / std | 06L global max_abs_diff / std | 06F max_abs_diff / range | 06L global max_abs_diff / range |
|---|---:|---:|---:|---:|
| ref_readout_0 | 2.60% | 42.17% | 0.292% | 4.725% |
| ref_readout_1 | 2.62% | 42.51% | 0.319% | 5.175% |

Readout-only Phase 06L errors are smaller than the global max, but still materially larger than the 06F baseline:

| channel | readout-only max_abs_diff | readout-only mean_abs_diff | readout-only max / std |
|---|---:|---:|---:|
| readout_0 | 0.081710 | 0.014852 | 29.33% |
| readout_1 | 0.056871 | 0.005745 | 20.58% |

## Interpretation

Phase 06L Q16.8 passes the current C-Sim tolerance with zero mismatches, but the numerical error is much larger than the Phase 06F Q24.12 baseline. The global max error rises from `0.00725275` to `0.11749`, or about `16.20x` larger. Relative to readout standard deviation, the global max error is about `42%` of one readout standard deviation.

Conclusion: Q16.8 is a useful stress experiment, but this evidence does not support replacing the 06F Q24.12 baseline. Any future Q16.8 path would need policy-level and cosim-level equivalence evidence before it can be considered safe.
