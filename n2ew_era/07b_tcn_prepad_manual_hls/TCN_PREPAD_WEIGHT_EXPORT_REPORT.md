# TCN PrePad Weight Export Report

- Variants exported: `scratch, distilled`.
- Fixed-point scheme: `ap_fixed<24,12>` for inputs/weights and `ap_fixed<40,16>` accumulators.
- Saturated values: `0`.
- ReLU is the only nonlinearity in `TCNPrePadRegressor`; no LUT/tanh/sigmoid core is needed.

## Audit

| name | shape | float_min | float_max | quantized_min | quantized_max | num_saturated | frac_bits | total_bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch:CONV1_W | 5x16x16x3 | -1.5292392 | 1.0965601 | -6264 | 4492 | 0 | 12 | 24 |
| scratch:CONV1_B | 5x16 | -0.25179362 | 0.47226506 | -1031 | 1934 | 0 | 12 | 24 |
| scratch:CONV2_W | 5x16x16x3 | -1.8689014 | 0.74286896 | -7655 | 3043 | 0 | 12 | 24 |
| scratch:CONV2_B | 5x16 | -0.33264533 | 0.51100725 | -1363 | 2093 | 0 | 12 | 24 |
| scratch:FC_W | 2x16 | -0.25583014 | 0.24302113 | -1048 | 995 | 0 | 12 | 24 |
| scratch:FC_B | 2 | -0.16358386 | 0.12600084 | -670 | 516 | 0 | 12 | 24 |
| distilled:CONV1_W | 5x16x16x3 | -2.7821047 | 1.6175845 | -11396 | 6626 | 0 | 12 | 24 |
| distilled:CONV1_B | 5x16 | -0.62949401 | 0.52430469 | -2578 | 2148 | 0 | 12 | 24 |
| distilled:CONV2_W | 5x16x16x3 | -1.8570987 | 0.63919824 | -7607 | 2618 | 0 | 12 | 24 |
| distilled:CONV2_B | 5x16 | -0.27003583 | 0.68523198 | -1106 | 2807 | 0 | 12 | 24 |
| distilled:FC_W | 2x16 | -0.26814312 | 0.23832582 | -1098 | 976 | 0 | 12 | 24 |
| distilled:FC_B | 2 | -0.014311556 | 0.096502371 | -59 | 395 | 0 | 12 | 24 |
