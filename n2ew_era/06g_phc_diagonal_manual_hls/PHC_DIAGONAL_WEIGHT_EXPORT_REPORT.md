# PHC Diagonal SSM Weight Export Report

Final quantization saturation status: `PASS`.

## Source

- Source checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05f_phc_diagonal_ssm\models\phc_diagonal_ssm_seed2_best_val.pt`.
- Selected policy JSON: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05f_phc_diagonal_ssm\results\phc_diagonal_ssm_selected_policy.json`.
- Selected seed: `2`.
- Selected best epoch: `157`.
- Exported header: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\hls_src\phc_diag_weights.hpp`.
- No training, no threshold selection, and no NEW5 access were performed.

## Quantization Scheme

`Q24.12 == ap_fixed<24,12>` is reused exactly from Phase 06B/06F for comparability.

```text
q_int = round(float_value * 2^12)
q_int = clip(q_int, -2^23, 2^23 - 1)
ap_fixed_value = q_int / 2^12
```

## Diagonal Recurrence Export Decision

`DIAG_DECAY` and `DIAG_GAIN` are exported as `sigmoid(raw_param)` constants. This removes runtime sigmoid operations from the HLS recurrence stage. The raw parameters are still audited below but are intentionally not exported to the HLS header.

There are no gate/candidate/synapse/gain-from-encoded/gain-from-state tensors in PHCDiagonalSSM, so none are exported or invented.

## Exported Tensor Audit

| phase   | name                     | cpp_name       | shape   |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | note                                 |
|:--------|:-------------------------|:---------------|:--------|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:-------------------------------------|
| 06g     | encoder.0.weight         | ENC_LINEAR_W   | 16x16   |          256 |  -1.02133   |    0.807634 |           -4183 |            3308 |               0 |           24 |          12 | True                 |                                      |
| 06g     | encoder.0.bias           | ENC_LINEAR_B   | 16      |           16 |  -0.606813  |    0.303344 |           -2486 |            1242 |               0 |           24 |          12 | True                 |                                      |
| 06g     | encoder.1.weight         | ENC_NORM_GAMMA | 16      |           16 |   0.785521  |    0.988402 |            3217 |            4048 |               0 |           24 |          12 | True                 |                                      |
| 06g     | encoder.1.bias           | ENC_NORM_BETA  | 16      |           16 |  -0.0621345 |    0.191721 |            -255 |             785 |               0 |           24 |          12 | True                 |                                      |
| 06g     | diag_decay_baked_sigmoid | DIAG_DECAY     | 16      |           16 |   0.822606  |    0.918752 |            3369 |            3763 |               0 |           24 |          12 | True                 | sigmoid baked into exported constant |
| 06g     | diag_gain_baked_sigmoid  | DIAG_GAIN      | 16      |           16 |   0.474065  |    0.518719 |            1942 |            2125 |               0 |           24 |          12 | True                 | sigmoid baked into exported constant |
| 06g     | readout.weight           | READOUT_W      | 2x16    |           32 |  -0.256461  |    0.263358 |           -1050 |            1079 |               0 |           24 |          12 | True                 |                                      |
| 06g     | readout.bias             | READOUT_B      | 2       |            2 |   0.011036  |    0.215284 |              45 |             882 |               0 |           24 |          12 | True                 |                                      |

## Raw Parameter Audit

| phase   | name             | cpp_name   |   shape |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | note                                                    |
|:--------|:-----------------|:-----------|--------:|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:--------------------------------------------------------|
| 06g     | diag_decay_param |            |      16 |           16 |    1.5341   |   2.42551   |            6284 |            9935 |               0 |           24 |          12 | False                | raw parameter; sigmoid is baked into DIAG_DECAY for HLS |
| 06g     | diag_gain_param  |            |      16 |           16 |   -0.103834 |   0.0749105 |            -425 |             307 |               0 |           24 |          12 | False                | raw parameter; sigmoid is baked into DIAG_GAIN for HLS  |

## Saturation Decision

- Total saturated values: `0`.
- PASS/FAIL: `PASS`.
No tensor values saturated under Q24.12.

## Evidence Links

- Phase 05F report: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05f_phc_diagonal_ssm\PHC_DIAGONAL_SSM_ABLATION_REPORT.md`.
- Phase 06B fixed-point evidence: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06b_phc_lite_streaming_fixed_point_equivalence\PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md`.

This phase exports constants only. It does not prove FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.
