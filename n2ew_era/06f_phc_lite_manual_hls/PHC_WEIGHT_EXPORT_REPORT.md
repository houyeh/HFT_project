# PHC Weight Export Report

Final quantization saturation status: `PASS`.

## Source

- Source checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt`.
- Step checkpoint loaded directly: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06e_phc_stp_step_hls4ml\models\phc_stp_step_hls_seed2.pt`.
- Exported header: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\hls_src\phc_weights.hpp`.
- No training, no threshold selection, and no NEW5 access were performed.
- `alpha_scalar` confirmed as raw alpha: `0.591699958`. It is not `sigmoid(alpha)`.

## Quantization Scheme

`Q24.12 == ap_fixed<24,12>` is reused exactly from Phase 06B. The 06B fixed-point equivalence report showed action agreement around `0.998`, selected-action agreement around `0.998`, and max absolute causal-fill PnL delta `2,455` versus multi-million FP32 baseline PnL; material degradation was `False`.

Quantization rule:

```text
q_int = round(float_value * 2^12)
q_int = clip(q_int, -2^23, 2^23 - 1)
ap_fixed_value = q_int / 2^12
```

## Routing Tensor Decision

`input_x`, `input_state`, and `output_pack` are fixed routing/identity matrices introduced only to make the 06E step graph frontend-friendly. They are audited below but intentionally omitted from `phc_weights.hpp`; the HLS datapath should implement them as direct input slicing, state passthrough, and output packing. This avoids wasting multipliers or storage on identity matrices.

## Exported Tensor Audit

| phase   | name                        | cpp_name               | shape   |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | omitted_direct_hls_plumbing   | note   |
|:--------|:----------------------------|:-----------------------|:--------|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:------------------------------|:-------|
| 06f     | decay                       | DECAY                  | 16      |           16 |   0.89401   |   0.903637  |            3662 |            3701 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | alpha_scalar                | ALPHA_SCALAR           | 1       |            1 |   0.5917    |   0.5917    |            2424 |            2424 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | enc_linear.weight           | ENC_LINEAR_W           | 16x16   |          256 |  -0.893887  |   0.810118  |           -3661 |            3318 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | enc_linear.bias             | ENC_LINEAR_B           | 16      |           16 |  -0.238208  |   0.241288  |            -976 |             988 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | enc_norm.weight             | ENC_NORM_GAMMA         | 16      |           16 |   0.941397  |   1.56719   |            3856 |            6419 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | enc_norm.bias               | ENC_NORM_BETA          | 16      |           16 |  -0.0459466 |   0.231621  |            -188 |             949 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | gate.weight                 | GATE_W                 | 16x16   |          256 |  -0.654347  |   0.566356  |           -2680 |            2320 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | gate.bias                   | GATE_B                 | 16      |           16 |  -0.219386  |   0.279102  |            -899 |            1143 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | candidate.weight            | CANDIDATE_W            | 16x16   |          256 |  -0.370782  |   0.402549  |           -1519 |            1649 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | candidate.bias              | CANDIDATE_B            | 16      |           16 |  -0.249336  |   0.236309  |           -1021 |             968 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | gain_from_encoded.weight    | GAIN_FROM_ENCODED_W    | 16x16   |          256 |  -0.850529  |   0.739105  |           -3484 |            3027 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | gain_from_encoded.bias      | GAIN_FROM_ENCODED_B    | 16      |           16 |  -0.28649   |   0.22475   |           -1173 |             921 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | gain_from_prev_state.weight | GAIN_FROM_PREV_STATE_W | 16x16   |          256 |  -0.430176  |   0.460959  |           -1762 |            1888 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | synapse.weight              | SYNAPSE_W              | 16x16   |          256 |  -0.350872  |   0.405936  |           -1437 |            1663 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | synapse.bias                | SYNAPSE_B              | 16      |           16 |  -0.199505  |   0.247485  |            -817 |            1014 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | loop_norm.weight            | LOOP_NORM_GAMMA        | 16      |           16 |   0.941701  |   1.03835   |            3857 |            4253 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | loop_norm.bias              | LOOP_NORM_BETA         | 16      |           16 |  -0.0315271 |   0.0372738 |            -129 |             153 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | readout.weight              | READOUT_W              | 2x16    |           32 |  -0.336822  |   0.348035  |           -1380 |            1426 |               0 |           24 |          12 | True                 | False                         |        |
| 06f     | readout.bias                | READOUT_B              | 2       |            2 |  -0.201776  |   0.0143739 |            -826 |              59 |               0 |           24 |          12 | True                 | False                         |        |

## Omitted Routing Tensor Audit

| phase   | name               | cpp_name   | shape   |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | omitted_direct_hls_plumbing   | note                                                                  |
|:--------|:-------------------|:-----------|:--------|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:------------------------------|:----------------------------------------------------------------------|
| 06f     | input_x.weight     |            | 16x32   |          512 |    0        |   1         |               0 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | input_state.weight |            | 16x32   |          512 |    0        |   1         |               0 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | output_pack.weight |            | 18x16   |          288 |   -0.336822 |   1         |           -1380 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | output_pack.bias   |            | 18      |           18 |   -0.201776 |   0.0143739 |            -826 |              59 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |

## Full Audit Table

| phase   | name                        | cpp_name               | shape   |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | omitted_direct_hls_plumbing   | note                                                                  |
|:--------|:----------------------------|:-----------------------|:--------|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:------------------------------|:----------------------------------------------------------------------|
| 06f     | decay                       | DECAY                  | 16      |           16 |   0.89401   |   0.903637  |            3662 |            3701 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | alpha_scalar                | ALPHA_SCALAR           | 1       |            1 |   0.5917    |   0.5917    |            2424 |            2424 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | input_x.weight              |                        | 16x32   |          512 |   0         |   1         |               0 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | input_state.weight          |                        | 16x32   |          512 |   0         |   1         |               0 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | enc_linear.weight           | ENC_LINEAR_W           | 16x16   |          256 |  -0.893887  |   0.810118  |           -3661 |            3318 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | enc_linear.bias             | ENC_LINEAR_B           | 16      |           16 |  -0.238208  |   0.241288  |            -976 |             988 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | enc_norm.weight             | ENC_NORM_GAMMA         | 16      |           16 |   0.941397  |   1.56719   |            3856 |            6419 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | enc_norm.bias               | ENC_NORM_BETA          | 16      |           16 |  -0.0459466 |   0.231621  |            -188 |             949 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | gate.weight                 | GATE_W                 | 16x16   |          256 |  -0.654347  |   0.566356  |           -2680 |            2320 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | gate.bias                   | GATE_B                 | 16      |           16 |  -0.219386  |   0.279102  |            -899 |            1143 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | candidate.weight            | CANDIDATE_W            | 16x16   |          256 |  -0.370782  |   0.402549  |           -1519 |            1649 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | candidate.bias              | CANDIDATE_B            | 16      |           16 |  -0.249336  |   0.236309  |           -1021 |             968 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | gain_from_encoded.weight    | GAIN_FROM_ENCODED_W    | 16x16   |          256 |  -0.850529  |   0.739105  |           -3484 |            3027 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | gain_from_encoded.bias      | GAIN_FROM_ENCODED_B    | 16      |           16 |  -0.28649   |   0.22475   |           -1173 |             921 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | gain_from_prev_state.weight | GAIN_FROM_PREV_STATE_W | 16x16   |          256 |  -0.430176  |   0.460959  |           -1762 |            1888 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | synapse.weight              | SYNAPSE_W              | 16x16   |          256 |  -0.350872  |   0.405936  |           -1437 |            1663 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | synapse.bias                | SYNAPSE_B              | 16      |           16 |  -0.199505  |   0.247485  |            -817 |            1014 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | loop_norm.weight            | LOOP_NORM_GAMMA        | 16      |           16 |   0.941701  |   1.03835   |            3857 |            4253 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | loop_norm.bias              | LOOP_NORM_BETA         | 16      |           16 |  -0.0315271 |   0.0372738 |            -129 |             153 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | readout.weight              | READOUT_W              | 2x16    |           32 |  -0.336822  |   0.348035  |           -1380 |            1426 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | readout.bias                | READOUT_B              | 2       |            2 |  -0.201776  |   0.0143739 |            -826 |              59 |               0 |           24 |          12 | True                 | False                         |                                                                       |
| 06f     | output_pack.weight          |                        | 18x16   |          288 |  -0.336822  |   1         |           -1380 |            4096 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |
| 06f     | output_pack.bias            |                        | 18      |           18 |  -0.201776  |   0.0143739 |            -826 |              59 |               0 |           24 |          12 | False                | True                          | Implemented directly as slicing/state/output packing in C++ datapath. |

## Saturation Decision

- Total saturated values: `0`.
- PASS/FAIL: `PASS`.

No tensor values saturated under Q24.12. The manual HLS weight export is ready for the next C++ datapath implementation step.

## Evidence Links

- Phase 06E step equivalence report: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06e_phc_stp_step_hls4ml\PHC_STP_STEP_HLS4ML_FINAL_REPORT.md`.
- Phase 06B fixed-point evidence: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06b_phc_lite_streaming_fixed_point_equivalence\PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md`.

This phase exports constants only. It does not prove FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.
