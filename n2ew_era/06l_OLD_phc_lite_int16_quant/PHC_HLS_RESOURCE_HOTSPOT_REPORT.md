# PHC HLS Resource Hotspot Report

Status: analysis_only_no_hls_source_edits

## Claim Boundary

This report follows `n2ew_era/00_project_reset_and_data_contract/ALLOWED_CLAIMS.md` and `FORBIDDEN_CLAIMS.md`.

Allowed framing: this is an HLS resource analysis of a quote-control primitive under research proxy assumptions.

Forbidden framing: no real-world profitability claim, no true exchange replay claim, no FPGA PnL improvement claim, and no production-readiness claim.

## Inputs Compared

| item | Q24.12 baseline | Q16.8 INT16 experiment |
| --- | --- | --- |
| Phase folder | `n2ew_era/06f_phc_lite_manual_hls/` | `n2ew_era/06l_phc_lite_int16_quant/` |
| csynth report | `hls_component/hls_component/hls/syn/report/csynth.rpt` | `hls_component/hls_component/hls/syn/report/csynth.rpt` |
| Report date | Mon Jun 15 02:25:12 2026 | Sun Jun 21 15:05:33 2026 |
| Vitis version | 2025.2 build 6295257 | 2025.2 build 6295257 |
| Top function | `phc_step` | `phc_step` |
| Device | `xc7z020-clg400-1` | `xc7z020-clg400-1` |

Version caveat: these reports are comparable because the Vitis version, device, top function, and copied source lineage match, but they are not the same synthesis run. The Q24.12 report is from 2026-06-15; the Q16.8 report is from 2026-06-21.

06L intentionally changed external/storage fixed point width from `ap_fixed<24,12>` to `ap_fixed<16,8>`. It did not narrow the internal accumulator types:

| type | 06F Q24.12 | 06L Q16.8 |
| --- | --- | --- |
| `phc_fixed_t` | `ap_fixed<24,12>` | `ap_fixed<16,8>` |
| `phc_acc_t` | `ap_fixed<40,16>` | `ap_fixed<40,16>` |
| `phc_var_t` | `ap_fixed<48,24>` | `ap_fixed<48,24>` |

That type boundary is the central reason INT16 has limited resource impact.

## Top-Level Result

| resource / metric | Q24.12 | Q16.8 | delta | pct change |
| --- | ---: | ---: | ---: | ---: |
| BRAM | 6 | 6 | 0 | 0.00% |
| DSP | 46 | 46 | 0 | 0.00% |
| FF | 14,518 | 14,332 | -186 | -1.28% |
| LUT | 24,836 | 24,746 | -90 | -0.36% |
| Latency cycles | 8,316 | 8,316 | 0 | 0.00% |
| Interval cycles | 8,317 | 8,317 | 0 | 0.00% |

The total result is not "INT16 is useless." It is more specific: the width reduction affects some storage and selected linear datapath nodes, but the largest LUT/FF and DSP consumers are dominated by unchanged internal 40/48-bit LayerNorm and activation machinery.

## Function / Outlined Loop Breakdown

Rows below are from the csynth `Performance & Resource Estimates` table under `Modules & Loops`. Plain `o` loop rows in this report do not carry direct BRAM/DSP/FF/LUT allocations; the resource-bearing rows are top/module/outlined-loop rows.

### Unchanged Resource Rows

These nodes are bottleneck candidates because their FF/LUT/DSP/BRAM did not move after Q16.8 storage quantization.

| csynth node | source operation mapping | Q24 FF | Q16 FF | Q24 LUT | Q16 LUT | Q24 DSP | Q16 DSP | latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sqrt_fixed_49_25_s` | `hls::sqrt(var + LN_EPS)` inside `layernorm_16` | 4,241 | 4,241 | 11,388 | 11,388 | 0 | 0 | 28 |
| `phc_step_Outline_VITIS_LOOP_72_1` | `sigmoid_16` loop at source line 72 | 312 | 312 | 1,081 | 1,081 | 0 | 0 | 145 |
| `phc_step_Outline_VITIS_LOOP_80_1` | `tanh_16` loop at source line 80 | 318 | 318 | 1,037 | 1,037 | 0 | 0 | 145 |
| `phc_step_Outline_VITIS_LOOP_241_2` | `gain_pre[i] = gain_encoded[i] + gain_prev_state[i]` | 54 | 54 | 108 | 108 | 0 | 0 | 49 |
| `phc_step_Outline_VITIS_LOOP_88_1` | `silu_16` loop at source line 88 | 354 | 354 | 1,087 | 1,087 | 0 | 0 | 177 |
| `phc_step_Outline_VITIS_LOOP_72_12` | second outlined `sigmoid_16` loop | 312 | 312 | 1,081 | 1,081 | 0 | 0 | 145 |
| `phc_step_Outline_VITIS_LOOP_262_5` | `state_next[i] = (phc_fixed_t)z[i]` | 13 | 13 | 56 | 56 | 0 | 0 | 33 |

The `sqrt_fixed_49_25_s` row alone accounts for 11,388 of 24,836 Q24 LUTs, or 45.85% of top-level LUT. In Q16.8 it is still 11,388 of 24,746 LUTs, or 46.02%. This is the strongest resource-hotspot evidence in the reports.

The activation outlined loops above account for 4,286 LUTs unchanged. They are tied to the 65-point piecewise-linear tanh LUT family and the per-element sigmoid/tanh/SiLU loops.

### Changed Resource Rows

These are the places where INT16 had a measurable but small effect.

| csynth node | likely source operation | Q24 FF | Q16 FF | delta FF | Q24 LUT | Q16 LUT | delta LUT | DSP / BRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `phc_step` | top | 14,518 | 14,332 | -186 | 24,836 | 24,746 | -90 | unchanged 46 DSP / 6 BRAM |
| `phc_step_Outline_VITIS_LOOP_119_1` | 16x16 accumulator linear loop | 248 | 240 | -8 | 227 | 226 | -1 | BRAM unchanged 1 |
| `phc_step_Outline_VITIS_LOOP_119_11` | 16x16 accumulator linear loop | 250 | 242 | -8 | 227 | 226 | -1 | BRAM unchanged 1 |
| `phc_step_Outline_VITIS_LOOP_233_1` | `state_temporal` update | 240 | 228 | -12 | 147 | 146 | -1 | unchanged |
| `phc_step_Outline_VITIS_LOOP_136_1` | `linear_16_16_fixed_no_bias` loop | 148 | 140 | -8 | 209 | 209 | 0 | BRAM unchanged 1 |
| `phc_step_Outline_VITIS_LOOP_119_13` | 16x16 accumulator linear loop | 248 | 240 | -8 | 227 | 226 | -1 | BRAM unchanged 1 |
| `phc_step_Outline_VITIS_LOOP_258_4` | readout cast to `phc_fixed_t` | 4 | 4 | 0 | 78 | 70 | -8 | unchanged |

The changes are real but small. They are not where the LUT/FF budget is concentrated.

## LayerNorm Evidence

`layernorm_16` uses `phc_acc_t` input/mean/output and `phc_var_t` variance:

- mean loop: source line 173
- variance loop: source line 180, `var += (phc_var_t)d * (phc_var_t)d`
- reciprocal standard deviation: source line 186, `1.0 / hls::sqrt(var + LN_EPS)`
- output normalization: source line 188, `gamma * (in - mean) * inv_std + beta`

The reports support the LayerNorm bottleneck hypothesis:

| operation | source line | Q24 operation / DSP / latency | Q16 operation / DSP / latency | result |
| --- | ---: | --- | --- | --- |
| variance square | 183 | `mul_40s_40s_72_2_1`, DSP 5, latency 1 | `mul_40s_40s_72_2_1`, DSP 5, latency 1 | unchanged |
| variance division / inv std path | 186 | `udiv_49s_37ns_40_53_seq`, latency 52 | `udiv_49s_37ns_40_53_seq`, latency 52 | unchanged |
| sqrt core | 186 | FF 4,241, LUT 11,388, latency 28 | FF 4,241, LUT 11,388, latency 28 | unchanged |
| output multiply 1 | 190 | `mul_41s_25ns_65_2_1`, DSP 3, latency 1 | same, DSP 3, latency 1 | unchanged |
| output multiply 2 | 190 | `mul_65s_40s_88_5_1`, DSP 8, latency 4 | same, DSP 8, latency 4 | unchanged |
| second LayerNorm variance square | 183 | `mul_40s_40s_72_2_1`, DSP 5, latency 1 | same, DSP 5, latency 1 | unchanged |
| second LayerNorm division | 186 | `udiv_49s_37ns_40_53_seq`, latency 52 | same, latency 52 | unchanged |
| second LayerNorm output multiply pair | 190 | DSP 3 + 8, latency 1 / 4 | DSP 3 + 8, latency 1 / 4 | unchanged |

This is evidence, not speculation: the highest LUT module and the key LayerNorm arithmetic operations are bit-identical in the csynth operation table because the internal accumulator and variance widths stayed at 40/48 bits.

## Nonlinearity / PWL LUT Evidence

The activation functions use the 65-point piecewise-linear tanh table in `tanh_lut_scalar`:

- `TANH_LUT[65]`, complete partition
- index path: `scaled = (x + 4) * 8`, `idx = (int)scaled`
- interpolation: `y0 + frac * (y1 - y0)`
- sigmoid uses `0.5 * (tanh(x / 2) + 1)`
- SiLU uses `in[i] * sigmoid_scalar(in[i])`

Resource-bearing activation rows are unchanged:

| node / operation | Q24 | Q16 | result |
| --- | --- | --- | --- |
| `phc_step_Outline_VITIS_LOOP_72_1` | FF 312, LUT 1,081, latency 145 | FF 312, LUT 1,081, latency 145 | unchanged |
| `phc_step_Outline_VITIS_LOOP_80_1` | FF 318, LUT 1,037, latency 145 | FF 318, LUT 1,037, latency 145 | unchanged |
| `phc_step_Outline_VITIS_LOOP_88_1` | FF 354, LUT 1,087, latency 177 | FF 354, LUT 1,087, latency 177 | unchanged |
| `phc_step_Outline_VITIS_LOOP_72_12` | FF 312, LUT 1,081, latency 145 | FF 312, LUT 1,081, latency 145 | unchanged |
| PWL interpolation multiply `mul_ln57` | `mul_26s_40s_64`, DSP 4, latency 1 | same, DSP 4, latency 1 | unchanged |
| SiLU multiply `mul_ln90` | `mul_40s_40s_64`, DSP 5, latency 1 | same, DSP 5, latency 1 | unchanged |

This explains why reducing stored weights and external values does not materially shrink the activation block: the activation block operates on `phc_acc_t` values, not on the narrowed external `phc_fixed_t`.

## Linear Datapath Evidence

The linear datapath is the part that does react to Q16.8, but the reaction is too small to move top-level resources much.

| operation | source line | Q24 operation | Q16 operation | DSP result |
| --- | ---: | --- | --- | --- |
| first fixed input/weight multiply | 106 | `mul_36s_25s_61`, DSP 2 | `mul_32s_25s_57`, DSP 2 | unchanged DSP |
| accumulator input/weight multiply | 124 | `mul_25s_40s_64`, DSP 3 | same, DSP 3 | unchanged |
| readout multiply | 159 | `mul_24s_40s_64`, DSP 3 | same, DSP 3 | unchanged |

The first row proves that Q16.8 did narrow a multiplier input, but it did not cross a DSP packing/resource threshold. Other multipliers remain dominated by 40-bit accumulator inputs.

## BRAM Evidence

BRAM stayed at 6 because the major 256-entry weight ROMs still map to one BRAM each after narrowing.

| weight ROM | Q24 stored bit/depth | Q16 stored bit/depth | BRAM result |
| --- | --- | --- | --- |
| `ENC_LINEAR_W` | 13, 256, 1 | 9, 256, 1 | 1 -> 1 |
| `GATE_W` | 13, 256, 1 | 9, 256, 1 | 1 -> 1 |
| `CANDIDATE_W` | 12, 256, 1 | 8, 256, 1 | 1 -> 1 |
| `GAIN_FROM_ENCODED_W` | 13, 256, 1 | 9, 256, 1 | 1 -> 1 |
| `GAIN_FROM_PREV_STATE_W` | 12, 256, 1 | 8, 256, 1 | 1 -> 1 |
| `SYNAPSE_W` | 12, 256, 1 | 8, 256, 1 | 1 -> 1 |

So Q16.8 shrank ROM widths but did not reduce the count of BRAM-mapped memories.

## 06M Error-Decomposition Check

`n2ew_era/06m_phc_lite_error_decomposition/` contains `hls_component_lut_only` and `hls_component_quant_only`, but they are C-sim control runs only. There are no `csynth.rpt` files under 06M.

Existing 06M result summary:

| label | max_abs_diff | mean_abs_diff | purpose |
| --- | ---: | ---: | --- |
| `06f_original_cosim_combined` | 0.007252753 | 0.0003725533494 | combined 06F fixed-point plus LUT behavior |
| `only_lut_error_csim` | 0.005462141831 | 0.0002923981572 | isolates nonlinear LUT interpolation error |
| `only_quant_error_csim` | 0.0064746145 | 0.0002502181146 | isolates fixed-point quantization/noise |

This is useful context but not a replacement for the resource hotspot comparison above. It says the 65-point activation LUT is numerically relevant; it does not say the activation LUT is the largest HLS resource consumer. The resource reports show sqrt/LayerNorm is the larger LUT hotspot.

## Root Cause

INT16 has limited resource benefit because it narrows the edges of the datapath while leaving the dominant internal arithmetic unchanged.

Evidence:

- The largest LUT block, `sqrt_fixed_49_25_s`, is exactly unchanged at FF 4,241 and LUT 11,388.
- The LayerNorm variance and normalization multipliers remain 40/48-bit dominated and keep identical DSP/latency.
- The PWL activation loops and interpolation multiply are exactly unchanged because activations use `phc_acc_t`.
- Linear and cast-related nodes shrink slightly, but those nodes are not the dominant LUT/FF consumers.
- BRAM-mapped 256-entry weight ROMs keep one BRAM each even after narrower stored widths.

## Priority For Real LUT/FF Reduction

1. Replace or approximate the LayerNorm reciprocal-standard-deviation path first.

   Target: `layernorm_16`, especially variance, division, and `hls::sqrt(var + LN_EPS)`.

   Candidate directions: approximate reciprocal square root with a small LUT plus one refinement step, RMSNorm-style simplification, fixed bounded variance approximation, or model-side removal/folding of LayerNorm. Any of these would require a new numerical and policy-proxy validation pass.

2. Reduce `phc_var_t` / LayerNorm intermediate width only if error analysis supports it.

   The current Q16.8 experiment left `phc_var_t=ap_fixed<48,24>` and `phc_acc_t=ap_fixed<40,16>`, so LayerNorm cannot shrink. A direct reduction could reduce resources, but it is risky because 06M already shows quantization noise is numerically material.

3. Revisit activation PWL after LayerNorm.

   The activation loops are unchanged and nontrivial, but smaller than the sqrt core. If LayerNorm is improved, then the next candidates are `sigmoid_16`, `tanh_16`, `silu_16`, and `tanh_lut_scalar` interpolation. Options include coarser LUT, cheaper interpolation, or activation redesign.

4. Do not expect BRAM reduction from Q16.8 alone.

   The ROM widths shrink, but the memories still map to six one-BRAM 256-entry weight ROMs. BRAM reduction needs memory packing, alternative ROM implementation, sharing, or layout changes, not only narrower stored coefficients.

## Bottom Line

The csynth breakdown supports the hypothesis that INT16 failed to help because the design is not primarily limited by external/weight storage width. It is dominated by unchanged internal LayerNorm and activation arithmetic, especially the generated `sqrt_fixed_49_25_s` core. The first optimization experiment worth running is therefore not another blanket INT16 pass; it is a targeted LayerNorm approximation/removal experiment with resource and numerical-error tracking.
