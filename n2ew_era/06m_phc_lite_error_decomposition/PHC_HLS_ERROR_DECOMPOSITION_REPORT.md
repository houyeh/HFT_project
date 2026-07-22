# PHC-lite HLS Error Decomposition Report

This phase does not modify Phase 06F source code, reports, test vectors, or tolerance. It adds two isolated Vitis HLS C-sim control runs using the same 25,600 test-vector rows and the same `PHC_ABS_TOL=1.0` testbench tolerance.

## Control Definitions

- `only_lut_error_csim`: inputs, previous state, weights, layernorm, dot products, and outputs are evaluated as unquantized `double`; sigmoid/tanh/SiLU still use the same 65-point piecewise-linear tanh LUT family as the 06F HLS implementation. This isolates nonlinear LUT interpolation error up to floating/double-vs-float32 roundoff.
- `only_quant_error_csim`: inputs, previous state, weights, accumulators, layernorm, and final outputs retain the 06F `ap_fixed<24,12>`/fixed-point datapath; sigmoid/tanh/SiLU are replaced by true floating `exp`/`tanh` evaluations cast back to the fixed accumulator type. This isolates fixed-point quantization/noise with LUT interpolation removed.
- `06f_original_cosim_combined`: existing 06F C/RTL co-sim output, where fixed-point quantization and LUT nonlinearities are both present. This is the cited `0.00725275` max-absolute-error baseline.

## Raw Metrics

| label | source | total_rows | dimensions/row | mismatches | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|---:|---:|---:|
| 06f_original_cosim_combined | existing_06f_cosim_csv | 25600 | 18 | 0 | 0.007252753 | 0.0003725533494 |
| 06f_original_csim_combined | existing_06f_csim_csv | 25600 | 18 | 0 | 0.007252753 | 0.0003725533494 |
| only_lut_error_csim | phase_06m_vitis_csim | 25600 | 18 | 0 | 0.005462141831 | 0.0002923981572 |
| only_quant_error_csim | phase_06m_vitis_csim | 25600 | 18 | 0 | 0.0064746145 | 0.0002502181146 |

## Component Split Against 06F Original

- 06F original combined reference used here: `06f_original_cosim_combined`, max_abs_diff `0.007252753`, mean_abs_diff `0.0003725533494`.
- only-LUT-error max_abs_diff: `0.005462141831`; mean_abs_diff: `0.0002923981572`.
- only-quant-error max_abs_diff: `0.0064746145`; mean_abs_diff: `0.0002502181146`.
- Component max sum: `0.01193675633` (1.646x original max).
- Component mean sum: `0.0005426162718` (1.456x original mean).
- Max-error component shares, normalized by component sum: LUT `45.76%`, quantization `54.24%`.
- Mean-error component shares, normalized by component sum: LUT `53.89%`, quantization `46.11%`.
- Dominant component by max_abs_diff: `quantization_noise`.
- Dominant component by mean_abs_diff: `lut_interpolation`.
- Interaction residual by max_abs_diff (`original - component_sum`): `-0.004684003331`.
- Interaction residual by mean_abs_diff (`original - component_sum`): `-0.0001700629224`.

## Interpretation

The component sum is not within roughly 20% of the original max error, so interaction effects or non-identical worst-case rows are material.

The percentages above are component-share diagnostics, not a proof that both error sources combine linearly on the exact same row. If the component sum differs from the combined 06F result, that residual is reported as interaction/non-aligned-worst-case behavior rather than hidden.

## Claim Boundary

- These are C-sim numerical controls only, not P&R.
- The controls use the same test vectors as 06F and do not alter the 06F implementation.
- This decomposition does not make claims about real exchange replay, real-world profitability, or FPGA-caused PnL improvement.
