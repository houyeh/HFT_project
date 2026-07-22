# PHC-lite Manual HLS Resource Patch 2 Report

Status: `code_patch_ready_for_user_csim_csynth`

This patch addresses the second-round resource issue: excessive fixed-point multiplier replication caused by nested loop unrolling in PHC-lite matrix-vector products.

## Scope

Modified file:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp
```

Unmodified files:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_weights.hpp
n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_testbench.cpp
```

The tolerance `PHC_ABS_TOL = 1.0` was not changed.

## Changes Applied

1. Matrix-vector functions:

```text
linear_16_16_fixed
linear_16_16_acc
linear_16_16_fixed_no_bias
linear_16_2_acc
```

- Removed inner-loop `#pragma HLS UNROLL factor=4`.
- Changed 16-output outer-loop unroll from `factor=4` to `factor=2`.
- Kept the 2-output readout outer loop fully unrolled.

2. Activation vector functions:

```text
sigmoid_16
tanh_16
silu_16
```

- Removed `#pragma HLS UNROLL factor=4` from the 16-element element-wise loops.
- Preserved the 65-point LUT-based `tanh_lut_scalar` and sigmoid definition.

3. LayerNorm:

```text
layernorm_16
```

- Removed `#pragma HLS UNROLL factor=4` from the mean, variance, and output loops.
- Preserved `hls::sqrt(var + LN_EPS)` without double casting.

4. Preserved as requested:

- Top-level `#pragma HLS PIPELINE II=1`.
- `#pragma HLS INLINE`.
- Array partition pragmas.
- `TANH_LUT[65]` contents.
- Weight constants.

## Source Checks

The patched source still contains no synthesizable calls to:

```text
hls::exp
hls::tanh
std::*
(double) casts
```

The only nonlinear paths are:

```text
tanh_lut_scalar
sigmoid_scalar = 0.5 * (tanh_lut_scalar(x / 2) + 1)
hls::sqrt(var + LN_EPS)
```

## Vitis Run Status

The Codex shell could not run Vitis because `vitis-run` incorrectly resolved the workspace path:

```text
C:/Users/holbo/00VS_projjects/00_HFT_project
```

as:

```text
C:/Users/holbo//00_HFT_project
```

and therefore failed before reading the config file. No csim/csynth/cosim result was fabricated in this report.

Run these commands from:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_component/
```

```bash
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --syn --config hls_config.cfg --work_dir hls_component
```

If C-Synthesis resources pass, run:

```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

## Expected Validation Fields

Please record:

```text
C-Sim total / mismatches / max_abs_diff / mean_abs_diff
C-Synth latency cycles
C-Synth II
C-Synth LUT / FF / DSP / BRAM usage and utilization
C/RTL Cosim PASS/FAIL and error metrics
```

Acceptance target:

```text
DSP < 220
LUT < 53,200
C-Sim mismatches = 0
C-Sim max_abs_diff <= 0.010
II remains 1, or any II regression is explicitly documented
```

Latency is expected to increase relative to the previous 297-cycle estimate because inner-loop multiplier parallelism was intentionally reduced.
