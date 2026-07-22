# PHC-lite Manual HLS Resource Patch 3 Report

Status: `code_patch_ready_for_user_csim_csynth`

This patch applies the third-round resource fix for PHC-lite manual HLS. It changes matrix weight partitioning from full partitioning to cyclic banking so Vitis HLS cannot expose all 16x16 weight elements as fully parallel datapaths under top-level `PIPELINE II=1`.

## Modified File

```text
n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp
```

## Changes Applied

### Matrix Weight Partitioning

Changed all four matrix-vector helper functions from:

```cpp
#pragma HLS ARRAY_PARTITION variable=w complete dim=0
```

to:

```cpp
#pragma HLS ARRAY_PARTITION variable=w cyclic dim=2 factor=2
```

Affected functions:

```text
linear_16_16_fixed
linear_16_16_acc
linear_16_16_fixed_no_bias
linear_16_2_acc
```

The 16-output functions retain:

```cpp
#pragma HLS UNROLL factor=2
```

on the outer output loop. The inner input loop has no `UNROLL` pragma.

The 2-output readout keeps the outer `j < 2` loop fully unrolled, with no inner-loop unroll.

### Activation Loops

Confirmed `sigmoid_16`, `tanh_16`, and `silu_16` have no loop unroll pragmas. This avoids replicating the 65-point LUT datapath across multiple parallel lanes.

### Preserved Logic

The following were intentionally left unchanged:

```text
Top-level #pragma HLS PIPELINE II=1
1D ARRAY_PARTITION complete pragmas
INLINE pragmas
TANH_LUT[65] values
phc_weights.hpp values
PHC_ABS_TOL in phc_step_testbench.cpp
LayerNorm mean/variance/output loops with UNROLL factor=4
hls::sqrt(var + LN_EPS) on phc_var_t/ap_fixed, with no double cast
```

## Static Checks

Confirmed:

```text
No "ARRAY_PARTITION variable=w complete dim=0" remains.
No synthesizable hls::exp or hls::tanh calls remain.
No (double) casts remain in phc_step_hls.cpp.
The HLS config uses part=xc7z020clg400-1 and syn.top=phc_step.
```

## User-Run Commands

Run from:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_component/
```

```bash
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --syn --config hls_config.cfg --work_dir hls_component
```

If resources pass:

```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

## Acceptance Targets

```text
C-Sim: mismatches=0, max_abs_diff <= 0.015
C-Synth: DSP < 220 and LUT < 53,200
C-Synth: report II honestly if II > 1
C/RTL Cosim: run only after resources pass
```

Latency may increase because the inner matrix loop is no longer fully parallel. That is expected and acceptable if resources fit the target device.
