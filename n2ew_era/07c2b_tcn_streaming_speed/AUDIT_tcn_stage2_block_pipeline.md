# TCN Stage2 Block-Level Pipeline Audit

Status: `resolved_negative`

## Scope

This audit tests one narrow change in `n2ew_era/07c2b_tcn_streaming_speed/`: add a `TCN_STAGE2_BLOCK_PIPELINE` macro and enable `#pragma HLS PIPELINE II=1` only on the CONV1 and CONV2 `oc` loops inside `stage2_resource_stream_block_step`.

No 07C2D closure artifacts were modified. No training, NEW5 access, HLS P&R, or model changes were performed.

## Code Changes

- Added `TCN_STAGE2_BLOCK_PIPELINE`, default `0`, in `hls_src/tcn_stage2_resource_impl.hpp`.
- Wrapped the CONV1 and CONV2 `oc` loop pragmas:
  - `TCN_STAGE2_BLOCK_PIPELINE=1`: `#pragma HLS PIPELINE II=1`
  - otherwise: `#pragma HLS PIPELINE off`
- Added `hls_src/tcn_stage2_resource_distilled_shared_u4_blockpipe.cpp`.
- Added `distilled_shared_u4_blockpipe` to `01_prepare_speed_components.py`.

The inner `ic` loops, their `PIPELINE off` pragmas, `UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL`, history buffer loops, weights, and testbench were left unchanged.

## C-Sim

Command:

```text
vitis-run.bat --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

Result:

| metric | value |
|---|---:|
| samples | 4 |
| readout_rows | 4 |
| readout_max_abs_diff | 0.4200745125 |
| readout_rmse | 0.308342212205 |
| sign_agreement_readout_0 | 1 |
| sign_agreement_readout_1 | 1 |
| sign_agreement_joint | 1 |
| valid_alignment_errors | 0 |

Evidence: `results/stage2_speed_distilled_shared_u4_blockpipe_event1008_tb_summary.csv`.

## C-Synthesis

The Vitis 2025.2 command-line flag `--syn` is not supported on this machine, so synthesis was run through the existing generated Tcl script:

```text
vitis-run.bat --mode hls --tcl --input_file run_csynth_only.tcl
```

Top report: `artifacts/distilled_shared_u4_blockpipe/syn_report/tcn_streaming_step_csynth.rpt`.

## RTL Co-Simulation Attempt

Because the csynth resource gate passed (`DSP=100 < 220`, `LUT=37,475 < 53,200`), RTL co-simulation was attempted:

```text
vitis-run.bat --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

The C testbench phase completed and reproduced the same 4-window summary as C-Sim:

| metric | value |
|---|---:|
| samples | 4 |
| readout_rows | 4 |
| readout_max_abs_diff | 0.420075 |
| readout_rmse | 0.308342 |
| sign_agreement_readout_0 | 1 |
| sign_agreement_readout_1 | 1 |
| sign_agreement_joint | 1 |
| valid_alignment_errors | 0 |

The RTL simulation did not reach a mismatch/pass comparison. It failed during XSIM/xelab static elaboration because generated helper modules were not found:

```text
ERROR: [VRFC 10-2063] Module <tcn_streaming_step_tcn_streaming_step_ap_fixed_const_bool_ap_fixed_bool_ap_fixed_bool_ap_fixed_1Zio> not found while processing module instance <tcn_streaming_step_ap_fixed_const_bool_ap_fixed_bool_ap_fixed_bool_ap_fixed_19_U>
ERROR: [VRFC 10-2063] Module <tcn_streaming_step_p_anonymous_namespace_stage2_resource_stream_block_step_0_1_3_s_input_hist_4_bkb> not found while processing module instance <input_hist_4_U>
ERROR: [VRFC 10-2063] Module <tcn_streaming_step_p_anonymous_namespace_stage2_resource_stream_block_step_0_1_3_s_p_ZN11tcn_weidEe> not found while processing module instance <p_ZN11tcn_weightsL7CONV1_BE_0_U>
ERROR: [XSIM 43-3322] Static elaboration of top level Verilog design unit(s) in library work failed.
ERROR: [COSIM 212-4] *** C/RTL co-simulation finished: FAIL ***
```

Evidence:

- `artifacts/distilled_shared_u4_blockpipe/cosim_attempt/hls_run_cosim.log`
- `artifacts/distilled_shared_u4_blockpipe/cosim_attempt/hls_component.hlsrun_cosim_summary`

## Baseline Comparison

07C2B same-infrastructure baseline:

| variant | block pipeline | PipelineType | latency cycles | interval cycles | DSP | LUT | FF | BRAM_18K | fit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| distilled_shared_u4 | false | no | 7438 | 7439 | 100 (45.45%) | 37475 (70.44%) | 24369 (22.90%) | 66 (23.57%) | true |
| distilled_shared_u4_blockpipe | true | no | 7438 | 7439 | 100 (45.45%) | 37475 (70.44%) | 24369 (22.90%) | 66 (23.57%) | true |

07C2D routed closure reference:

| variant | clock | latency cycles | interval cycles | latency | DSP | LUT | FF | BRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distilled shared_u4 routed closure | 100 MHz | 7438 | 7439 | 74.38 us | 49.09% | 32.00% | 14.44% | 23.57% |
| distilled shared_u4_blockpipe 07C2B csynth | 100 MHz csynth target | 7438 | 7439 | 74.38 us | 45.45% | 70.44% | 22.90% | 23.57% |

The 07C2D line is a routed iface-slim closure point, while this audit is a 07C2B speed-scan csynth experiment. The key comparison for this audit is the 07C2B same-infrastructure baseline.

## Pragma Conflict

The acceptance criterion required that the previous pipeline-conflict warning disappear. It did not. The new conflict is now localized to the attempted block-level `oc` pipeline versus the still-present inner `ic` loop `PIPELINE off` pragmas:

```text
WARNING: [HLS 214-275] In loop 'VITIS_LOOP_145_5'(../hls_src/tcn_stage2_resource_impl.hpp:145:35), pragma conflict between PIPELINE (../hls_src/tcn_stage2_resource_impl.hpp:140:1) and PIPELINE off (../hls_src/tcn_stage2_resource_impl.hpp:146:1) in a sub-region. Ignoring PIPELINE to honor PIPELINE off
WARNING: [HLS 214-275] In loop 'VITIS_LOOP_201_8'(../hls_src/tcn_stage2_resource_impl.hpp:201:35), pragma conflict between PIPELINE (../hls_src/tcn_stage2_resource_impl.hpp:196:1) and PIPELINE off (../hls_src/tcn_stage2_resource_impl.hpp:202:1) in a sub-region. Ignoring PIPELINE to honor PIPELINE off
```

Evidence: `artifacts/distilled_shared_u4_blockpipe/hls.log`.

## Conclusion

`block-level pipeline` as defined in this test is ineffective on this design because Vitis ignores the `oc` loop pipeline to honor the nested `PIPELINE off` pragmas in the `ic` loop sub-regions. Latency and interval are unchanged at 7438 / 7439 cycles, and the top report still shows `PipelineType=no`.

The second-round `u2_blockpipe` experiment was not run because this was not a resource explosion; it was a pragma conflict. Lowering `TCN_STAGE2_CHANNEL_UNROLL` from 4 to 2 would not, by itself, remove the nested `PIPELINE off` conflict. A future attempt would need an explicitly different hypothesis, such as reconciling or conditionally removing the inner `ic` `PIPELINE off` pragmas while preserving the intended unroll/resource budget discipline.

Cosim was attempted after the resource gate passed, but it failed in XSIM/xelab static elaboration before RTL output comparison. Therefore the functional evidence level for this variant remains C-Sim plus csynth, not passing RTL cosim.
