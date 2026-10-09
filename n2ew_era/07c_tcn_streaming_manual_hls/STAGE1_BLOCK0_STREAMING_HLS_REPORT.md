status: stage1_block0_complete_stop_for_review
finding: block0_streaming_hls_measured_against_stage0_reference
# Stage 1 Block0 Streaming HLS Report

## Scope

This stage implements only block0 (`dilation=1`) as a streaming HLS step function. It does not modify 07a, 07b, or Stage 0 reference outputs. It does not use NEW5. It does not implement blocks 1-4, cosim, implementation, P&R, bitstream, or board tests.

CSim compares every valid block0 timestep and all 16 channels against the Stage 0 Python streaming reference. The csynth `latency_max_cycles` column is one event-step top-function call latency; `ii_max_cycles` is the conservative repeated-call event interval when invoking the step function back-to-back.

## CSim and Csynth Summary

| variant | csim_status | csim_output | csim_rows | samples | valid_timesteps_per_sample | compared_values | max_abs_diff | mean_abs_diff | rmse | valid_alignment_all_true | first_quarter_rmse | last_quarter_rmse | last_minus_first_quarter_rmse | last_over_first_quarter_rmse | max_timestep_mean_rmse | max_timestep_max_abs_diff | csynth_status | csynth_report | latency_min_cycles | latency_max_cycles | latency_absolute_min_reported | latency_absolute_max_reported | ii_min_cycles | ii_max_cycles | pipeline_type | latency_us_100mhz | interval_us_100mhz | BRAM_18K | DSP | FF | LUT | URAM | DSP_pct | LUT_pct | FF_pct | BRAM_18K_pct | speedup_vs_naive_block0_total | speedup_vs_naive_block0_amortized | throughput_speedup_vs_naive_block0_total | throughput_speedup_vs_naive_block0_amortized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage1_block0_scratch_csim_out.csv | 49600 | 200 | 248 | 793600 | 0.0056416969 | 0.00020399093 | 0.00045964187 | True | 9.1903956e-05 | 0.00063806173 | 0.00054615777 | 6.9427014 | 0.00060509413 | 0.0056416969 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_block0_scratch\hls_component\hls\syn\report\tcn_block0_stream_step_csynth.rpt | 1 | 95 | 10.000 ns | 0.950 us | 2 | 96 | no | 0.95 | 0.96 | 0 | 134 | 34567 | 17861 | 0 | 60.909091 | 33.573308 | 32.487782 | 0 | 3365.1474 | 13.569143 | 3330.0938 | 13.427797 |
| distilled | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage1_block0_distilled_csim_out.csv | 49600 | 200 | 248 | 793600 | 0.0071830744 | 0.00028748451 | 0.00060144092 | True | 0.00013302885 | 0.00083198024 | 0.00069895139 | 6.2541337 | 0.0008039552 | 0.0071830744 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_block0_distilled\hls_component\hls\syn\report\tcn_block0_stream_step_csynth.rpt | 1 | 95 | 10.000 ns | 0.950 us | 2 | 96 | no | 0.95 | 0.96 | 0 | 120 | 33364 | 17610 | 0 | 54.545455 | 33.101504 | 31.357143 | 0 | 3365.1474 | 13.569143 | 3330.0938 | 13.427797 |

## Baseline Comparison

| artifact | latency | verification | note |
| --- | --- | --- | --- |
| 07b naive full TCN | 13,703.85 us | HLS csynth estimate | Full-window recomputation baseline. |
| 07b naive block0 | 319,689 cycles total; 1289.07 cycles/output timestep amortized | HLS csynth instance estimate | Report row grp_p_anonymous_namespace_tcn_block_252_1_s_fu_442. |
| 07a TCN analytical estimate | 32.656 us | pre-synthesis MAC estimate, not HLS-verified | Not a measured FPGA number. |
| PHC-lite v8 | 83.16 us | routed P&R at 100MHz | Existing PHC baseline. |

## Naive Block0 Improvement

| variant | latency_max_cycles | ii_max_cycles | latency_us_100mhz | interval_us_100mhz | pipeline_type | speedup_vs_naive_block0_total | speedup_vs_naive_block0_amortized | throughput_speedup_vs_naive_block0_total | throughput_speedup_vs_naive_block0_amortized | csynth_report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 95 | 96 | 0.95 | 0.96 | no | 3365.1474 | 13.569143 | 3330.0938 | 13.427797 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_block0_scratch\hls_component\hls\syn\report\tcn_block0_stream_step_csynth.rpt |
| distilled | 95 | 96 | 0.95 | 0.96 | no | 3365.1474 | 13.569143 | 3330.0938 | 13.427797 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_block0_distilled\hls_component\hls\syn\report\tcn_block0_stream_step_csynth.rpt |

## Error-Stability Readout

The `last_minus_first_quarter_rmse` and `last_over_first_quarter_rmse` columns are a simple check for timestep-wise error growth. They are diagnostic only; Stage 1 did not introduce a new tuned threshold.

## Stage Boundary

STOP HERE. Blocks 1-4 are intentionally not implemented until this Stage 1 block0 result is reviewed.
