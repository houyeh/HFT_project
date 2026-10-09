status: stage2_mixed_csim_passed_csynth_partial_failure
finding: five_block_streaming_csim_passed_but_csynth_not_fully_available
# Stage 2 Full Streaming HLS Report

## Scope

This report covers the new 07c Stage 2 streaming HLS implementation only. It does not modify 07a, 07b, Stage 0 outputs, or Stage 1 outputs. It does not use NEW5. It stops at CSim/csynth; no cosim, implementation, P&R, bitstream, or board result is claimed.

Stage 2 uses `TCN_STAGE2_CHANNEL_UNROLL=4` to keep the five-block implementation resource-feasible; this differs from the Stage 1 single-block diagnostic that used the inherited unroll-16 default.

## Preflight

- Preflight report: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\STAGE2_PREFLIGHT_BLOCK0_ERROR_DIAGNOSTIC_REPORT.md`
- Preflight curve CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_preflight_block0_timestep_rmse.csv`

## Per-Block CSim Diagnostics

| variant | stage | block_csim_status | csim_block_output | rows | timesteps | compared_values | max_abs_diff | mean_abs_diff | rmse | first_quarter_mean_rmse | last_quarter_mean_rmse | last_over_first_quarter_rmse | post10_rmse_max | post10_rmse_slope_per_timestep | real_data_start_out_t | post_real10_rmse_max | post_real10_rmse_median | post_real10_rmse_slope_per_timestep | block0_preflight_ratio | ratio_warning | ratio_warning_basis | stage2_block_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | block0 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_block_csim_out.csv | 49600 | 248 | 793600 | 0.0056416969 | 0.00020399093 | 0.00045964187 | 9.1903956e-05 | 0.00063793982 | 6.941375 | 0.00066688632 | 3.425883e-06 | 120 | 0.00066688632 | 0.00063817925 | 6.1413883e-08 | 6.941375 | False | >2x block0 preflight ratio (13.8827) | True |
| scratch | block1 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_block_csim_out.csv | 48000 | 240 | 768000 | 0.027232172 | 0.00059727816 | 0.0012464302 | 0.000195667 | 0.0017297018 | 8.8400281 | 0.0018861243 | 9.8986273e-06 | 112 | 0.0018861243 | 0.0017132547 | 1.0798495e-07 | 6.941375 | False | >2x block0 preflight ratio (13.8827) | True |
| scratch | block2 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_block_csim_out.csv | 44800 | 224 | 716800 | 0.042239186 | 0.00093314036 | 0.0020141987 | 0.00020313112 | 0.0027439802 | 13.508419 | 0.0031063609 | 1.7385569e-05 | 96 | 0.0031063609 | 0.0027051561 | 9.9670294e-07 | 6.941375 | False | >2x block0 preflight ratio (13.8827) | True |
| scratch | block3 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_block_csim_out.csv | 38400 | 192 | 614400 | 0.054208731 | 0.0013348487 | 0.0027417149 | 0.00021254211 | 0.0035291071 | 16.604273 | 0.0039526358 | 2.4166231e-05 | 64 | 0.0039526358 | 0.0034378628 | 3.1792122e-06 | 6.941375 | True | >2x block0 preflight ratio (13.8827) | True |
| scratch | block4 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_block_csim_out.csv | 25600 | 128 | 409600 | 0.062042239 | 0.0022742107 | 0.0038638225 | 0.0028765451 | 0.0041982995 | 1.4594937 | 0.0046765309 | 9.2575933e-06 | 0 | 0.0046765309 | 0.00403777 | 9.2575933e-06 | 6.941375 | False | >2x block0 preflight ratio (13.8827) | True |
| distilled | block0 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_block_csim_out.csv | 49600 | 248 | 793600 | 0.0071830744 | 0.00028748451 | 0.00060144092 | 0.00013302885 | 0.00083159331 | 6.2512251 | 0.00089148852 | 4.3910907e-06 | 120 | 0.00089148852 | 0.00083283306 | 2.0248939e-08 | 6.2512251 | False | >2x block0 preflight ratio (12.5025) | True |
| distilled | block1 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_block_csim_out.csv | 48000 | 240 | 768000 | 0.026138307 | 0.00086088571 | 0.0018595424 | 0.00028424496 | 0.0025661241 | 9.0278612 | 0.0028133869 | 1.4828596e-05 | 112 | 0.0028133869 | 0.0025573846 | 8.2457312e-08 | 6.2512251 | False | >2x block0 preflight ratio (12.5025) | True |
| distilled | block2 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_block_csim_out.csv | 44800 | 224 | 716800 | 0.039681431 | 0.0012179918 | 0.0025420826 | 0.00032563454 | 0.0034217492 | 10.507943 | 0.0038026717 | 2.1200431e-05 | 96 | 0.0038026717 | 0.0034159624 | 3.482663e-07 | 6.2512251 | False | >2x block0 preflight ratio (12.5025) | True |
| distilled | block3 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_block_csim_out.csv | 38400 | 192 | 614400 | 0.044420238 | 0.0016250564 | 0.0030998335 | 0.00036565563 | 0.0039287918 | 10.744513 | 0.0043231654 | 2.5850391e-05 | 64 | 0.0043231654 | 0.0039031089 | 2.1527381e-06 | 6.2512251 | False | >2x block0 preflight ratio (12.5025) | True |
| distilled | block4 | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_block_csim_out.csv | 25600 | 128 | 409600 | 0.042496683 | 0.0026370493 | 0.0041458907 | 0.0032842238 | 0.0044694047 | 1.3608709 | 0.0048672632 | 7.3958254e-06 | 0 | 0.0048672632 | 0.0042903967 | 7.3958254e-06 | 6.2512251 | False | >2x block0 preflight ratio (12.5025) | True |

## Final Readout CSim Diagnostics

| variant | readout_csim_status | readout_csim_output | readout_rows | readout_max_abs_diff | readout_mean_abs_diff | readout_rmse | sign_agreement_readout_0 | sign_agreement_readout_1 | sign_agreement_joint | readout_gate_pass | component_csim_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_scratch_readout_csim_out.csv | 200 | 0.34269338 | 0.069059692 | 0.085448245 | 1 | 1 | 1 | True | CS_PASSED |
| distilled | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_streaming_distilled_readout_csim_out.csv | 200 | 0.57189946 | 0.27712666 | 0.30506537 | 1 | 1 | 1 | True | CS_PASSED |

## Csynth Summary

| variant | csynth_status | csynth_report | latency_min_cycles | latency_max_cycles | latency_absolute_min_reported | latency_absolute_max_reported | ii_min_cycles | ii_max_cycles | pipeline_type | latency_us_100mhz | interval_us_100mhz | BRAM_18K | DSP | FF | LUT | URAM | DSP_pct | LUT_pct | FF_pct | BRAM_18K_pct | resource_fit_zynq7020 | csynth_failure_error | csynth_partial_reports_count | csynth_failure_log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | present | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_scratch\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt | 15 | 558 | 0.150 us | 5.580 us | 16 | 559 | no | 5.58 | 5.59 | 16 | 688 | 223737 | 141327 | 0 | 312.72727 | 265.65226 | 210.27914 | 5.7142857 | False |  |  |  |
| distilled | failed_before_top_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_distilled\hls_component\logs\hls_compile.log: ERROR: [v++ 17-180] Spawn failed: The operation completed successfully. | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_distilled_vpp_csynth_stdout.log: ERROR: [v++ 17-180] Spawn failed: The operation completed successfully. | 12 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_distilled\hls_component\logs\hls_compile.log |

## Baseline Comparison

| artifact | latency | verification | note |
| --- | --- | --- | --- |
| 07b naive full TCN | 13,703.85 us | HLS csynth estimate | Full-window recomputation baseline. |
| 07a TCN analytical estimate | 32.656 us | pre-synthesis MAC estimate, not HLS-verified | Analytical MAC-rate estimate only. |
| PHC-lite v8 | 83.16 us | routed P&R at 100MHz | Existing PHC baseline. |

## Stage Boundary

This stage intentionally stops after CSim/csynth. No Stage 2 cosim, Vivado implementation, P&R, routed timing, or board-level claim is made here.
