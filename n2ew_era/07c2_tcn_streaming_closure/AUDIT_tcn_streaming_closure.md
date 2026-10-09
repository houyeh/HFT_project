status: mixed_resource_fit_and_pnr_met_but_1008_event_cosim_timeout
finding: shared_u1 BRAM-backed history storage fixes the 07C streaming TCN resource overflow and routes at 100MHz, but full 1008-event RTL cosim did not complete within the one-hour budget, so the closure is not deployable yet.

**Scope**
This audit lives in `n2ew_era/07c2_tcn_streaming_closure/`. It copied 07C Stage 2 HLS source/results into a new workspace and did not modify `n2ew_era/07c_tcn_streaming_manual_hls/` or any other prior phase. No model weights were changed, no retraining was performed, and NEW5 was not touched.

The latency object here is streaming TCN per-event top-call latency/interval. It is not the same object as PHC recurrent single-step latency or 07B naive whole-window recomputation latency.

**Environment Fix**
The original 07C distilled csynth failure (`Spawn failed: The operation completed successfully`) did not reproduce after rebuilding in a clean 07C2 shortpath workdir under `C:\tmp`. Distilled `shared_u1` produced a complete csynth top report, so the likely cause was stale/long-path/tool workdir state rather than invalid distilled source or weights.

**Resource Sweep**
These rows are the carried-forward 07C sweep plus the 07C2-verified selected setting. The resource-fit point is `shared_u1`: `TCN_STAGE2_CHANNEL_UNROLL=1` with BRAM-backed shared history storage.

| variant | setting | CSim state | csynth latency us @100MHz | II max cycles | DSP | LUT | FF | BRAM_18K | fit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| scratch | u2 | CS_PASSED | 98.60 | 9861 | 48.18% | 154.77% | 119.52% | 16.43% | False |
| scratch | u1 | CS_PASSED | 163.40 | 16341 | 40.91% | 140.01% | 116.97% | 16.43% | False |
| scratch | shared_u1 | CS_PASSED | 165.10 | 16511 | 40.91% | 52.12% | 19.87% | 23.57% | True |
| distilled | u2 | present_unknown | 98.60 | 9861 | 45.45% | 154.63% | 119.27% | 16.43% | False |
| distilled | u1 | missing | 163.40 | 16341 | 41.82% | 140.22% | 117.21% | 16.43% | False |
| distilled | shared_u1 | CS_PASSED | 165.10 | 16511 | 41.82% | 52.33% | 20.12% | 23.57% | True |

**Selected Variant CSim**
07C2 reran CSim on the first 4 full windows. The testbench internally calls `tcn_streaming_step` over 252 events per window, so this is 1008 streaming event calls.

| variant | readout rows | event calls | readout max abs diff | readout RMSE | joint sign agreement | alignment errors |
|---|---:|---:|---:|---:|---:|---:|
| scratch shared_u1 | 4 | 1008 | 0.149958 | 0.076498 | 1.000 | 0 |
| distilled shared_u1 | 4 | 1008 | 0.420075 | 0.308342 | 1.000 | 0 |

**Csynth And P&R**

| variant | csynth latency cycles | csynth interval cycles | csynth DSP/LUT/FF/BRAM | routed timing @100MHz | routed resources | verification tier |
|---|---:|---:|---|---|---|---|
| scratch shared_u1 | 16510 | 16511 | DSP 40.91%, LUT 52.12%, FF 19.87%, BRAM 23.57% | WNS 0.683 ns, TNS 0.000 ns, failing endpoints 0, timing_met=True | LUT 25.55%, FF 14.03%, DSP 37.27%, BRAM 23.57% | csim_csynth_pnr_met_cosim1008_timeout |
| distilled shared_u1 | 16510 | 16511 | DSP 41.82%, LUT 52.33%, FF 20.12%, BRAM 23.57% | WNS 0.482 ns, TNS 0.000 ns, failing endpoints 0, timing_met=True | LUT 25.69%, FF 14.11%, DSP 38.18%, BRAM 23.57% | csim_csynth_pnr_met_cosim1008_timeout |

100MHz P&R met timing for both selected variants, so the 90MHz fallback was not run.

**Speed Cost**
The resource-fit mechanism is effective but expensive. The original over-budget scratch `u4` point was 558 cycles / 5.58 us at csynth, while `shared_u1` is 16,510 cycles / 165.10 us. This is still much faster than 07B naive whole-window recomputation, but it is slower than the routed PHC 06F v8 baseline and does not meet the sub-10us goal. The selected rows therefore close resource and timing, not the latency objective.

**Function Hotspot**
The selected `shared_u1` design still spends most csynth resources inside the five TCN stream blocks, but binding the history buffers to BRAM removes the LUT/FF explosion seen in `u1` and `u2`.

| variant | module | BRAM_18K | DSP | LUT | FF |
|---|---|---:|---:|---:|---:|
| scratch | stage2_resource_stream_block_step_0_1_3_s | 10 | 16 | 4054 | 2736 |
| scratch | stage2_resource_stream_block_step_1_2_5_s | 10 | 16 | 4245 | 2824 |
| scratch | stage2_resource_stream_block_step_2_4_9_s | 10 | 14 | 4258 | 2847 |
| scratch | stage2_resource_stream_block_step_3_8_17_s | 10 | 22 | 4211 | 3066 |
| scratch | stage2_resource_stream_block_step_4_16_33_s | 10 | 18 | 4115 | 2825 |
| distilled | stage2_resource_stream_block_step_0_1_3_s | 10 | 20 | 4128 | 3002 |
| distilled | stage2_resource_stream_block_step_1_2_5_s | 10 | 18 | 4303 | 3089 |
| distilled | stage2_resource_stream_block_step_2_4_9_s | 10 | 14 | 4277 | 2844 |
| distilled | stage2_resource_stream_block_step_3_8_17_s | 10 | 18 | 4138 | 2800 |
| distilled | stage2_resource_stream_block_step_4_16_33_s | 10 | 18 | 4149 | 2823 |

**Cosim Boundary**
The requested 1000-event cosim was attempted on 1008 streaming event calls. Both variants got past RTL elaboration in the shortpath flow, unlike the first long-path attempt, but neither completed final compare within one hour:

| variant | attempted event calls | total RTL transactions | completed transactions at timeout | final compare report |
|---|---:|---:|---:|---|
| scratch shared_u1 | 1008 | 1012 | 78 | missing_timeout |
| distilled shared_u1 | 1008 | 1012 | 55 | missing_timeout |

This means the P&R-met rows are timing/resource evidence, not deployable hardware closure. They must not be listed as `csim+cosim+P&R_met`, and no RTL mismatch count can be claimed from the timed-out 1008-event attempts.

**Comparison Table**

| implementation | latency definition | verification level | latency | steady-state interval / throughput | routed timing | routed resources | note |
|---|---|---|---:|---:|---|---|---|
| PHC-lite 06F v8 baseline | PHC recurrent single step | existing_full_cosim_and_routed_pnr | 83.160 us | 83.160 us interval | WNS +0.020 ns @100MHz | LUT 14.20%, FF 7.72%, DSP 21.36%, BRAM 2.14% | Existing routed PHC baseline. |
| PHC-lite 06U V1 unroll16 | PHC recurrent single step | full_csim_1000_cosim_and_routed_pnr | 22.360 us | 22.360 us interval | WNS +0.004 ns @100MHz | LUT 28.34%, FF 19.73%, DSP 68.64%, BRAM 0.00% | Dense PHC speedup point with narrow WNS. |
| PHC-lite 06U V3 diagonal u8 localpart | PHC recurrent single step | full_csim_1000_cosim_and_routed_pnr | 5.300 us | 5.300 us interval | WNS +0.130 ns @100MHz | LUT 21.33%, FF 11.99%, DSP 89.09%, BRAM 0.00% | Fastest fully verified PHC-family point. |
| 07B2 naive TCN distilled | whole receptive-window recomputation | CSim pass + cosim timeout + P&R failed | 13,703.85 us | same as whole-window recomputation | 100MHz WNS -1.112 ns; 90MHz WNS -0.358 ns | LUT 24.85%, FF 11.94%, DSP 21.82%, BRAM18K equiv 22.50% | Not timing-closed. |
| 07C2 streaming TCN scratch shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met, but 1008-event cosim timeout | 165.100 us csynth / timing object | 165.110 us interval | WNS +0.683 ns @100MHz | LUT 25.55%, FF 14.03%, DSP 37.27%, BRAM 23.57% | Diagnostic only until cosim completes. |
| 07C2 streaming TCN distilled shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met, but 1008-event cosim timeout | 165.100 us csynth / timing object | 165.110 us interval | WNS +0.482 ns @100MHz | LUT 25.69%, FF 14.11%, DSP 38.18%, BRAM 23.57% | Diagnostic only until cosim completes. |

**Evidence Files**
Primary generated CSVs:
- `results/stage2_closure_hls_summary.csv`
- `results/stage2_closure_pnr_summary.csv`
- `results/stage2_closure_cosim_attempts.csv`
- `results/stage2_closure_comparison.csv`

Primary report artifacts:
- `artifacts/shortpath_scratch_shared_u1/syn_report/tcn_streaming_step_csynth.rpt`
- `artifacts/shortpath_scratch_shared_u1/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt`
- `artifacts/shortpath_scratch_shared_u1/cosim_1008_timeout_xsim.log`
- `artifacts/shortpath_distilled_shared_u1/syn_report/tcn_streaming_step_csynth.rpt`
- `artifacts/shortpath_distilled_shared_u1/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt`
- `artifacts/shortpath_distilled_shared_u1/cosim_1008_timeout_xsim.log`

**Honest Conclusion**
The resource bottleneck in 07C Stage 2 was not DSP after unroll reduction; it was LUT/FF pressure from fully materialized history storage. The `shared_u1` BRAM-backed history setting solves that resource problem and the routed RTL meets 100MHz for both scratch and distilled variants. The cost is severe: the resource-fit point regresses from the 5.58 us over-budget `u4` point to 165.10 us/event, so it is not the desired low-latency streaming solution. In addition, the required 1000-event RTL cosim did not finish within the one-hour attempts, so this task is only partially closed: resource-fit and P&R timing are resolved, but low-latency and full cosim closure remain open. The current valid claim is `csim + csynth + routed P&R timing-met, diagnostic only because cosim1008 timed out`.
