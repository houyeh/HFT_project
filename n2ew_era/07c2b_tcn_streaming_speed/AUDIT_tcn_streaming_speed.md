status: mixed_speed_improved_resource_fit_and_pnr_met_but_cosim1008_timeout
finding: shared-history plus unroll=4 is the fastest tested resource-fit point, improving 07C2 shared_u1 from 16,510 cycles to 7,438 cycles, but 1008-event RTL cosim timed out after a two-hour budget. The P&R-met rows are diagnostic only, not deployable closure.

**Scope Boundary**
This audit lives in `n2ew_era/07c2b_tcn_streaming_speed/`. The workdir was created by copying 07C2 outputs as a starting point; `n2ew_era/07c_tcn_streaming_manual_hls/` and `n2ew_era/07c2_tcn_streaming_closure/` were not modified. The only HLS source changes are inside this 07C2B copy. No model weights were changed, no retraining was performed, and NEW5 was not touched.

The latency object here is streaming TCN per-event top-call latency and interval. PHC rows are recurrent single-step latency; 07B naive TCN rows are whole receptive-window recomputation. These objects are not directly rankable as one common latency number.

**Shared-History Unroll Scan**
The shared BRAM history architecture stayed enabled while scanning channel unroll. CSim used 4 full windows, i.e. 1008 streaming event calls, and zero valid-alignment errors was required before csynth.

| variant | unroll | run source | CSim | align err | lat cycles | lat us | interval cycles | DSP | LUT | FF | BRAM | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distilled | 1 | 07c2_carried_forward | pass | 0 | 16510 | 165.10 | 16511 | 41.82% | 52.33% | 20.12% | 23.57% | true |
| distilled | 2 | 07c2b_current | pass | 0 | 10030 | 100.30 | 10031 | 45.45% | 58.61% | 21.50% | 23.57% | true |
| distilled | 4 | 07c2b_current | pass | 0 | 7438 | 74.38 | 7439 | 45.45% | 70.44% | 22.90% | 23.57% | true |
| scratch | 1 | 07c2_carried_forward | pass | 0 | 16510 | 165.10 | 16511 | 40.91% | 52.12% | 19.87% | 23.57% | true |
| scratch | 2 | 07c2b_current | pass | 0 | 10030 | 100.30 | 10031 | 48.18% | 58.74% | 21.75% | 23.57% | true |
| scratch | 4 | 07c2b_current | pass | 0 | 7438 | 74.38 | 7439 | 48.18% | 70.62% | 23.15% | 23.57% | true |

Result: the highest tested resource-fit unroll is `TCN_STAGE2_CHANNEL_UNROLL=4` for both scratch and distilled. Against the carried-forward 07C2 shared_u1 point, scratch improved by 2.22x and distilled improved by 2.22x in csynth latency cycles.

**Latency vs Throughput Pipeline**
Top-level `#pragma HLS PIPELINE` was attempted on the selected u4 setting, but Vitis did not actually apply it because it conflicts with existing inner `PIPELINE off` pragmas in `tcn_stage2_resource_impl.hpp`. The csynth top report still says `PipelineType=no`, so this run is a pragma-conflict diagnostic, not a measurement of pipeline effectiveness.

| variant | top PIPELINE | PipelineType | lat cycles | interval cycles | delta lat | delta interval | DSP | LUT | FF | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | false | no | 7438 | 7439 | 0 | 0 | 48.18% | 70.62% | 23.15% | true |
| scratch | true | no | 7438 | 7439 | 0 | 0 | 48.18% | 70.62% | 23.15% | true |
| distilled | false | no | 7438 | 7439 | 0 | 0 | 45.45% | 70.44% | 22.90% | true |
| distilled | true | no | 7438 | 7439 | 0 | 0 | 45.45% | 70.44% | 22.90% | true |

Pipeline warning observed in both pipeline variants: `pragma conflict between PIPELINE ... and PIPELINE off ... Ignoring PIPELINE to honor PIPELINE off`. Therefore the reported steady-state interval remains 7,439 cycles. The true throughput effect of a reconciled pipeline design has not yet been tested, and this audit should not be read as evidence that pipelining is ineffective for this architecture.

**Local Array Partition Diagnostic**
Because u4 was still much slower than the sub-10us target, one local-tap buffering experiment was run on scratch. It passed CSim, but worsened both latency and LUT/FF, so the distilled localtap run was skipped as diagnostic-only.

| variant | status | CSim | lat cycles | interval cycles | delta lat vs scratch u4 | DSP | LUT | FF | fit | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | completed_diagnostic | pass | 7928 | 7929 | 490 | 48.18% | 75.77% | 36.33% | true | local fully-partitioned tap buffer; active only in this wrapper |
| distilled | skipped_after_scratch_regression | not_run |  |  |  |  |  |  |  | staged but not run because scratch localtap worsened latency and LUT/FF |

**Selected U4 Hotspot Snapshot**
The five streaming blocks remain the dominant csynth instance-level resource consumers. Shared BRAM history keeps total resources within budget, but it serializes enough work that u4 still costs 7,438 cycles.

| variant | module | BRAM_18K | DSP | LUT | FF |
| --- | --- | --- | --- | --- | --- |
| scratch_shared_u4 | stage2_resource_stream_block_step_0_1_3_s | 10 | 20 | 5880 | 3366 |
| scratch_shared_u4 | stage2_resource_stream_block_step_1_2_5_s | 10 | 22 | 6148 | 3716 |
| scratch_shared_u4 | stage2_resource_stream_block_step_2_4_9_s | 10 | 14 | 6001 | 3211 |
| scratch_shared_u4 | stage2_resource_stream_block_step_3_8_17_s | 10 | 24 | 6054 | 3692 |
| scratch_shared_u4 | stage2_resource_stream_block_step_4_16_33_s | 10 | 18 | 5857 | 3189 |
| distilled_shared_u4 | stage2_resource_stream_block_step_0_1_3_s | 10 | 20 | 5869 | 3369 |
| distilled_shared_u4 | stage2_resource_stream_block_step_1_2_5_s | 10 | 18 | 6056 | 3455 |
| distilled_shared_u4 | stage2_resource_stream_block_step_2_4_9_s | 10 | 16 | 6103 | 3471 |
| distilled_shared_u4 | stage2_resource_stream_block_step_3_8_17_s | 10 | 20 | 5963 | 3427 |
| distilled_shared_u4 | stage2_resource_stream_block_step_4_16_33_s | 10 | 18 | 5857 | 3187 |

**Cosim Completeness**
The final selected u4 setting was launched for 1008 streaming event calls with a two-hour timeout budget for each variant. Both attempts entered RTL simulation, but neither produced `hls_cosim.rpt` or a final compare result.

| variant | event calls | timeout s | RTL transactions | completed at timeout | final report | accepted pass |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | 1008 | 7200 | 1012 | 132 | missing_timeout | false |
| distilled | 1008 | 7200 | 1012 | 177 | missing_timeout | false |

This is the closure boundary: the u4 P&R data below is valid timing/resource evidence, but must not be cited as deployable `csim+cosim+P&R_met`.

**P&R**
100MHz P&R met timing for both selected u4 variants, so the 90MHz fallback was not run.

| variant | clock | WNS | TNS | failing endpoints | timing met | routed DSP | routed LUT | routed FF | routed BRAM | tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 100MHz | 0.203 ns | 0.000 ns | 0 | true | 44.55% | 35.36% | 16.09% | 23.57% | diagnostic_pnr_met_cosim_timeout |
| distilled | 100MHz | 0.153 ns | 0.000 ns | 0 | true | 41.82% | 35.19% | 16.01% | 23.57% | diagnostic_pnr_met_cosim_timeout |

**Comparison Boundary Table**
Each row states its latency definition. Streaming TCN, naive TCN, and PHC should be read as methodology comparisons, not one direct speed ranking.

| implementation | latency definition | verification | latency us | interval us | P&R timing | post-route resources | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHC-lite 06F v8 baseline | PHC recurrent single step | existing_full_cosim_and_routed_pnr | 83.160 | 83.160 | WNS +0.020 ns, timing_met=true | LUT 14.20%, FF 7.72%, DSP 21.36%, BRAM 2.14% | existing PHC-family reference |
| PHC-lite 06U V1 unroll16 | PHC recurrent single step | full_csim_1000_cosim_and_routed_pnr | 22.360 | 22.360 | WNS +0.004 ns, timing_met=true | LUT 28.34%, FF 19.73%, DSP 68.64%, BRAM 0.00% | existing PHC-family reference |
| PHC-lite 06U V3 diagonal u8 localpart | PHC recurrent single step | full_csim_1000_cosim_and_routed_pnr | 5.300 | 5.300 | WNS +0.130 ns, timing_met=true | LUT 21.33%, FF 11.99%, DSP 89.09%, BRAM 0.00% | existing PHC-family reference |
| 07B2 naive TCN distilled | whole receptive-window recomputation | CSim pass + cosim timeout + P&R failed | 13703.850 | 13703.860 | WNS -1.112 ns, timing_met=False | LUT 24.85%, FF 11.94%, DSP 21.82%, BRAM 22.50% | naive whole-window TCN is not timing-closed |
| 07C2 streaming TCN scratch shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met; cosim1008_timeout | 165.100 | 165.110 | WNS +0.683 ns, timing_met=True | LUT 25.55%, FF 14.03%, DSP 37.27%, BRAM 23.57% | 07C2 carried-forward baseline |
| 07C2 streaming TCN distilled shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met; cosim1008_timeout | 165.100 | 165.110 | WNS +0.482 ns, timing_met=True | LUT 25.69%, FF 14.11%, DSP 38.18%, BRAM 23.57% | 07C2 carried-forward baseline |
| 07C2B streaming TCN scratch shared_u4 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met; cosim1008_timeout | 74.380 | 74.390 | WNS +0.203 ns, timing_met=True | LUT 35.36%, FF 16.09%, DSP 44.55%, BRAM 23.57% | diagnostic only; cosim timed out at 132/1012 transactions |
| 07C2B streaming TCN distilled shared_u4 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met; cosim1008_timeout | 74.380 | 74.390 | WNS +0.153 ns, timing_met=True | LUT 35.19%, FF 16.01%, DSP 41.82%, BRAM 23.57% | diagnostic only; cosim timed out at 177/1012 transactions |

**Evidence Files**
| artifact | path |
| --- | --- |
| unroll scan CSV | results/stage2_speed_unroll_scan.csv |
| pipeline scan CSV | results/stage2_speed_pipeline_scan.csv |
| local partition diagnostic CSV | results/stage2_speed_local_partition_diagnostic.csv |
| cosim attempts CSV | results/stage2_speed_cosim_attempts.csv |
| P&R summary CSV | results/stage2_speed_pnr_summary.csv |
| comparison CSV | results/stage2_speed_comparison.csv |
| scratch u4 csynth report | artifacts/scratch_shared_u4/syn_report/tcn_streaming_step_csynth.rpt |
| distilled u4 csynth report | artifacts/distilled_shared_u4/syn_report/tcn_streaming_step_csynth.rpt |
| scratch u4 routed timing | artifacts/scratch_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt |
| distilled u4 routed timing | artifacts/distilled_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt |
| scratch u4 timeout log | artifacts/scratch_shared_u4/cosim_1008_timeout_xsim.log |
| distilled u4 timeout log | artifacts/distilled_shared_u4/cosim_1008_timeout_xsim.log |

**Honest Conclusion**
07C2B answered the specific speed question: shared-history can be combined with higher unroll, and unroll=4 is the fastest tested resource-fit point. It cuts the 07C2 shared_u1 latency from 165.10 us/event to 74.38 us/event at 100MHz while still meeting Zynq-7020 resource budgets and routed timing. However, this remains far above the <10us objective, the top-level pipeline pragma was ignored due to inner `PIPELINE off` conflicts, and the local-tap partition experiment regressed. Most importantly, the required 1000-event RTL cosim did not finish in the two-hour attempts, so the final status is diagnosed but not deployable: resource fit and 100MHz P&R are resolved, full cosim closure remains open.
