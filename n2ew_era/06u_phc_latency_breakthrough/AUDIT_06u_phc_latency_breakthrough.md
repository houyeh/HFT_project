status: mixed
finding: fastest_valid_candidate_is_v3_diagonal_unroll8_localpart_100mhz_530_rtl_cycles_5p300us;_111_125mhz_are_timing_only_due_cosim_failure
# AUDIT 06U PHC Latency Breakthrough

## Scope

This audit records the 06U PHC latency design-space scan. The work was isolated under:

`n2ew_era/06u_phc_latency_breakthrough/`

No existing phase directory was modified. No model was retrained. NEW5 was not loaded, read, or used. Each variant was gated by CSim before csynth. PnL/signal-quality numbers below are imported only from frozen existing phase result CSVs.

## Summary

The fastest valid routed candidate is:

| item | value |
| --- | ---: |
| variant | `v3_diagonal16_unroll8_localpart_100mhz` |
| model source | 05F diagonal SSM, seed2, best_epoch 157 |
| CSim | 25,600 vectors, 0 mismatch |
| csynth | 532 cycles, 5.320 us |
| cosim | 1,000 vectors, Verilog pass, 0 mismatch |
| RTL cosim latency used for deployable comparison | 530 cycles |
| routed frequency | 100 MHz |
| routed latency | 5.300 us |
| routed WNS/TNS | +0.130 ns / 0.000 ns |
| post-route resources | LUT 11,348 (21.33%), FF 12,760 (11.99%), DSP 196 (89.09%), BRAM_18K 0 (0.00%) |
| speedup vs 06F 83.16 us baseline | 15.69x |

This is a latency result, not a new signal-quality result. Its signal-quality source is the already frozen 05F diagonal SSM result, which is weaker than the 05C4 dense PHC/STP signal source on val/NEW3/NEW4. Therefore the result should be described as a latency/resource/signal-quality tradeoff, not a strict dominance result.

## Verification Boundary

| category | boundary |
| --- | --- |
| Data | Only existing HLS vectors and existing frozen signal-quality CSVs were used. NEW5 was not touched. |
| Models | Existing 05C4/05E/05F checkpoints only. No retraining. |
| Existing phases | Read-only references only. 06F/06G/06L/06N2/05C4/05E/05F were not modified. |
| CSim gate | Every variant that entered csynth first passed CSim with 0 mismatches on the available 25,600-vector set. |
| Cosim gate | A variant is called deployable only if RTL cosim passed. V4 111/125 MHz are timing-only diagnostics because cosim failed. |
| P&R label | Routed timing reports are separated from csynth estimates and from cosim validity. |

## Final Valid Candidates

| variant | model/signal source | MHz | RTL cycles | latency_us | routed WNS/TNS | post-route resources | verification level | primary reports |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 06F v8 baseline | 05C4 dense PHC/STP seed2 | 100 | 8,316 | 83.160 | +0.020 / 0.000 | LUT 14.20%, FF 7.72%, DSP 21.36%, BRAM 2.14% | existing full cosim + routed P&R | csynth `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`; cosim `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/sim/report/phc_step_cosim.rpt`; P&R `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/report/phc_step_timing_routed.rpt` |
| V1 unroll16 + dataflow | 05C4 dense PHC/STP seed2 | 90 | 2,230 | 24.778 | +0.352 / 0.000 | LUT 28.61%, FF 19.46%, DSP 68.64%, BRAM 0.00% | full CSim + full cosim + routed P&R | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_90mhz_shortpath/syn/phc_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_90mhz_shortpath/cosim/phc_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_90mhz_shortpath/impl/phc_step_timing_routed.rpt` |
| V1 unroll16 + dataflow | 05C4 dense PHC/STP seed2 | 100 | 2,236 | 22.360 | +0.004 / 0.000 | LUT 28.34%, FF 19.73%, DSP 68.64%, BRAM 0.00% | full CSim + 1,000-vector cosim + routed P&R | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_100mhz_shortpath/syn/phc_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_100mhz_shortpath/cosim/phc_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v1_unroll16_dataflow_100mhz_shortpath/impl/phc_step_timing_routed.rpt` |
| V2 hidden8 unroll8 | 05E hidden8 seed1, median val seed | 100 | 1,202 | 12.020 | +0.445 / 0.000 | LUT 22.87%, FF 13.68%, DSP 59.55%, BRAM 0.00% | full CSim + 1,000-vector cosim + routed P&R | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v2_hidden8_unroll8_100mhz_shortpath/syn/phc_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v2_hidden8_unroll8_100mhz_shortpath/cosim/phc_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v2_hidden8_unroll8_100mhz_shortpath/impl/phc_step_timing_routed.rpt` |
| V3 diagonal unroll8 + local array partition | 05F diagonal SSM seed2 | 100 | 530 | 5.300 | +0.130 / 0.000 | LUT 21.33%, FF 11.99%, DSP 89.09%, BRAM 0.00% | full CSim + 1,000-vector cosim + routed P&R | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll8_localpart_100mhz_shortpath/syn/phc_diag_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll8_localpart_100mhz_shortpath/cosim/phc_diag_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll8_localpart_100mhz_shortpath/impl/phc_diag_step_timing_routed.rpt` |

Machine-readable table:

`n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv`

## Invalid Or Timing-Only Diagnostics

These runs are useful diagnostics but should not be reported as deployable accelerator points.

| variant | CSim | csynth | cosim/P&R outcome | status | evidence |
| --- | --- | --- | --- | --- | --- |
| V3 diagonal unroll16 100 MHz | 25,600 vectors, 0 mismatch | 383 cycles, 3.830 us, DSP 452/220 | P&R not pursued | invalid: over DSP budget | `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll16_100mhz_shortpath/syn/phc_diag_step_csynth.rpt` |
| V3 diagonal unroll8 without local partition | 25,600 vectors, 0 mismatch | 556 cycles, 5.560 us, DSP 66/220 | cosim failed: 268 mismatches / 1,000 | invalid: C/RTL mismatch | `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll8_100mhz_shortpath/cosim/phc_diag_step_cosim.rpt` |
| V3 diagonal unroll4 without local partition | 25,600 vectors, 0 mismatch | 750 cycles, 7.500 us, DSP 55/220 | cosim failed: 657 mismatches / 1,000 | invalid: C/RTL mismatch | `n2ew_era/06u_phc_latency_breakthrough/artifacts/v3_diagonal16_unroll4_100mhz_shortpath/cosim/phc_diag_step_cosim.rpt` |
| V4 best V3 u8 localpart 111 MHz | 25,600 vectors, 0 mismatch | 723 cycles, 6.514 us | cosim failed: 429 mismatches / 1,000; routed timing met WNS +0.472 ns | timing-only, not deployable | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_111mhz_shortpath/syn/phc_diag_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_111mhz_shortpath/cosim/phc_diag_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_111mhz_shortpath/impl/phc_diag_step_timing_routed.rpt` |
| V4 best V3 u8 localpart 125 MHz | 25,600 vectors, 0 mismatch | 733 cycles, 5.864 us | cosim failed: 429 mismatches / 1,000; routed timing met WNS +0.259 ns | timing-only, not deployable | csynth `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_125mhz_shortpath/syn/phc_diag_step_csynth.rpt`; cosim `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_125mhz_shortpath/cosim/phc_diag_step_cosim.rpt`; P&R `n2ew_era/06u_phc_latency_breakthrough/artifacts/v4_best_v3_u8lp_125mhz_shortpath/impl/phc_diag_step_timing_routed.rpt` |

## Signal Quality Sources

These are existing frozen evaluation results, not newly rerun selection or training.

| hardware variant family | signal-quality source | selected seed/checkpoint | val action_pnl | NEW3 action_pnl | NEW4 action_pnl | selection flags |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 06F baseline and V1 dense PHC/STP | `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_v1_hgb_queue.csv` | seed2, best_epoch 171, threshold 0.0 | 2,037,925 | 5,237,410 | 1,871,685 | NEW3/NEW4/NEW5 not used for selection |
| V2 hidden8 | `n2ew_era/05e_phc_hidden8_ablation/results/phc_stp_v2_seed_stability.csv` | seed1 median val, best_epoch 196, threshold 1.2433674931526184 | 1,908,845 | 5,032,255 | 1,755,610 | NEW3/NEW4/NEW5 not used for selection |
| V3/V4 diagonal SSM | `n2ew_era/05f_phc_diagonal_ssm/results/phc_diagonal_ssm_vs_v1_hgb_queue.csv` | seed2, best_epoch 157, threshold 0.0 | 1,771,525 | 4,727,425 | 1,642,680 | NEW3/NEW4/NEW5 not used for selection |

## Mechanism Notes

V1 tested whether 06L-style unroll16 and 06N2-style main-chain/gain-chain dataflow stack. They do partly stack: the deployable dense PHC point moves from the 06F 8,316-cycle baseline to 2,236 RTL cycles at 100 MHz. The cost is a much larger DSP footprint and very narrow 100 MHz routed WNS (+0.004 ns).

V2 tested whether reducing hidden dimension from 16 to 8 gives a cleaner latency/resource point. It does: 1,202 RTL cycles at 100 MHz with +0.445 ns WNS and lower post-route DSP than V1. The tradeoff is that it uses the weaker but still frozen 05E hidden8 signal source.

V3 is the largest latency lever. Removing dense 16x16 recurrent matmuls and using the diagonal SSM recurrence reduces valid routed latency to 530 RTL cycles at 100 MHz. The best working implementation needed local array partitioning in the unroll8 variant; without that, CSim passed but RTL cosim failed, so the non-localpartition variants are invalid. The unroll16 variant was faster in csynth only but exceeded the Zynq-7020 DSP budget.

V4 showed that routed timing closure at 111/125 MHz is not the immediate blocker for the best V3 design. Both runs met routed timing, but both failed RTL cosim with 429 mismatches out of 1,000. These numbers must be treated as timing-only diagnostics, not valid accelerator latency claims.

## Conclusion

The 06U scan found a valid sub-10us PHC-family hardware point: V3 diagonal SSM with unroll8 and local array partitioning reaches 530 RTL cycles, 5.300 us at 100 MHz, with routed timing met and 1,000-vector RTL cosim passing. That is 15.69x faster than the existing 83.16 us 06F dense PHC-lite baseline. The honest caveat is signal quality: this fastest point relies on the existing 05F diagonal SSM model, whose frozen val/NEW3/NEW4 action_pnl is below the 05C4 dense PHC/STP source used by the 06F baseline and V1. The clean final claim is therefore a latency-resource-signal-quality tradeoff, with V3 as the fastest verified hardware point and V1/V2 as intermediate points that preserve more of the dense/hidden PHC family behavior.
