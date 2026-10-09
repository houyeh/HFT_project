status: closure_blocked_by_cosim_timeout_and_pnr_timing_failure
finding: 07b2_distilled_naive_full_window_tcn_csim_passes_1000_rows_but_cosim_did_not_complete_and_pnr_fails_at_100mhz_and_90mhz
# 07B2 TCN HLS Closure Audit

## Scope Boundary

- Work was confined to `n2ew_era/07b2_tcn_hls_closure/`; the original 07B directory was read/copied but not modified.
- No model retraining, checkpoint change, threshold change, or NEW5 access was performed.
- The selected closure target is the 07A/07B `distilled` TCN PrePad model because it is the stronger 07A policy row; scratch artifacts are copied into 07B2 but not accepted as closure evidence in this pass.
- Latency definition remains 07B naive full-window recomputation: one new event supplies a 128x16 window, the HLS top prepends 124 zeros, recomputes all five valid-conv TCN blocks, then emits bid/ask readout.

## Verification Evidence

| item | result | rows | max abs diff | mean abs diff | joint sign agreement | evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CSim | pass | 1000/1000 | 0.676136 | 0.271577 | 1.000000 | `n2ew_era/07b2_tcn_hls_closure/results/tcn_prepad_distilled_csim_out.csv` |
| RTL cosim | timeout/incomplete, no `hls_cosim.rpt` | 568/1000 partial CSV rows | 0.571899 | 0.267630 | 1.000000 | `n2ew_era/07b2_tcn_hls_closure/results/tcn_prepad_distilled_cosim_out.csv` plus missing `n2ew_era/07b2_tcn_hls_closure/hls_component_distilled/hls_component/reports/hls_cosim.rpt` |

The 1000-row cosim command was attempted and stopped after a 30-minute timeout. It reached C TB testing and left a partial CSV, but no HLS cosim report was produced. Therefore no RTL transaction latency/II is accepted from cosim.

## HLS Csynth Estimate

| latency cycles | interval cycles | estimated clock ns | latency @100MHz estimate | interval @100MHz estimate | BRAM_18K | DSP | FF | LUT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1370385 | 1370386 | 8.223 | 13703.850 us | 13703.860 us | 63 | 48 | 18325 | 25619 |

These latency/II numbers are csynth estimates only. They are kept for diagnosis but are not used as measured closure latency.

## Routed P&R

| target | WNS ns | TNS ns | setup failing endpoints | timing met | routed DSP | routed LUT | routed FF | routed BRAM_18K equiv | evidence |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 100MHz | -1.112 | -144.069 | 387 | false | 48 (21.82%) | 13218 (24.85%) | 12706 (11.94%) | 63.0 (22.50%) | `n2ew_era/07b2_tcn_hls_closure/hls_component_distilled/hls_component/hls/syn/verilog/direct_tcn_prepad_reports_distilled_100mhz/tcn_prepad_infer_timing_summary_routed.rpt` |
| 90MHz | -0.358 | -23.158 | 96 | false | 48 (21.82%) | 13201 (24.81%) | 12529 (11.78%) | 63.0 (22.50%) | `n2ew_era/07b2_tcn_hls_closure/hls_component_distilled/hls_component/hls/syn/verilog/direct_tcn_prepad_reports_distilled_90mhz/tcn_prepad_infer_timing_summary_routed.rpt` |

Resource fit is not the blocker: DSP/LUT/FF/BRAM all fit comfortably after routing. The closure blocker is timing: 100MHz fails with negative WNS, and the required 90MHz fallback also fails.

## PHC-Style Latency Comparison

| implementation | latency definition | verification level | single-decision latency | steady-state interval / throughput | routed timing | note |
| --- | --- | --- | ---: | ---: | --- | --- |
| PHC-lite v8 baseline | single recurrent step | routed P&R + cosim | 83.160 us | 83.170 us interval | met @100MHz | Source: 06F/06S carried-forward PHC baseline. |
| PHC-lite unroll16 seed2 | single recurrent step | routed P&R + cosim1000/full evidence mix | 25.733 us | 25.744 us interval | met @90MHz | Source: `n2ew_era/06s_unroll16_verification_closure/AUDIT_unroll16_verification_closure.md`. |
| TCN PrePad distilled 07B2 | naive full-window recomputation, receptive field 125 | CSim1000 pass; cosim timeout; P&R timing fail | not accepted | not accepted | fail @100MHz and @90MHz | Csynth estimate is 13703.850 us @100MHz / 15226.348 us @90MHz, but this is not measured closure latency. |

Important: TCN and PHC rows are not the same latency object. TCN here recomputes a whole window per decision; PHC processes one recurrent step. The table is a verification/closure comparison, not a direct architectural speed ranking.

## Streaming II=1 Follow-up Boundary

A separate 07C streaming effort already exists and reaches a resource-fitting csynth-only point (`shared_u1`, 16510 cycles), but it has no cosim or routed timing closure. Because the 07B2 baseline failed both the 1000-row cosim gate and 90MHz P&R gate, this pass did not spend additional time creating a new II=1 streaming variant inside 07B2. Any future streaming result must be reported separately as single-decision latency vs steady-state throughput.

## Honest Conclusion

The 07B2 closure task is diagnosed but not closed. The distilled naive full-window TCN C model passes 1000 validation rows, so the copied fixed-point datapath is numerically usable at CSim level. However, 1000-row RTL cosim did not complete and produced no `hls_cosim.rpt`, and routed P&R fails timing at both 100MHz and the requested 90MHz fallback. Therefore the existing 07B TCN latency cannot be promoted to a PHC-style routed/cosim-verified headline. The next engineering target should be the streaming/cached-activation TCN path, not further citation of the naive full-window recompute path.
