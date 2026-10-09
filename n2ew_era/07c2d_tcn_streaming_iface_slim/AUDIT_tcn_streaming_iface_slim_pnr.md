status: resolved_iface_slim_pnr_met

# 07C2D TCN Streaming Interface-Slim P&R Audit

07C2D is the first TCN streaming line from 07C onward to carry the same shared_u4 RTL through CSim, csynth, 1008-event RTL cosim, and routed P&R timing closure at 90MHz or better.

## Scope Boundary

- Work directory: `n2ew_era/07c2d_tcn_streaming_iface_slim/`.
- No changes were made to `n2ew_era/07c2_tcn_streaming_closure/`, `n2ew_era/07c2b_tcn_streaming_speed/`, or `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/`.
- No model retraining, no weight changes, no NEW5 access, and no change to shared_u4 unroll/BRAM settings or the slim top interface.
- The 1008-event 07C2D slim csynth/RTL product was reused because every actually synthesized source file matched the current 07C2D workspace source by SHA-256: `true`.
- P&R was run with the 07C2B-style direct non-project Vivado flow using `run_direct_tcn_streaming_pnr_custom_clock.tcl`. The only script correction in the 07C2D copy was to load the generated clock via XDC, matching the 07C2B flow.

## CSynth Reuse Check

| check | matched | sha256 |
| --- | --- | --- |
| scratch wrapper | true | A1BB01811EAD... |
| distilled wrapper | true | 7A963BE9741E... |
| prepad scratch | true | BB4AFFC6ABBF... |
| prepad distilled | true | BB4AFFC6ABBF... |
| streaming header scratch | true | 38509B9759EF... |
| streaming header distilled | true | 38509B9759EF... |
| impl scratch | true | A840F69FA783... |
| impl distilled | true | A840F69FA783... |
| weights scratch | true | CCF2315DBF40... |
| weights distilled | true | 0FD3FE86E2D4... |

The HLS configs for the reused 1008-event products synthesize only the slim wrapper, `tcn_prepad_hls.hpp`, `tcn_stage2_streaming_hls.hpp`, `tcn_stage2_resource_impl.hpp`, and the matching weight header. Legacy copied files left in the temporary source directory were not referenced by those configs.

## 1008-Event Cosim Evidence

| variant | RTL tx | readout rows | xsim peak KB | xsim wall | readout max diff | readout RMSE | joint sign | valid errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 1012/1012 | 4 | 234784.0 | 1000 s | 0.149958 | 0.076498 | 1 | 0 |
| distilled | 1012/1012 | 4 | 234812.0 | 954 s | 0.420075 | 0.308342 | 1 | 0 |

The cosim evidence is the existing 07C2D slim-interface 1008-event run. It is readout-only at the top interface, with final readout sign agreement of 1.0 and zero valid-alignment errors for both variants.

## Routed P&R Results

| run | variant | clock | WNS | TNS | setup fail | hold fail | timing met | LUT | FF | DSP | BRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_shared_u4_100mhz | scratch | 100.0 MHz | -0.150 | -7.213 | 48 | 0 | false | 31.39% | 14.45% | 51.36% | 23.57% |
| scratch_shared_u4_90mhz | scratch | 90.0 MHz | 0.746 | 0.000 | 0 | 0 | true | 31.39% | 14.36% | 51.36% | 23.57% |
| distilled_shared_u4_100mhz | distilled | 100.0 MHz | 0.187 | 0.000 | 0 | 0 | true | 32.00% | 14.44% | 49.09% | 23.57% |

Scratch did not meet 100MHz after interface slimming, so the required 90MHz fallback was run and met timing. Distilled met timing at 100MHz, so no 90MHz fallback was required for distilled.

## Selected Closure Points

| variant | selected clock | verification tier | latency cycles | latency at selected clock | interval cycles | timing met |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | 90.0 MHz | csim+csynth+cosim_1008event+routed_pnr_met | 7438 | 82.64 us | 7439 | true |
| distilled | 100.0 MHz | csim+csynth+cosim_1008event+routed_pnr_met | 7438 | 74.38 us | 7439 | true |

Latency/interval are still the HLS streaming-step latency and interval objects from csynth. They should not be ranked directly against PHC recurrent single-step latency or naive TCN whole-window recompute latency.

## Interface Effect Versus 07C2B Old Top

| variant | old/new 100MHz timing | WNS delta | LUT old -> new | FF old -> new | DSP old -> new | BRAM old -> new |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | true -> false | -0.353 ns | 35.36% -> 31.39% (-3.97 pp) | 16.09% -> 14.45% (-1.64 pp) | 44.55% -> 51.36% (+6.81 pp) | 23.57% -> 23.57% (+0.00 pp) |
| distilled | true -> true | +0.034 ns | 35.19% -> 32.00% (-3.19 pp) | 16.01% -> 14.44% (-1.57 pp) | 41.82% -> 49.09% (+7.27 pp) | 23.57% -> 23.57% (+0.00 pp) |

At 100MHz, interface slimming reduced routed LUT and FF use for both variants, left BRAM unchanged, and increased routed DSP use. Timing moved in different directions: distilled improved slightly and still met 100MHz, while scratch moved from passing in 07C2B to failing by -0.150ns in 07C2D. Therefore the measured effect is not simply "fewer ports means everything improves"; the reliable conclusion is narrower: the slim interface solved cosim memory scaling and reduced LUT/FF, but scratch needs 90MHz for routed timing under this flow.

## Evidence Files

| artifact | path |
| --- | --- |
| P&R summary CSV | results/iface_slim_pnr_summary.csv |
| old vs new comparison CSV | results/iface_slim_pnr_old_vs_new.csv |
| csynth reuse hash CSV | results/iface_slim_csynth_reuse_hash_check.csv |
| scratch 100MHz routed timing | artifacts/scratch_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt |
| scratch 90MHz routed timing | artifacts/scratch_shared_u4/pnr_reports_90mhz/tcn_streaming_step_timing_summary_routed.rpt |
| distilled 100MHz routed timing | artifacts/distilled_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt |
| Vivado logs | artifacts/pnr_logs/ |
| 1008-event cosim metrics | results/iface_slim_cosim_metrics.csv |

## Honest Conclusion

The 07C2D interface-slim route closes the verification chain that 07C2B could not close: both shared_u4 variants have CSim, csynth, 1008-event RTL cosim, and routed P&R timing-met evidence. The final routed clocks are 90MHz for scratch and 100MHz for distilled. This is a full validation milestone for the slim-interface streaming TCN, but it does not change the latency definition: TCN streaming remains an amortized per-event streaming-step result, not the same latency object as PHC-lite's recurrent single-step path.
