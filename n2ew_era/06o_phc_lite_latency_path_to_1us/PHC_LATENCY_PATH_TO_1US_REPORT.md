# PHC-lite Latency Bottleneck and Path to 1us Report

Status: `existing_tool_output_analysis_no_new_PnR_run`.

This report analyzes the frozen PHC-lite manual HLS implementation on Zynq-7020 and the already-run 06L/06G experiments. It does not rerun P&R, retrain a model, change HLS source, or touch NEW5.

## Executive Conclusion

The current 83.16us latency is not a timing-closure problem. The routed 100MHz implementation meets timing with WNS +0.020ns, but the single-step schedule has 8316 cycles and interval 8317. Because the recurrent state update makes the next event depend on the previous event's `state_next`, throughput II effectively equals single-step latency for one live state stream.

The dominant bottleneck is the repeated unpipelined 16x16 linear/matrix-vector loops, not LayerNorm sqrt/division. Six 16x16 linear blocks account for about 7956 visible cycles, or 95.67% of the 8316-cycle budget if counted as raw block latency. In the scheduler-attributed view, they still dominate the critical path.

The best existing measured lever is 06L inner-loop unroll factor 16: 2316 cycles / 23.16us at 100MHz, with csim/cosim passing and csynth resources still inside PYNQ-Z2 budget. This is a real 3.59x improvement over v8, but it is still 23.16x away from 1us at 100MHz and has not yet received routed P&R timing closure.

## Baseline Evidence

| item | value | source |
|---|---:|---|
| latency | 8316 cycles / 83.16us | 06F `phc_step_csynth.rpt` and RTL cosim |
| interval / II | 8317 cycles | 06F `phc_step_csynth.rpt` |
| P&R WNS | +0.020ns | 06F routed timing report |
| clock | 10.000ns / 100MHz | 06F routed timing report |
| inferred routed critical period | 9.980ns | 10.000ns - 0.020ns WNS |
| critical path shape | FSM register to DSP48E1 PCIN, data path delay 8.496ns | 06F routed timing report |
| routed utilization fingerprint | LUT 14.20%, FD/FF 7.72%, DSP 21.36%, RAMB/FIFO 2.14% | 06F `hls_impl_pnr.rpt` |

## Stage / Loop Latency Breakdown

| stage | kind | cycles | pct of 8316 | csynth entry |
|---|---|---:|---:|---|
| ENC_LINEAR | 16x16 linear | 1328 | 15.97% | VITIS_LOOP_101_1 |
| ENC_LAYER_NORM_VISIBLE | LayerNorm visible subtotal | 300 | 3.61% | mean+var+sqrt+normalize |
| ENC_SILU | SiLU activation | 176 | 2.12% | VITIS_LOOP_88_1 |
| GATE_LINEAR | 16x16 linear | 1328 | 15.97% | VITIS_LOOP_119_1 |
| CANDIDATE_LINEAR | 16x16 linear | 1329 | 15.98% | Outline_VITIS_LOOP_119_1 |
| GAIN_FROM_ENCODED_LINEAR | 16x16 linear | 1329 | 15.98% | Outline_VITIS_LOOP_119_11 |
| GAIN_FROM_PREV_STATE_LINEAR | 16x16 linear no-bias | 1313 | 15.79% | Outline_VITIS_LOOP_136_1 |
| SYNAPSE_LINEAR | 16x16 linear | 1329 | 15.98% | Outline_VITIS_LOOP_119_13 |
| DECAY_STATE_UPDATE | elementwise recurrence | 81 | 0.97% | Outline_VITIS_LOOP_233_1 |
| GAIN_ADD | elementwise add | 49 | 0.59% | Outline_VITIS_LOOP_241_2 |
| GAIN_SIGMOID | sigmoid activation | 145 | 1.74% | Outline_VITIS_LOOP_72_12 |
| MESSAGE_SILU | SiLU activation | 177 | 2.13% | Outline_VITIS_LOOP_88_1 |
| LOOP_INPUT_COMBINE | elementwise combine | 160 | 1.92% | VITIS_LOOP_250_3 |
| LOOP_LAYER_NORM_VISIBLE | LayerNorm visible subtotal | 300 | 3.61% | mean+var+sqrt+normalize |
| READOUT_LINEAR | 2x16 linear | 164 | 1.97% | VITIS_LOOP_154_1 |
| OUTPUT_COPY | readout/state copy wait group | 33 | 0.40% | max(3,33) |
| RESIDUAL_CONTROL | control/divider/handshake residual | 137 | 1.65% | 8316 - attributed total |

Category summary:

| category | cycles | pct of 8316 |
|---|---:|---:|
| six 16x16 linear blocks | 7956 | 95.67% |
| two LayerNorm visible subtotals | 600 | 7.22% |
| activation and elementwise loops | 821 | 9.87% |
| readout/output/residual | 334 | 4.02% |

Interpretation: The 16x16 matrix-vector loops are the latency wall. LayerNorm is visible and should not be ignored, but the current report data does not support treating it as the primary bottleneck.

## Clock Frequency Lever

This is a linear cycle-time estimate only. The current routed critical path is already 9.980ns. Therefore the unchanged 06F netlist would be expected to fail 150MHz and 200MHz timing unless the design is retimed/reimplemented and a fresh P&R timing report shows nonnegative WNS.

| frequency | period ns | latency if 8316 cycles unchanged | distance to 1us | timing evidence | slack proxy |
|---:|---:|---:|---:|---|---:|
| 100MHz | 10.000 | 83.16us | 83.16x | verified_pass | 0.020ns |
| 150MHz | 6.667 | 55.44us | 55.44x | not_verified_expected_fail_without_retiming | -3.313ns |
| 200MHz | 5.000 | 41.58us | 41.58x | not_verified_expected_fail_without_retiming | -4.980ns |

Conclusion: clocking alone cannot reach 1us. Even a hypothetical timing-clean 200MHz implementation of the current 8316-cycle schedule would be 41.58us.

## Existing Unroll Sweep Evidence

| factor | latency cycles | latency us | DSP | LUT | FF | BRAM | fits budget | speedup vs v8 | cosim |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|---|
| 1 | 8316 | 83.16 | 46 | 24836 | 14518 | 6 | True | 1.00x | PASS |
| 2 | 5036 | 50.36 | 56 | 25601 | 16162 | 12 | True | 1.65x | PASS |
| 4 | 3724 | 37.24 | 57 | 27456 | 17000 | 0 | True | 2.23x | PASS |
| 8 | 3068 | 30.68 | 70 | 30392 | 19704 | 0 | True | 2.71x | PASS |
| 16 | 2316 | 23.16 | 121 | 36792 | 35774 | 0 | True | 3.59x | PASS |

The unroll factor 16 point is the best existing data-backed latency/resource tradeoff: it uses 121 DSP (55%), 36,792 LUT (69%), and 35,774 FF (33%), all under Zynq-7020 budget in csynth. It still needs routed timing closure before it can replace the 06F frozen baseline.

## Lever Tradeoff Table

| lever | best supported latency | improvement vs v8 | distance to 1us | resource cost | evidence / status | recommendation |
|---|---:|---:|---:|---|---|---|
| clock 100->150MHz on unchanged 06F netlist | 55.44us | 1.50x | 55.44x | none if same netlist | no new P&R; existing 100MHz WNS implies likely fail; unverified_expected_fail_without_retiming | do not claim until fresh 150MHz P&R shows nonnegative WNS |
| clock 100->200MHz on unchanged 06F netlist | 41.58us | 2.00x | 41.58x | none if same netlist | no new P&R; existing 100MHz WNS implies likely fail; unverified_expected_fail_without_retiming | not credible without major retiming/re-P&R |
| linear helper unroll factor 16 (06L) | 23.16us | 3.59x | 23.16x | DSP 121/220, LUT 36792/53200, FF 35774/106400, BRAM 0/280 | 06L csim+csynth+cosim, no P&R timing yet; synthesis_and_cosim_pass_route_unverified | best measured latency lever so far; next candidate for P&R timing |
| Q16.8 quantization only | 83.16us | 1.00x | 83.16x | DSP unchanged 46, LUT slightly lower 24746 vs 24836; numerical error much worse | 06l_OLD phase2 csim+csynth, no P&R; not_recommended | not useful for latency; do not use as path to 1us |
| diagonal/structured recurrence replacement (06G) | 20.87us | 3.98x | 20.87x | DSP 34/220, LUT 16058/53200, FF 9066/106400, BRAM 1/280 | 06G csynth only; different model, not PHC-lite production; route_unverified_and_model_regresses_vs_best_sequence_model | architectural clue, not a drop-in replacement |

## Partial Time-Step Unroll / Recurrent Pipeline Assessment

For a single live market state stream, partial unroll across time is not causally valid: event t+1 depends on `state_next` from event t. This is why the design has interval approximately equal to latency. Multi-stream interleaving could improve aggregate throughput across independent symbols/sessions, but it would not reduce the single-stream decision latency needed for an HFT quote gate.

To reach 1us at 100MHz, the step must be at most 100 cycles. The best existing PHC-lite-style datapoint is 2316 cycles, so the gap is still about 23x. At 200MHz, that same factor-16 design would still be 11.58us if timing closed, which remains about 11.6x from 1us.

## Concrete Technical Path Toward 1us

1. First route the 06L factor-16 design. This is the lowest-risk next experiment because csim/cosim already pass and csynth fits resources. It can potentially reduce 83.16us to 23.16us, but timing closure must be verified.
2. If factor-16 routes, test modest clock targets only after routing: 125MHz or 150MHz before 200MHz. The current 06F routed path implies unchanged 150MHz/200MHz would fail, so fresh P&R WNS is mandatory.
3. To move below ~10us, the model architecture must reduce the number of dense 16x16 linear blocks or replace them with structured/low-rank/diagonal transforms. The 06G diagonal experiment gives a 2087-cycle csynth datapoint, but it is a different model and not a drop-in replacement.
4. To approach 1us, dense PHC-lite needs a deeper architecture/HLS redesign: e.g. fewer dense blocks, parallel MAC trees with more DSP than factor16, carefully placed pipeline registers inside state-dependent computations, and potentially a simplified activation/normalization path. This is not achievable by clock frequency or Q16.8 quantization alone on the current Zynq-7020 evidence.

## Claim Boundary

- This report uses existing csynth/cosim/P&R artifacts only; no new P&R was run here.
- 150MHz/200MHz numbers are cycle-time projections, not verified timing-closed implementations.
- No FPGA PnL, exchange replay, or real-world profitability claim is made.
- The 06L factor-16 result is not a frozen replacement until it receives routed P&R timing closure and a separate freeze decision.
