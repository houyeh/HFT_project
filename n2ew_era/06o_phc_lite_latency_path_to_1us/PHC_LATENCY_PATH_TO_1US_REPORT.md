# PHC-lite Latency Bottleneck and Path to 1us Report

Status: `updated_with_90mhz_routed_unroll16_result_no_new_runs`.

This report analyzes the frozen PHC-lite manual HLS implementation on Zynq-7020 and the already-run 06L/06G experiments. It does not rerun P&R, retrain a model, change HLS source, or touch NEW5.

Resource percentages use the PYNQ-Z2 / Zynq-7020 budget: LUT 53,200, FF 106,400, DSP 220, BRAM_18K 280.

## Verification Level Labels

Latency and resource numbers in this roadmap use the following evidence labels:

| Label | Meaning |
|---|---|
| `estimate_only` | Arithmetic or architectural estimate only; no HLS synthesis result for that exact point. |
| `csynth_only` | Vitis HLS synthesis report exists, but no RTL cosim or routed P&R closure for that exact point. |
| `csim_cosim_pass` | C simulation and RTL cosimulation passed; routed timing is not met or not verified. |
| `routed_pnr_failed` | Routed implementation was attempted and timing did not close. |
| `routed_pnr_met` | Routed implementation timing met with nonnegative WNS/TNS and zero failing endpoints. |

## Executive Conclusion

The original 06F frozen baseline remains a valid routed implementation: `8316 cycles / 83.16us @ 100MHz [routed_pnr_met@100MHz]`. It meets timing with WNS `+0.020ns`, but the single-step schedule is long and the recurrent state feedback makes the next event depend on the previous event's `state_next`.

The best current **routed timing-met** latency point is now the 06L unroll16 fallback: `2316 cycles / 25.733us @ 90.001MHz [routed_pnr_met@90MHz]`, with WNS `+0.151ns`, TNS `0.000ns`, and `0` failing endpoints. This supersedes the earlier roadmap headline that treated `23.16us @ 100MHz` as the best route candidate.

The `2316 cycles / 23.160us @ 100MHz [csim_cosim_pass, routed_pnr_failed@100MHz]` point remains useful as a verified functional target, but it is **not** a timing-closed P&R result. Three 100MHz routed attempts failed timing: baseline WNS `-0.235ns` with 144 failing endpoints, aggressive physical optimization WNS `-0.159ns` with 96 failing endpoints, and fanout duplication WNS `-0.383ns` with 240 failing endpoints. A 95MHz routed attempt came close but still failed: WNS `-0.033ns` with 48 failing endpoints.

Clock and quantization alone cannot reach 1us. Even the routed 90MHz unroll16 result is still `25.733x` away from 1us. Sub-10us and especially ~1us require architecture-level changes: fewer dense 16x16 blocks, structured/diagonal transforms, or a feedforward replacement whose signal quality must be revalidated.

## Baseline Evidence

| item | value | verification level | source |
|---|---:|---|---|
| 06F latency | 8316 cycles / 83.16us | `routed_pnr_met@100MHz` | 06F `phc_step_csynth.rpt`, RTL cosim, routed timing |
| 06F interval / II | 8317 cycles | `routed_pnr_met@100MHz` | 06F `phc_step_csynth.rpt` |
| 06F P&R WNS | +0.020ns | `routed_pnr_met@100MHz` | 06F routed timing report |
| 06F clock | 10.000ns / 100MHz | `routed_pnr_met@100MHz` | 06F routed timing report |
| 06F inferred routed critical period | 9.980ns | `routed_pnr_met@100MHz` | 10.000ns - 0.020ns WNS |
| 06F critical path shape | FSM register to DSP48E1 PCIN, data path delay 8.496ns | `routed_pnr_met@100MHz` | 06F routed timing report |
| 06F routed utilization fingerprint | LUT 7555 (14.2%), FF 8217 (7.7%), DSP 47 (21.4%), BRAM_18K 6 (2.1%) | `routed_pnr_met@100MHz` | 06F `hls_impl_pnr.rpt` |

## Stage / Loop Latency Breakdown

These stage numbers remain from the 06F 8316-cycle schedule and explain why the frozen baseline is slow.

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

Interpretation: the 16x16 matrix-vector loops are the latency wall. LayerNorm is visible and should not be ignored, but the current report data does not support treating it as the primary bottleneck.

## Clock Frequency Lever

The table below separates arithmetic projections from routed evidence.

| frequency | period ns | latency if 8316-cycle 06F schedule unchanged | distance to 1us | verification level | timing evidence |
|---:|---:|---:|---:|---|---|
| 100MHz | 10.000 | 83.16us | 83.16x | `routed_pnr_met@100MHz` | WNS +0.020ns |
| 150MHz | 6.667 | 55.44us | 55.44x | `estimate_only` | no routed attempt for unchanged 06F; existing 100MHz critical path implies likely fail without retiming |
| 200MHz | 5.000 | 41.58us | 41.58x | `estimate_only` | no routed attempt for unchanged 06F; existing 100MHz critical path implies likely fail without retiming |

Conclusion: clocking alone cannot reach 1us. Even a hypothetical timing-clean 200MHz implementation of the current 8316-cycle schedule would be 41.58us.

## Existing Unroll Sweep Evidence

The unroll sweep originally established functional correctness and HLS-level resource feasibility. The P&R update changes the interpretation of the factor16 point.

| factor | latency cycles | latency us at 100MHz | DSP (% of 220) | LUT (% of 53,200) | FF (% of 106,400) | BRAM_18K (% of 280) | fits HLS budget | speedup vs 06F cycles | cosim | routed status |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|---|---|
| 1 | 8316 | 83.16 | 46 (20.9%) | 24836 (46.7%) | 14518 (13.6%) | 6 (2.1%) | True | 1.00x | PASS | 06F v8 baseline is `routed_pnr_met@100MHz`; 06L factor1 is synthesis/cosim comparison only |
| 2 | 5036 | 50.36 | 56 (25.5%) | 25601 (48.1%) | 16162 (15.2%) | 12 (4.3%) | True | 1.65x | PASS | `csim_cosim_pass`; no routed P&R evidence |
| 4 | 3724 | 37.24 | 57 (25.9%) | 27456 (51.6%) | 17000 (16.0%) | 0 (0.0%) | True | 2.23x | PASS | `csim_cosim_pass`; no routed P&R evidence |
| 8 | 3068 | 30.68 | 70 (31.8%) | 30392 (57.1%) | 19704 (18.5%) | 0 (0.0%) | True | 2.71x | PASS | `csim_cosim_pass`; no routed P&R evidence |
| 16 | 2316 | 23.160 | 121 (55.0%) | 36792 (69.2%) | 35774 (33.6%) | 0 (0.0%) | True | 3.59x | PASS | `csim_cosim_pass`; 100MHz P&R failed in three attempts; 90MHz P&R met at 25.733us |

### Factor16 Routed Attempts

| attempt | clock | latency interpretation | WNS | TNS | failing endpoints | verification level |
|---|---:|---:|---:|---:|---:|---|
| 100MHz baseline | 100.000MHz / 10.000ns | 2316 cycles / 23.160us | -0.235ns | -13.737ns | 144 | `routed_pnr_failed` |
| 100MHz aggressive physical optimization | 100.000MHz / 10.000ns | 2316 cycles / 23.160us | -0.159ns | -13.956ns | 96 | `routed_pnr_failed` |
| 100MHz fanout duplication | 100.000MHz / 10.000ns | 2316 cycles / 23.160us | -0.383ns | -53.903ns | 240 | `routed_pnr_failed` |
| 95MHz baseline | 95.003MHz / 10.526ns | 2316 cycles / 24.378us | -0.033ns | -1.578ns | 48 | `routed_pnr_failed` |
| 90MHz baseline | 90.001MHz / 11.111ns | 2316 cycles / 25.733us | +0.151ns | 0.000ns | 0 | `routed_pnr_met` |

Critical-path diagnosis from `n2ew_era/06l_phc_lite_pipeline_budget_sweep/AUDIT_unroll16_pnr_timing.md`: the 100MHz failures are not random unrelated timing misses. They are dominated by the same high-fanout control path shape: FSM state register `ap_CS_fsm_reg[9]` drives selection logic feeding a family of DSP48E1 cascade inputs (`PCIN`) across the unrolled datapath. The 100MHz baseline critical path has 8.751ns data path delay, with route delay 3.718ns, about 42% of the data path. It missed by only 0.235ns in the baseline run, while fanout/replication made timing worse. The 90MHz run passes because relaxing the period to 11.111ns absorbs this routing delay; it does not structurally remove the high-fanout mechanism.

The best usable headline is now:

`2316 cycles / 25.733us @ 90.001MHz [routed_pnr_met]`

The 23.160us number should be retained only as:

`2316 cycles / 23.160us @ 100MHz [csim_cosim_pass; routed_pnr_failed in three 100MHz attempts]`

## Lever Tradeoff Table

| lever | best supported latency | verification level | improvement vs 83.16us routed 06F | distance to 1us | resource cost | recommendation |
|---|---:|---|---:|---:|---|---|
| 06F frozen baseline | 83.16us @ 100MHz | `routed_pnr_met@100MHz` | 1.00x | 83.16x | routed LUT 7555 (14.2%), FF 8217 (7.7%), DSP 47 (21.4%), BRAM_18K 6 (2.1%) | safe baseline for board wrapper work |
| 06L unroll16 fallback | 25.733us @ 90.001MHz | `routed_pnr_met@90MHz` | 3.23x | 25.733x | routed LUT 14826 (27.9%), FF 19805 (18.6%), DSP 125 (56.8%), BRAM_18K 0 (0.0%) | best verified latency point |
| 06L unroll16 100MHz target | 23.160us @ 100MHz | `csim_cosim_pass`; 100MHz `routed_pnr_failed` | 3.59x if timing closes | 23.160x | routed failed attempts about LUT 14.8k (27.8%), FF 19.8k (18.6%), DSP 125 (56.8%), BRAM_18K 0 (0.0%) | keep as open timing-closure target; do not claim routed |
| clock 100->150MHz on unchanged 06F | 55.44us | `estimate_only` | 1.50x theoretical | 55.44x | no resource change if same netlist | do not claim without fresh P&R |
| clock 100->200MHz on unchanged 06F | 41.58us | `estimate_only` | 2.00x theoretical | 41.58x | no resource change if same netlist | not credible without major retiming/re-P&R |
| Q16.8 quantization only | 83.16us | `csynth_only` / phase2 numerical checks only | 1.00x | 83.16x | DSP unchanged 46 (20.9%), LUT 24746 (46.5%) vs 24836 (46.7%); full FF/BRAM row not used for latency conclusion | not useful for latency; do not use as path to 1us |
| diagonal/structured recurrence replacement (06G) | 20.87us | `csynth_only` | 3.98x theoretical vs 06F | 20.87x | DSP 34 (15.5%), LUT 16058 (30.2%), FF 9066 (8.5%), BRAM_18K 1 (0.4%) | architectural clue only; model accuracy regresses vs production sequence model and it is not a drop-in replacement |

## Partial Time-Step Unroll / Recurrent Pipeline Assessment

For a single live market state stream, partial unroll across time is not causally valid: event t+1 depends on `state_next` from event t. This is why the design has interval approximately equal to latency. Multi-stream interleaving could improve aggregate throughput across independent symbols/sessions, but it would not reduce the single-stream decision latency needed for an HFT quote gate.

To reach 1us at 100MHz, the step must be at most 100 cycles. The best routed PHC-lite-style datapoint is now 2316 cycles / 25.733us at 90.001MHz, so the gap is still about 25.7x in wall-clock latency. The 100MHz version would be 23.160us if timing closed, still 23.16x from 1us.

## Concrete Technical Path Toward 1us

1. Use the 06F v8 baseline when the priority is most conservative routed evidence: `83.16us [routed_pnr_met@100MHz]`.
2. Use the 06L unroll16 90MHz point as the best verified low-latency PHC-lite implementation: `25.733us [routed_pnr_met@90MHz]`. This is the strongest current replacement candidate for latency discussions, subject to the separate wrapper/interface decision.
3. Keep `23.160us [csim_cosim_pass]` as an open 100MHz timing-closure target, not as a routed claim. The exact P&R evidence is: 100MHz baseline WNS `-0.235ns` / 144 failing endpoints, 100MHz aggressive physical optimization WNS `-0.159ns` / 96 failing endpoints, and 100MHz fanout duplication WNS `-0.383ns` / 240 failing endpoints. The 95MHz near miss is WNS `-0.033ns` / 48 failing endpoints. The root mechanism is high-fanout control routing into DSP48E1 PCIN selection logic; 90MHz works by absorbing that route delay with a longer period.
4. Treat the 06G diagonal/structured experiment as a research clue only: `20.87us [csynth_only]`. It is not a drop-in PHC-lite replacement because the model accuracy and strategy quality regress relative to the production sequence model.
5. To move below ~10us, reduce the number of dense 16x16 blocks or replace them with structured/low-rank/diagonal transforms, then revalidate signal quality and routed timing.
6. To approach ~1us, clock changes and quantization are insufficient by themselves. The current evidence says this requires architecture-level redesign, not just another implementation directive pass.

## Claim Boundary

- This report uses existing csynth/cosim/P&R artifacts only; no new P&R was run here.
- `25.733us @ 90.001MHz` is routed timing-met evidence.
- `23.160us @ 100MHz` is csim/cosim-pass evidence with routed P&R failures at 100MHz.
- 150MHz/200MHz numbers are estimates unless a future routed timing report proves nonnegative WNS/TNS.
- 06G diagonal/structured numbers are not production PHC-lite latency claims and must be marked `csynth_only`.
- No FPGA PnL, exchange replay, or real-world profitability claim is made.
