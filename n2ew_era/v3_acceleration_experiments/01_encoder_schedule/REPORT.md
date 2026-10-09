# V3 Encoder Output-Channel Scheduling Experiment

Status: `resolved_positive`

Verification tier reached for the selected `oc2` candidate: `core_ooc_routed_pnr_met_after_1000_vector_cosim`. This is a core out-of-context result, not a board-wrapper or on-board measurement.

## Scope and provenance

The experiment copied, rather than modified, the frozen V3 diagonal16 source from `n2ew_era/06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/`. The model is the frozen 05F diagonal SSM seed 2 checkpoint at best epoch 157, selected on `val` with threshold `0.0`; NEW3, NEW4, and NEW5 were not used for selection. Exact hashes are in `source_hash_manifest.json`.

Only `linear_16_16_fixed`, which implements the encoder matrix multiplication, differs in the candidates. LayerNorm, SiLU, recurrence, readout, fixed-point types, per-channel bias initialization, product casts, and accumulator update order are unchanged.

## Baseline bottleneck

The frozen source uses an outer 16-channel loop and an inner 16-input loop with `UNROLL_FACTOR=8`. Baseline csynth reports:

- encoder outer loop: 320 cycles, 16 trips, 20 cycles per output channel;
- inner loop: 16 cycles, two trips after factor-8 unrolling, iteration latency 8;
- full core: 532-cycle latency and 533-cycle interval;
- encoder weights: cyclic factor-8 partition on dimension 2, implemented as eight ROM banks;
- `x_t`: dual-port `ap_memory` at the top-level RTL; the same input values are broadcast across candidate output lanes;
- accumulation dependency: each output channel retains its own loop-carried `acc += cast(weight) * cast(input)` chain in increasing input-index order.

The candidates add dimension-1 weight banking and two or four independent accumulators. No pipeline pragma was added to the encoder outer loop. Compile logs confirm factor-8 input-loop unrolling and complete unrolling only of the new lane loops.

## Numerical equivalence

| Variant | Independent vectors | Closed-loop replay | Result |
|---|---:|---:|---|
| baseline | 25,600 | 4,096 | reference |
| oc2 | 25,600 | 4,096 | bit-exact to baseline |
| oc4 | 25,600 | 4,096 | bit-exact to baseline |

For the independent test, all three output CSV files have SHA-256 `99F66AB6A019F5F179C3D4C9FB6BA95A78E182A743891A0F5EA7E9FF2E98D4FE`. For the recurrent state-feedback replay, all three have SHA-256 `842C048B9BA89CCF4D3C03029D410E5FAC66395EB27348DFA1E4F98595E9CE19`. The baseline also retained the existing float-reference result: 25,600 vectors, zero tolerance failures, maximum absolute difference `0.00206566`, and mean absolute difference `0.000386541` at the unchanged tolerance of `1.0`.

This bit-exact result applies to the tested 25,600 independent vectors and 4,096-event closed-loop replay. It is not a formal proof over every possible fixed-point input/state combination.

## Synthesis scan

All values below are Vitis HLS 2025.2 csynth estimates for `xc7z020-clg400-1` at a 10 ns target. They are not routed timing results.

Host note: temporary drive mappings were used to keep Vitis-generated implementation paths below the Windows path-length limit. All retained experiment outputs reside under this phase directory; the model source was not changed.

| Variant | Encoder cycles | Full latency / II | Latency at 100 MHz | DSP | LUT | FF | BRAM_18K | Strict core budget |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 320 | 532 / 533 | 5.320 us | 198 (90.0%) | 35,617 (66.95%) | 28,485 (26.77%) | 0 | pass |
| oc2 | 160 | 372 / 373 | 3.720 us | 208 (94.55%) | 36,681 (68.95%) | 30,708 (28.86%) | 0 | pass |
| oc4 | 76 | 288 / 289 | 2.880 us | 220 (100.0%) | 38,567 (72.49%) | 33,066 (31.08%) | 0 | **fail: DSP must be <220** |

`oc2` reduces encoder latency by exactly 2.00x and full-core latency by 1.430x while retaining 12 DSPs of core headroom. `oc4` reduces the encoder loop by 4.21x and the full core by 1.847x, but consumes every DSP and therefore fails the project's strict core budget before wrapper/debug resources are considered.

The benefit is both lower single-event latency and lower service interval: II falls from 533 to 373 or 289 cycles. However, each design remains non-pipelined at the top level with interval equal to latency plus one; this change shortens sequential service time rather than creating overlapped recurrent-event execution.

## Selected-candidate RTL verification

Only `oc2`, the sole candidate that passed the strict csynth resource gate, was advanced. RTL co-simulation used the first 1,000 vectors from the unchanged independent-vector set. It reported zero mismatches against the C model and matched the corresponding 1,000 CSim CSV rows exactly.

| Evidence | Result | Verification meaning |
|---|---:|---|
| RTL cosim vectors | 1,000 | Verilog/XSim |
| RTL cosim latency | 370 cycles | observed RTL transaction latency |
| RTL cosim interval | 371 cycles | observed RTL service interval |
| RTL-derived latency at 100 MHz | 3.700 us | simulated cycles divided by routed target frequency |
| Float-reference max / mean absolute difference | 0.00153518 / 0.000338227 | unchanged tolerance `1.0`; zero failures |

The 370-cycle RTL result and 372-cycle csynth estimate are retained separately because they are different latency objects. The scan table's `latency_cycles=372` remains the HLS scheduling estimate, while `cosim_latency_cycles=370` is the Verilog simulation observation.

## Core out-of-context place and route

The same generated `oc2` RTL was implemented for `xc7z020-clg400-1` at a 10 ns target. The first implementation launch hit only a Windows path-length failure before implementation; the preserved retry used a shorter temporary drive mapping and made no source, pragma, precision, or constraint change.

| Metric | Routed result |
|---|---:|
| Target frequency | 100.000 MHz |
| WNS / TNS | +0.384 ns / 0.000 ns |
| Failing setup endpoints | 0 |
| Post-synthesis / post-implementation period | 9.259 ns / 9.616 ns |
| LUT | 11,892 / 53,200 (22.35%) |
| FF | 13,724 / 106,400 (12.90%) |
| DSP | 213 / 220 (96.82%) |
| DSP headroom | 7 |
| BRAM_18K | 0 / 280 (0.00%) |

All user timing constraints were met. This establishes 100 MHz timing closure for the HLS core out of context. It does not include the replay wrapper, BRAM player, clock generator, ILA, board I/O, or a physical on-board latency counter. The `3.700 us` figure therefore combines RTL-simulated cycles with a routed clock target; it is not an on-silicon latency measurement.

The routed critical path starts at `ap_CS_fsm_reg[22]` and ends at a DSP48E1 `PCIN` input, with fanout 172 on the first routed net. Timing still closes, but the routed DSP count rises from the csynth estimate of 208 to 213, leaving only seven DSPs for any surrounding board logic.

## Decision and stop point

`oc2` is the only viable candidate from this scan. It is bit-exact on the 25,600-vector CSim test and the 4,096-event closed-loop CSim replay, passes a 1,000-vector RTL cosim, and closes core OOC timing at 100 MHz. `oc4` is retained as a useful boundary result but was not advanced because its csynth DSP usage is exactly 220/220.

No PnL rerun, wrapper integration, board-level P&R, bitstream generation, or on-board work was performed. The next hardware step would need a separate wrapper resource and timing check because the routed core already uses 213/220 DSPs. Model quality remains the frozen 05F diagonal-SSM result; this scheduling experiment changes arithmetic parallelism only.

## Evidence paths

- Machine-readable scan: `results/schedule_scan.csv`
- Frozen hashes and model provenance: `source_hash_manifest.json`
- Candidate patches: `patches/oc2.patch`, `patches/oc4.patch`
- Full logs: `logs/*_csim.log`, `logs/*_csynth.log`
- Copied csynth reports: `results/<variant>_phc_diag_step_csynth.rpt`
- Selected-candidate cosim: `results/oc2_phc_diag_step_cosim.rpt`, `logs/oc2_cosim.log`
- Selected-candidate routed reports: `results/oc2_phc_diag_step_export.rpt`, `results/oc2_phc_diag_step_timing_routed.rpt`, `results/oc2_phc_diag_step_utilization_routed.rpt`
- Preserved implementation logs: `logs/oc2_impl.log`, `logs/oc2_impl_pathlength_failure.log`, `logs/oc2_vivado_pathlength_failure.log`
- Reproducible generated projects (ignored by Git): `variants/<variant>/hls_component/hls_work_*/`
- Exact CSim outputs: `results/<variant>_independent.csv`, `results/<variant>_replay.csv`
