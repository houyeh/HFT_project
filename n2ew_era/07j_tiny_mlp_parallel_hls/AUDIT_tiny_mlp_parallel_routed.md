# Tiny MLP Parallel HLS Routed Audit

status: `resolved_positive`

## Scope

This phase lowers the latency of the frozen 07G `tiny_mlp_direct_action_h8_seed1` policy by spending parallel FPGA resources. It does not retrain the model, change its 163 parameters, change the action rule, or load NEW5. The arithmetic contract remains the 07H-selected `ap_fixed<28,14>` input/weight/bias format with `ap_fixed<40,16>` accumulators and `AP_TRN`/`AP_WRAP` semantics.

The successful implementation computes all eight hidden neurons concurrently. Each 16-input dot product uses an explicit balanced `16 -> 8 -> 4 -> 2 -> 1` adder tree, and each output dot product uses an `8 -> 4 -> 2 -> 1` tree. This removes the long combinational accumulation chain while preserving the fixed-point operation widths and final raw accumulator result.

## Candidate ablation

| Candidate | CSim | C synthesis latency | Estimated clock | HLS resources (DSP/LUT/FF/BRAM) | Decision |
|---|---|---:|---:|---:|---|
| Linear accumulation chain | 25,600 vectors, 0 mismatches | 77 cycles | 25.506 ns | 50 / 11,172 / 8,640 / 0 | Rejected: cannot support 100 MHz according to C synthesis |
| Explicit balanced trees | 25,600 vectors, 0 mismatches | 18 cycles, interval 19 | 8.248 ns | 184 / 18,856 / 20,598 / 0 | Advanced to RTL cosim and P&R |

The first candidate demonstrates that unrolling alone is insufficient: it reduced cycle count but exposed an eight-term serial adder path. The balanced tree changed the schedule, not the learned model or fixed-point contract.

## Functional verification

| Stage | Vectors | Result | Mismatches | Maximum raw-logit difference | Latency / interval | Tier |
|---|---:|---|---:|---:|---:|---|
| Vitis CSim | 25,600 val rows | pass | 0 | 0 LSB | n/a | simulated |
| Vitis C synthesis | n/a | pass | n/a | n/a | 18 / 19 cycles | csynth estimate |
| Verilog C/RTL co-simulation | 25,600 val rows | pass | 0 | 0 LSB | 18 / 19 cycles | `csim_cosim_pass` |
| Vivado OOC P&R at 100 MHz | core RTL | timing met | n/a | n/a | 18 cycles / 0.180 us | `core_level_routed` |

The testbench uses exact integer comparison, not a widened tolerance. CSim and Verilog cosim both agree bit for bit with the 07H integer emulator for all 25,600 vectors. The independent cosim report gives min/average/max latency `18/18/18` cycles and interval `19/19/19` cycles.

## Routed result

Target: `xc7z020-clg400-1`, Vivado/Vitis 2025.2, 10.000 ns constraint.

| Metric | Routed result | Zynq-7020 budget | Utilization |
|---|---:|---:|---:|
| Latency | 18 cycles / 0.180 us at 100 MHz | n/a | cycle-derived from routed timing closure |
| Interval | 19 cycles / 0.190 us at 100 MHz | n/a | verified by RTL cosim |
| WNS / TNS | +0.331 ns / 0.000 ns | timing constraint | 0 failing endpoints |
| LUT | 8,760 | 53,200 | 16.47% |
| FF | 9,052 | 106,400 | 8.51% |
| DSP48E1 | 96 | 220 | 43.64% |
| BRAM_18K | 0 | 280 | 0.00% |

Vivado reports that all user-specified timing constraints are met. The final run explicitly sets `HD.CLK_SRC=BUFGCTRL_X0Y0`; its setup WNS is +0.331 ns, hold slack is +0.038 ns, and both TNS and THS are zero. The worst setup path is internal, from `ap_CS_fsm_reg[4]` to the clock-enable pin of an adder register. The first routed run without an explicit clock-source location also met timing at +0.313 ns and is retained in `pnr_100mhz/output/` as provenance.

The HLS resource estimate (184 DSP, 18,856 LUT, 20,598 FF) is deliberately kept separate from the routed result. With constant frozen weights, Vivado remapped a substantial subset of multiplications into LUT logic and reduced the final DSP count to 96.

## Task-value trade-off

| Split | Fixed Tiny MLP sequential excess PnL | PHC reference PnL | Retention vs PHC |
|---|---:|---:|---:|
| val | 139,515 | 171,835 | 81.19% |
| NEW3 (post-selection diagnostic) | 472,055 | 611,095 | 77.25% |
| NEW4 (post-selection diagnostic) | 154,400 | 219,310 | 70.40% |

Relative to the 07I resource-shared Tiny MLP implementation, latency falls from 1.720 us to 0.180 us, a 9.56x reduction, while the final routed core uses 96 DSPs instead of the 07I C-synthesis estimate of 4 DSPs. Relative to the routed 25.733 us recurrent PHC design point, the numerical latency ratio is 142.96x, but this is not an acceleration of the same model: it substitutes a much smaller feedforward policy and accepts the PnL retention shown above.

## Verification boundary

- C synthesis report: `hls_component/tiny_mlp_parallel_csynth_work/hls/syn/report/tiny_mlp_parallel_csynth.rpt`
- C/RTL co-simulation report: `hls_component/tiny_mlp_parallel_csynth_work/hls/sim/report/tiny_mlp_parallel_cosim.rpt`
- Primary routed timing report: `pnr_100mhz/output_clksrc/timing_summary_routed.rpt`
- Primary routed utilization report: `pnr_100mhz/output_clksrc/utilization_routed.rpt`
- Primary routed checkpoint: `pnr_100mhz/output_clksrc/tiny_mlp_parallel_routed_clksrc.dcp`
- Initial routed run without explicit `HD.CLK_SRC`: `pnr_100mhz/output/`
- Machine-readable results: `results/parallel_candidate_sweep.csv`, `results/tiny_mlp_parallel_hls_summary.csv`
- Fixed-point quality: `../07h_tiny_mlp_fixed_point_validation/AUDIT_tiny_mlp_mixed_precision.md`
- Corrected task value: `../07g_tiny_mlp_policy/AUDIT_terminal_settlement_correction.md`

This P&R run is out-of-context at the core boundary. The report has zero unconstrained internal endpoints and binds the OOC clock source to `BUFGCTRL_X0Y0`, but top-level input/output delays are not specified. Consequently, `0.180 us` is a core-level routed, cycle-derived latency. It is not a board-level routed result, a bitstream result, an on-board latency measurement, or an end-to-end market-data latency. A registered board wrapper with physical I/O and clock constraints would be required for those stronger claims.

## Decision

The balanced-tree Tiny MLP clears the requested nanosecond-scale core-routing experiment. It is bit-exact in CSim and RTL cosim, fits the Zynq-7020, and closes 100 MHz timing with 18-cycle latency. The result supports a `180 ns core-level routed` claim under the stated boundary. The cost is model substitution: fixed-point causal-fill PnL retains approximately 70.40% to 81.19% of the frozen PHC reference across the three reported splits.
