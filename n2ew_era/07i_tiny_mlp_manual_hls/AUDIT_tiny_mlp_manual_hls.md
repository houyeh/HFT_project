# Tiny MLP Manual HLS Audit

status: `resolved_positive`

## Scope

This phase implements the frozen 07G `tiny_mlp_direct_action_h8_seed1` policy as a hand-written Vitis HLS core. The architecture is `16 -> 8 ReLU -> 3 logits -> argmax` with 163 trainable parameters. It reuses the 07H-selected `ap_fixed<28,14>` input/weight/bias contract, `ap_fixed<40,16>` accumulators, and `AP_TRN`/`AP_WRAP` semantics. No model was retrained, thresholds were not changed, and NEW5 was not loaded.

Train-derived feature standardization remains folded algebraically into the first dense layer. The HLS test vectors contain the first 25,600 val rows in raw causal-feature space, quantized once to signed Q28.14 integers. Reference logits are the raw signed Q40.24 accumulator outputs from the 07H integer emulator.

## Verification results

| Stage | Vectors | Result | Mismatches | Maximum raw-logit difference | Latency / interval | Verification tier |
|---|---:|---|---:|---:|---|---|
| Software fixed-point | val/NEW3/NEW4 full diagnostics | pass | action flips versus float: 26 / 60 / 29 | n/a | n/a | software fixed-point emulation |
| Vitis CSim | 25,600 val rows | pass | 0 | 0 LSB | n/a | simulated |
| Vitis C synthesis | n/a | pass | n/a | n/a | 172 / 173 cycles | csynth estimate |
| Verilog C/RTL co-simulation | 25,600 val rows | pass | 0 | 0 LSB | 172 / 173 cycles | `csim_cosim_pass` |

The Verilog co-simulation report independently reports min/avg/max latency of 172 cycles and min/avg/max interval of 173 cycles. Therefore the C implementation, generated RTL, and 07H integer arithmetic agree bit for bit for all 25,600 vectors. No tolerance is used by the testbench.

## Synthesis result

Target: `xc7z020-clg400-1`, Vitis HLS 2025.2, 10.000 ns target clock.

| Metric | Result | Zynq-7020 budget | Utilization |
|---|---:|---:|---:|
| Latency | 172 cycles / 1.720 us at 100 MHz | n/a | cycle-derived, not routed |
| Interval | 173 cycles / 1.730 us at 100 MHz | n/a | non-pipelined top-level transaction |
| DSP | 4 | 220 | 1.82% |
| LUT | 1,268 | 53,200 | 2.38% |
| FF | 1,187 | 106,400 | 1.12% |
| BRAM_18K | 1 | 280 | 0.36% |
| Estimated clock | 7.080 ns | 10.000 ns target | estimated Fmax 141.24 MHz |

The feared 28-bit multiplier explosion did not occur. Vitis inferred two time-shared 28-bit constant-weight multiplier structures, each using two DSP48E1 blocks, for four DSPs total. The hidden and output MAC nests both achieved their requested inner-loop II of 1. The design stores the 128 first-layer constants in one BRAM_18K and implements the smaller arrays without additional BRAM.

## Task-value context

The fixed Tiny MLP retained the corrected float Tiny MLP causal-fill result within boundary noise: val `139,515` versus `139,400`, NEW3 `472,055` versus `471,655`, and NEW4 `154,400` versus `153,970`. Action agreement with the float Tiny MLP was 99.9687%, 99.9661%, and 99.9577%, respectively. These small positive PnL deltas are consequences of a few boundary action flips and are not evidence that quantization improves the policy.

Relative to the corrected PHC teacher results, the float Tiny MLP retained 81.12% on val, 77.18% on NEW3, and 70.21% on NEW4. Thus the 1.720 us HLS estimate is obtained by accepting a measurable task-value reduction, not by accelerating the original recurrent PHC computation unchanged.

## Evidence and boundary

- C synthesis report: `hls_component/tiny_mlp_csynth_work/hls/syn/report/tiny_mlp_infer_csynth.rpt`
- C/RTL co-simulation report: `hls_component/tiny_mlp_csynth_work/hls/sim/report/tiny_mlp_infer_cosim.rpt`
- Machine-readable summary: `results/tiny_mlp_hls_summary.csv`
- Fixed-point quality source: `../07h_tiny_mlp_fixed_point_validation/AUDIT_tiny_mlp_mixed_precision.md`
- Corrected task-value source: `../07g_tiny_mlp_policy/AUDIT_terminal_settlement_correction.md`

This phase does not include routed P&R, static timing closure, power analysis, bitstream generation, or on-board measurement. `1.720 us` is the 172-cycle C-synthesis and RTL-cosimulation latency evaluated at the 100 MHz target clock. It must not be presented as a routed or on-silicon latency until those later verification tiers pass.

## Decision

The Tiny MLP clears the manual-HLS correctness and synthesis-feasibility gate. Its RTL is bit-exact to the selected fixed-point emulator, fits the Zynq-7020 resource budget with substantial margin, and has a cycle-derived latency below 2 us at the target clock. The next justified step is Vivado routed P&R at 100 MHz; only after timing closure should this become a routed latency claim.
