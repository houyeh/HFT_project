# 07L Ultrafast Policy Sweep

**Status: resolved_positive for QAT Tiny MLP; resolved_negative for the tested LUT policy.**

This phase tested two low-latency replacements for the frozen event-wise PHC policy. Candidate selection used `train` and `val` only. `new3` and `new4` were frozen post-selection diagnostics, and NEW5 was not read. PnL values below use the corrected terminal-settlement sequential simulator inherited from 07G.

## 1. Tested approaches

1. **Low-bit QAT Tiny MLP:** the existing 16-input, 8-hidden-unit, 3-action Tiny MLP was initialized from the frozen 07G champion and fine-tuned with fake quantization at Q8.4, Q10.5, and Q12.6. The val split selected Q12.6.
2. **Quantized LUT policy:** train-only ANOVA F scores selected 2--4 causal features. Score-mean and action-majority tables with at most 512 cells were evaluated. The val split selected a 256-cell table using `queue_imbalance` and `log1p_bid_sz_1`.

The LUT policy did not clear the task-value gate, so it was not sent through HLS. The Q12.6 Tiny MLP cleared the task, fixed-point, CSim, RTL cosim, and core-level route gates.

## 2. Task-level result

| Policy | Split | Corrected sequential excess PnL | Retention vs PHC | Delta vs tuned static queue rule | Selection role |
|---|---:|---:|---:|---:|---|
| Frozen PHC reference | val | 171,835 | 100.00% | +57,985 | reference |
| Q12.6 Tiny MLP | val | 136,990 | 79.72% | +23,140 | val-selected QAT champion |
| Q12.6 Tiny MLP | new3 | 473,410 | 77.47% | +122,455 | frozen diagnostic |
| Q12.6 Tiny MLP | new4 | 152,195 | 69.40% | +36,320 | frozen diagnostic |
| 256-cell LUT | val | 109,805 | 63.90% | -4,045 | val-selected LUT champion |
| 256-cell LUT | new3 | 347,500 | 56.87% | -3,455 | frozen diagnostic |
| 256-cell LUT | new4 | 114,670 | 52.29% | -1,205 | frozen diagnostic |

The Q12.6 model retains 69.40--79.72% of PHC PnL and remains above the tuned static queue rule on all three reported splits. Relative to the unquantized Tiny MLP, its PnL retention is 98.27% on val, 100.37% on new3, and 98.85% on new4. The LUT policy is smaller and human-inspectable, but it does not beat the tuned static rule; that branch is therefore a negative result.

Q12.6 input/activation saturation counts were 32, 30, and 5 on val/new3/new4. These are recorded rather than hidden; the task-level results already include their effect.

## 3. Fixed-point and RTL equivalence

The HLS core uses `ap_fixed<12,6,AP_TRN,AP_SAT>` for inputs, weights, biases, and hidden activations, with `ap_fixed<32,16,AP_TRN,AP_SAT>` accumulators. The implementation is a fully unrolled 16x8x3 MLP with explicit balanced adder trees and ReLU.

| Check | Vectors | Mismatches | Maximum raw integer difference | Tier |
|---|---:|---:|---:|---|
| Python integer reference vs HLS CSim | 25,600 | 0 | 0 | `simulated_csim` |
| HLS CSim vs RTL cosim | 25,600 | 0 | 0 | `simulated_cosim` |

Both action and logits are bit-exact for the complete 25,600-vector val test set. The generated HLS schedule reports 13-cycle latency and 14-cycle interval.

## 4. Synthesis and routed result

| Metric | HLS estimate | Routed OOC core | Zynq-7020 budget |
|---|---:|---:|---:|
| Latency | 13 cycles / 0.130 us | 13 cycles / 0.130 us (cycle-derived) | n/a |
| Interval | 14 cycles | 14 cycles (HLS schedule) | n/a |
| Clock | 7.151 ns estimated | 10.000 ns target, WNS +0.467 ns | 100 MHz target |
| DSP48E1 | 40 (18.18%) | 36 (16.36%) | 220 |
| LUT | 13,115 (24.65%) | 4,466 (8.39%) | 53,200 |
| FF | 3,183 (2.99%) | 2,160 (2.03%) | 106,400 |
| BRAM18K | 0 (0.00%) | 0 (0.00%) | 280 |

The routed timing report has WNS `+0.467 ns`, TNS `0.000 ns`, and zero setup and hold failing endpoints. The verification tier is therefore **`core_level_routed`**. The `0.130 us` value is derived from 13 scheduled cycles at the routed 100 MHz constraint; it is not an independent on-chip latency measurement.

Compared with the 07J 28-bit parallel Tiny MLP, Q12.6 reduces scheduled latency from 18 to 13 cycles (0.180 to 0.130 us, 27.8%) and routed DSP use from 96 to 36 (62.5%). This is a same-family comparison. The much larger ratio against recurrent PHC endpoints is a model-substitution trade-off, not an acceleration of the same model.

## 5. Claim boundary

This core accepts already standardized 16-feature inputs. Feature extraction, train-stat standardization, market-data transport, wrapper control, and board I/O are outside this latency number. Standardization was not folded into the Q12 weights because the scale factors can widen the coefficient range; a board implementation must either include that preprocessing upstream or separately synthesize and time it.

The tested model is a QAT Tiny MLP, not PHC-lite. It trades away roughly 20--31% of PHC corrected sequential PnL for a routed 130 ns core and low resource use. No board-level wrapper, bitstream, or on-board equivalence test was run in this phase.

## 6. Evidence paths

- Software sweep: `results/ultrafast_policy_sweep.csv`
- Frozen selection record: `results/selection_manifest.json`
- HLS summary: `results/qat12_hls_summary.csv`
- CSim/cosim outputs: `results/qat12_csim_out.csv`, `results/qat12_cosim_out.csv`
- C-synthesis report: `hls_component/qat12_csynth_work/hls/syn/report/qat12_tiny_mlp_csynth.rpt`
- Routed timing: `pnr_100mhz/output/timing_summary_routed.rpt`
- Routed utilization: `pnr_100mhz/output/utilization_routed.rpt`

## 7. Decision

The Q12.6 route is worth carrying forward as the current fastest validated policy core. The tested LUT route is not worth hardware work because it fails to preserve even the tuned static-rule PnL. A wider QAT MLP is the next useful accuracy-for-resource experiment; a cascade is relevant only for average compute and does not improve deterministic worst-case latency.
