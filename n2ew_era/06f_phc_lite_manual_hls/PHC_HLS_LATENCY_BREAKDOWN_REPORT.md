# PHC HLS Latency Breakdown Report

Status: `resolved_top_level_verified_stage_breakdown_csynth_only`

Date: 2026-06-21

This report explains the existing PHC-lite HLS single-step latency of
`8316` cycles, or `83.160 us` at the configured `10 ns` clock. It does not
change HLS source, pragmas, test vectors, or generated reports.

## Contract Boundary

Read before this audit:

- `n2ew_era/00_project_reset_and_data_contract/ALLOWED_CLAIMS.md`
- `n2ew_era/00_project_reset_and_data_contract/FORBIDDEN_CLAIMS.md`
- `n2ew_era/00_project_reset_and_data_contract/EVALUATION_CONTRACT.md`

Boundary applied here:

- This is an HLS quote-control primitive latency audit only.
- No real-world profitability, true exchange replay, true queue priority, FPGA
  PnL improvement, or production-readiness claim is made.
- No policy/model selection or trading evaluation was performed.

## Completed Verification

### Final Component Identification

The component matching the requested routed fingerprint is:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component
```

Identifier evidence from
`hls_component/hls_component/reports/hls_impl_pnr.rpt`:

| item | value |
|---|---:|
| target device | `xc7z020-clg400-1` |
| target clock | `10.000 ns` |
| post-route period | `9.980 ns` |
| routed path slack | `+0.020 ns` |
| LUT utilization | `14.20%` |
| FD/FF utilization | `7.72%` |
| DSP utilization | `21.36%` |
| RAMB/FIFO utilization | `2.14%` |

This fingerprint identifies `06f_phc_lite_manual_hls`, not a later int16
phase. The `06l_phc_lite_int16_quant` phase does not contain this routed P&R
result.

### Top-Level Latency

From `hls/syn/report/phc_step_csynth.rpt`:

| source | min cycles | max cycles | min absolute | max absolute | interval | pipeline |
|---|---:|---:|---:|---:|---:|---|
| csynth `phc_step` summary | 8316 | 8316 | 83.160 us | 83.160 us | 8317 | no |

The same report gives timing estimate `7.147 ns` for target `10.00 ns`
before implementation.

### RTL Co-Simulation Check

From `hls/sim/report/phc_step_cosim.rpt`,
`hls/sim/report/verilog/lat.rpt`, and
`hls/sim/report/verilog/result.transaction.rpt`:

| source | status | latency min | latency avg | latency max | interval min/avg/max | total execution cycles |
|---|---|---:|---:|---:|---:|---:|
| Verilog RTL cosim summary | Pass | 8316 | 8316 | 8316 | 8317 / 8317 / 8317 | 212915199 |
| `result.transaction.rpt` | per transaction | 8316 | 8316 | 8316 | 8317 / 8317 / 8317 | n/a |

Conclusion: the top-level `8316` cycle csynth latency is confirmed by RTL
co-simulation. The cosim reports available here do not provide per-stage RTL
start/end timestamps; per-stage rows below are csynth schedule estimates.

## How To Read The Breakdown

The raw function/loop rows below are not all additive. Vitis scheduled some
independent subcalls concurrently and then waited for the slower child module.
For example, the generated RTL contains wait states that block on pairs such
as:

- `Outline_VITIS_LOOP_119_1` and `Outline_VITIS_LOOP_72_1`
- `Outline_VITIS_LOOP_119_11` and `Outline_VITIS_LOOP_80_1`
- `Outline_VITIS_LOOP_119_13` and `Outline_VITIS_LOOP_136_1`
- `Outline_VITIS_LOOP_88_1` and `Outline_VITIS_LOOP_241_2`

Therefore this report provides two views:

1. Raw csynth function/loop latency entries.
2. A scheduler critical-path attribution that sums to `8316` cycles with an
   explicit residual bucket for scalar divider/control/handshake cycles that
   are not split out by the csynth performance table.

All percentages use `8316` cycles as the denominator.

## Raw Csynth Function And Loop Entries

Source mapping is from
`n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp`.
All listed loops have `#pragma HLS PIPELINE off` at the loop body. The helper
functions are `#pragma HLS INLINE`; there are no active `UNROLL` pragmas in
the current source file.

| stage / operation | source call or loop | csynth entry | min cycles | max cycles | pct of 8316 | source pragma / note |
|---|---|---|---:|---:|---:|---|
| `ENC_LINEAR` | call line 223, helper loop lines 101/104 | top loop `VITIS_LOOP_101_1` | 1328 | 1328 | 15.97% | outer and inner loops `PIPELINE off` |
| `ENC_LAYER_NORM` visible subtotal | call line 224; mean/var/output loops lines 173/180/188 plus sqrt line 186 | `VITIS_LOOP_173_1` 32 + `VITIS_LOOP_180_2` 80 + `sqrt_fixed_49_25_s` 28 + `VITIS_LOOP_188_3` 160 | 300 | 300 | 3.61% | loops `PIPELINE off`; sqrt is pipelined; divider not separately listed as performance latency |
| `encoded` SiLU | call line 225, helper loop line 88 | top loop `VITIS_LOOP_88_1` | 176 | 176 | 2.12% | `PIPELINE off`, tanh LUT scalar inline |
| `GATE_LINEAR` | call line 227, helper loop lines 119/122 | top loop `VITIS_LOOP_119_1` | 1328 | 1328 | 15.97% | outer and inner loops `PIPELINE off` |
| gate sigmoid | call line 228, helper loop line 72 | `Outline_VITIS_LOOP_72_1` | 145 | 145 | 1.74% | `PIPELINE off`, tanh LUT scalar inline |
| `CANDIDATE_LINEAR` | call line 230; ROM names `CANDIDATE_W/B` | `Outline_VITIS_LOOP_119_1` | 1329 | 1329 | 15.98% | outer and inner loops `PIPELINE off` |
| candidate tanh | call line 231, helper loop line 80 | `Outline_VITIS_LOOP_80_1` | 145 | 145 | 1.74% | `PIPELINE off`, tanh LUT scalar inline |
| `DECAY` state update | loop lines 233-236; ROM name `DECAY` | `Outline_VITIS_LOOP_233_1` | 81 | 81 | 0.97% | `PIPELINE off` |
| `GAIN_FROM_ENCODED_LINEAR` | call line 239; ROM names `GAIN_FROM_ENCODED_W/B` | `Outline_VITIS_LOOP_119_11` | 1329 | 1329 | 15.98% | outer and inner loops `PIPELINE off` |
| `GAIN_FROM_PREV_STATE_LINEAR` | call line 240; ROM name `GAIN_FROM_PREV_STATE_W` | `Outline_VITIS_LOOP_136_1` | 1313 | 1313 | 15.79% | outer and inner loops `PIPELINE off` |
| gain add | loop lines 241-243 | `Outline_VITIS_LOOP_241_2` | 49 | 49 | 0.59% | `PIPELINE off` |
| gain sigmoid | call line 245, helper loop line 72 | `Outline_VITIS_LOOP_72_12` | 145 | 145 | 1.74% | `PIPELINE off`, tanh LUT scalar inline |
| `SYNAPSE_LINEAR` | call line 247; ROM names `SYNAPSE_W/B` | `Outline_VITIS_LOOP_119_13` | 1329 | 1329 | 15.98% | outer and inner loops `PIPELINE off` |
| message SiLU | call line 248, helper loop line 88 | `Outline_VITIS_LOOP_88_1` | 177 | 177 | 2.13% | `PIPELINE off`, tanh LUT scalar inline |
| loop input combine | loop lines 250-252 | top loop `VITIS_LOOP_250_3` | 160 | 160 | 1.92% | `PIPELINE off` |
| `LOOP_LAYER_NORM` visible subtotal | call line 254; mean/var/output loops lines 173/180/188 plus sqrt line 186 | `VITIS_LOOP_173_1` 32 + `VITIS_LOOP_180_2` 80 + `sqrt_fixed_49_25_s` 28 + `VITIS_LOOP_188_3` 160 | 300 | 300 | 3.61% | loops `PIPELINE off`; sqrt is pipelined; divider not separately listed as performance latency |
| `READOUT_LINEAR` | call line 256, helper loop lines 154/157 | top loop `VITIS_LOOP_154_1` | 164 | 164 | 1.97% | outer and inner loops `PIPELINE off` |
| readout output copy | loop lines 258-260 | `Outline_VITIS_LOOP_258_4` | 3 | 3 | 0.04% | `PIPELINE off` |
| `state_next` output copy | loop lines 262-264 | `Outline_VITIS_LOOP_262_5` | 33 | 33 | 0.40% | `PIPELINE off` |

Additional LayerNorm scalar evidence:

- `phc_step_csynth.rpt` lists `sqrt_fixed_49_25_s` as a `28` cycle pipelined
  function.
- The same csynth resource table lists two divider instances,
  `udiv_49s_37ns_40_53_seq_1_U74` and
  `udiv_49s_37ns_40_53_seq_1_U78`, but the performance estimate instance
  table does not give them standalone latency rows.
- Generated RTL `phc_step_udiv_49s_37ns_40_53_seq_1.v` implements the divider
  with a sequential `r_stage[in0_WIDTH]` done path and default
  `in0_WIDTH = 32`. This verifies the divider is real, but the available
  cosim report still does not provide a per-stage RTL measured divider
  timestamp.

## Scheduler Critical-Path Attribution

This view accounts for HLS overlap by charging only the slower child in each
observed wait-on-subcall group. It is the best additive explanation available
from the frozen reports without re-running cosim waveform instrumentation.

| critical-path bucket | child entries represented | charged cycles | pct of 8316 |
|---|---|---:|---:|
| `ENC_LINEAR` | `VITIS_LOOP_101_1` | 1328 | 15.97% |
| first LayerNorm visible work | mean 32 + variance 80 + sqrt 28 + output normalize 160 | 300 | 3.61% |
| encoded SiLU | top `VITIS_LOOP_88_1` | 176 | 2.12% |
| `GATE_LINEAR` | top `VITIS_LOOP_119_1` | 1328 | 15.97% |
| wait group: gate sigmoid vs `CANDIDATE_LINEAR` | max(145, 1329) | 1329 | 15.98% |
| wait group: candidate tanh vs `GAIN_FROM_ENCODED_LINEAR` | max(145, 1329) | 1329 | 15.98% |
| `DECAY` state update | `Outline_VITIS_LOOP_233_1` | 81 | 0.97% |
| wait group: `GAIN_FROM_PREV_STATE_LINEAR` vs `SYNAPSE_LINEAR` | max(1313, 1329) | 1329 | 15.98% |
| wait group: gain add vs message SiLU | max(49, 177) | 177 | 2.13% |
| gain sigmoid | `Outline_VITIS_LOOP_72_12` | 145 | 1.74% |
| loop input combine | `VITIS_LOOP_250_3` | 160 | 1.92% |
| second LayerNorm visible work | mean 32 + variance 80 + sqrt 28 + output normalize 160 | 300 | 3.61% |
| `READOUT_LINEAR` | `VITIS_LOOP_154_1` | 164 | 1.97% |
| output write wait group | max(readout copy 3, state copy 33) | 33 | 0.40% |
| residual scalar divider/control/handshake not split by performance table | `8316 - 8179` | 137 | 1.65% |
| **total** |  | **8316** | **100.00%** |

The residual bucket is intentionally not assigned to a single operation. It
contains schedule/control/call-boundary effects and scalar arithmetic,
including divider handling that is present in RTL but not listed as an
individual latency row in the csynth performance table.

## Root Cause Of The 8316 Cycles

The largest single visible contributors are the 16x16 linear/matrix-vector
loops with `PIPELINE off` and no active unroll:

- `CANDIDATE_LINEAR`: `1329` cycles, `15.98%`
- `GAIN_FROM_ENCODED_LINEAR`: `1329` cycles, `15.98%`
- `SYNAPSE_LINEAR`: `1329` cycles, `15.98%`
- `ENC_LINEAR`: `1328` cycles, `15.97%`
- `GATE_LINEAR`: `1328` cycles, `15.97%`
- `GAIN_FROM_PREV_STATE_LINEAR`: `1313` cycles, `15.79%`

Mechanism: each 16x16 helper has a 16-iteration output loop and a nested
16-iteration input loop. With both loops marked `#pragma HLS PIPELINE off`,
the report shows about `82-83` cycles per output element, giving about
`1313-1329` cycles per 16-output linear block.

## LayerNorm Hypothesis Check

Hypothesis checked: the `8316` cycle latency might be dominated by LayerNorm
division or square root.

Evidence against this as the largest contributor:

- `sqrt_fixed_49_25_s` is `28` cycles in the csynth performance table.
- Each visible LayerNorm subtotal is `300` cycles:
  `32 + 80 + 28 + 160`.
- Both LayerNorm calls together account for `600` visible cycles, or `7.22%`
  of top-level latency.
- Even if the entire `137` cycle residual were conservatively assigned to
  LayerNorm scalar divider/control, the LayerNorm pair would be at most
  `737` cycles, or `8.86%`, still below a single `1329` cycle 16x16 linear
  block.

Conclusion: the available reports refute the suspicion that LayerNorm sqrt or
division is the largest single latency contributor. LayerNorm is nontrivial
and has sequential divider hardware in RTL, but the dominant contributors are
the unpipelined/ununrolled 16x16 linear loops.

## Pending Verification

The following were not available from the existing frozen artifacts:

- Per-child RTL cosim start/end cycle timestamps.
- Waveform-derived exact divider contribution inside each LayerNorm call.
- A fully annotated RTL timeline that assigns the `137` residual cycles to
  named FSM states.

Obtaining those would require waveform or transaction instrumentation and a
new cosim analysis run. That was intentionally not done in this audit because
the task is to explain the current frozen result without modifying or
optimizing the design.

## Honest Conclusion

The `8316` cycle number is not a black box at top level: csynth reports exactly
`8316` cycles and RTL cosim confirms every transaction at `8316` cycles. At
stage level, the best available breakdown shows that the design is dominated
by repeated 16x16 linear loops under `#pragma HLS PIPELINE off`; these loops
contribute about `1.3k` cycles each and form the largest visible critical-path
blocks. LayerNorm sqrt/division exists and should remain on the optimization
watchlist, but it is not the largest single contributor in the current reports.
