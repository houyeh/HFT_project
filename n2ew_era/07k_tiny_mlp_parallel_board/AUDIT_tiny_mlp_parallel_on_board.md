# Tiny MLP Parallel PYNQ-Z2 Board Audit

status: `resolved_positive`

## Scope

This phase validates the frozen 07J fully parallel Tiny MLP on a physical
PYNQ-Z2. The policy, 163 learned parameters, fixed-point arithmetic, action
rule, and 4,096 replay rows were not changed. The board wrapper reads one
packed row from initialized BRAM, starts the HLS core, compares all three
40-bit logits and the 2-bit action exactly, and exposes the accumulated result
through LEDs and a VIO core.

The tested bitstream is
`results/board_100mhz/tiny_mlp_parallel_100mhz.bit`; its matching probe file is
`results/board_100mhz/tiny_mlp_parallel_100mhz.ltx`.

## Functional Evidence

| Check | Result | Tier |
|---|---:|---|
| Device | `xc7z020_1` | on-board |
| Replay rows completed | 4,096 (`current_index=0x00000fff`) | on-board |
| Exact logit mismatch rows | 0 | on-board |
| Action mismatch rows | 0 | on-board |
| Maximum logit difference | 0 LSB | on-board |
| Completion / pass / fail | `1 / 1 / 0` | on-board |
| Core wait counter | `0x13` = 19 clock edges | on-board |

The board therefore reproduced every stored fixed-point logit and action
exactly. The 19-edge wrapper observation is consistent with the independently
verified HLS schedule of 18-cycle latency and 19-cycle interval: the wrapper
counts the handshake observation edge as well as the scheduled core latency.
It is an on-chip cycle observation, not a measurement of network, feature
construction, or end-to-end order path latency.

The raw Hardware Manager transcript is retained in
`results/board_100mhz/program_and_read.log`.

## Board-Level P&R

Target: `xc7z020-clg400-1`, Vivado 2025.2, 100 MHz generated PL clock.

| Metric | Board-level routed result | Zynq-7020 budget | Utilization |
|---|---:|---:|---:|
| WNS / TNS | +0.544 ns / 0.000 ns | timing constraint | 0 failing endpoints |
| Slice LUTs | 9,423 | 53,200 | 17.71% |
| Slice registers | 11,492 | 106,400 | 10.80% |
| DSP48E1 | 96 | 220 | 43.64% |
| BRAM_18K equivalent | 72 | 280 | 25.71% |
| Core latency | 18 cycles / 180 ns | n/a | cycle-derived from routed 100 MHz clock |
| Core interval | 19 cycles / 190 ns | n/a | RTL cosim schedule |

All user-specified setup and hold constraints are met. The reports are
`results/board_100mhz/timing_summary_routed.rpt` and
`results/board_100mhz/utilization_routed.rpt`.

Vivado's vectorless power report estimates 0.326 W total on-chip power,
comprising 0.215 W dynamic and 0.111 W static. These are activity-propagation
estimates, not board power measurements; the report also warns that reset-net
activity may reduce estimate accuracy. The source is
`results/board_100mhz/power_routed.rpt`.

## Timing Failure and Repair

The first board implementation failed at WNS -5.065 ns and TNS -838.926 ns
with 517 failing endpoints. The worst path began at a Tiny MLP output register
and ended at the clock-enable of `max_logit_lsb_diff`, crossing a 40-bit
absolute difference, mismatch aggregation, and maximum comparison in one
wrapper cycle. This was a validation-wrapper path, not the inference schedule.

The checker was split into registered capture, three per-logit difference
stages, two maximum-reduction stages, and one counter-update stage. A repeated
4,096-row XSim remained exact (`mismatch=0`, `action_mismatch=0`,
`max_diff=0`) and retained the 19-edge core wait observation. The revised
board implementation then closed at WNS +0.544 ns. These extra checker stages
only extend replay bookkeeping after `ap_done`; they do not change the HLS
core's 18-cycle inference latency.

## Task-Value Boundary

This hardware point implements the 07G Tiny MLP policy, not the recurrent PHC
model. Its fixed-point causal-fill PnL is 139,515 / 472,055 / 154,400 on
val / NEW3 / NEW4, retaining 81.19% / 77.25% / 70.40% of the corresponding
frozen PHC reference. The 180 ns result is therefore a latency-value trade-off,
not an acceleration of the same PHC computation.

## Decision

The Tiny MLP design is now verified through CSim, RTL cosim, board-level routed
timing closure, bitstream generation, and physical PYNQ-Z2 replay. All 4,096
on-board outputs matched exactly, with no action flips. The defensible headline
is `180 ns core latency, board-level routed and functionally verified on-board
at 100 MHz`, with the explicit policy-substitution and end-to-end boundaries
above.
