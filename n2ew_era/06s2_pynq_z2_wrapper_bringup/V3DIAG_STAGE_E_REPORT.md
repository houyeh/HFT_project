# V3 Diagonal PYNQ-Z2 Stage E Report

status: `stage_e_pass_on_board_functional_bit_equivalence`

Date: 2026-10-04

## Verification Boundary

The V3 diagonal SSM replay core was programmed onto a physical PYNQ-Z2 and
completed all 4,096 replay rows. The on-board evidence covers functional
execution and bit-level readout equivalence through VIO. The `5.300 us`
latency remains a routed derivation (`530 cycles / 100 MHz`), not an
independent on-silicon latency measurement; this design does not contain a
cycle counter or ILA latency measurement path.

## Timing-Closed Diagnostic Builds

The VIO readback logic is a diagnostic companion to the no-VIO demo
bitstream. Both positive and negative companions use the same V3 core and
100 MHz board clock.

| build | initial routed WNS / TNS | bounded post-route action | final WNS / TNS | failing endpoints |
|---|---:|---|---:|---:|
| positive reference | `-0.027 ns / -0.248 ns` | `phys_opt_design -directive AggressiveExplore` | `+0.019 ns / 0.000 ns` | 0 |
| negative injection | `-0.238 ns / -9.420 ns` | `phys_opt_design -directive AggressiveExplore` | `+0.013 ns / 0.000 ns` | 0 |

Positive diagnostic resources are `14,107 LUT (26.52%)`, `16,464 FF
(15.47%)`, `196 DSP (89.09%)`, and `28.5 Block RAM Tile (20.36%)` on the
Zynq-7020. These are board-level resources including wrapper and VIO, not the
standalone HLS-core figures.

## Positive On-Board Result

The timing-closed positive VIO bitstream was programmed to `xc7z020_1`. The
replay reached index `4095`, so all 4,096 rows completed.

| metric | on-board value |
|---|---:|
| Built-in 1-LSB tolerance gate | `pass=1`, `fail=0` |
| Tolerance-gate mismatch count | 0 |
| Zero-tolerance drift rows | 0 / 4,096 |
| Maximum readout difference | 0 LSB |
| Difference direction | `-1: 0`, `+1: 0`, `other: 0` |
| Decision flips | 0 |
| Minimum policy margin | 1 LSB |

This V3 result is exactly equal to its V3 HLS CSim-derived expected readout
stream. It is not expected to reproduce the dense model's earlier `27 / 4096`
one-LSB drift count because the arithmetic and recurrence paths differ.

Evidence: `logs/vivado_v3diag_program_and_read_positive.log`.
After the negative control, the positive bitstream was programmed again and
the same all-pass counters were re-read. The board was therefore left in the
positive demo state. Evidence: `logs/vivado_v3diag_restore_positive_final.log`.

## Negative On-Board Comparator Test

A separate timing-closed bitstream was built with a compile-time fault that
adds 16 LSB to the expected readout at replay index 17. Synthesis confirms
`INJECT_NEGATIVE_TEST=1`; XSim localizes the sole mismatch to index 17. The
on-board VIO does not expose `first_mismatch_index`, so the hardware claim is
limited to detecting exactly one injected fault, while its location is
established by the compile-time parameter and XSim.

| metric | on-board value |
|---|---:|
| Built-in 1-LSB tolerance gate | `pass=0`, `fail=1` |
| Mismatch count | 1 |
| Zero-tolerance drift rows | 1 |
| Maximum readout difference | 16 LSB (`0x10`) |
| Difference direction | `-1: 0`, `+1: 0`, `other: 1` |
| Decision flips | 0 |

This establishes that the positive zero-mismatch result is not caused by a
disconnected or permanently passing comparator.

Evidence: `logs/vivado_v3diag_program_and_read_negative.log` and
`logs/xsim_v3diag_readback_negative_xsim.log`.

## Stage E Result

```text
[E] programming: success, device=xc7z020_1
[E] total rows: 4096
[E] built-in 1-LSB gate: pass=1, fail=0, mismatches=0
[E] exact zero-tolerance drift rows: 0
[E] maximum difference: 0 LSB
[E] difference direction: -1=0, +1=0, other=0
[E] decision flips: 0
[E] minimum policy margin: 1 LSB
[E] negative control: caught 1 injected 16-LSB fault at configured index 17
[E] negative control decision flips: 0
[E] functional verification tier: on-board (functional + bit-level equivalence)
[E] latency verification tier: routed (derived from post-route timing closure)
```

The acceptance condition `decision_flip_count == 0` is satisfied.

## Programming Artifacts

| artifact | SHA-256 |
|---|---|
| `results_v3_diag/pnr_100mhz_readback_vio_postroute_physopt/phc_v3diag_demo_100mhz_readback_vio_postroute_physopt.bit` | `10676F79F11734E8DB92CCAF302C8E3A596417F0C707B5FE6CE5C5DDF34FFE79` |
| `results_v3_diag/pnr_100mhz_readback_vio_postroute_physopt/phc_v3diag_demo_100mhz_readback_vio_postroute_physopt.ltx` | `9A61C23AC7E2092C53882CB7A6D3CE73F22BE66E990E5A28AB9666FAD5C97546` |
| `results_v3_diag/pnr_100mhz_readback_vio_negative_postroute_physopt/phc_v3diag_demo_100mhz_readback_vio_negative_postroute_physopt.bit` | `DE6EB4FAF630FF4939E1140508A4050E2C4410DFA95A083ABE465E437E7D7A4D` |
| `results_v3_diag/pnr_100mhz_readback_vio_negative_postroute_physopt/phc_v3diag_demo_100mhz_readback_vio_negative_postroute_physopt.ltx` | `9A61C23AC7E2092C53882CB7A6D3CE73F22BE66E990E5A28AB9666FAD5C97546` |
