# PYNQ-Z2 Wrapper Bring-up Status

status: `v3diag_100mhz_on_board_functional_bit_equivalence_pass`

## V3 Diagonal Deployment Track (2026-10-04)

The 06U V3 diagonal core has now completed Stages A-E in isolated V3-specific
directories. Its no-VIO board-level wrapper met 100 MHz timing with WNS
`+0.148 ns`, TNS `0.000 ns`, and zero setup/hold failing endpoints. The VIO
readback companion also met timing after one bounded post-route physical
optimization (WNS `+0.019 ns`, TNS `0.000 ns`). The derived core latency is
`530 / 100 MHz = 5.300 us`; this remains a routed timing derivation, not an
independent on-silicon latency measurement.

The positive on-board readback bitstream is:

`results_v3_diag/pnr_100mhz_readback_vio_postroute_physopt/phc_v3diag_demo_100mhz_readback_vio_postroute_physopt.bit`

It was programmed to `xc7z020_1` and completed all 4,096 events. The built-in
1-LSB gate reported `pass=1`, `fail=0`, and zero mismatches; the independent
zero-tolerance counters reported zero drift rows, zero maximum LSB difference,
and zero decision flips. A separate timing-closed negative bitstream injected
one 16-LSB expected-readout fault at index 17 and the board reported exactly
one mismatch with `fail=1`. Full evidence is in `V3DIAG_STAGE_E_REPORT.md`.

## Scope

This workspace prepares the demo replay wrapper for PYNQ-Z2. It does not modify the frozen 06F, 06L, 06U, or 06S IP-level outputs, and it does not retrain any model.

The physical PYNQ-Z2 has been detected and programmed. The first programmed bitstream exposed a LED usability bug: `done` was only held for one 90 MHz clock cycle, so LD0 was not human-visible even though LD1 `pass` stayed high. The wrapper has since been fixed, re-simulated, re-routed, regenerated, and reprogrammed. Visual LED observation of the new bitstream is pending user confirmation.

## Current Decision

The wrapper now uses the complete 06S seed2 / 06L unroll16 90 MHz RTL source set, not the preferred 06U V1 22.36 us RTL package.

Reason: the repo-local 06U V1 RTL export is not self-contained for standalone wrapper XSim. Missing generated submodules include local RAM/ROM helper modules such as:

```text
phc_step_p_anonymous_namespace_step_dataflow_fn_state_prev_for_main_RAM_AUTO_1R1W_memcore
phc_step_p_anonymous_namespace_main_chain_fn_p_ZN16phc_step_weightsL9SYNAPSE_WE_0_ROM_yd2
phc_step_p_anonymous_namespace_gate_branch_p_ZN16phc_step_weightsL6GATE_WE_0_ROM_AUTO_1R
```

This does not invalidate the 06U latency report, but it blocks this standalone board wrapper from using 06U V1 tonight.

## Demo Narrative Baselines

- Replay burst remains `NEW4_20260112_TXFA6`.
- Burst selection rule: choose the highest 1-second rolling event count among the eligible `val/new3/new4` sessions. This is an objective reproducible criterion, not a hand-picked dramatic interval for demo effect.
- CPU comparison baseline is N=4 concurrent LibTorch/TorchScript, not N=1. This better represents a realistic laptop/edge deployment where multiple strategies or models contend for CPU resources.
- Fixed CPU mouthpiece: `n2ew_era\06f_phc_lite_manual_hls\results\cpu_concurrent_benchmark_N4.json`, round 1: mean `72.198630 us`, p50 `65.600000 us`, p95 `126.300000 us`, p99 `262.600000 us`, deadline `83.160000 us`, deadline miss rate `15.1250%`.
- N=4 round 2 exists with deadline miss rate `15.1375%`; the demo uses round 1 consistently because `DEMO_REPLAY_STALENESS_REPORT.md` and the generated staleness replay already use the N=4 empirical distribution.

## Verification Summary

| item | status | evidence |
|---|---:|---|
| Demo replay burst selection | done | `results/demo_replay_window_selection.csv` |
| CPU-vs-FPGA staleness scoreboard | done | `results/demo_replay_staleness_timeseries.csv`, `results/demo_replay_summary.csv` |
| Scoreboard latency aligned to selected board RTL | done | FPGA fixed latency now `25.733 us` in `00_build_demo_replay_staleness.py` |
| BRAM replay input/reference packing | done | `bram/demo_replay_x_readout_q24_12.mem`, `bram/BRAM_VECTOR_PACKING.md` |
| Selected HLS replay CSim | pass | `results/demo_replay_seed2_90_hls_csim_out.csv`; `total=4096`, `mismatches=0` |
| Wrapper positive XSim | pass | `logs/xsim_direct_positive_xsim.log`; `POSITIVE_TEST_PASS`, `4096` vectors |
| Wrapper negative XSim | pass | `logs/xsim_direct_negative_xsim.log`; `NEGATIVE_TEST_PASS`, mismatch caught at index `17` |
| Done LED latch fix | done | `rtl/pynq_z2_phc_demo_wrapper.sv`; `S_DONE` now holds `done_reg=1` until reset |
| Wrapper routed P&R after latch fix | pass | `results/wrapper_timing_summary_routed.rpt`; WNS `+0.198 ns`, 0 setup/hold failing endpoints |
| Bitstream generation after latch fix | pass | `results/phc_demo_wrapper.bit`; `Bitgen Completed Successfully` in `logs/wrapper_pnr_vivado_done_latch_fix.log` |
| Hardware/JTAG detection | pass | `vivado_13024.backup.log`; detected `arm_dap_0`, `xc7z020_1` |
| Hardware program before latch fix | pass | `vivado.log`; `PYNQ_Z2_PROGRAM_STATUS: programmed` |
| Hardware program after latch fix | pass | `logs/program_pynq_z2_demo_done_latch_fix_retry.log`; `PYNQ_Z2_PROGRAM_STATUS: programmed` |
| V3 positive VIO companion timing | pass | `results_v3_diag/pnr_100mhz_readback_vio_postroute_physopt/v3diag_readback_postroute_physopt_timing_summary.rpt`; WNS `+0.019 ns`, TNS `0.000 ns` |
| V3 on-board positive replay | pass | `logs/vivado_v3diag_program_and_read_positive.log`; 4,096 rows, zero exact drift, zero decision flips |
| V3 negative comparator control | pass | `logs/vivado_v3diag_program_and_read_negative.log`; one injected 16-LSB mismatch caught, `fail=1` |

## XSim Details

Positive XSim uses the real 4096-event replay vector corpus and the complete 06S seed2 / 06L unroll16 90 MHz RTL:

```text
DEMO_WRAPPER_DONE current_index=4095 pass=1 fail=0 first_mismatch_index=4294967295
MISMATCH_STATS count=0 max_readout_lsb_diff=1 readout_tol_lsb=1
POSITIVE_TEST_PASS
```

Negative XSim intentionally corrupts one expected readout value by 16 Q24.12 LSBs:

```text
DEMO_WRAPPER_DONE current_index=31 pass=0 fail=1 first_mismatch_index=17
FIRST_MISMATCH actual0=0xfffe9d expected0=0xfffe8d actual1=0xfffede expected1=0xfffede
MISMATCH_STATS count=1 max_readout_lsb_diff=16 readout_tol_lsb=1
NEGATIVE_TEST_PASS
```

The comparator uses a fixed 1-LSB readout tolerance because exact zero-LSB comparison showed a maximum 1-LSB drift under RTL-feedback recurrence. This is still strict at Q24.12 precision; the negative test verifies that real faults are not masked.

## Routed P&R Details

Selected board-level top: `rtl/pynq_z2_phc_demo_board_top.sv`

Bitstream: `results/phc_demo_wrapper.bit`

Verification tier: `routed_pnr_met_no_board_test`

| metric | value |
|---|---:|
| Input clock | 125 MHz on PYNQ-Z2 `sysclk` pin H16 |
| Core clock | 90 MHz generated by MMCM |
| Core latency headline | 2316 cycles / 25.733 us |
| Setup WNS | +0.198 ns |
| Setup TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.008 ns |
| Hold THS | 0.000 ns |
| Hold failing endpoints | 0 |
| LUT | 15,693 / 53,200 = 29.50% |
| FF | 20,285 / 106,400 = 19.06% |
| DSP | 131 / 220 = 59.55% |
| Block RAM Tile | 28.5 / 140 = 20.36% |
| MMCM | 1 / 4 = 25.00% |

Summary CSV: `results/wrapper_pnr_summary.csv`

## ILA Status

The generated bitstream is a no-ILA LED sanity bitstream. The top-level RTL leaves internal signals marked with `MARK_DEBUG`, and an optional `PHC_ENABLE_ILA` block exists, but the automated direct-flow ILA build was not completed because Vivado could not synthesize the generated ILA submodule set cleanly in this harness.

Therefore:

- No `.ltx` debug-probe file is produced for the current bitstream.
- Hardware Manager programming can still be done.
- Live ILA capture cannot be claimed from this bitstream.
- If ILA is required tomorrow, add/debug the ILA from the Vivado GUI or restore project-mode `launch_runs` after fixing the local Vivado/TclStore/Windows Script Host environment.

## Board-Dependent Items

Already observed on the first programmed bitstream:

- `connect_hw_server` / `get_hw_devices`: pass
- `program_hw_devices`: pass
- LD3 heartbeat: observed blinking
- LD1 pass: observed lit
- LD0 done: not human-visible because the old bitstream pulsed `done` for one cycle only

Still pending after programming the latch-fixed bitstream:

- Confirm LD0 stays lit after replay completes
- Confirm LD1 pass stays lit and LD2 fail stays dark
- ILA or VIO sanity capture

Prepared Tcl scripts:

- Presence check: `tcl/check_pynq_z2_hw.tcl`
- Program bitstream: `tcl/program_pynq_z2_demo.tcl`

## Claim Boundary

- This is not a proof that FPGA improves PnL.
- This is not live exchange replay.
- The V3 replay is on-board verified for functional execution and bit-level equivalence.
- The `5.300 us` value is still routed-derived; no independent on-silicon cycle counter or ILA latency measurement was implemented.
