# TCN Stage2 Inner Pipeline Partial-Sum Audit

Status: `resolved_mixed`

## Scope

This audit tests the third-round 07C2B streaming-speed hypothesis: keep the round-2 `TCN_STAGE2_INNER_PIPELINE=1` inner `ic` loop pipeline, leave the outer `oc` loops non-pipelined, and restructure the three consecutive tap additions into local partial sums so the loop-carried accumulator only sees one `acc += tap_sum` dependency.

No 07C2D artifacts were modified. No training, NEW5 access, implementation, or P&R was performed.

## Code Changes

- Rewrote the CONV1 accumulator body in `hls_src/tcn_stage2_resource_impl.hpp:165` through `:169` from three consecutive `acc +=` operations to `tap_m2`, `tap_m1`, `tap_cur`, `tap_sum`, then one `acc += tap_sum`.
- Rewrote the CONV2 accumulator body in `hls_src/tcn_stage2_resource_impl.hpp:227` through `:231` in the same form with `c1_m2`, `c1_m1`, and `c1_cur`.
- Added the requested long-name wrapper `hls_src/tcn_stage2_resource_distilled_shared_u4_innerpipe_partialsum.cpp`.
- Added a short equivalent wrapper `hls_src/tcn_s2_dsu4_ip_ps.cpp` for the actual Windows HLS run. The long wrapper path triggered the same `mingw32-make no rule` path-handling failure seen in earlier Windows CSim diagnostics; the short wrapper has the same four macro definitions and includes the same implementation header.
- Added `distilled_shared_u4_innerpipe_partialsum` to `01_prepare_speed_components.py`, with a variant-local short work root `C:/tmp/07c2b_ipps/c`.

The new variant does not define `TCN_STAGE2_BLOCK_PIPELINE`, so the outer `oc` loops remain on the existing `PIPELINE off` path. No new HLS macro branch was introduced for the arithmetic rewrite.

## Prepared Component

Command:

```text
python 01_prepare_speed_components.py --variant distilled_shared_u4_innerpipe_partialsum --clean
```

Prepared short-path component:

```text
C:\tmp\07c2b_ipps\c
```

Workspace evidence copy:

```text
n2ew_era/07c2b_tcn_streaming_speed/artifacts/distilled_shared_u4_innerpipe_partialsum/
```

## C-Sim

Command:

```text
vitis-run.bat --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

Result:

| metric | value |
|---|---:|
| samples | 4 |
| event timesteps | 1008 |
| readout_rows | 4 |
| readout_max_abs_diff | 0.420075 |
| readout_rmse | 0.308342 |
| sign_agreement_readout_0 | 1 |
| sign_agreement_readout_1 | 1 |
| sign_agreement_joint | 1 |
| valid_alignment_errors | 0 |

Per-stage fixed-point comparison from the CSim summary:

| stage | rows | compared_values | max_abs_diff | rmse |
|---|---:|---:|---:|---:|
| block0 | 992 | 15,872 | 0.004620076875 | 0.000579061365921 |
| block1 | 960 | 15,360 | 0.021152975 | 0.00177219787195 |
| block2 | 896 | 14,336 | 0.029714581875 | 0.00265998846775 |
| block3 | 768 | 12,288 | 0.037681575625 | 0.00324995159233 |
| block4 | 512 | 8,192 | 0.037681575625 | 0.00427384802503 |
| readout | 4 | 8 | 0.4200745125 | 0.308342212205 |

Evidence:

- `artifacts/distilled_shared_u4_innerpipe_partialsum/tcn_streaming_step_csim.log`
- `artifacts/distilled_shared_u4_innerpipe_partialsum/stage2_speed_distilled_shared_u4_innerpipe_partialsum_event1008_tb_summary.csv`

## C-Synthesis

The Vitis 2025.2 command-line flag `--syn` is not supported on this machine, so synthesis was run through the generated Tcl script:

```text
vitis-run.bat --mode hls --tcl --input_file run_csynth_only.tcl
```

Top report:

```text
artifacts/distilled_shared_u4_innerpipe_partialsum/csynth_reports/tcn_streaming_step_csynth.rpt
```

Result: C-synthesis passed and produced top-level latency/resource reports.

| metric | value |
|---|---:|
| target clock | 10.00 ns |
| clock uncertainty | 2.70 ns |
| estimated clock | 7.040 ns |
| estimated Fmax | 142.05 MHz |
| latency min | 15 cycles |
| latency max | 5,036 cycles |
| latency max @100 MHz | 50.360 us |
| interval min | 16 cycles |
| interval max | 5,037 cycles |
| interval max @100 MHz | 50.370 us |
| top Pipeline Type | no |

Instance latency:

| instance | max latency cycles | max latency @100 MHz |
|---|---:|---:|
| block0 `stage2_resource_stream_block_step_0_1_3` | 944 | 9.440 us |
| block1 `stage2_resource_stream_block_step_1_2_5` | 944 | 9.440 us |
| block2 `stage2_resource_stream_block_step_2_4_9` | 976 | 9.760 us |
| block3 `stage2_resource_stream_block_step_3_8_17` | 982 | 9.820 us |
| block4 `stage2_resource_stream_block_step_4_16_33` | 1,124 | 11.240 us |
| readout `Outline_VITIS_LOOP_312_1` | 129 | 1.290 us |

Resource estimate:

| resource | used | available | utilization |
|---|---:|---:|---:|
| DSP | 100 | 220 | 45.45% |
| LUT | 34,089 | 53,200 | 64.08% |
| FF | 23,579 | 106,400 | 22.16% |
| BRAM_18K | 180 | 280 | 64.29% |

## Timing Gate

The round-2 failure strings are absent from the synthesis log:

| search string | result |
|---|---|
| `Cannot meet target clock period` | 0 matches |
| `Estimated clock period` | 0 matches |
| `critical path` | 0 matches |
| `pragma conflict` | 0 matches |
| `Ignoring PIPELINE` | 0 matches |

Positive synthesis evidence:

```text
INFO: [HLS 200-790] **** Loop Constraint Status: All loop constraints were satisfied.
INFO: [HLS 200-789] **** Estimated Fmax: 142.05 MHz
```

Evidence lines: `artifacts/distilled_shared_u4_innerpipe_partialsum/hls_run_tcl.log:939` through `:940`.

The inner loop pipeline evidence appears for all five template instantiations of CONV1 and CONV2:

```text
INFO: [SCHED 204-61] Pipelining loop 'VITIS_LOOP_149_5'.
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 6, loop 'VITIS_LOOP_149_5'
INFO: [SCHED 204-61] Pipelining loop 'VITIS_LOOP_211_8'.
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 6, loop 'VITIS_LOOP_211_8'
```

Evidence lines: `artifacts/distilled_shared_u4_innerpipe_partialsum/hls_run_tcl.log:183` through `:362`.

## Baseline Comparison

| variant | verification level | latency cycles | interval cycles | latency @100 MHz | interval @100 MHz | DSP | LUT | FF | BRAM_18K | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilled_shared_u4 07C2B baseline | csim+csynth | 7,438 | 7,439 | 74.38 us | 74.39 us | 100 | 37,475 | 24,369 | 66 | same-infrastructure baseline |
| distilled_shared_u4_innerpipe round 2 | CSim pass, csynth fail | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | failed on 12.148 ns accumulator chain |
| distilled_shared_u4_innerpipe_partialsum | CSim pass, csynth pass, cosim attempted OOM | 5,036 | 5,037 | 50.36 us | 50.37 us | 100 | 34,089 | 23,579 | 180 | this audit |
| distilled shared_u4 07C2D routed closure | routed P&R | 7,438 | 7,439 | 74.38 us | 74.39 us | 49.09% | 32.00% | 14.44% | 23.57% | iface-slim routed reference, not the same 07C2B csynth resource accounting |

Relative to the 07C2B same-infrastructure baseline, this partial-sum variant reduces top max latency from 7,438 to 5,036 cycles, a 2,402-cycle reduction, or 32.29% fewer cycles. The latency speedup is 1.48x at the csynth estimate level.

The DSP total remains 100 and still fits the 220-DSP Zynq-7020 budget. The partial-sum inner pipeline is applied to all five instantiated blocks in this run, so the 100-DSP top-level number already includes the full 5-block design rather than a single-template extrapolation. BRAM_18K increases from 66 to 180, likely due to additional memory copies/ports inferred to satisfy the II=1 loop schedules; it remains under the 280 BRAM_18K device budget.

## RTL Co-Simulation Attempt

Because CSim and csynth passed and resources fit, RTL cosim was attempted:

```text
vitis-run.bat --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

The C testbench phase completed and reproduced the same 4-window summary:

| metric | value |
|---|---:|
| samples | 4 |
| readout_rows | 4 |
| readout_max_abs_diff | 0.420075 |
| readout_rmse | 0.308342 |
| sign_agreement_readout_0 | 1 |
| sign_agreement_readout_1 | 1 |
| sign_agreement_joint | 1 |
| valid_alignment_errors | 0 |

XSIM reached real RTL simulation and completed transactions through `918 / 1012`, then terminated due to simulator memory exhaustion:

```text
// RTL Simulation : 918 / 1012 [100.00%] @ "34363585000"
Out of memory on request for a fresh 8388608 bytes
Total memory consumed so far :3931333132 bytes
ERROR: [COSIM-4] *** C/RTL co-simulation finished: FAIL ***
```

Evidence:

- `artifacts/distilled_shared_u4_innerpipe_partialsum/cosim_logs/hls_run_cosim.log`
- `artifacts/distilled_shared_u4_innerpipe_partialsum/cosim_logs/xsim.log`
- `artifacts/distilled_shared_u4_innerpipe_partialsum/cosim_logs/tcn_streaming_step.log`

This is not recorded as RTL cosim pass. No RTL mismatch was reported before the simulator OOM termination.

## Conclusion

Partial-sum restructuring did solve the round-2 12.148 ns scheduling/timing problem at the csynth level: the `Cannot meet target clock period` warnings disappeared, all loop constraints were satisfied, and the estimate improved to 7.040 ns / 142.05 MHz.

The speed result is materially better than the 07C2B shared-u4 baseline at the same 100 MHz csynth target: 5,036 cycles / 50.36 us versus 7,438 cycles / 74.38 us. The resource tradeoff is acceptable but nontrivial: DSP stays at 100/220, LUT and FF fit, while BRAM rises to 180/280.

The honest verification level is `CSim pass + csynth pass + RTL cosim attempted but not completed`. The cosim failure was an XSIM out-of-memory event at 918/1012 transactions, not a reported functional mismatch. This variant is therefore a promising csynth-level speed result, but it is not yet cosim- or routed-P&R-verified.
