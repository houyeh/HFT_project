# TCN Stage2 Inner Reduction Pipeline Audit

Status: `resolved_negative`

## Scope

This audit tests the second-round 07C2B streaming-speed hypothesis: keep the CONV1/CONV2 outer `oc` loops non-pipelined and apply `#pragma HLS PIPELINE II=1` only to the inner `ic` reduction loops. The goal is to determine whether a single non-nested reduction loop can be pipelined without the previous block-level pragma conflict.

No 07C2D artifacts were modified. No training, NEW5 access, RTL cosim beyond the gate, implementation, or P&R was performed.

## Code Changes

- Added `TCN_STAGE2_INNER_PIPELINE`, default `0`, in `hls_src/tcn_stage2_resource_impl.hpp`.
- Wrapped the CONV1 and CONV2 inner `ic` loop pragmas:
  - `TCN_STAGE2_INNER_PIPELINE=1`: `#pragma HLS PIPELINE II=1`
  - otherwise: `#pragma HLS PIPELINE off` plus `#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL`
- Added `hls_src/tcn_stage2_resource_distilled_shared_u4_innerpipe.cpp`.
- Added `distilled_shared_u4_innerpipe` to `01_prepare_speed_components.py`.

The new variant does not define `TCN_STAGE2_BLOCK_PIPELINE`, so the outer `oc` loops remain on the existing non-pipelined path.

## Prepared Component

The component was generated with the local script's current argument name, `--variant`:

```text
python 01_prepare_speed_components.py --variant distilled_shared_u4_innerpipe --clean
```

Prepared short-path component:

```text
C:\tmp\07c2b_tcn_streaming_speed_shortpath\distilled_shared_u4_innerpipe\hls_component_stage2_resource_distilled_shared_u4_innerpipe
```

Workspace evidence copy:

```text
n2ew_era/07c2b_tcn_streaming_speed/artifacts/distilled_shared_u4_innerpipe/
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

- `artifacts/distilled_shared_u4_innerpipe/tcn_streaming_step_csim.log`
- `artifacts/distilled_shared_u4_innerpipe/stage2_speed_distilled_shared_u4_innerpipe_event1008_tb_summary.csv`

## C-Synthesis

The Vitis 2025.2 command-line flag `--syn` is not supported on this machine, so synthesis was run through the generated Tcl script:

```text
vitis-run.bat --mode hls --tcl --input_file run_csynth_only.tcl
```

Result: C-synthesis failed during scheduling/hardware synthesis. No top-level `tcn_streaming_step_csynth.rpt` latency/resource report was produced.

Evidence:

- `artifacts/distilled_shared_u4_innerpipe/hls.log`
- `artifacts/distilled_shared_u4_innerpipe/hls_run_tcl.log`
- `artifacts/distilled_shared_u4_innerpipe/csynth_design_size.rpt`

## Pragma Conflict Gate

The explicit conflict strings were absent:

| search string | result |
|---|---|
| `pragma conflict` | 0 matches |
| `Ignoring PIPELINE` | 0 matches |

The relevant positive pipeline evidence is:

```text
INFO: [SCHED 204-61] Pipelining loop 'VITIS_LOOP_209_8'.
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 4, loop 'VITIS_LOOP_209_8'
```

Evidence: `artifacts/distilled_shared_u4_innerpipe/hls.log`.

## Failure Mechanism

The run did not fail because Vitis rejected the pragma. It failed because the direct accumulator reduction path could not meet the 10 ns target:

```text
WARNING: [HLS 200-887] Cannot meet target clock period from 'add' operation 64 bit ('add_ln225', ../hls_src/tcn_stage2_resource_impl.hpp:225) to 'add' operation 64 bit ('add_ln227', ../hls_src/tcn_stage2_resource_impl.hpp:227) (combination delay: 10.56 ns) to honor II or Latency constraint in region 'VITIS_LOOP_209_8'.
WARNING: [HLS 200-887] Cannot meet target clock period from 'add' operation 64 bit ('add_ln225', ../hls_src/tcn_stage2_resource_impl.hpp:225) to 'store' operation ('acc_write_ln208', ../hls_src/tcn_stage2_resource_impl.hpp:208) of variable 'acc', ../hls_src/tcn_stage2_resource_impl.hpp:227 40 bit on local variable 'acc', ../hls_src/tcn_stage2_resource_impl.hpp:208 (combination delay: 12.148 ns) to honor II or Latency constraint in region 'VITIS_LOOP_209_8'.
WARNING: [HLS 200-871] Estimated clock period (12.148 ns) exceeds the target (target clock period: 10.000 ns, clock uncertainty: 2.700 ns, effective delay budget: 7.300 ns).
```

The reported critical path is:

| operation | source location | delay |
|---|---|---:|
| `acc_load` | `tcn_stage2_resource_impl.hpp:225` | 0.000 ns |
| `add_ln225` | `tcn_stage2_resource_impl.hpp:225` | 3.520 ns |
| `add_ln226` | `tcn_stage2_resource_impl.hpp:226` | 3.520 ns |
| `add_ln227` | `tcn_stage2_resource_impl.hpp:227` | 3.520 ns |
| `acc_write_ln208` | `tcn_stage2_resource_impl.hpp:208/227` | 1.588 ns |

Evidence lines: `artifacts/distilled_shared_u4_innerpipe/hls.log:163` through `hls.log:173`, and `artifacts/distilled_shared_u4_innerpipe/hls_run_tcl.log:179` through `hls_run_tcl.log:197`.

## Baseline Comparison

Because this variant failed before a top csynth report, latency and resource numbers are not available.

| variant | verification level | latency cycles | interval cycles | DSP | LUT | FF | BRAM_18K | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| distilled_shared_u4 07C2B baseline | csynth | 7,438 | 7,439 | 100 | 37,475 | 24,369 | 66 | same-infrastructure baseline from previous audit |
| distilled shared_u4 07C2D routed closure | routed P&R | 7,438 | 7,439 | 49.09% | 32.00% | 14.44% | 23.57% | iface-slim routed reference |
| distilled_shared_u4_innerpipe | CSim pass, csynth fail | n/a | n/a | n/a | n/a | n/a | n/a | failed during scheduling on `VITIS_LOOP_209_8` |

## Cosim Gate

RTL cosim was not run. The gate was not satisfied because csynth failed and no RTL artifact/resource report was produced.

## Conclusion

A single non-nested inner `ic` reduction loop can be accepted by Vitis as a pipeline attempt: there was no `pragma conflict`, no `Ignoring PIPELINE`, and the log reports `Target II = 1, Final II = 1` for `VITIS_LOOP_209_8`.

However, this is not a deployable speed variant. In the current direct-accumulator form, the 40-bit accumulator load/write plus three 64-bit add operations create a 12.148 ns estimated critical path against the 10 ns clock target, and csynth stops with `Synthesis failed`. The next meaningful experiment would need to change the reduction structure, such as partial sums or a tree reduction, rather than adding more pipeline pragmas around the same accumulator loop.
