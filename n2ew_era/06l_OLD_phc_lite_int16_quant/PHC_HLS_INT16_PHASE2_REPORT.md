# PHC-lite HLS Phase 2 Q16.8 Quantization Report

Final status: `q16_8_csim_and_csynth_complete_but_not_recommended`.

## Purpose

This phase tests whether the Phase 06F manual HLS PHC-lite step implementation can reduce external/storage fixed-point width from Q24.12 to Q16.8 while preserving numerical behavior and PYNQ-Z2 synthesis feasibility.

This is a quantization feasibility check only. It does not retrain the model, does not run P&R / Vivado implementation, does not evaluate PnL, and does not touch NEW5.

## Source And Changes

Phase 06L was created as a separate copy under:

`n2ew_era/06l_phc_lite_int16_quant/`

Phase 06F was not modified.

Copied from Phase 06F:

- `hls_src/phc_step_hls.cpp`
- `hls_src/phc_step_hls.hpp`
- `hls_src/phc_weights.hpp`
- `hls_src/phc_step_testbench.cpp`
- `hls_component/hls_config.cfg`
- `results/phc_step_test_vectors.csv`

The copied test vector file is byte-identical to Phase 06F:

`MD5 = F7130075A7479F2099B6FD2857135F92`

Intentional Phase 06L changes:

- `phc_fixed_t` changed from `ap_fixed<24,12>` to `ap_fixed<16,8>`.
- `phc_weights.hpp` stored weight type changed from Q24.12 to Q16.8.
- `phc_acc_t = ap_fixed<40,16>` and `phc_var_t = ap_fixed<48,24>` were left unchanged.
- `hls_config.cfg` argv paths were redirected to the 06L results directory.

No algorithmic loops, activation LUT values, weights source, testbench tolerance, or HLS scheduling pragmas were changed relative to the 06F implementation.

## Commands Run

C-Sim:

```powershell
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

C-Synthesis:

```powershell
v++ -c --mode hls --config hls_config.cfg --work_dir hls_component
```

No `--impl` command was run.

## C-Sim Result

| phase | scheme | rows | mismatches | max_abs_diff | mean_abs_diff | tolerance |
|---|---|---:|---:|---:|---:|---:|
| 06F baseline | Q24.12 | 25,600 | 0 | 0.00725275 | 0.000372553 | 1.0 |
| 06L phase2 | Q16.8 | 25,600 | 0 | 0.11749 | 0.00589889 | 1.0 |

The Q16.8 C-Sim run passes the current tolerance with zero mismatches. However, the maximum absolute error is `16.20x` larger than the Q24.12 baseline, and the mean absolute error is `15.83x` larger.

See:

`results/readout_error_vs_std_phase2.md`

Archived C-Sim log:

`results/hls_run_csim.log`

## C-Synthesis Result

Target device: `xc7z020-clg400-1`

Clock target: `10.00 ns`

Estimated clock: `7.147 ns`

| phase | scheme | BRAM_18K | DSP | FF | LUT | URAM | latency_cycles | latency_us | interval_cycles | pipeline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 06F baseline | Q24.12 | 6 | 46 | 14,518 | 24,836 | 0 | 8,316 | 83.160 | 8,317 | no |
| 06L phase2 | Q16.8 | 6 | 46 | 14,332 | 24,746 | 0 | 8,316 | 83.160 | 8,317 | no |

Utilization for 06L:

| resource | used | available | utilization |
|---|---:|---:|---:|
| BRAM_18K | 6 | 280 | 2% |
| DSP | 46 | 220 | 20% |
| FF | 14,332 | 106,400 | 13% |
| LUT | 24,746 | 53,200 | 46% |
| URAM | 0 | 0 | n/a |

Resource interpretation: Q16.8 did not materially reduce the already-sequential 06F HLS resource footprint. DSP and BRAM are unchanged; LUT and FF reduce only slightly.

Archived C-Synthesis report:

`results/phc_step_csynth.rpt`

## Numerical Risk

The readout reference standard deviations are approximately:

- `ref_readout_0 std = 0.278582`
- `ref_readout_1 std = 0.276395`

The Phase 06L global max absolute error is `0.11749`, which is:

- `42.17%` of `ref_readout_0` std
- `42.51%` of `ref_readout_1` std

By contrast, the 06F baseline max error was about `2.60%` of readout std.

Readout-only Phase 06L max errors are:

- `readout_0 max_abs_diff = 0.081710`, or `29.33%` of `ref_readout_0` std
- `readout_1 max_abs_diff = 0.056871`, or `20.58%` of `ref_readout_1` std

This is a substantial numerical degradation even though the current C-Sim tolerance reports zero mismatches.

## Decision

Q16.8 is not recommended as a replacement for the Phase 06F Q24.12 implementation.

Reason:

- It does not meaningfully improve HLS resource utilization.
- It preserves latency at the same 8,316-cycle level.
- It increases numerical error by roughly 16x on the testbench summary metric.
- Its global max error is large relative to readout dynamic range.

The safer frozen implementation remains Phase 06F Q24.12 unless a future model-side or arithmetic redesign proves that Q16.8 preserves policy-level behavior.

## Claim Boundary

Allowed claims:

- Phase 06L Q16.8 C-Sim passed with zero mismatches under the existing tolerance.
- Phase 06L Q16.8 C-Synthesis fits the xc7z020 resource budget.
- Phase 06L Q16.8 does not materially reduce resources relative to Phase 06F Q24.12.
- Phase 06L Q16.8 materially increases numerical error relative to Phase 06F Q24.12.

Forbidden claims:

- Do not claim Q16.8 preserves policy-level PnL.
- Do not claim Q16.8 is deployment-ready.
- Do not claim FPGA improves PnL.
- Do not claim this is real-world exchange replay.
- Do not claim P&R timing closure; implementation/P&R was not run in this phase.
