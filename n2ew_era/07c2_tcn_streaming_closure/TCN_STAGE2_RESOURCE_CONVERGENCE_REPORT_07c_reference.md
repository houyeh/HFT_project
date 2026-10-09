status: resolved_at_csynth_resource_fit_no_pnr
finding: shared_history_u1_fits_zynq7020_with_16510_cycle_csynth_latency
# TCN Stage 2 Resource Convergence Report

## Scope

This pass only addresses 07C Stage 2 streaming HLS resource convergence. It does not modify 07a, 07b, Stage 0 outputs, or Stage 1 outputs. It does not use NEW5. It stops at CSim/csynth; no RTL cosim, Vivado implementation, P&R, routed timing, bitstream, or board claim is made here.

The arithmetic and FIFO/delay topology are kept aligned with the already validated Stage 2 design. The resource variants change only HLS-level scheduling/storage controls: channel unroll factor, local pipeline disabling to prevent forced complete unroll, and optional BRAM binding for history buffers.

## Reproducible Artifacts

| artifact | path |
| --- | --- |
| resource implementation header | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_src\tcn_stage2_resource_impl.hpp` |
| resource CSim testbench | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_src\tcn_stage2_resource_testbench.cpp` |
| prepare script | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\07_stage2_prepare_resource_sweep.py` |
| collector script | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\08_stage2_collect_resource_sweep.py` |
| summary CSV | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_resource_sweep_summary.csv` |
| summary JSON | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_resource_sweep_summary.json` |

## HLS Knobs Tested

| suffix | mode | channel unroll | history storage | intent |
| --- | --- | ---: | --- | --- |
| `u2` | `unroll_only` | 2 | complete partition registers | Reduce multiplier replication from the Stage 2 unroll-4 baseline. |
| `u1` | `unroll_only` | 1 | complete partition registers | Minimum channel unroll before adding storage sharing. |
| `shared_u1` | `shared_history` | 1 | `BIND_STORAGE ram_2p bram` for block history buffers | Trade BRAM and latency for LUT/FF reduction. |

All sweep csynth runs use part `xc7z020-clg400-1` and `clock=10ns`. All six sweep csynth runs completed with `complete_unroll_warning_count=0`.

## Prior Stage 2 Baseline

| variant | setting | CSim | csynth | latency cycles | latency at 100MHz | DSP | LUT | FF | BRAM_18K | fit Zynq-7020 | evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| scratch | unroll-4 baseline | pass | present | 558 | 5.58 us | 688 / 312.73% | 141327 / 265.65% | 223737 / 210.28% | 16 / 5.71% | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\STAGE2_STREAMING_FULL_HLS_REPORT.md` |
| distilled | unroll-4 baseline | pass | failed before top report | n/a | n/a | n/a | n/a | n/a | n/a | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_distilled\hls_component\logs\hls_compile.log` |

The distilled baseline failure was `ERROR: [v++ 17-180] Spawn failed: The operation completed successfully.` at `hls_compile.log` line 2463. This failure was not reproduced in the clean resource-sweep csynth projects below.

## Resource Sweep Results

| variant | suffix | CSim status | csynth status | latency cycles | latency at 100MHz | II cycles | interval at 100MHz | DSP | LUT | FF | BRAM_18K | fit Zynq-7020 | csynth report |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| scratch | `u2` | pass | present | 9860 | 98.60 us | 9861 | 98.61 us | 106 / 48.18% | 82336 / 154.77% | 127173 / 119.52% | 46 / 16.43% | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_scratch_u2\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |
| scratch | `u1` | pass | present | 16340 | 163.40 us | 16341 | 163.41 us | 90 / 40.91% | 74487 / 140.01% | 124455 / 116.97% | 46 / 16.43% | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_scratch_u1\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |
| scratch | `shared_u1` | pass | present | 16510 | 165.10 us | 16511 | 165.11 us | 90 / 40.91% | 27730 / 52.12% | 21143 / 19.87% | 66 / 23.57% | True | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_scratch_shared_u1\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |
| distilled | `u2` | timeout/empty summary | present | 9860 | 98.60 us | 9861 | 98.61 us | 100 / 45.45% | 82262 / 154.63% | 126908 / 119.27% | 46 / 16.43% | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_distilled_u2\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |
| distilled | `u1` | not rerun | present | 16340 | 163.40 us | 16341 | 163.41 us | 92 / 41.82% | 74597 / 140.22% | 124715 / 117.21% | 46 / 16.43% | False | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_distilled_u1\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |
| distilled | `shared_u1` | pass | present | 16510 | 165.10 us | 16511 | 165.11 us | 92 / 41.82% | 27840 / 52.33% | 21403 / 20.12% | 66 / 23.57% | True | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_distilled_shared_u1\hls_component\hls\syn\report\tcn_streaming_step_csynth.rpt` |

## CSim Checks For Resource-Fit Settings

| variant | setting | CSim result | readout RMSE | readout max abs diff | sign agreement joint | valid alignment errors | CSim evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| scratch | `shared_u1` | pass | 0.0854482445833 | 0.342693378125 | 1.0 | 0 | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_resource_scratch_shared_u1_csim_summary.csv` |
| distilled | `shared_u1` | pass | 0.305065372699 | 0.571899459375 | 1.0 | 0 | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_resource_distilled_shared_u1_csim_summary.csv` |

The distilled `shared_u1` Vitis CSim log also records `CSim done with 0 errors` in `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\hls_component_stage2_resource_distilled_shared_u1\hls_component\hls\csim\report\tcn_streaming_step_csim.log`.

## Mechanism

Lowering `TCN_STAGE2_CHANNEL_UNROLL` from 4 to 2 and 1 removed the DSP overuse, but it did not solve area. With complete-partitioned history storage, the late dilated blocks still materialize a large register/mux structure, so LUT and FF remain over budget even at unroll 1.

The first resource-fit setting is `shared_u1`. Binding the per-block history buffers to dual-port BRAM increases BRAM from 46 to 66 BRAM_18K, but cuts scratch LUT from 74487 to 27730 and FF from 124455 to 21143 relative to `u1`. Distilled shows the same mechanism: LUT drops from 74597 to 27840 and FF from 124715 to 21403. The cost is a small latency increase from 16340 to 16510 cycles.

## Latency Comparison

| artifact | latency | verification level | resource status | note |
| --- | ---: | --- | --- | --- |
| 07b naive full-window TCN | 13703.85 us | HLS csynth estimate | fits in 07b report | Recomputes the full window per event. |
| 07a TCN analytical estimate | 32.656 us | pre-synthesis MAC estimate, not HLS-verified | resource unknown | Must not be cited as measured FPGA latency. |
| PHC-lite v8 | 83.16 us | routed P&R at 100MHz | routed baseline | Stronger evidence level than any 07C csynth number here. |
| 07C Stage 2 unroll-4 scratch | 5.58 us | HLS csynth estimate only | does not fit | Fast but unusable on Zynq-7020 resource budget. |
| 07C Stage 2 `shared_u1` scratch | 165.10 us | CSim pass + HLS csynth estimate only | fits csynth budget | 83.00x faster than 07b naive, 1.99x slower than PHC-lite, 5.06x slower than the 07a analytical estimate. |
| 07C Stage 2 `shared_u1` distilled | 165.10 us | CSim pass + HLS csynth estimate only | fits csynth budget | Same latency schedule as scratch; resource counts differ slightly by weights. |

## Commands Used

The sweep directories were generated with:

```powershell
& 'C:\Users\holbo\AppData\Local\Programs\Python\Python313\python.exe' n2ew_era\07c_tcn_streaming_manual_hls\07_stage2_prepare_resource_sweep.py
```

Each csynth run used Vitis HLS Tcl execution from its component directory:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --tcl run_csynth_only.tcl
```

Resource-fit CSim checks used:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --config hls_config.cfg --work_dir hls_component --csim
```

The final summary was regenerated with:

```powershell
& 'C:\Users\holbo\AppData\Local\Programs\Python\Python313\python.exe' n2ew_era\07c_tcn_streaming_manual_hls\08_stage2_collect_resource_sweep.py
```

## Honest Conclusion

At the csynth resource-estimate level, the Stage 2 streaming TCN resource problem is resolved by `shared_u1` for both scratch and distilled variants. The mechanism is not additional arithmetic pruning; it is moving history storage out of complete-partitioned register fabric and into BRAM while keeping channel unroll at 1. This produces a resource-fit 16510-cycle, 165.10 us per-event design at 100MHz.

This is still not a routed timing result. It must be cited as `CSim pass + HLS csynth estimate only`, not as cosim-verified, P&R-timing-met, deployed, or board-validated latency.
