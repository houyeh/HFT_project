# hls4ml Upgrade Diagnostic Report

Status: `resolved_negative`.

Conclusion in one line: upgrading from hls4ml `1.3.0` to GitHub main `1.4.0.dev32+gb90fb067` removed the old visible `Syntax check failed in clang-tidy` termination string, but did **not** produce a csynth report. The upgraded toolchain now fails earlier in clang-16 preprocessing / performance-pragma detection with the same zero-diagnostics signature (`num-err-msgs=0`, `num-stderr-lines=0`).

## Scope

- This is an environment/tool-version diagnostic for 06H only.
- It reuses `models/phc_distilled_student.keras`; it does not retrain or re-distill the student.
- It does not modify `06f_phc_lite_manual_hls`, `06g_phc_hls4ml_a3d3_retry`, `05c4_phc_stp_cosine_lr`, or NEW5.
- The tested model is a distilled Keras student, not PHC-lite itself.

## Version Evidence

- PyPI version check: `pip index versions hls4ml` reported latest stable `1.3.0`; the PyPI project page also lists `1.3.0` as the current release. Therefore no newer PyPI stable release exists above the project baseline.
- Upgrade-test version: GitHub main installed as `hls4ml 1.4.0.dev32+gb90fb067` from commit `b90fb06736baa0908a8995fc1cf4ac4a7d1c241f`.
- GitHub release notes for `1.3.0` include Vitis-related updates, including "Add compatibility with new Vitis command-line compilation flow", but the search did not reveal a directly named `Vitis 2025.2 clang-tidy nnet_utils` fix. This test is therefore an empirical main-branch compatibility probe, not a test of a known fixed release.
- The requested 06H-local venv path hit Windows long-path installation failure while installing TensorFlow. The successful isolated venv is the shorter repo-root `.h6`; it still does not touch the main project `.venv`.
- The first in-sandbox run reached `vitis-run` resolution failure/permission behavior and was preserved as `results/06h_build_full_log_clamped32_plain_hls4ml_main_nocsim_vitisrun_not_found.txt` and `results/06h_build_full_log_clamped32_plain_hls4ml_main_nocsim_sandbox_permission_denied.txt`. The effective result below is from the unsandboxed diagnostic run after preserving those preflight logs.

## Environment

- Python executable: `C:\Users\holbo\00VS_projjects\00_HFT_project\.h6\Scripts\python.exe`
- Python version: `3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:09:58) [MSC v.1929 64 bit (AMD64)]`
- TensorFlow: `2.21.0`
- Keras: `3.15.1`
- hls4ml: `1.4.0.dev32+gb90fb067`
- hls4ml path: `C:\Users\holbo\00VS_projjects\00_HFT_project\.h6\Lib\site-packages\hls4ml\__init__.py`
- Vitis: `C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.BAT`
- Vitis settings: `C:\AMDDesignTools\2025.2\Vitis\settings64.bat`, applied before build.

## Build Result

| method | convertible | build_run | build_success | csim_skipped | csynth_report_count | old clang-tidy syntax line present? | new terminal signature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `keras_distilled_student_clamped32_plain_hls4ml_main_nocsim` | True | True | False | True | 0 | No | `run_clang result=failed num-err-msgs=0 num-stderr-lines=0` in clang-16 performance detector and preprocessing compiler |

The full build log is `results/06h_build_full_log_clamped32_plain_hls4ml_main_nocsim.txt`.

The full flow log is `hls4ml_prj/phc_distilled_student_keras_clamped32_plain_hls4ml_main/phc_distilled_student_keras_clamped32_plain_hls4ml_main_prj/solution1/.autopilot/db/autopilot.flow.log`.

## 06F Reference Endpoint Comparison

| implementation                                             | model_identity                                   | verification                        | latency_cycles | latency_us     | DSP | LUT   | FF    | BRAM_18K |
| ---------------------------------------------------------- | ------------------------------------------------ | ----------------------------------- | -------------- | -------------- | --- | ----- | ----- | -------- |
| 06F manual HLS PHC-STP                                     | production PHC-lite/STP seed2                    | csim+cosim+routed P&R in 06F        | 8316           | 83.16 @ 100MHz | 46  | 24836 | 14518 | 6        |
| keras_distilled_student_clamped32_plain_hls4ml_main_nocsim | 06H distilled Keras student, not PHC-lite itself | upgrade-test no-CSim csynth attempt | /              |                |     |       |       |          |

## Distillation Quality Boundary

Existing 06H quality metrics are reproduced below only to prevent confusion with 06F bit-exact PHC HLS. The key val split number is `joint_sign_agreement=0.9503707885742188`.

```csv
split,row_count,rmse_all,rmse_readout,rmse_state,max_abs_diff,sign_agreement_readout_0,sign_agreement_readout_1,joint_sign_agreement,new5_used
val,131072,0.09756776690483093,0.04120087996125221,0.10245590656995773,2.626870632171631,0.97418212890625,0.9761581420898438,0.9503707885742188,False
```

## Build Termination Excerpt

```text
[06H upgrade build] post_patch_vitis_run=C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.BAT
****** vitis-run v2025.2 (64-bit)
INFO: [HLS 200-1510] Running: config_array_partition -maximum_size 4096 
ERROR: [HLS 200-101] config_array_partition: Unknown option '-maximum_size'.
ERROR: [HLS 200-101] config_array_partition: Unknown option '4096'.
***** C/RTL SYNTHESIS *****
INFO: [HLS 200-1510] Running: csynth_design 
INFO: [HLS 200-2191] C-Synthesis will use clang-16 as the compiler
INFO-FLOW: run_clang phc_distilled_student_keras_clamped32_plain_hls4ml_main.cpp.xilinx-performance-pragma-detector exec: C:/AMDDesignTools/2025.2/Vitis/win64/tools/clang-16/bin/xilinx-performance-pragma-detector ...
INFO-FLOW: run_clang result=failed num-err-msgs=0 num-stderr-lines=0
INFO-FLOW: run_clang phc_distilled_student_keras_clamped32_plain_hls4ml_main.cpp.clang exec: C:/AMDDesignTools/2025.2/Vitis/win64/tools/clang-16/bin/clang ...
INFO-FLOW: run_clang result=failed num-err-msgs=0 num-stderr-lines=0
INFO: [HLS 200-2161] Finished Command csynth_design Elapsed time: 00:00:00; Allocated memory: 0.605 MB.
compilation failed: 
INFO: [vitis-run 60-1662] Stopping dispatch session having empty uuid.
[06H upgrade build] exception propagated to Python
Exception: Build failed for phc_distilled_student_keras_clamped32_plain_hls4ml_main. See logs for details.
    raise Exception(f'Build failed for {model.config.get_project_name()}. See logs for details.')
Exception: Build failed for phc_distilled_student_keras_clamped32_plain_hls4ml_main. See logs for details.
```

## Conclusion

升版 hls4ml 到 `1.4.0.dev32+gb90fb067` **沒有**完成驗收標準，因為 `csynth_report_count=0`。判斷依據是：舊版可見的 `Syntax check failed in clang-tidy` 字串不再出現，但 Vitis 2025.2 的 clang-16 子程序仍以 zero-diagnostics 方式提前失敗，這次停在 performance-pragma detector / preprocessing compiler，而不是完成 csynth。

This remains a toolchain compatibility problem, not a PHC-lite training or checkpoint problem. It also remains a distilled-student result only; it cannot be cited as PHC-lite/06F bit-exact hardware validation.
