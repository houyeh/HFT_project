# 06H clang-tidy settings64fix Diagnostic

## Scope

This diagnostic only inspects the 06H clamped32 hls4ml/Vitis build artifacts. It does not modify 06F, 06G, or 05C4 artifacts and does not use NEW5.

## Current Run Examined

- Build log: `results/06h_build_full_log_clamped32_nocsim_settings64fix.txt`
- Flow log: `hls4ml_prj/phc_distilled_student_keras_clamped32/phc_distilled_student_keras_clamped32_prj/solution1/.autopilot/db/autopilot.flow.log`
- PATH diagnostic: `results/06h_subprocess_env_path_check.txt`

## settings64 Environment Propagation Check

`resolve_vitis_for_hls4ml()` was called and the resulting process environment was recorded immediately before the hls4ml build would inherit it.

Key result from `results/06h_subprocess_env_path_check.txt`:

| Check | Result |
|---|---:|
| `settings64_applied` | `true` |
| `settings64_env_keys_applied` | `59` |
| `XILINX_VITIS` | `C:\AMDDesignTools\2025.2\Vitis` |
| `XILINX_HLS` | `C:\AMDDesignTools\2025.2\Vitis` |
| `XILINX_VIVADO` | `C:\AMDDesignTools\2025.2\Vivado` |
| `RDI_PLATFORM` | `win64` |
| manual settings64 PATH segments | `48` |
| subprocess PATH segments | `49` |
| unique PATH segments missing from subprocess vs manual | `0` |
| unique extra PATH segments in subprocess vs manual | `0` |

The segment count differs by one because the subprocess PATH contains a duplicate `C:\AMDDesignTools\2025.2\Vitis\bin`; there are no unique missing or extra segments compared with the manually verified `call settings64.bat` environment.

Conclusion: the settings64 environment is being applied to the Python/hls4ml build process. This is not consistent with a simple "settings64 did not propagate at all" failure.

## Current `autopilot.flow.log` clang/clang-tidy Evidence

The current clamped32 settings64fix no-CSim build reaches `csynth_design`, preprocesses the design, and then fails at the loop-label clang-tidy stage.

Relevant flow-log signatures:

```text
line 572: INFO-FLOW: run_clang result=failed num-err-msgs=0 num-stderr-lines=0
```

This first failure is for `xilinx-performance-pragma-detector`. It is a silent auxiliary-tool failure but does not stop the run.

```text
line 589: INFO-FLOW: run_clang result=success num-err-msgs=0 num-stderr-lines=0
line 597: INFO-FLOW: run_clang result=success num-err-msgs=0 num-stderr-lines=0
```

The normal `clang` preprocessing/syntax checks succeed.

```text
line 600: Command         clang_tidy done; 1.901 sec.
line 604: Command         clang_tidy done; 2.437 sec.
```

Two earlier clang-tidy preprocessing passes complete successfully.

```text
line 609: INFO-FLOW: run_clang loop-label exec: C:/AMDDesignTools/2025.2/Vitis/win64/tools/clang-16/bin/clang-tidy ...
line 610: INFO-FLOW: run_clang result=failed num-err-msgs=0 num-stderr-lines=0
line 611: INFO-FLOW: Caught error in clang_tidy: clang-tidy failed:
line 615: INFO-FLOW: Caught error in elaborate: Syntax check failed in clang-tidy
line 641: INFO-FLOW: Caught error in csynth_design: Syntax check failed in clang-tidy
```

The failing stage is specifically the loop-label clang-tidy invocation. It fails with zero error messages and zero stderr lines.

## loop-label Output Files

The following expected loop-label output files are missing in the current `.autopilot/db` directory:

```text
phc_distilled_student_keras_clamped32.pp.0.cpp.clang-tidy.loop-label.out.log
phc_distilled_student_keras_clamped32.pp.0.cpp.clang-tidy.loop-label.err.log
phc_distilled_student_keras_clamped32.pp.0.cpp.clang-tidy.loop-label.diag.yml
```

Because the files do not exist, there is no loop-label stdout/stderr/diagnostic payload to inspect.

The ordinary clang diagnostic files do exist. They contain warnings only:

- `firmware/nnet_utils/nnet_activation.h:402`: unexpected pragma argument `softmax`
- `firmware/nnet_utils/nnet_dense_resource.h`: deprecated `Resource pragma`

No C++ syntax error is reported by the normal clang syntax path.

## Comparison With Prior No-settings no-CSim Attempt

The previous no-settings no-CSim `.autopilot/db/autopilot.flow.log` was not preserved as a separate artifact because the clamped32 project directory is rebuilt in place. Therefore, a line-by-line `autopilot.flow.log` comparison to the previous no-settings run is not possible from the current filesystem state.

The preserved top-level logs do show the same terminal symptom before and after settings64fix:

- `results/06h_build_full_log_clamped_nocsim.txt`
- `results/06h_build_full_log_clamped32_nocsim_settings64fix.txt`

Both terminate with:

```text
Syntax check failed in clang-tidy
Exception: Build failed for phc_distilled_student_keras_clamped32. See logs for details.
```

The current `autopilot.flow.log` gives the more detailed signature: loop-label clang-tidy fails with `num-err-msgs=0` and `num-stderr-lines=0`.

## Control Comparison: Unclamped Project

The existing unclamped project has a preserved `autopilot.flow.log`:

`hls4ml_prj/phc_distilled_student_keras/phc_distilled_student_keras_prj/solution1/.autopilot/db/autopilot.flow.log`

Its loop-label clang-tidy stage succeeds:

```text
line 623: Execute clang_tidy -errorcheck -desc loop-label ...
line 627: INFO-FLOW: run_clang result=success num-err-msgs=0 num-stderr-lines=544
```

This confirms that the Vitis flow can run the loop-label clang-tidy stage in this workspace/toolchain for a nearby hls4ml project.

## Diagnosis

The settings64 environment is being passed into the Python/hls4ml process: the recorded PATH has no unique missing segments relative to the manually verified settings64 shell, and the expected Vitis environment variables are present.

The current failure is also not behaving like a normal generated-C++ syntax error. Ordinary clang syntax checking succeeds, earlier clang-tidy preprocessing passes succeed, and the failing loop-label clang-tidy invocation produces no stderr, no error messages, and no diagnostic file.

Best current classification:

```text
settings64 environment propagation: PASS
ordinary C++ syntax error evidence: NOT FOUND
remaining failure mode: loop-label clang-tidy silent failure specific to the clamped32 generated project
```

This does not prove the exact lower-level root cause. The evidence points away from "settings64 not passed" and away from a normal syntax error, and toward an internal clang-tidy/tool invocation failure at the loop-label checker stage for the clamped32 project.

