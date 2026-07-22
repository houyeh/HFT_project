# 06H clamped32 Preprocessed Complexity Audit

## Scope

This audit compares the preprocessed Vitis HLS C++ sources for the unclamped and clamped32 hls4ml student projects. It does not modify the Vitis installation, does not modify 06F/06G/05C4, and does not use NEW5.

## Compared Files

| Project | Preprocessed file |
|---|---|
| unclamped | `hls4ml_prj/phc_distilled_student_keras/phc_distilled_student_keras_prj/solution1/.autopilot/db/phc_distilled_student_keras.pp.0.cpp` |
| clamped32 | `hls4ml_prj/phc_distilled_student_keras_clamped32/phc_distilled_student_keras_clamped32_prj/solution1/.autopilot/db/phc_distilled_student_keras_clamped32.pp.0.cpp` |

## Step 1 Metrics

| Project | File size bytes | Lines | `ap_fixed<` tokens | Longest `ap_fixed`/`ap_ufixed` token length |
|---|---:|---:|---:|---:|
| unclamped | 2,526,158 | 58,801 | 150 | 102 |
| clamped32 | 2,524,118 | 58,801 | 150 | 102 |

Longest token in both files:

```text
ap_fixed<2 * (decltype(w_T::weight)::width + x_T::width), (decltype(w_T::weight)::width + x_T::width)>
```

Additional clamp-related counts:

| Project | `ap_fixed<32,16` | `AP_RND_CONV` | `AP_SAT` | `ap_fixed<156,76` |
|---|---:|---:|---:|---:|
| unclamped | 0 | 2 | 9 | 3 |
| clamped32 | 34 | 36 | 43 | 0 |

## Interpretation

The clamped32 file does contain the expected explicit clamp precision markers: more `ap_fixed<32,16`, more `AP_RND_CONV`, and more `AP_SAT`. It also removes the previous `ap_fixed<156,76` occurrences.

However, the preprocessed source is not materially larger or deeper by the requested metrics:

- clamped32 is slightly smaller by 2,040 bytes.
- line count is identical.
- total `ap_fixed<` token count is identical.
- longest `ap_fixed`/`ap_ufixed` token length is identical.

Therefore the specific complexity hypothesis in Step 1, namely that clamped32 visibly inflates the preprocessed C++ size or type-token length enough to explain a clang-tidy stack/complexity failure, is not supported by these metrics.

## Step 2 Decision

Step 2 was not executed.

Reason:

```text
complexity_hypothesis_initial_check = not_supported
editbin_stack_experiment = skipped
```

Because the requested precondition for modifying `clang-tidy.exe` was not met, no Vitis installation file was backed up or edited.

