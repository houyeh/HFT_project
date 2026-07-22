# PHC hls4ml A3D3 Retry Report

Final status: `hls4ml_still_blocked_at_conversion`.

## Scope

- New isolated phase: `n2ew_era/06g_phc_hls4ml_a3d3_retry/`.
- Source weights: adopted Phase 05C4 seed=2 step checkpoint produced in Phase 06E.
- No training, no threshold selection, and no NEW5 usage.
- The frozen manual HLS project under `06f_phc_lite_manual_hls` was not modified.

## Pre-Conversion Equivalence

| model                               |   sample_count |   max_abs_diff |   mean_abs_diff | allclose_atol_1e_6   |
|:------------------------------------|---------------:|---------------:|----------------:|:---------------------|
| manual_layernorm_vectorized_mean    |             64 |    5.96046e-07 |     5.02861e-08 | True                 |
| manual_layernorm_unrolled_reduction |             64 |    4.76837e-07 |     6.38824e-08 | True                 |

The manual-LayerNorm wrapper must match the original step model before it is handed to hls4ml.

## Retry Status

| method                                | convertible   | compile_success   | build_run   | build_success   | first_error_line                                   | difference_vs_06e                                                                                                         |
|:--------------------------------------|:--------------|:------------------|:------------|:----------------|:---------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
| torch_fx_manual_layernorm             | False         | False             | False       | False           | Exception: Unsupported function mean               | 06E PyTorch FX failed on nn.LayerNorm shape [None,16]; 06G moved past nn.LayerNorm but stopped at unsupported torch.mean. |
| torch_fx_unrolled_layernorm           | False         | False             | False       | False           | KeyError: 'getitem'                                | Second FX attempt avoids torch.mean by explicit 16-way reduction; compare error line for next unsupported primitive.      |
| onnx_shape_inference_original_step    | False         | False             | False       | False           | Exception: ERROR: Unsupported operation type: Gemm | Initializer shape annotation moved past 06E's MatMul_72 shape blocker; hls4ml now stops at unsupported ONNX Gemm.         |
| onnx_shape_inference_manual_layernorm | False         | False             | False       | False           | Exception: ERROR: Unsupported operation type: Gemm | Initializer shape annotation moved past MatMul_74 shape lookup; hls4ml now stops at unsupported ONNX Gemm.                |

Full tracebacks are stored in `results/hls4ml_retry_errors.txt`; the raw wide status table is stored in `results/hls4ml_retry_status.csv`.

## Parsed hls4ml Layers

No hls4ml layer table was produced.

## Comparison Endpoint

The frozen hand-written 06F endpoint remains the reference if hls4ml cannot produce a csynth report:

| implementation   |   latency_cycles |   latency_us_at_100mhz |   interval |   DSP |   DSP_pct |   LUT | LUT_pct_note                                                                |    FF |   FF_pct |   BRAM_18K |   BRAM_pct | source                                                                                                            |
|:-----------------|-----------------:|-----------------------:|-----------:|------:|----------:|------:|:----------------------------------------------------------------------------|------:|---------:|-----------:|-----------:|:------------------------------------------------------------------------------------------------------------------|
| 06f_manual_hls   |             8316 |                  83.16 |       8317 |    46 |      20.9 | 24836 | 06f routed report gives 14.20%; csynth utilization varies by report context | 14518 |     13.6 |          6 |        2.1 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\PHC_HLS_LATENCY_BREAKDOWN_REPORT.md |

## Interpretation

Both bounded retry routes remain blocked before a usable hls4ml synthesis report is produced.
This is still useful evidence: the error messages below show whether the failure moved past the original 06E LayerNorm and ONNX MatMul shape blockers.

## Protocol Boundaries

- This phase does not claim PYNQ-Z2 deployment.
- This phase does not claim FPGA PnL improvement.
- This phase does not prove real-world profitability or true exchange replay.
- If hls4ml succeeds, the result is a hardware-conversion artifact only; strategy evidence remains Phase 05C4.
