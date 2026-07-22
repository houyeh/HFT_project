"""
匯出 TCN 模型的 INT8 權重、reference input/output，以及 HLS 用的固定 header。

輸出：
  1. tcn_int8_weights.wts                 - C++ 動態版可讀的 INT8 權重檔
  2. tensor_manifest.json                 - 權重清單與統計
  3. reference_input.txt                  - 一筆 (F*T) 的 int8 量化輸入
  4. reference_output.txt                 - 原始 FP32 模型輸出
  5. fixed_tcn_weights.hpp                - HLS-friendly 固定權重 header
  6. fixed_tcn_test_vectors.hpp           - HLS testbench 固定輸入與 golden output
"""

from __future__ import annotations

# Module note: this is the bridge between the trained PyTorch model and the
# fixed-shape C++/HLS flow. It quantizes weights, emits portable artifacts, and
# computes a golden reference using the same integer math as the hardware path.

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.tcn import build_default_model


FIXED_NUM_INPUTS = 12
FIXED_NUM_CLASSES = 3
FIXED_CHANNELS = 16
FIXED_KERNEL_SIZE = 3
FIXED_SEQ_LEN = 100
FIXED_ACT_SCALE = 1.0 / 64.0

# 把 float tensor 轉成 int8 + scale（對稱量化）
# 量化後一定有誤差（quantization error），此誤差由 scale、rounding、clamp 共同決定。
def quantize_tensor_symmetric(tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(array)))
    scale = max(max_abs / 127.0, 1e-8)
    quantized = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
    return quantized, scale


def clamp_int8_array(array: np.ndarray) -> np.ndarray:
    return np.clip(array, -127, 127).astype(np.int8)


def bias_to_accumulator(
    bias_q: np.ndarray,
    bias_scale: float,
    input_scale: float,
    weight_scale: float,
) -> np.ndarray:
    # 將 bias 轉成 accumulator 尺度，確保 int32 乘加時和權重/輸入同尺度
    ratio = bias_scale / (input_scale * weight_scale)
    return np.round(bias_q.astype(np.float32) * ratio).astype(np.int32)


def requantize_to_fixed_scale(acc: np.ndarray, acc_scale: float) -> np.ndarray:
    # 把 int32 accumulator 重新縮放回固定 activation scale，再做 int8 飽和截斷
    ratio = acc_scale / FIXED_ACT_SCALE
    return clamp_int8_array(np.round(acc.astype(np.float32) * ratio))

# 固定整數推論核心
def causal_conv_accumulate(
    input_q: np.ndarray,
    weights_q: np.ndarray,
    bias_acc: np.ndarray,
    dilation: int,
) -> np.ndarray:
    in_channels, seq_len = input_q.shape
    out_channels, _, kernel_size = weights_q.shape
    output = np.zeros((out_channels, seq_len), dtype=np.int32)
    for out_ch in range(out_channels):
        for t in range(seq_len):
            acc = int(bias_acc[out_ch])
            for in_ch in range(in_channels):
                for k in range(kernel_size):
                    src_t = t - ((kernel_size - 1 - k) * dilation)
                    if src_t < 0:
                        continue
                    acc += int(input_q[in_ch, src_t]) * int(weights_q[out_ch, in_ch, k])
            output[out_ch, t] = acc
    return output


def pointwise_conv_accumulate(
    input_q: np.ndarray,
    weights_q: np.ndarray,
    bias_acc: np.ndarray,
) -> np.ndarray:
    in_channels, seq_len = input_q.shape
    out_channels, _ = weights_q.shape
    output = np.zeros((out_channels, seq_len), dtype=np.int32)
    for out_ch in range(out_channels):
        for t in range(seq_len):
            acc = int(bias_acc[out_ch])
            for in_ch in range(in_channels):
                acc += int(input_q[in_ch, t]) * int(weights_q[out_ch, in_ch])
            output[out_ch, t] = acc
    return output


def residual_block(
    input_q: np.ndarray,
    conv1_w: np.ndarray,
    conv1_w_scale: float,
    conv1_b: np.ndarray,
    conv1_b_scale: float,
    conv2_w: np.ndarray,
    conv2_w_scale: float,
    conv2_b: np.ndarray,
    conv2_b_scale: float,
    dilation: int,
    downsample_w: np.ndarray | None = None,
    downsample_w_scale: float | None = None,
    downsample_b: np.ndarray | None = None,
    downsample_b_scale: float | None = None,
) -> np.ndarray:
    conv1_bias_acc = bias_to_accumulator(conv1_b, conv1_b_scale, FIXED_ACT_SCALE, conv1_w_scale)
    conv1_acc = causal_conv_accumulate(input_q, conv1_w, conv1_bias_acc, dilation=dilation)
    conv1_acc = np.maximum(conv1_acc, 0)
    q1 = requantize_to_fixed_scale(conv1_acc, FIXED_ACT_SCALE * conv1_w_scale)

    conv2_bias_acc = bias_to_accumulator(conv2_b, conv2_b_scale, FIXED_ACT_SCALE, conv2_w_scale)
    conv2_acc = causal_conv_accumulate(q1, conv2_w, conv2_bias_acc, dilation=dilation)
    conv2_acc = np.maximum(conv2_acc, 0)
    q2 = requantize_to_fixed_scale(conv2_acc, FIXED_ACT_SCALE * conv2_w_scale)

    if downsample_w is not None and downsample_b is not None and downsample_w_scale is not None and downsample_b_scale is not None:
        down_bias_acc = bias_to_accumulator(downsample_b, downsample_b_scale, FIXED_ACT_SCALE, downsample_w_scale)
        down_acc = pointwise_conv_accumulate(input_q, downsample_w, down_bias_acc)
        residual_q = requantize_to_fixed_scale(down_acc, FIXED_ACT_SCALE * downsample_w_scale)
    else:
        residual_q = input_q.astype(np.int8, copy=True)

    merged = q2.astype(np.int16) + residual_q.astype(np.int16)
    merged = np.maximum(merged, 0)
    return clamp_int8_array(merged)


def global_average_pool(hidden_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = hidden_q.astype(np.int32).sum(axis=1) // hidden_q.shape[1]
    pooled_q = clamp_int8_array(pooled)
    return pooled.astype(np.int32), pooled_q


def fully_connected(
    pooled_q: np.ndarray,
    fc_w: np.ndarray,
    fc_b: np.ndarray,
    fc_w_scale: float,
    fc_b_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc_bias_acc = bias_to_accumulator(fc_b, fc_b_scale, FIXED_ACT_SCALE, fc_w_scale)
    logits_int32 = fc_bias_acc + (fc_w.astype(np.int32) @ pooled_q.astype(np.int32))
    logits_float = logits_int32.astype(np.float32) * (FIXED_ACT_SCALE * fc_w_scale)
    logits_float = logits_float.astype(np.float32)
    shifted = logits_float - np.max(logits_float)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    return logits_int32.astype(np.int32), logits_float, probabilities.astype(np.float32)


def run_fixed_reference_inference(quantized_tensors: dict[str, dict], input_q_flat: np.ndarray) -> dict:
    input_q = input_q_flat.reshape(FIXED_NUM_INPUTS, FIXED_SEQ_LEN)

    block0 = residual_block(
        input_q=input_q,
        conv1_w=quantized_tensors["tcn.network.0.conv1.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_NUM_INPUTS, FIXED_KERNEL_SIZE),
        conv1_w_scale=quantized_tensors["tcn.network.0.conv1.conv.weight"]["scale"],
        conv1_b=quantized_tensors["tcn.network.0.conv1.conv.bias"]["array"],
        conv1_b_scale=quantized_tensors["tcn.network.0.conv1.conv.bias"]["scale"],
        conv2_w=quantized_tensors["tcn.network.0.conv2.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_CHANNELS, FIXED_KERNEL_SIZE),
        conv2_w_scale=quantized_tensors["tcn.network.0.conv2.conv.weight"]["scale"],
        conv2_b=quantized_tensors["tcn.network.0.conv2.conv.bias"]["array"],
        conv2_b_scale=quantized_tensors["tcn.network.0.conv2.conv.bias"]["scale"],
        dilation=1,
        downsample_w=quantized_tensors["tcn.network.0.downsample.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_NUM_INPUTS),
        downsample_w_scale=quantized_tensors["tcn.network.0.downsample.weight"]["scale"],
        downsample_b=quantized_tensors["tcn.network.0.downsample.bias"]["array"],
        downsample_b_scale=quantized_tensors["tcn.network.0.downsample.bias"]["scale"],
    )

    block1 = residual_block(
        input_q=block0,
        conv1_w=quantized_tensors["tcn.network.1.conv1.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_CHANNELS, FIXED_KERNEL_SIZE),
        conv1_w_scale=quantized_tensors["tcn.network.1.conv1.conv.weight"]["scale"],
        conv1_b=quantized_tensors["tcn.network.1.conv1.conv.bias"]["array"],
        conv1_b_scale=quantized_tensors["tcn.network.1.conv1.conv.bias"]["scale"],
        conv2_w=quantized_tensors["tcn.network.1.conv2.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_CHANNELS, FIXED_KERNEL_SIZE),
        conv2_w_scale=quantized_tensors["tcn.network.1.conv2.conv.weight"]["scale"],
        conv2_b=quantized_tensors["tcn.network.1.conv2.conv.bias"]["array"],
        conv2_b_scale=quantized_tensors["tcn.network.1.conv2.conv.bias"]["scale"],
        dilation=2,
    )

    block2 = residual_block(
        input_q=block1,
        conv1_w=quantized_tensors["tcn.network.2.conv1.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_CHANNELS, FIXED_KERNEL_SIZE),
        conv1_w_scale=quantized_tensors["tcn.network.2.conv1.conv.weight"]["scale"],
        conv1_b=quantized_tensors["tcn.network.2.conv1.conv.bias"]["array"],
        conv1_b_scale=quantized_tensors["tcn.network.2.conv1.conv.bias"]["scale"],
        conv2_w=quantized_tensors["tcn.network.2.conv2.conv.weight"]["array"].reshape(FIXED_CHANNELS, FIXED_CHANNELS, FIXED_KERNEL_SIZE),
        conv2_w_scale=quantized_tensors["tcn.network.2.conv2.conv.weight"]["scale"],
        conv2_b=quantized_tensors["tcn.network.2.conv2.conv.bias"]["array"],
        conv2_b_scale=quantized_tensors["tcn.network.2.conv2.conv.bias"]["scale"],
        dilation=4,
    )

    pooled_int32, pooled_q = global_average_pool(block2)
    logits_int32, logits_float, probabilities = fully_connected(
        pooled_q=pooled_q,
        fc_w=quantized_tensors["fc.weight"]["array"].reshape(FIXED_NUM_CLASSES, FIXED_CHANNELS),
        fc_b=quantized_tensors["fc.bias"]["array"],
        fc_w_scale=quantized_tensors["fc.weight"]["scale"],
        fc_b_scale=quantized_tensors["fc.bias"]["scale"],
    )

    return {
        "pooled_int32": pooled_int32,
        "pooled_q": pooled_q,
        "logits_int32": logits_int32,
        "logits_float": logits_float,
        "probabilities": probabilities,
    }


def format_cpp_scalar(value, kind: str) -> str:
    if kind == "int8":
        return str(int(value))
    if kind == "int32":
        return str(int(value))
    if kind == "float":
        return f"{float(value):.8f}f"
    raise ValueError(f"Unsupported scalar kind: {kind}")


def format_cpp_array(array: np.ndarray | list, kind: str, per_line: int = 16) -> str:
    flat = np.asarray(array).reshape(-1).tolist()
    chunks = []
    for start in range(0, len(flat), per_line):
        chunk = ", ".join(format_cpp_scalar(value, kind) for value in flat[start:start + per_line])
        chunks.append(f"    {chunk}")
    return "{\n" + ",\n".join(chunks) + "\n}"


def write_fixed_weights_header(output_path: Path, quantized_tensors: dict[str, dict]) -> None:
    lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        '#include "fixed_tcn.hpp"',
        "",
        "namespace tcn_hls::generated {",
        "",
        "inline const FixedTCNWeights kFixedWeights = [] {",
        "    FixedTCNWeights weights{};",
    ]

    mapping = [
        ("block0_conv1_w", "tcn.network.0.conv1.conv.weight", "int8"),
        ("block0_conv1_b", "tcn.network.0.conv1.conv.bias", "int8"),
        ("block0_conv2_w", "tcn.network.0.conv2.conv.weight", "int8"),
        ("block0_conv2_b", "tcn.network.0.conv2.conv.bias", "int8"),
        ("block0_downsample_w", "tcn.network.0.downsample.weight", "int8"),
        ("block0_downsample_b", "tcn.network.0.downsample.bias", "int8"),
        ("block1_conv1_w", "tcn.network.1.conv1.conv.weight", "int8"),
        ("block1_conv1_b", "tcn.network.1.conv1.conv.bias", "int8"),
        ("block1_conv2_w", "tcn.network.1.conv2.conv.weight", "int8"),
        ("block1_conv2_b", "tcn.network.1.conv2.conv.bias", "int8"),
        ("block2_conv1_w", "tcn.network.2.conv1.conv.weight", "int8"),
        ("block2_conv1_b", "tcn.network.2.conv1.conv.bias", "int8"),
        ("block2_conv2_w", "tcn.network.2.conv2.conv.weight", "int8"),
        ("block2_conv2_b", "tcn.network.2.conv2.conv.bias", "int8"),
        ("fc_w", "fc.weight", "int8"),
        ("fc_b", "fc.bias", "int8"),
    ]
    scale_mapping = [
        ("block0_conv1_w_scale", "tcn.network.0.conv1.conv.weight"),
        ("block0_conv1_b_scale", "tcn.network.0.conv1.conv.bias"),
        ("block0_conv2_w_scale", "tcn.network.0.conv2.conv.weight"),
        ("block0_conv2_b_scale", "tcn.network.0.conv2.conv.bias"),
        ("block0_downsample_w_scale", "tcn.network.0.downsample.weight"),
        ("block0_downsample_b_scale", "tcn.network.0.downsample.bias"),
        ("block1_conv1_w_scale", "tcn.network.1.conv1.conv.weight"),
        ("block1_conv1_b_scale", "tcn.network.1.conv1.conv.bias"),
        ("block1_conv2_w_scale", "tcn.network.1.conv2.conv.weight"),
        ("block1_conv2_b_scale", "tcn.network.1.conv2.conv.bias"),
        ("block2_conv1_w_scale", "tcn.network.2.conv1.conv.weight"),
        ("block2_conv1_b_scale", "tcn.network.2.conv1.conv.bias"),
        ("block2_conv2_w_scale", "tcn.network.2.conv2.conv.weight"),
        ("block2_conv2_b_scale", "tcn.network.2.conv2.conv.bias"),
        ("fc_w_scale", "fc.weight"),
        ("fc_b_scale", "fc.bias"),
    ]

    for field_name, tensor_name, kind in mapping:
        tensor_array = quantized_tensors[tensor_name]["array"]
        lines.append(f"    weights.{field_name} = {format_cpp_array(tensor_array, kind)};")

    for field_name, tensor_name in scale_mapping:
        lines.append(f"    weights.{field_name} = {format_cpp_scalar(quantized_tensors[tensor_name]['scale'], 'float')};")

    lines.extend([
        "    return weights;",
        "}();",
        "",
        "}  // namespace tcn_hls::generated",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixed_test_vectors_header(output_path: Path, input_q: np.ndarray, fixed_reference: dict) -> None:
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        '#include "fixed_tcn.hpp"',
        "",
        "namespace tcn_hls::generated {",
        "",
        f"inline const std::array<int8_t, kFixedNumInputs * kFixedSeqLen> kReferenceInput = {format_cpp_array(input_q, 'int8')};",
        f"inline const std::array<int32_t, kFixedNumClasses> kGoldenLogitsInt32 = {format_cpp_array(fixed_reference['logits_int32'], 'int32', per_line=8)};",
        f"inline const std::array<float, kFixedNumClasses> kGoldenLogitsFloat = {format_cpp_array(fixed_reference['logits_float'], 'float', per_line=8)};",
        f"inline const std::array<float, kFixedNumClasses> kGoldenProbabilities = {format_cpp_array(fixed_reference['probabilities'], 'float', per_line=8)};",
        "inline constexpr float kProbabilityTolerance = 1e-6f;",
        "",
        "}  // namespace tcn_hls::generated",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("artifacts/04_hardware_handoff/int8_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = Path("cpp_hls_prep/include/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)

    model = build_default_model(num_inputs=12)
    state_dict = torch.load("artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Quantize every parameter symmetrically and emit a `.wts` file that the
    # C++ reference path can load without any PyTorch dependency.
    manifest = []
    quantized_tensors: dict[str, dict] = {}
    wts_path = output_dir / "tcn_int8_weights.wts"
    with wts_path.open("w", encoding="utf-8") as handle:
        named_params = list(model.state_dict().items())
        handle.write(f"{len(named_params)}\n")
        for name, tensor in named_params:
            quantized, scale = quantize_tensor_symmetric(tensor)
            flat = quantized.reshape(-1)
            shape = list(quantized.shape)
            manifest.append(
                {
                    "name": name,
                    "shape": shape,
                    "count": int(flat.size),
                    "scale": scale,
                    "dtype": "int8",
                }
            )
            quantized_tensors[name] = {
                "array": flat.copy(),
                "shape": shape,
                "scale": scale,
            }
            values = " ".join(str(int(value)) for value in flat.tolist())
            handle.write(f"{name} {flat.size} {scale:.12g} {'x'.join(str(dim) for dim in shape)} {values}\n")

    torch.manual_seed(99)
    sample_input_btf = torch.randn(1, 100, 12)
    sample_input_cf = sample_input_btf.squeeze(0).permute(1, 0)

    # Emit one deterministic reference input/output pair so the dynamic C++ path,
    # fixed-shape core, and HLS testbench all compare against the same sample.
    input_np = sample_input_cf.numpy().reshape(-1)
    input_max_abs = float(np.max(np.abs(input_np)))
    input_scale = max(input_max_abs / 127.0, 1e-8)
    input_q = np.clip(np.round(input_np / input_scale), -127, 127).astype(np.int8)

    ref_input_path = output_dir / "reference_input.txt"
    with ref_input_path.open("w", encoding="utf-8") as f:
        for val in input_q:
            f.write(f"{int(val)}\n")

    with torch.no_grad():
        logits = model(sample_input_btf)
        proba = torch.softmax(logits, dim=1)

    ref_output_path = output_dir / "reference_output.txt"
    with ref_output_path.open("w", encoding="utf-8") as f:
        f.write("# FP32 logits\n")
        for val in logits[0].tolist():
            f.write(f"{val:.8f}\n")
        f.write("# FP32 probabilities (softmax)\n")
        for val in proba[0].tolist():
            f.write(f"{val:.8f}\n")

    fixed_reference = run_fixed_reference_inference(quantized_tensors, input_q)

    # Generate headers consumed by the HLS testbench and the Vitis HLS top
    # wrapper so synthesis can run without parsing external text files.
    weights_header_path = generated_dir / "fixed_tcn_weights.hpp"
    test_vectors_header_path = generated_dir / "fixed_tcn_test_vectors.hpp"
    write_fixed_weights_header(weights_header_path, quantized_tensors)
    write_fixed_test_vectors_header(test_vectors_header_path, input_q, fixed_reference)

    summary = {
        "source_model": str(Path("artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt").resolve()),
        "weight_file": str(wts_path.resolve()),
        "tensor_count": len(manifest),
        "total_parameters": int(sum(item["count"] for item in manifest)),
        "reference_input": str(ref_input_path.resolve()),
        "reference_output": str(ref_output_path.resolve()),
        "generated_weights_header": str(weights_header_path.resolve()),
        "generated_test_vectors_header": str(test_vectors_header_path.resolve()),
        "input_scale": float(input_scale),
        "fp32_logits": logits[0].tolist(),
        "fp32_probabilities": proba[0].tolist(),
        "fixed_logits_int32": fixed_reference["logits_int32"].tolist(),
        "fixed_logits_float": fixed_reference["logits_float"].tolist(),
        "fixed_probabilities": fixed_reference["probabilities"].tolist(),
    }
    with (output_dir / "tensor_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "tensors": manifest}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
