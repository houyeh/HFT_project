// 動態 INT8 推論輔助函式：在模型被凍結成固定形狀 HLS 導向表示之前，
// 使用這些函式在 CPU 上進行推論。

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tcn_hls {

// ---------------------------------------------------------------------------
//  Data structures
// ---------------------------------------------------------------------------

struct TensorBlob {
    std::string name;
    float scale = 1.0f;
    std::vector<int> shape;
    std::vector<int8_t> values;
};

// Model architecture constants (must match Python TCNClassifier defaults)
struct TCNConfig {
    int num_inputs    = 12;
    int num_classes   = 3;
    int num_blocks    = 3;     // [16, 16, 16]
    int channels      = 16;
    int kernel_size   = 3;
    int seq_len       = 100;

    // derived
    int dilation(int block) const { return 1 << block; }
    int causal_pad(int block) const { return (kernel_size - 1) * dilation(block); }
};

// ---------------------------------------------------------------------------
//  Weight file I/O
// ---------------------------------------------------------------------------

std::unordered_map<std::string, TensorBlob> load_wts_file(const std::string& path);

// ---------------------------------------------------------------------------
//  Low-level kernels
// ---------------------------------------------------------------------------

// Conv1d with 'valid' mode (no padding)
// input   layout: [in_channels  × input_length]
// weights layout: [out_channels × in_channels × kernel_size]
// output  layout: [out_channels × output_length]
std::vector<int32_t> conv1d_valid_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int dilation);

// Conv1d with left (causal) zero-padding
// Prepends `pad_len = (kernel_size - 1) * dilation` zeros on the time axis,
// then calls conv1d_valid so that output_length == input_length.
std::vector<int32_t> conv1d_causal_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int dilation);

// GEMM: output[r][c] = sum_k (input[r][k] * weights[c][k])
// input  layout: [rows × inner_dim]
// weights layout: [cols × inner_dim]
// output layout: [rows × cols]
std::vector<int32_t> gemm_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int rows,
    int cols,
    int inner_dim);

// ---------------------------------------------------------------------------
//  Element-wise ops (all in-place on int32 accumulators)
// ---------------------------------------------------------------------------

// ReLU: clamp all values to >= 0
void relu_inplace_int32(std::vector<int32_t>& data);

// Add bias per channel: data[c * length + t] += bias[c]
void add_bias_int32(
    std::vector<int32_t>& data,
    const std::vector<int32_t>& bias,
    int channels,
    int length);

// Element-wise addition: out[i] += residual[i]
void residual_add_int32(
    std::vector<int32_t>& out,
    const std::vector<int32_t>& residual);

// ---------------------------------------------------------------------------
//  Requantization — compress int32 accumulator back to int8
// ---------------------------------------------------------------------------

// out_scale = (in_scale * weight_scale) / target_scale
// q_out = clamp(round(acc * out_scale), -127, 127)
std::vector<int8_t> requantize_int32_to_int8(
    const std::vector<int32_t>& acc,
    float acc_scale,
    float target_scale);

// ---------------------------------------------------------------------------
//  Pooling
// ---------------------------------------------------------------------------

// Global Average Pooling along time axis: (C, T) → (C,)
// Returns int32 sums (caller divides by length for the true average).
std::vector<int32_t> global_avg_pool_int32(
    const std::vector<int32_t>& data,
    int channels,
    int length);

// ---------------------------------------------------------------------------
//  Full TCN inference
// ---------------------------------------------------------------------------

// Runs the complete inference chain:
//   causal_conv → ReLU → causal_conv → ReLU → residual add  (×3 blocks)
//   → global average pool → FC → 3-class logits
//
// input:   (num_inputs × seq_len) in int8 (channel-first, already quantized)
// output:  (num_classes,) in int32 (raw logits, caller applies scale + softmax)
//
// Returns a struct with both raw int32 logits and float probabilities.
struct InferenceResult {
    std::vector<int32_t> logits_int32;   // [num_classes]
    std::vector<float>   logits_float;   // [num_classes] after scale recovery
    std::vector<float>   probabilities;  // [num_classes] after softmax
};

InferenceResult tcn_inference_int8(
    const std::vector<int8_t>& input,
    const std::unordered_map<std::string, TensorBlob>& weights,
    const TCNConfig& config = TCNConfig{});

}  // namespace tcn_hls
