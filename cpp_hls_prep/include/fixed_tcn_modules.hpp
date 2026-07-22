// 核心算子宣告：CausalConv1D、ResidualBlock、GlobalAvgPool、FC、Softmax
// 這些都是 HLS 友善的純 C++ 實作，沒有 PyTorch 依賴。

#pragma once

#include "fixed_tcn.hpp"

#include <array>
#include <cstdint>

namespace tcn_hls {

using FixedInputBuffer = std::array<int8_t, kFixedNumInputs * kFixedSeqLen>;
using FixedHiddenBuffer = std::array<int8_t, kFixedChannels * kFixedSeqLen>;
using FixedHiddenAccBuffer = std::array<int32_t, kFixedChannels * kFixedSeqLen>;
using FixedHiddenBiasBuffer = std::array<int32_t, kFixedChannels>;
using FixedClassBiasBuffer = std::array<int32_t, kFixedNumClasses>;
using FixedPoolBuffer = std::array<int32_t, kFixedChannels>;
using FixedClassIntBuffer = std::array<int32_t, kFixedNumClasses>;
using FixedClassFloatBuffer = std::array<float, kFixedNumClasses>;

int8_t clamp_int8(int value);

void bias_to_accumulator_hidden(
    const std::array<int8_t, kFixedChannels>& bias_q,
    float bias_scale,
    float input_scale,
    float weight_scale,
    FixedHiddenBiasBuffer& bias_acc);

void bias_to_accumulator_output(
    const std::array<int8_t, kFixedNumClasses>& bias_q,
    float bias_scale,
    float input_scale,
    float weight_scale,
    FixedClassBiasBuffer& bias_acc);

void requantize_hidden_to_fixed(
    const FixedHiddenAccBuffer& src,
    float acc_scale,
    FixedHiddenBuffer& dst);

void causal_conv1d_fixed(
    const int8_t* input,
    int in_channels,
    int out_channels,
    int dilation,
    const int8_t* weights,
    const int32_t* bias,
    int32_t* output);

void pointwise_conv1d_fixed(
    const int8_t* input,
    int in_channels,
    int out_channels,
    const int8_t* weights,
    const int32_t* bias,
    int32_t* output);

void residual_block_fixed(
    const int8_t* input,
    int in_channels,
    int dilation,
    const int8_t* conv1_weights,
    float conv1_w_scale,
    const std::array<int8_t, kFixedChannels>& conv1_bias_q,
    float conv1_b_scale,
    const int8_t* conv2_weights,
    float conv2_w_scale,
    const std::array<int8_t, kFixedChannels>& conv2_bias_q,
    float conv2_b_scale,
    const int8_t* downsample_weights,
    float downsample_w_scale,
    const std::array<int8_t, kFixedChannels>& downsample_bias_q,
    float downsample_b_scale,
    bool use_downsample,
    FixedHiddenBuffer& output);

void global_average_pool_fixed(
    const FixedHiddenBuffer& input,
    FixedPoolBuffer& pooled,
    std::array<int8_t, kFixedChannels>& pooled_q);

void fc_fixed(
    const std::array<int8_t, kFixedChannels>& pooled_q,
    const std::array<int8_t, kFixedNumClasses * kFixedChannels>& weights,
    const FixedClassBiasBuffer& bias,
    FixedClassIntBuffer& logits);

void softmax3_fixed(
    const FixedClassFloatBuffer& logits,
    FixedClassFloatBuffer& probabilities);

}  // namespace tcn_hls
