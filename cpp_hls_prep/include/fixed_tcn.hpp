// 固定形狀 TCN 佈局與頂層權重/結果容器（用於 Vitis HLS 轉移路徑）

#pragma once

#include <array>
#include <cstdint>

namespace tcn_hls {

constexpr int kFixedNumInputs = 12;
constexpr int kFixedNumClasses = 3;
constexpr int kFixedNumBlocks = 3;
constexpr int kFixedChannels = 16;
constexpr int kFixedKernelSize = 3;
constexpr int kFixedSeqLen = 100;
constexpr float kFixedActivationScale = 1.0f / 64.0f;

struct FixedTCNWeights {
    std::array<int8_t, kFixedChannels * kFixedNumInputs * kFixedKernelSize> block0_conv1_w{};
    std::array<int8_t, kFixedChannels> block0_conv1_b{};
    float block0_conv1_w_scale = 1.0f;
    float block0_conv1_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedChannels * kFixedKernelSize> block0_conv2_w{};
    std::array<int8_t, kFixedChannels> block0_conv2_b{};
    float block0_conv2_w_scale = 1.0f;
    float block0_conv2_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedNumInputs> block0_downsample_w{};
    std::array<int8_t, kFixedChannels> block0_downsample_b{};
    float block0_downsample_w_scale = 1.0f;
    float block0_downsample_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedChannels * kFixedKernelSize> block1_conv1_w{};
    std::array<int8_t, kFixedChannels> block1_conv1_b{};
    float block1_conv1_w_scale = 1.0f;
    float block1_conv1_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedChannels * kFixedKernelSize> block1_conv2_w{};
    std::array<int8_t, kFixedChannels> block1_conv2_b{};
    float block1_conv2_w_scale = 1.0f;
    float block1_conv2_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedChannels * kFixedKernelSize> block2_conv1_w{};
    std::array<int8_t, kFixedChannels> block2_conv1_b{};
    float block2_conv1_w_scale = 1.0f;
    float block2_conv1_b_scale = 1.0f;

    std::array<int8_t, kFixedChannels * kFixedChannels * kFixedKernelSize> block2_conv2_w{};
    std::array<int8_t, kFixedChannels> block2_conv2_b{};
    float block2_conv2_w_scale = 1.0f;
    float block2_conv2_b_scale = 1.0f;

    std::array<int8_t, kFixedNumClasses * kFixedChannels> fc_w{};
    std::array<int8_t, kFixedNumClasses> fc_b{};
    float fc_w_scale = 1.0f;
    float fc_b_scale = 1.0f;
};

struct FixedInferenceResult {
    std::array<int32_t, kFixedNumClasses> logits_int32{};
    std::array<float, kFixedNumClasses> logits_float{};
    std::array<float, kFixedNumClasses> probabilities{};
};

FixedInferenceResult tcn_inference_fixed_int8(
    const std::array<int8_t, kFixedNumInputs * kFixedSeqLen>& input,
    const FixedTCNWeights& weights);

}  // namespace tcn_hls
