// Fixed-point building blocks for the hand-written HLS-friendly TCN
// implementation. These functions intentionally keep shapes static and math
// explicit so they can be traced back to the exported integer reference.

#include "fixed_tcn_modules.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tcn_hls {

// Clamp to the symmetric INT8 range used by the exported artifacts.
int8_t clamp_int8(int value) {
    if (value > 127) {
        return static_cast<int8_t>(127);
    }
    if (value < -127) {
        return static_cast<int8_t>(-127);
    }
    return static_cast<int8_t>(value);
}

void bias_to_accumulator_hidden(
    const std::array<int8_t, kFixedChannels>& bias_q,
    float bias_scale,
    float input_scale,
    float weight_scale,
    FixedHiddenBiasBuffer& bias_acc) {
    const float ratio = bias_scale / (input_scale * weight_scale);
    for (int idx = 0; idx < kFixedChannels; ++idx) {
        bias_acc[idx] = static_cast<int32_t>(std::round(static_cast<float>(bias_q[idx]) * ratio));
    }
}

// Convert output-layer bias values into the same accumulator scale used by the
// final fully connected stage.
void bias_to_accumulator_output(
    const std::array<int8_t, kFixedNumClasses>& bias_q,
    float bias_scale,
    float input_scale,
    float weight_scale,
    FixedClassBiasBuffer& bias_acc) {
    const float ratio = bias_scale / (input_scale * weight_scale);
    for (int idx = 0; idx < kFixedNumClasses; ++idx) {
        bias_acc[idx] = static_cast<int32_t>(std::round(static_cast<float>(bias_q[idx]) * ratio));
    }
}

void requantize_hidden_to_fixed(
    const FixedHiddenAccBuffer& src,
    float acc_scale,
    FixedHiddenBuffer& dst) {
    const float ratio = acc_scale / kFixedActivationScale;
    for (size_t idx = 0; idx < src.size(); ++idx) {
        const int rounded = static_cast<int>(std::round(static_cast<float>(src[idx]) * ratio));
        dst[idx] = clamp_int8(rounded);
    }
}

// Causal convolution with explicit index arithmetic so the implementation stays
// easy to compare against the Python reference and the generated HLS RTL.
void causal_conv1d_fixed(
    const int8_t* input,
    int in_channels,
    int out_channels,
    int dilation,
    const int8_t* weights,
    const int32_t* bias,
    int32_t* output) {
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        for (int t = 0; t < kFixedSeqLen; ++t) {
            int32_t acc = bias[out_ch];
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                for (int k = 0; k < kFixedKernelSize; ++k) {
                    const int src_t = t - ((kFixedKernelSize - 1 - k) * dilation);
                    if (src_t < 0) {
                        continue;
                    }
                    const int input_idx = in_ch * kFixedSeqLen + src_t;
                    const int weight_idx = (out_ch * in_channels + in_ch) * kFixedKernelSize + k;
                    acc += static_cast<int32_t>(input[input_idx]) * static_cast<int32_t>(weights[weight_idx]);
                }
            }
            output[out_ch * kFixedSeqLen + t] = acc;
        }
    }
}

void pointwise_conv1d_fixed(
    const int8_t* input,
    int in_channels,
    int out_channels,
    const int8_t* weights,
    const int32_t* bias,
    int32_t* output) {
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        for (int t = 0; t < kFixedSeqLen; ++t) {
            int32_t acc = bias[out_ch];
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                const int input_idx = in_ch * kFixedSeqLen + t;
                const int weight_idx = out_ch * in_channels + in_ch;
                acc += static_cast<int32_t>(input[input_idx]) * static_cast<int32_t>(weights[weight_idx]);
            }
            output[out_ch * kFixedSeqLen + t] = acc;
        }
    }
}

// A residual TCN block implemented with explicit integer stages:
// conv1 -> ReLU -> requantize -> conv2 -> ReLU -> requantize -> residual add.
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
    FixedHiddenBuffer& output) {
    FixedHiddenBiasBuffer conv1_bias{};
    FixedHiddenBiasBuffer conv2_bias{};
    FixedHiddenBiasBuffer downsample_bias{};
    FixedHiddenAccBuffer acc1{};
    FixedHiddenAccBuffer acc2{};
    FixedHiddenAccBuffer residual_acc{};
    FixedHiddenBuffer q1{};
    FixedHiddenBuffer q2{};
    FixedHiddenBuffer residual_q{};

    bias_to_accumulator_hidden(conv1_bias_q, conv1_b_scale, kFixedActivationScale, conv1_w_scale, conv1_bias);
    causal_conv1d_fixed(input, in_channels, kFixedChannels, dilation, conv1_weights, conv1_bias.data(), acc1.data());
    for (auto& value : acc1) {
        if (value < 0) {
            value = 0;
        }
    }
    requantize_hidden_to_fixed(acc1, kFixedActivationScale * conv1_w_scale, q1);

    bias_to_accumulator_hidden(conv2_bias_q, conv2_b_scale, kFixedActivationScale, conv2_w_scale, conv2_bias);
    causal_conv1d_fixed(q1.data(), kFixedChannels, kFixedChannels, dilation, conv2_weights, conv2_bias.data(), acc2.data());
    for (auto& value : acc2) {
        if (value < 0) {
            value = 0;
        }
    }
    requantize_hidden_to_fixed(acc2, kFixedActivationScale * conv2_w_scale, q2);

    if (use_downsample) {
        bias_to_accumulator_hidden(
            downsample_bias_q,
            downsample_b_scale,
            kFixedActivationScale,
            downsample_w_scale,
            downsample_bias);
        pointwise_conv1d_fixed(
            input,
            in_channels,
            kFixedChannels,
            downsample_weights,
            downsample_bias.data(),
            residual_acc.data());
        requantize_hidden_to_fixed(
            residual_acc,
            kFixedActivationScale * downsample_w_scale,
            residual_q);
    } else {
        std::copy(input, input + (kFixedChannels * kFixedSeqLen), residual_q.begin());
    }

    for (int idx = 0; idx < kFixedChannels * kFixedSeqLen; ++idx) {
        const int merged = static_cast<int>(q2[idx]) + static_cast<int>(residual_q[idx]);
        output[idx] = clamp_int8(std::max(0, merged));
    }
}

void global_average_pool_fixed(
    const FixedHiddenBuffer& input,
    FixedPoolBuffer& pooled,
    std::array<int8_t, kFixedChannels>& pooled_q) {
    for (int ch = 0; ch < kFixedChannels; ++ch) {
        int64_t sum = 0;
        for (int t = 0; t < kFixedSeqLen; ++t) {
            sum += input[ch * kFixedSeqLen + t];
        }
        pooled[ch] = static_cast<int32_t>(sum / kFixedSeqLen);
        pooled_q[ch] = clamp_int8(pooled[ch]);
    }
}

// Final linear classifier operating on the pooled hidden activations.
void fc_fixed(
    const std::array<int8_t, kFixedChannels>& pooled_q,
    const std::array<int8_t, kFixedNumClasses * kFixedChannels>& weights,
    const FixedClassBiasBuffer& bias,
    FixedClassIntBuffer& logits) {
    for (int out_class = 0; out_class < kFixedNumClasses; ++out_class) {
        int32_t acc = bias[out_class];
        for (int ch = 0; ch < kFixedChannels; ++ch) {
            const int weight_idx = out_class * kFixedChannels + ch;
            acc += static_cast<int32_t>(pooled_q[ch]) * static_cast<int32_t>(weights[weight_idx]);
        }
        logits[out_class] = acc;
    }
}

void softmax3_fixed(
    const FixedClassFloatBuffer& logits,
    FixedClassFloatBuffer& probabilities) {
    const float max_logit = std::max(logits[0], std::max(logits[1], logits[2]));
    float denom = 0.0f;
    for (int idx = 0; idx < kFixedNumClasses; ++idx) {
        probabilities[idx] = std::exp(logits[idx] - max_logit);
        denom += probabilities[idx];
    }
    for (int idx = 0; idx < kFixedNumClasses; ++idx) {
        probabilities[idx] /= denom;
    }
}

}  // namespace tcn_hls
