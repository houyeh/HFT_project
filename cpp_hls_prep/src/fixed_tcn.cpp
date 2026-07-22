// Fixed-shape reference inference that mirrors the exported INT8 TCN exactly
// and is used as the golden model for the HLS testbench.

#include "fixed_tcn.hpp"
#include "fixed_tcn_modules.hpp"

namespace tcn_hls {

FixedInferenceResult tcn_inference_fixed_int8(
    const std::array<int8_t, kFixedNumInputs * kFixedSeqLen>& input,
    const FixedTCNWeights& weights) {
    FixedInferenceResult result;
    FixedHiddenBuffer block0_out{};
    FixedHiddenBuffer block1_out{};
    FixedHiddenBuffer block2_out{};
    FixedPoolBuffer pooled{};
    std::array<int8_t, kFixedChannels> pooled_q{};
    FixedClassBiasBuffer fc_bias{};

    residual_block_fixed(
        input.data(),
        kFixedNumInputs,
        1,
        weights.block0_conv1_w.data(),
        weights.block0_conv1_w_scale,
        weights.block0_conv1_b,
        weights.block0_conv1_b_scale,
        weights.block0_conv2_w.data(),
        weights.block0_conv2_w_scale,
        weights.block0_conv2_b,
        weights.block0_conv2_b_scale,
        weights.block0_downsample_w.data(),
        weights.block0_downsample_w_scale,
        weights.block0_downsample_b,
        weights.block0_downsample_b_scale,
        true,
        block0_out);

    residual_block_fixed(
        block0_out.data(),
        kFixedChannels,
        2,
        weights.block1_conv1_w.data(),
        weights.block1_conv1_w_scale,
        weights.block1_conv1_b,
        weights.block1_conv1_b_scale,
        weights.block1_conv2_w.data(),
        weights.block1_conv2_w_scale,
        weights.block1_conv2_b,
        weights.block1_conv2_b_scale,
        nullptr,
        1.0f,
        {},
        1.0f,
        false,
        block1_out);

    residual_block_fixed(
        block1_out.data(),
        kFixedChannels,
        4,
        weights.block2_conv1_w.data(),
        weights.block2_conv1_w_scale,
        weights.block2_conv1_b,
        weights.block2_conv1_b_scale,
        weights.block2_conv2_w.data(),
        weights.block2_conv2_w_scale,
        weights.block2_conv2_b,
        weights.block2_conv2_b_scale,
        nullptr,
        1.0f,
        {},
        1.0f,
        false,
        block2_out);

    global_average_pool_fixed(block2_out, pooled, pooled_q);
    bias_to_accumulator_output(weights.fc_b, weights.fc_b_scale, kFixedActivationScale, weights.fc_w_scale, fc_bias);
    fc_fixed(pooled_q, weights.fc_w, fc_bias, result.logits_int32);
    for (int out_class = 0; out_class < kFixedNumClasses; ++out_class) {
        result.logits_float[out_class] =
            static_cast<float>(result.logits_int32[out_class]) * (kFixedActivationScale * weights.fc_w_scale);
    }
    softmax3_fixed(result.logits_float, result.probabilities);
    return result;
}

}  // namespace tcn_hls
