// Dynamic INT8 kernels that are used to validate the exported `.wts` file on
// CPU before freezing the model into the fixed-shape HLS-oriented form.

#include "int8_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tcn_hls {

// ---------------------------------------------------------------------------
//  Weight file I/O
// ---------------------------------------------------------------------------

std::unordered_map<std::string, TensorBlob> load_wts_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open weight file: " + path);
    }

    size_t tensor_count = 0;
    input >> tensor_count;
    std::unordered_map<std::string, TensorBlob> tensors;
    for (size_t index = 0; index < tensor_count; ++index) {
        TensorBlob blob;
        size_t count = 0;
        std::string shape_text;
        input >> blob.name >> count >> blob.scale >> shape_text;
        blob.values.resize(count);

        std::stringstream shape_stream(shape_text);
        std::string token;
        while (std::getline(shape_stream, token, 'x')) {
            blob.shape.push_back(std::stoi(token));
        }
        for (size_t value_index = 0; value_index < count; ++value_index) {
            int value = 0;
            input >> value;
            blob.values[value_index] = static_cast<int8_t>(value);
        }
        tensors.emplace(blob.name, std::move(blob));
    }
    return tensors;
}

// ---------------------------------------------------------------------------
//  Conv1d – valid mode
// ---------------------------------------------------------------------------

std::vector<int32_t> conv1d_valid_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int dilation) {
    const int effective_kernel = (kernel_size - 1) * dilation + 1;
    const int output_length = 1 + (input_length - effective_kernel) / stride;
    std::vector<int32_t> output(out_channels * output_length, 0);

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int out_t = 0; out_t < output_length; ++out_t) {
            int32_t acc = 0;
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    const int in_t = out_t * stride + k * dilation;
                    const int input_idx = ic * input_length + in_t;
                    const int weight_idx = (oc * in_channels + ic) * kernel_size + k;
                    acc += static_cast<int32_t>(input[input_idx]) *
                           static_cast<int32_t>(weights[weight_idx]);
                }
            }
            output[oc * output_length + out_t] = acc;
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
//  Conv1d – causal mode (left zero-padding)
// ---------------------------------------------------------------------------

std::vector<int32_t> conv1d_causal_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int dilation) {

    const int pad_len = (kernel_size - 1) * dilation;
    const int padded_length = input_length + pad_len;

    // Build padded input: zeros on the left, original data on the right
    std::vector<int8_t> padded(in_channels * padded_length, 0);
    for (int c = 0; c < in_channels; ++c) {
        for (int t = 0; t < input_length; ++t) {
            padded[c * padded_length + pad_len + t] = input[c * input_length + t];
        }
    }

    // Valid convolution on padded input gives output_length == input_length
    return conv1d_valid_int8(
        padded, weights, in_channels, out_channels,
        padded_length, kernel_size, /*stride=*/1, dilation);
}

// ---------------------------------------------------------------------------
//  GEMM
// ---------------------------------------------------------------------------

std::vector<int32_t> gemm_int8(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int rows,
    int cols,
    int inner_dim) {
    std::vector<int32_t> output(rows * cols, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int32_t acc = 0;
            for (int k = 0; k < inner_dim; ++k) {
                acc += static_cast<int32_t>(input[r * inner_dim + k]) *
                       static_cast<int32_t>(weights[c * inner_dim + k]);
            }
            output[r * cols + c] = acc;
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
//  Element-wise ops
// ---------------------------------------------------------------------------

void relu_inplace_int32(std::vector<int32_t>& data) {
    for (auto& val : data) {
        if (val < 0) val = 0;
    }
}

void add_bias_int32(
    std::vector<int32_t>& data,
    const std::vector<int32_t>& bias,
    int channels,
    int length) {
    for (int c = 0; c < channels; ++c) {
        for (int t = 0; t < length; ++t) {
            data[c * length + t] += bias[c];
        }
    }
}

void residual_add_int32(
    std::vector<int32_t>& out,
    const std::vector<int32_t>& residual) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] += residual[i];
    }
}

// ---------------------------------------------------------------------------
//  Requantization
// ---------------------------------------------------------------------------

std::vector<int8_t> requantize_int32_to_int8(
    const std::vector<int32_t>& acc,
    float acc_scale,
    float target_scale) {
    const float ratio = acc_scale / target_scale;
    std::vector<int8_t> out(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) {
        float val = static_cast<float>(acc[i]) * ratio;
        int rounded = static_cast<int>(std::round(val));
        if (rounded > 127)  rounded = 127;
        if (rounded < -127) rounded = -127;
        out[i] = static_cast<int8_t>(rounded);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Global Average Pooling
// ---------------------------------------------------------------------------

std::vector<int32_t> global_avg_pool_int32(
    const std::vector<int32_t>& data,
    int channels,
    int length) {
    std::vector<int32_t> out(channels, 0);
    for (int c = 0; c < channels; ++c) {
        int64_t sum = 0;  // use int64 to avoid overflow during summation
        for (int t = 0; t < length; ++t) {
            sum += data[c * length + t];
        }
        // Integer division for the average (matches floor behavior)
        out[c] = static_cast<int32_t>(sum / length);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Helper: extract and convert bias from int8 blob to int32
//  (bias is stored as int8 with its own scale)
// ---------------------------------------------------------------------------

static std::vector<int32_t> bias_to_int32(
    const TensorBlob& blob,
    float input_scale,
    float weight_scale) {
    // In proper INT8 inference, bias is stored at the accumulator scale
    // bias_real = blob.values[i] * blob.scale
    // bias_in_acc_units = bias_real / (input_scale * weight_scale)
    const float bias_to_acc = blob.scale / (input_scale * weight_scale);
    std::vector<int32_t> result(blob.values.size());
    for (size_t i = 0; i < blob.values.size(); ++i) {
        result[i] = static_cast<int32_t>(std::round(
            static_cast<float>(blob.values[i]) * bias_to_acc));
    }
    return result;
}

// ---------------------------------------------------------------------------
//  Full TCN inference
// ---------------------------------------------------------------------------

InferenceResult tcn_inference_int8(
    const std::vector<int8_t>& input,
    const std::unordered_map<std::string, TensorBlob>& weights,
    const TCNConfig& config) {

    const int T = config.seq_len;

    // Helper to look up a weight tensor by name
    auto get = [&](const std::string& name) -> const TensorBlob& {
        auto it = weights.find(name);
        if (it == weights.end()) {
            throw std::runtime_error("Missing weight tensor: " + name);
        }
        return it->second;
    };

    // Current activation is kept as int8.
    // After each layer we requantize back to int8.
    std::vector<int8_t> current = input;
    float current_scale = 1.0f;  // will be set from input quantization

    // In a proper pipeline this would come from calibration.
    // For now we use 1/64 which keeps most LOB features in range.
    const float FIXED_ACT_SCALE = 1.0f / 64.0f;
    current_scale = FIXED_ACT_SCALE;

    for (int blk = 0; blk < config.num_blocks; ++blk) {
        const std::string prefix = "tcn.network." + std::to_string(blk) + ".";
        const int in_ch  = (blk == 0) ? config.num_inputs : config.channels;
        const int out_ch = config.channels;
        const int dil    = config.dilation(blk);

        // --- conv1 ---
        const auto& w1 = get(prefix + "conv1.conv.weight");
        const auto& b1 = get(prefix + "conv1.conv.bias");
        float w1_scale = w1.scale;
        float acc1_scale = current_scale * w1_scale;
        auto out1 = conv1d_causal_int8(current, w1.values,
                                       in_ch, out_ch, T, config.kernel_size, dil);
        auto bias1 = bias_to_int32(b1, current_scale, w1_scale);
        add_bias_int32(out1, bias1, out_ch, T);
        relu_inplace_int32(out1);

        // Requantize acc → int8 for next layer
        float target_scale_1 = FIXED_ACT_SCALE;  // use fixed calibration scale
        auto q1 = requantize_int32_to_int8(out1, acc1_scale, target_scale_1);

        // --- conv2 ---
        const auto& w2 = get(prefix + "conv2.conv.weight");
        const auto& b2 = get(prefix + "conv2.conv.bias");
        float w2_scale = w2.scale;
        float acc2_scale = target_scale_1 * w2_scale;
        auto out2 = conv1d_causal_int8(q1, w2.values,
                                       out_ch, out_ch, T, config.kernel_size, dil);
        auto bias2 = bias_to_int32(b2, target_scale_1, w2_scale);
        add_bias_int32(out2, bias2, out_ch, T);
        relu_inplace_int32(out2);

        // --- residual path ---
        std::vector<int32_t> residual;
        float residual_scale;

        if (in_ch != out_ch) {
            // 1×1 downsample conv
            const auto& wd = get(prefix + "downsample.weight");
            const auto& bd = get(prefix + "downsample.bias");
            float wd_scale = wd.scale;
            residual_scale = current_scale * wd_scale;
            residual = conv1d_valid_int8(current, wd.values,
                                         in_ch, out_ch, T,
                                         /*kernel_size=*/1, /*stride=*/1, /*dilation=*/1);
            auto bias_d = bias_to_int32(bd, current_scale, wd_scale);
            add_bias_int32(residual, bias_d, out_ch, T);
        } else {
            // Identity residual: promote current int8 to int32
            residual.resize(out_ch * T);
            for (int i = 0; i < out_ch * T; ++i) {
                residual[i] = static_cast<int32_t>(current[i]);
            }
            residual_scale = current_scale;
        }

        // Scale-align residual to match out2's accumulator scale, then add
        // We requantize both to a fixed target scale BEFORE adding
        float block_out_scale = FIXED_ACT_SCALE;
        auto res_q = requantize_int32_to_int8(residual, residual_scale, block_out_scale);
        auto out2_q = requantize_int32_to_int8(out2, acc2_scale, block_out_scale);

        // Add residual (both now in same quantized domain)
        std::vector<int32_t> merged(out_ch * T);
        for (int i = 0; i < out_ch * T; ++i) {
            merged[i] = static_cast<int32_t>(out2_q[i]) + static_cast<int32_t>(res_q[i]);
        }

        // Final ReLU for the block
        relu_inplace_int32(merged);

        // the merged result is already scaled at block_out_scale, just cast back to int8
        current.resize(out_ch * T);
        for (int i = 0; i < out_ch * T; ++i) {
            int val = merged[i];
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            current[i] = static_cast<int8_t>(val);
        }
        current_scale = block_out_scale;
    }

    // --- Global Average Pooling ---
    // Promote current int8 to int32 for pooling
    std::vector<int32_t> gap_input(config.channels * T);
    for (int i = 0; i < config.channels * T; ++i) {
        gap_input[i] = static_cast<int32_t>(current[i]);
    }
    auto pooled = global_avg_pool_int32(gap_input, config.channels, T);
    // pooled: [channels] in int32

    // Requantize pooled → int8 for FC layer
    auto pooled_q = requantize_int32_to_int8(pooled, current_scale, current_scale);

    // --- Fully Connected Layer ---
    const auto& fc_w = get("fc.weight");
    const auto& fc_b = get("fc.bias");
    float fc_w_scale = fc_w.scale;
    float fc_acc_scale = current_scale * fc_w_scale;

    auto logits_i32 = gemm_int8(pooled_q, fc_w.values,
                                /*rows=*/1, config.num_classes, config.channels);
    auto fc_bias = bias_to_int32(fc_b, current_scale, fc_w_scale);
    for (int c = 0; c < config.num_classes; ++c) {
        logits_i32[c] += fc_bias[c];
    }

    // --- Convert to float and apply softmax ---
    InferenceResult result;
    result.logits_int32 = logits_i32;
    result.logits_float.resize(config.num_classes);
    result.probabilities.resize(config.num_classes);

    for (int c = 0; c < config.num_classes; ++c) {
        result.logits_float[c] = static_cast<float>(logits_i32[c]) * fc_acc_scale;
    }

    // Softmax
    float max_logit = *std::max_element(result.logits_float.begin(), result.logits_float.end());
    float sum_exp = 0.0f;
    for (int c = 0; c < config.num_classes; ++c) {
        result.probabilities[c] = std::exp(result.logits_float[c] - max_logit);
        sum_exp += result.probabilities[c];
    }
    for (int c = 0; c < config.num_classes; ++c) {
        result.probabilities[c] /= sum_exp;
    }

    return result;
}

}  // namespace tcn_hls
