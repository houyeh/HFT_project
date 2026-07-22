// Loaders that copy named tensors from the dynamic `.wts` representation into
// the compile-time fixed-shape weight container expected by the HLS core.

#include "fixed_tcn_host.hpp"

#include <algorithm>

namespace tcn_hls {

namespace {

template <size_t N>
bool copy_tensor_values(
    const std::unordered_map<std::string, TensorBlob>& tensors,
    const std::string& name,
    std::array<int8_t, N>& dest,
    float& scale,
    std::string* error) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        if (error != nullptr) {
            *error = "Missing tensor: " + name;
        }
        return false;
    }
    if (it->second.values.size() != N) {
        if (error != nullptr) {
            *error = "Tensor size mismatch for " + name;
        }
        return false;
    }
    std::copy(it->second.values.begin(), it->second.values.end(), dest.begin());
    scale = it->second.scale;
    return true;
}

}  // namespace

bool load_fixed_tcn_weights(
    const std::unordered_map<std::string, TensorBlob>& tensors,
    FixedTCNWeights& weights,
    std::string* error) {
    return
        copy_tensor_values(tensors, "tcn.network.0.conv1.conv.weight", weights.block0_conv1_w, weights.block0_conv1_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.0.conv1.conv.bias", weights.block0_conv1_b, weights.block0_conv1_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.0.conv2.conv.weight", weights.block0_conv2_w, weights.block0_conv2_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.0.conv2.conv.bias", weights.block0_conv2_b, weights.block0_conv2_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.0.downsample.weight", weights.block0_downsample_w, weights.block0_downsample_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.0.downsample.bias", weights.block0_downsample_b, weights.block0_downsample_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.1.conv1.conv.weight", weights.block1_conv1_w, weights.block1_conv1_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.1.conv1.conv.bias", weights.block1_conv1_b, weights.block1_conv1_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.1.conv2.conv.weight", weights.block1_conv2_w, weights.block1_conv2_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.1.conv2.conv.bias", weights.block1_conv2_b, weights.block1_conv2_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.2.conv1.conv.weight", weights.block2_conv1_w, weights.block2_conv1_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.2.conv1.conv.bias", weights.block2_conv1_b, weights.block2_conv1_b_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.2.conv2.conv.weight", weights.block2_conv2_w, weights.block2_conv2_w_scale, error) &&
        copy_tensor_values(tensors, "tcn.network.2.conv2.conv.bias", weights.block2_conv2_b, weights.block2_conv2_b_scale, error) &&
        copy_tensor_values(tensors, "fc.weight", weights.fc_w, weights.fc_w_scale, error) &&
        copy_tensor_values(tensors, "fc.bias", weights.fc_b, weights.fc_b_scale, error);
}

bool load_fixed_tcn_weights_from_wts(
    const std::string& path,
    FixedTCNWeights& weights,
    std::string* error) {
    try {
        const auto tensors = load_wts_file(path);
        return load_fixed_tcn_weights(tensors, weights, error);
    } catch (const std::exception& exc) {
        if (error != nullptr) {
            *error = exc.what();
        }
        return false;
    }
}

}  // namespace tcn_hls
