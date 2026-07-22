// 輔助函式：將 PyTorch 匯出的動態 INT8 Tensor 載入到固定形狀的 HLS 權重結構中
// 供硬體導向的推論核心使用。

#pragma once

#include "fixed_tcn.hpp"
#include "int8_kernels.hpp"

#include <string>

namespace tcn_hls {

bool load_fixed_tcn_weights(
    const std::unordered_map<std::string, TensorBlob>& tensors,
    FixedTCNWeights& weights,
    std::string* error = nullptr);

bool load_fixed_tcn_weights_from_wts(
    const std::string& path,
    FixedTCNWeights& weights,
    std::string* error = nullptr);

}  // namespace tcn_hls
