// 頂層 Vitis HLS 進入點宣告：固定形狀 TCN 核心

#pragma once

#include "fixed_tcn.hpp"

#include <cstdint>

namespace tcn_hls {

void tcn_fixed_top(
    const int8_t input[kFixedNumInputs * kFixedSeqLen],
    int32_t logits[kFixedNumClasses],
    float probabilities[kFixedNumClasses]);

}  // namespace tcn_hls
