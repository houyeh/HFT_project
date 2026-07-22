// Vitis HLS top wrapper that copies array ports into the fixed-shape C++
// containers and runs the golden fixed-point inference core.

#include "fixed_tcn_top.hpp"

#include "generated/fixed_tcn_weights.hpp"

namespace tcn_hls {

void tcn_fixed_top(
    const int8_t input[kFixedNumInputs * kFixedSeqLen],
    int32_t logits[kFixedNumClasses],
    float probabilities[kFixedNumClasses]) {
#pragma HLS INTERFACE ap_memory port=input
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_memory port=probabilities
#pragma HLS INTERFACE ap_ctrl_hs port=return

    std::array<int8_t, kFixedNumInputs * kFixedSeqLen> input_array{};
    for (int idx = 0; idx < kFixedNumInputs * kFixedSeqLen; ++idx) {
        input_array[idx] = input[idx];
    }

    const auto result = tcn_inference_fixed_int8(input_array, generated::kFixedWeights);

    for (int idx = 0; idx < kFixedNumClasses; ++idx) {
        logits[idx] = result.logits_int32[idx];
        probabilities[idx] = result.probabilities[idx];
    }
}

}  // namespace tcn_hls
