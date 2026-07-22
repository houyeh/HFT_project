// Standalone C simulation testbench that checks the HLS top module against the
// generated golden logits and probabilities.

#include "fixed_tcn_top.hpp"

#include "generated/fixed_tcn_test_vectors.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    int32_t logits[tcn_hls::kFixedNumClasses] = {0};
    float probabilities[tcn_hls::kFixedNumClasses] = {0.0f};

    tcn_hls::tcn_fixed_top(
        tcn_hls::generated::kReferenceInput.data(),
        logits,
        probabilities);

    bool pass = true;
    for (int idx = 0; idx < tcn_hls::kFixedNumClasses; ++idx) {
        if (logits[idx] != tcn_hls::generated::kGoldenLogitsInt32[idx]) {
            std::cerr << "[FAIL] logit mismatch at class " << idx
                      << ": got " << logits[idx]
                      << ", expected " << tcn_hls::generated::kGoldenLogitsInt32[idx]
                      << "\n";
            pass = false;
        }
    }

    for (int idx = 0; idx < tcn_hls::kFixedNumClasses; ++idx) {
        const float diff = std::fabs(probabilities[idx] - tcn_hls::generated::kGoldenProbabilities[idx]);
        if (diff > tcn_hls::generated::kProbabilityTolerance) {
            std::cerr << "[FAIL] probability mismatch at class " << idx
                      << ": got " << probabilities[idx]
                      << ", expected " << tcn_hls::generated::kGoldenProbabilities[idx]
                      << ", diff=" << diff
                      << "\n";
            pass = false;
        }
    }

    if (!pass) {
        return 1;
    }

    std::cout << "[PASS] HLS testbench matched fixed golden logits/probabilities.\n";
    return 0;
}
