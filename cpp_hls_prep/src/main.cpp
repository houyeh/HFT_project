// Command-line harness for loading exported INT8 weights, running the dynamic
// CPU inference path, and optionally comparing it against the fixed-shape HLS
// core used for Vitis HLS.

#include "int8_kernels.hpp"
#include "fixed_tcn.hpp"
#include "fixed_tcn_host.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

template <typename Func>
double benchmark_us(Func&& fn, int repeats) {
    auto start = Clock::now();
    for (int i = 0; i < repeats; ++i) {
        fn();
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() /
           static_cast<double>(repeats);
}

// Load a reference input binary file: F×T int8 values, one per line
static std::vector<int8_t> load_reference_input(const std::string& path, int expected_size) {
    std::ifstream file(path);
    if (!file.is_open()) return {};
    std::vector<int8_t> data;
    data.reserve(expected_size);
    int val;
    while (file >> val) {
        data.push_back(static_cast<int8_t>(val));
    }
    return data;
}

static std::array<int8_t, tcn_hls::kFixedNumInputs * tcn_hls::kFixedSeqLen> to_fixed_input(
    const std::vector<int8_t>& input) {
    if (static_cast<int>(input.size()) != tcn_hls::kFixedNumInputs * tcn_hls::kFixedSeqLen) {
        throw std::runtime_error("Reference input size does not match fixed TCN layout.");
    }
    std::array<int8_t, tcn_hls::kFixedNumInputs * tcn_hls::kFixedSeqLen> out{};
    std::copy(input.begin(), input.end(), out.begin());
    return out;
}

int main(int argc, char** argv) {
    try {
        tcn_hls::TCNConfig config;

        // --- Parse arguments ---
        std::string weight_path;
        std::string ref_input_path;
        bool run_kernel_bench = false;
        bool compare_fixed_core = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--weights" && i + 1 < argc) {
                weight_path = argv[++i];
            } else if (arg == "--ref-input" && i + 1 < argc) {
                ref_input_path = argv[++i];
            } else if (arg == "--kernel-bench") {
                run_kernel_bench = true;
            } else if (arg == "--compare-fixed") {
                compare_fixed_core = true;
            } else if (arg == "--help") {
                std::cout << "Usage: tcn_hls_prep [options]\n"
                          << "  --weights <path>     Path to INT8 weight file (.wts)\n"
                          << "  --ref-input <path>   Path to reference input file (optional)\n"
                          << "  --kernel-bench       Also run kernel microbenchmarks\n"
                          << "  --compare-fixed      Also run the fixed-shape HLS-friendly core\n";
                return 0;
            } else if (weight_path.empty()) {
                // Legacy: first positional arg = weight path
                weight_path = arg;
            }
        }

        // --- Load weights ---
        std::unordered_map<std::string, tcn_hls::TensorBlob> tensors;
        if (!weight_path.empty()) {
            tensors = tcn_hls::load_wts_file(weight_path);
            std::cout << "Loaded tensors: " << tensors.size() << "\n";

            // Print weight summary
            int total_params = 0;
            for (const auto& [name, blob] : tensors) {
                total_params += static_cast<int>(blob.values.size());
            }
            std::cout << "Total parameters: " << total_params << "\n\n";
        } else {
            std::cout << "No weight file provided. Use --weights <path>.\n";
            return 1;
        }

        // --- Prepare input ---
        const int input_size = config.num_inputs * config.seq_len;
        std::vector<int8_t> test_input;

        if (!ref_input_path.empty()) {
            test_input = load_reference_input(ref_input_path, input_size);
            if (static_cast<int>(test_input.size()) != input_size) {
                std::cerr << "Warning: reference input has " << test_input.size()
                          << " values, expected " << input_size << ". Using random.\n";
                test_input.clear();
            } else {
                std::cout << "Loaded reference input from: " << ref_input_path << "\n";
            }
        }

        if (test_input.empty()) {
            // Generate random input
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-8, 8);
            test_input.resize(input_size);
            for (auto& v : test_input) v = static_cast<int8_t>(dist(rng));
            std::cout << "Using random input (seed=42)\n";
        }

        // --- Full inference ---
        std::cout << "\n=== Full TCN Inference ===\n";
        std::cout << "Input shape: (" << config.num_inputs << ", " << config.seq_len << ")\n";

        auto result = tcn_hls::tcn_inference_int8(test_input, tensors, config);

        std::cout << "Output logits (int32):";
        for (auto v : result.logits_int32) std::cout << " " << v;
        std::cout << "\n";

        std::cout << "Output logits (float):";
        for (auto v : result.logits_float) std::cout << " " << v;
        std::cout << "\n";

        std::cout << "Probabilities (softmax):";
        for (auto v : result.probabilities) std::cout << " " << v;
        std::cout << "\n";

        // Find predicted class
        int pred_class = 0;
        for (int c = 1; c < config.num_classes; ++c) {
            if (result.probabilities[c] > result.probabilities[pred_class]) {
                pred_class = c;
            }
        }
        const char* class_names[] = {"Down", "Stationary", "Up"};
        std::cout << "Predicted: " << class_names[pred_class]
                  << " (p=" << result.probabilities[pred_class] << ")\n";

        // --- Inference benchmark ---
        std::cout << "\n=== Inference Latency Benchmark ===\n";
        const int warmup = 10;
        const int repeats = 100;

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            auto r = tcn_hls::tcn_inference_int8(test_input, tensors, config);
            (void)r;
        }

        const double avg_us = benchmark_us(
            [&]() {
                auto r = tcn_hls::tcn_inference_int8(test_input, tensors, config);
                (void)r;
            },
            repeats);

        std::cout << "Average inference latency: " << avg_us << " us\n";
        std::cout << "Average inference latency: " << avg_us / 1000.0 << " ms\n";

        // --- Optional fixed-shape HLS-friendly inference ---
        if (compare_fixed_core) {
            std::cout << "\n=== Fixed-Shape HLS-Friendly Core ===\n";
            tcn_hls::FixedTCNWeights fixed_weights;
            std::string fixed_error;
            if (!tcn_hls::load_fixed_tcn_weights(tensors, fixed_weights, &fixed_error)) {
                throw std::runtime_error("Unable to load fixed TCN weights: " + fixed_error);
            }

            const auto fixed_input = to_fixed_input(test_input);
            const auto fixed_result = tcn_hls::tcn_inference_fixed_int8(fixed_input, fixed_weights);

            std::cout << "Fixed core logits (int32):";
            for (auto v : fixed_result.logits_int32) std::cout << " " << v;
            std::cout << "\n";

            std::cout << "Fixed core probabilities:";
            for (auto v : fixed_result.probabilities) std::cout << " " << v;
            std::cout << "\n";

            double max_prob_diff = 0.0;
            for (int c = 0; c < config.num_classes; ++c) {
                max_prob_diff = std::max(
                    max_prob_diff,
                    std::abs(static_cast<double>(result.probabilities[c]) -
                             static_cast<double>(fixed_result.probabilities[c])));
            }
            std::cout << "Max probability diff vs dynamic core: " << max_prob_diff << "\n";

            const double fixed_avg_us = benchmark_us(
                [&]() {
                    auto r = tcn_hls::tcn_inference_fixed_int8(fixed_input, fixed_weights);
                    (void)r;
                },
                repeats);
            std::cout << "Fixed core latency: " << fixed_avg_us << " us\n";
        }

        // --- Optional kernel microbenchmark ---
        if (run_kernel_bench) {
            std::cout << "\n=== Kernel Microbenchmarks ===\n";
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-8, 8);

            // Conv1d benchmark
            const int in_ch = config.num_inputs;
            const int out_ch = config.channels;
            const int len = config.seq_len;
            const int ks = config.kernel_size;
            std::vector<int8_t> ci(in_ch * len), cw(out_ch * in_ch * ks);
            for (auto& v : ci) v = static_cast<int8_t>(dist(rng));
            for (auto& v : cw) v = static_cast<int8_t>(dist(rng));

            auto conv_us = benchmark_us(
                [&]() {
                    auto o = tcn_hls::conv1d_causal_int8(ci, cw, in_ch, out_ch, len, ks, 1);
                    (void)o;
                },
                500);
            std::cout << "  conv1d_causal (12->16, T=100, k=3, d=1): " << conv_us << " us\n";

            // GEMM benchmark
            std::vector<int8_t> gi(config.channels), gw(config.num_classes * config.channels);
            for (auto& v : gi) v = static_cast<int8_t>(dist(rng));
            for (auto& v : gw) v = static_cast<int8_t>(dist(rng));

            auto gemm_us = benchmark_us(
                [&]() {
                    auto o = tcn_hls::gemm_int8(gi, gw, 1, config.num_classes, config.channels);
                    (void)o;
                },
                5000);
            std::cout << "  gemm (1x16 -> 1x3): " << gemm_us << " us\n";
        }

        std::cout << "\nDone.\n";
        return 0;

    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
