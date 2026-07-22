#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDefaultWarmupIters = 100;
constexpr int kDefaultTimingItersPerStream = 10000;
constexpr int kDefaultRounds = 2;
constexpr int kDefaultTorchThreads = 8;
constexpr double kDefaultDeadlineUs = 83.16;

struct Stats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    std::size_t sample_count = 0;
    std::size_t deadline_miss_count = 0;
    double deadline_miss_rate = 0.0;
};

struct WorkerResult {
    std::vector<double> latencies_us;
    std::string error;
};

std::string json_escape(const std::string& value) {
    std::ostringstream os;
    for (const char ch : value) {
        switch (ch) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << ch; break;
        }
    }
    return os.str();
}

std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::size_t percentile_index(const std::size_t n, const double q) {
    if (n == 0) {
        return 0;
    }
    const auto idx = static_cast<std::size_t>(q * static_cast<double>(n));
    return std::min<std::size_t>(idx, n - 1);
}

Stats summarize(std::vector<double> samples, const double deadline_us) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }

    std::sort(samples.begin(), samples.end());
    stats.sample_count = samples.size();
    stats.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
                    static_cast<double>(samples.size());
    stats.p50_us = samples[samples.size() / 2];
    stats.p95_us = samples[percentile_index(samples.size(), 0.95)];
    stats.p99_us = samples[percentile_index(samples.size(), 0.99)];
    stats.min_us = samples.front();
    stats.max_us = samples.back();
    stats.deadline_miss_count = static_cast<std::size_t>(
        std::count_if(samples.begin(), samples.end(),
                      [deadline_us](const double latency_us) {
                          return latency_us > deadline_us;
                      }));
    stats.deadline_miss_rate =
        static_cast<double>(stats.deadline_miss_count) /
        static_cast<double>(samples.size());
    return stats;
}

std::vector<WorkerResult> run_round(const std::string& model_path,
                                    const int concurrency,
                                    const int warmup_iters,
                                    const int timing_iters_per_stream) {
    std::vector<WorkerResult> results(static_cast<std::size_t>(concurrency));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(concurrency));

    std::atomic<int> ready_count{0};
    std::atomic<bool> start_timing{false};

    for (int worker_id = 0; worker_id < concurrency; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            try {
                torch::NoGradGuard no_grad;
                torch::jit::Module module = torch::jit::load(model_path, torch::kCPU);
                module.eval();

                auto input = torch::zeros(
                    {1, 32},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                std::vector<torch::jit::IValue> inputs;
                inputs.emplace_back(input);

                for (int i = 0; i < warmup_iters; ++i) {
                    (void)module.forward(inputs);
                }

                auto& latencies = results[static_cast<std::size_t>(worker_id)].latencies_us;
                latencies.reserve(static_cast<std::size_t>(timing_iters_per_stream));
                ready_count.fetch_add(1, std::memory_order_release);

                while (!start_timing.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }

                for (int i = 0; i < timing_iters_per_stream; ++i) {
                    const auto begin = std::chrono::steady_clock::now();
                    (void)module.forward(inputs);
                    const auto end = std::chrono::steady_clock::now();
                    const double elapsed_us =
                        std::chrono::duration<double, std::micro>(end - begin).count();
                    latencies.push_back(elapsed_us);
                }
            } catch (const std::exception& ex) {
                results[static_cast<std::size_t>(worker_id)].error = ex.what();
                ready_count.fetch_add(1, std::memory_order_release);
            } catch (...) {
                results[static_cast<std::size_t>(worker_id)].error = "unknown exception";
                ready_count.fetch_add(1, std::memory_order_release);
            }
        });
    }

    while (ready_count.load(std::memory_order_acquire) < concurrency) {
        std::this_thread::yield();
    }
    start_timing.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    return results;
}

void print_usage() {
    std::cerr
        << "usage: cpp_libtorch_concurrent_benchmark <torchscript_model.pt> "
           "<out_json> <raw_latency_csv> <concurrency> [rounds] "
           "[timing_iters_per_stream] [warmup_iters] [torch_threads]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage();
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string out_json_path = argv[2];
    const std::string raw_csv_path = argv[3];
    const int concurrency = std::stoi(argv[4]);
    const int rounds = argc >= 6 ? std::stoi(argv[5]) : kDefaultRounds;
    const int timing_iters_per_stream =
        argc >= 7 ? std::stoi(argv[6]) : kDefaultTimingItersPerStream;
    const int warmup_iters = argc >= 8 ? std::stoi(argv[7]) : kDefaultWarmupIters;
    const int torch_threads = argc >= 9 ? std::stoi(argv[8]) : kDefaultTorchThreads;

    if (concurrency <= 0 || rounds <= 0 || timing_iters_per_stream <= 0 ||
        warmup_iters < 0 || torch_threads <= 0) {
        std::cerr << "all numeric arguments must be positive; warmup may be zero\n";
        return 1;
    }

    torch::set_num_threads(torch_threads);
    torch::set_num_interop_threads(torch_threads);

    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv) {
        std::cerr << "failed to open raw latency CSV: " << raw_csv_path << "\n";
        return 1;
    }
    raw_csv << "round,stream,iteration,latency_us,deadline_us,deadline_miss\n";

    struct RoundSummary {
        int round = 0;
        Stats stats;
    };
    std::vector<RoundSummary> round_summaries;
    round_summaries.reserve(static_cast<std::size_t>(rounds));

    for (int round = 1; round <= rounds; ++round) {
        auto worker_results =
            run_round(model_path, concurrency, warmup_iters, timing_iters_per_stream);

        std::vector<double> round_samples;
        round_samples.reserve(static_cast<std::size_t>(concurrency) *
                              static_cast<std::size_t>(timing_iters_per_stream));

        for (int worker_id = 0; worker_id < concurrency; ++worker_id) {
            const auto& worker = worker_results[static_cast<std::size_t>(worker_id)];
            if (!worker.error.empty()) {
                std::cerr << "worker " << worker_id << " failed: " << worker.error << "\n";
                return 2;
            }
            for (std::size_t i = 0; i < worker.latencies_us.size(); ++i) {
                const double latency_us = worker.latencies_us[i];
                raw_csv << round << ',' << worker_id << ',' << i << ','
                        << std::fixed << std::setprecision(6) << latency_us << ','
                        << kDefaultDeadlineUs << ','
                        << (latency_us > kDefaultDeadlineUs ? 1 : 0) << '\n';
                round_samples.push_back(latency_us);
            }
        }

        const auto stats = summarize(std::move(round_samples), kDefaultDeadlineUs);
        round_summaries.push_back({round, stats});
        std::cout << "N=" << concurrency << " round=" << round
                  << " samples=" << stats.sample_count
                  << " mean_us=" << stats.mean_us
                  << " p50_us=" << stats.p50_us
                  << " p95_us=" << stats.p95_us
                  << " p99_us=" << stats.p99_us
                  << " min_us=" << stats.min_us
                  << " miss_rate=" << stats.deadline_miss_rate << "\n";
    }

    std::ofstream os(out_json_path);
    if (!os) {
        std::cerr << "failed to open output JSON: " << out_json_path << "\n";
        return 1;
    }

    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"device\": \"CPU\",\n";
    os << "  \"model\": \"PHC-lite step model (TorchScript float32)\",\n";
    os << "  \"backend\": \"libtorch_cpp_concurrent\",\n";
    os << "  \"batch_size\": 1,\n";
    os << "  \"concurrency\": " << concurrency << ",\n";
    os << "  \"rounds\": " << rounds << ",\n";
    os << "  \"warmup_iters_per_stream\": " << warmup_iters << ",\n";
    os << "  \"timing_iters_per_stream\": " << timing_iters_per_stream << ",\n";
    os << "  \"samples_per_round\": "
       << static_cast<long long>(concurrency) * timing_iters_per_stream << ",\n";
    os << "  \"deadline_us\": " << kDefaultDeadlineUs << ",\n";
    os << "  \"torch_no_grad\": true,\n";
    os << "  \"new5_used\": false,\n";
    os << "  \"torch_num_threads\": " << torch_threads << ",\n";
    os << "  \"torch_num_interop_threads\": " << torch_threads << ",\n";
    os << "  \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n";
    os << "  \"cpu\": \"" << json_escape(getenv_string("PROCESSOR_IDENTIFIER")) << "\",\n";
    os << "  \"torchscript_model\": \"" << json_escape(model_path) << "\",\n";
    os << "  \"raw_latency_csv\": \"" << json_escape(raw_csv_path) << "\",\n";
    os << "  \"stream_module_instances\": \"one TorchScript module loaded per concurrent stream\",\n";
    os << "  \"round_summaries\": [\n";

    for (std::size_t i = 0; i < round_summaries.size(); ++i) {
        const auto& summary = round_summaries[i];
        os << "    {\n";
        os << "      \"round\": " << summary.round << ",\n";
        os << "      \"sample_count\": " << summary.stats.sample_count << ",\n";
        os << "      \"mean_us\": " << summary.stats.mean_us << ",\n";
        os << "      \"p50_us\": " << summary.stats.p50_us << ",\n";
        os << "      \"p95_us\": " << summary.stats.p95_us << ",\n";
        os << "      \"p99_us\": " << summary.stats.p99_us << ",\n";
        os << "      \"min_us\": " << summary.stats.min_us << ",\n";
        os << "      \"max_us\": " << summary.stats.max_us << ",\n";
        os << "      \"deadline_miss_count\": " << summary.stats.deadline_miss_count << ",\n";
        os << "      \"deadline_miss_rate\": " << summary.stats.deadline_miss_rate << "\n";
        os << "    }" << (i + 1 == round_summaries.size() ? "\n" : ",\n");
    }

    os << "  ]\n";
    os << "}\n";

    return 0;
}
