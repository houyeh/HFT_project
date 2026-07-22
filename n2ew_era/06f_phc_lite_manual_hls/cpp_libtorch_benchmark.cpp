#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWarmupIters = 100;
constexpr int kTimingIters = 10000;

struct Stats {
    double mean_us;
    double median_us;
    double p99_us;
    double min_us;
};

Stats summarize(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::runtime_error("No timing samples collected.");
    }
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    std::sort(samples.begin(), samples.end());
    const std::size_t p99_idx = std::min(samples.size() - 1, static_cast<std::size_t>(0.99 * samples.size()));
    return Stats{
        sum / static_cast<double>(samples.size()),
        samples[samples.size() / 2],
        samples[p99_idx],
        samples.front(),
    };
}

std::string json_escape(const std::string& value) {
    std::ostringstream os;
    for (const char c : value) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    return os.str();
}

std::string cpu_brand_from_env() {
#if defined(_WIN32)
    const char* identifier = std::getenv("PROCESSOR_IDENTIFIER");
    if (identifier != nullptr) {
        return std::string(identifier);
    }
#endif
    return "unknown";
}

bool is_integer_arg(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    std::size_t pos = 0;
    if (value[0] == '+' || value[0] == '-') {
        pos = 1;
    }
    if (pos >= value.size()) {
        return false;
    }
    for (; pos < value.size(); ++pos) {
        if (value[pos] < '0' || value[pos] > '9') {
            return false;
        }
    }
    return true;
}

void write_raw_latency_csv(const std::string& raw_csv, const std::vector<double>& samples_us) {
    if (raw_csv.empty()) {
        return;
    }
    std::ofstream os(raw_csv);
    if (!os) {
        throw std::runtime_error("Could not open raw latency CSV for writing: " + raw_csv);
    }
    os << "iteration,latency_us\n";
    os.setf(std::ios::fixed);
    os.precision(9);
    for (std::size_t i = 0; i < samples_us.size(); ++i) {
        os << i << "," << samples_us[i] << "\n";
    }
}

}  // namespace

int main(int argc, const char** argv) {
    if (argc < 3) {
        std::cerr << "usage: cpp_libtorch_benchmark <torchscript_model.pt> <out_json> [raw_latency_csv] [num_threads]\n";
        std::cerr << "       cpp_libtorch_benchmark <torchscript_model.pt> <out_json> [num_threads]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string out_json = argv[2];
    std::string raw_csv;
    int num_threads = 8;
    if (argc >= 4) {
        const std::string third_arg = argv[3];
        if (is_integer_arg(third_arg)) {
            num_threads = std::stoi(third_arg);
        } else {
            raw_csv = third_arg;
        }
    }
    if (argc >= 5) {
        num_threads = std::stoi(argv[4]);
    }

    torch::set_num_threads(num_threads);
    torch::set_num_interop_threads(num_threads);

    torch::jit::script::Module module = torch::jit::load(model_path, torch::kCPU);
    module.eval();
    torch::Tensor x_cat = torch::zeros({1, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x_cat);

    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < kWarmupIters; ++i) {
            (void)module.forward(inputs);
        }
    }

    std::vector<double> samples_us;
    samples_us.reserve(kTimingIters);
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < kTimingIters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void)module.forward(inputs);
            const auto t1 = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::micro> elapsed = t1 - t0;
            samples_us.push_back(elapsed.count());
        }
    }
    const Stats stats = summarize(samples_us);
    write_raw_latency_csv(raw_csv, samples_us);

    std::ofstream os(out_json);
    os << "{\n";
    os << "  \"device\": \"CPU\",\n";
    os << "  \"model\": \"PHC-lite step model (TorchScript float32)\",\n";
    os << "  \"backend\": \"libtorch_cpp\",\n";
    os << "  \"batch_size\": 1,\n";
    os << "  \"warmup_iters\": " << kWarmupIters << ",\n";
    os << "  \"timing_iters\": " << kTimingIters << ",\n";
    os << "  \"mean_us\": " << stats.mean_us << ",\n";
    os << "  \"median_us\": " << stats.median_us << ",\n";
    os << "  \"p99_us\": " << stats.p99_us << ",\n";
    os << "  \"min_us\": " << stats.min_us << ",\n";
    os << "  \"torch_no_grad\": true,\n";
    os << "  \"new5_used\": false,\n";
    os << "  \"torch_num_threads\": " << torch::get_num_threads() << ",\n";
    os << "  \"torch_num_interop_threads\": " << torch::get_num_interop_threads() << ",\n";
    os << "  \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n";
    os << "  \"cpu\": \"" << json_escape(cpu_brand_from_env()) << "\",\n";
    os << "  \"torchscript_model\": \"" << json_escape(model_path) << "\",\n";
    os << "  \"raw_latency_csv\": \"" << json_escape(raw_csv) << "\"\n";
    os << "}\n";
    os.close();

    std::cout << "mean_us=" << stats.mean_us
              << " median_us=" << stats.median_us
              << " p99_us=" << stats.p99_us
              << " min_us=" << stats.min_us << "\n";
    return 0;
}
