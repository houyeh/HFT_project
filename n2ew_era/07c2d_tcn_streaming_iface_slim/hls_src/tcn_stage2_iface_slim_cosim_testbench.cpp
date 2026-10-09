#include "tcn_stage2_streaming_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static const double TCN_STAGE2_WARN_ABS_TOL = 1.0;

struct Stats {
    long long rows = 0;
    long long values = 0;
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cells.push_back(item);
    }
    return cells;
}

double parse_double_cell(const std::vector<std::string>& cells, int idx) {
    if (idx < 0 || idx >= static_cast<int>(cells.size())) {
        return 0.0;
    }
    return std::strtod(cells[idx].c_str(), nullptr);
}

void zero_vec(tcn_fixed_t x[TCN_CHANNELS]) {
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        x[c] = (tcn_fixed_t)0;
    }
}

int sign_with_zero_positive(double x) {
    return x < 0.0 ? -1 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tcn_stage2_iface_slim_cosim_testbench <stage2_vectors.csv> <summary.csv> [event_limit]\n";
        return 2;
    }
    int event_limit = TCN_SEQ_LEN_IN;
    if (argc >= 4) {
        event_limit = std::atoi(argv[3]);
        if (event_limit <= 0 || event_limit > TCN_SEQ_LEN_IN) {
            std::cerr << "event_limit must be in [1, " << TCN_SEQ_LEN_IN << "], got "
                      << event_limit << "\n";
            return 2;
        }
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open input: " << argv[1] << "\n";
        return 2;
    }

    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "failed to open summary output: " << argv[2] << "\n";
        return 2;
    }
    out << std::setprecision(12);
    out << "scope,stage,rows,compared_values,max_abs_diff,mean_abs_diff,rmse,"
           "sign_agreement_readout_0,sign_agreement_readout_1,sign_agreement_joint,valid_alignment_errors\n";

    std::string header;
    std::getline(in, header);

    const int input_cols = TCN_SEQ_LEN * TCN_FEATURE_DIM;
    const int block0_base = 2 + input_cols;
    const int block1_base = block0_base + TCN_STAGE2_BLOCK0_OUT * TCN_CHANNELS;
    const int block2_base = block1_base + TCN_STAGE2_BLOCK1_OUT * TCN_CHANNELS;
    const int block3_base = block2_base + TCN_STAGE2_BLOCK2_OUT * TCN_CHANNELS;
    const int block4_base = block3_base + TCN_STAGE2_BLOCK3_OUT * TCN_CHANNELS;
    const int readout_base = block4_base + TCN_STAGE2_BLOCK4_OUT * TCN_CHANNELS;
    const int expected_cols = readout_base + TCN_READOUT_DIM;

    long long samples = 0;
    long long valid_alignment_errors = 0;
    long long warning_scopes = 0;
    long long sign_agree0 = 0;
    long long sign_agree1 = 0;
    long long sign_agree_joint = 0;
    Stats readout_stats;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        if (static_cast<int>(cells.size()) < expected_cols) {
            std::cerr << "malformed row with " << cells.size() << " columns at sample=" << samples
                      << " expected at least " << expected_cols << "\n";
            return 2;
        }

        tcn_fixed_t in_vec[TCN_CHANNELS];
        tcn_fixed_t readout[TCN_READOUT_DIM];
        bool readout_valid[1];

        zero_vec(in_vec);
        tcn_streaming_step(in_vec, true, readout, readout_valid);

        int readout_valid_count = 0;
        for (int event_t = 0; event_t < event_limit; ++event_t) {
            if (event_t < TCN_PREPAD) {
                zero_vec(in_vec);
            } else {
                const int src_t = event_t - TCN_PREPAD;
                const int col = 2 + src_t * TCN_FEATURE_DIM;
                for (int c = 0; c < TCN_FEATURE_DIM; ++c) {
                    in_vec[c] = (tcn_fixed_t)parse_double_cell(cells, col + c);
                }
            }

            tcn_streaming_step(in_vec, false, readout, readout_valid);

            const bool expected_valid = event_t == TCN_SEQ_LEN_IN - 1 && event_limit >= TCN_SEQ_LEN_IN;
            if (readout_valid[0] != expected_valid) {
                valid_alignment_errors++;
            }
            if (readout_valid[0]) {
                readout_valid_count++;
                for (int i = 0; i < TCN_READOUT_DIM; ++i) {
                    const double ref = parse_double_cell(cells, readout_base + i);
                    const double value = static_cast<double>(readout[i]);
                    const double diff = std::fabs(value - ref);
                    if (diff > readout_stats.max_abs_diff) {
                        readout_stats.max_abs_diff = diff;
                    }
                    readout_stats.sum_abs_diff += diff;
                    readout_stats.sum_sq_diff += diff * diff;
                    readout_stats.values++;
                }
                const double ref0 = parse_double_cell(cells, readout_base);
                const double ref1 = parse_double_cell(cells, readout_base + 1);
                const double dut0 = static_cast<double>(readout[0]);
                const double dut1 = static_cast<double>(readout[1]);
                const int agree0 = sign_with_zero_positive(ref0) == sign_with_zero_positive(dut0) ? 1 : 0;
                const int agree1 = sign_with_zero_positive(ref1) == sign_with_zero_positive(dut1) ? 1 : 0;
                sign_agree0 += agree0;
                sign_agree1 += agree1;
                sign_agree_joint += agree0 && agree1 ? 1 : 0;
                readout_stats.rows++;
            }
        }

        const int expected_readout_count = event_limit >= TCN_SEQ_LEN_IN ? 1 : 0;
        if (readout_valid_count != expected_readout_count) {
            std::cerr << "unexpected readout valid count at sample " << samples << ": "
                      << readout_valid_count << " expected " << expected_readout_count << "\n";
            return 2;
        }
        samples++;
    }

    const double readout_mean_abs_diff = readout_stats.values > 0
        ? readout_stats.sum_abs_diff / static_cast<double>(readout_stats.values)
        : 0.0;
    const double readout_rmse = readout_stats.values > 0
        ? std::sqrt(readout_stats.sum_sq_diff / static_cast<double>(readout_stats.values))
        : 0.0;
    const double agree0 = readout_stats.rows > 0 ? static_cast<double>(sign_agree0) / readout_stats.rows : 0.0;
    const double agree1 = readout_stats.rows > 0 ? static_cast<double>(sign_agree1) / readout_stats.rows : 0.0;
    const double agree_joint = readout_stats.rows > 0 ? static_cast<double>(sign_agree_joint) / readout_stats.rows : 0.0;
    if (readout_stats.rows > 0 && (readout_stats.max_abs_diff > TCN_STAGE2_WARN_ABS_TOL || agree_joint != 1.0)) {
        warning_scopes++;
    }
    out << "readout,final," << readout_stats.rows << "," << readout_stats.values << ","
        << readout_stats.max_abs_diff << "," << readout_mean_abs_diff << "," << readout_rmse << ","
        << agree0 << "," << agree1 << "," << agree_joint << "," << valid_alignment_errors << "\n";

    std::cout << "samples=" << samples << "\n";
    std::cout << "event_limit=" << event_limit << "\n";
    std::cout << "readout_rows=" << readout_stats.rows << " readout_max_abs_diff="
              << readout_stats.max_abs_diff << " readout_rmse=" << readout_rmse << "\n";
    std::cout << "sign_agreement_readout_0=" << agree0
              << " sign_agreement_readout_1=" << agree1
              << " joint=" << agree_joint << "\n";
    std::cout << "valid_alignment_errors=" << valid_alignment_errors << "\n";

    return (valid_alignment_errors == 0 && warning_scopes == 0) ? 0 : 1;
}
