#include "tcn_stage1_block0_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static const double TCN_STAGE1_WARN_ABS_TOL = 1.0;

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

long long parse_ll_cell(const std::vector<std::string>& cells, int idx) {
    if (idx < 0 || idx >= static_cast<int>(cells.size())) {
        return 0;
    }
    return std::strtoll(cells[idx].c_str(), nullptr, 10);
}

void zero_vec(tcn_fixed_t x[TCN_CHANNELS]) {
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        x[c] = (tcn_fixed_t)0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tcn_stage1_block0_testbench <stage1_vectors.csv> <output_results.csv>\n";
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open input: " << argv[1] << "\n";
        return 2;
    }
    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "failed to open output: " << argv[2] << "\n";
        return 2;
    }

    std::string header;
    std::getline(in, header);

    out << std::setprecision(10);
    out << "row_id,sample_idx,out_t";
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        out << ",ref_c" << c << ",dut_c" << c << ",abs_diff_c" << c;
    }
    out << ",max_abs_diff,rmse_timestep,valid\n";

    long long samples = 0;
    long long valid_timesteps = 0;
    long long malformed_valid = 0;
    long long warning_rows = 0;
    long long diff_count = 0;
    double max_abs_diff_global = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        const int expected_cols = 2 + TCN_SEQ_LEN * TCN_FEATURE_DIM + TCN_BLOCK0_VALID_OUTPUTS * TCN_CHANNELS;
        if (static_cast<int>(cells.size()) < expected_cols) {
            std::cerr << "malformed row with " << cells.size() << " columns at sample=" << samples
                      << " expected at least " << expected_cols << "\n";
            return 2;
        }

        const long long row_id = parse_ll_cell(cells, 0);
        const long long sample_idx = parse_ll_cell(cells, 1);
        double x_ref[TCN_SEQ_LEN][TCN_FEATURE_DIM];
        double block0_ref[TCN_BLOCK0_VALID_OUTPUTS][TCN_CHANNELS];
        int col = 2;
        for (int t = 0; t < TCN_SEQ_LEN; ++t) {
            for (int c = 0; c < TCN_FEATURE_DIM; ++c) {
                x_ref[t][c] = parse_double_cell(cells, col++);
            }
        }
        for (int t = 0; t < TCN_BLOCK0_VALID_OUTPUTS; ++t) {
            for (int c = 0; c < TCN_CHANNELS; ++c) {
                block0_ref[t][c] = parse_double_cell(cells, col++);
            }
        }

        tcn_fixed_t in_vec[TCN_CHANNELS];
        tcn_fixed_t out_vec[TCN_CHANNELS];
        bool valid[1];
        zero_vec(in_vec);
        tcn_block0_stream_step(in_vec, true, out_vec, valid);

        int out_t = 0;
        for (int event_t = 0; event_t < TCN_BLOCK0_INPUT_EVENTS; ++event_t) {
            if (event_t < TCN_PREPAD) {
                zero_vec(in_vec);
            } else {
                const int src_t = event_t - TCN_PREPAD;
                for (int c = 0; c < TCN_FEATURE_DIM; ++c) {
                    in_vec[c] = (tcn_fixed_t)x_ref[src_t][c];
                }
            }

            tcn_block0_stream_step(in_vec, false, out_vec, valid);
            const bool expected_valid = event_t >= 4;
            if (valid[0] != expected_valid) {
                malformed_valid++;
            }
            if (!expected_valid) {
                continue;
            }

            double max_abs_diff_row = 0.0;
            double sum_sq_row = 0.0;
            out << row_id << "," << sample_idx << "," << out_t;
            for (int c = 0; c < TCN_CHANNELS; ++c) {
                const double ref = block0_ref[out_t][c];
                const double dut = static_cast<double>(out_vec[c]);
                const double diff = std::fabs(dut - ref);
                if (diff > max_abs_diff_row) {
                    max_abs_diff_row = diff;
                }
                sum_abs_diff += diff;
                sum_sq_diff += diff * diff;
                sum_sq_row += diff * diff;
                diff_count++;
                out << "," << ref << "," << dut << "," << diff;
            }
            const double rmse_row = std::sqrt(sum_sq_row / static_cast<double>(TCN_CHANNELS));
            out << "," << max_abs_diff_row << "," << rmse_row << "," << (valid[0] ? 1 : 0) << "\n";
            if (max_abs_diff_row > max_abs_diff_global) {
                max_abs_diff_global = max_abs_diff_row;
            }
            if (max_abs_diff_row > TCN_STAGE1_WARN_ABS_TOL) {
                warning_rows++;
            }
            valid_timesteps++;
            out_t++;
        }
        if (out_t != TCN_BLOCK0_VALID_OUTPUTS) {
            std::cerr << "unexpected valid output count for sample " << sample_idx << ": " << out_t << "\n";
            return 2;
        }
        samples++;
    }

    const double mean_abs_diff = diff_count > 0 ? sum_abs_diff / static_cast<double>(diff_count) : 0.0;
    const double rmse = diff_count > 0 ? std::sqrt(sum_sq_diff / static_cast<double>(diff_count)) : 0.0;
    std::cout << "samples=" << samples << " valid_timesteps=" << valid_timesteps
              << " compared_values=" << diff_count << "\n";
    std::cout << "max_abs_diff=" << max_abs_diff_global << " mean_abs_diff=" << mean_abs_diff
              << " rmse=" << rmse << " warning_abs_tol=" << TCN_STAGE1_WARN_ABS_TOL
              << " warning_rows=" << warning_rows << "\n";
    std::cout << "valid_alignment_errors=" << malformed_valid << "\n";
    return (malformed_valid == 0 && warning_rows == 0) ? 0 : 1;
}
