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

int sign_with_zero_positive(double x) {
    return x < 0.0 ? -1 : 1;
}

struct BlockRefOffsets {
    int block0;
    int block1;
    int block2;
    int block3;
    int block4;
    int readout;
};

int block_len(int block) {
    switch (block) {
        case 0:
            return TCN_STAGE2_BLOCK0_OUT;
        case 1:
            return TCN_STAGE2_BLOCK1_OUT;
        case 2:
            return TCN_STAGE2_BLOCK2_OUT;
        case 3:
            return TCN_STAGE2_BLOCK3_OUT;
        default:
            return TCN_STAGE2_BLOCK4_OUT;
    }
}

int block_start_event(int block) {
    switch (block) {
        case 0:
            return 4;
        case 1:
            return 12;
        case 2:
            return 28;
        case 3:
            return 60;
        default:
            return 124;
    }
}

void compare_block_row(
    std::ofstream& out,
    const std::vector<std::string>& cells,
    int base_col,
    const char* stage,
    long long row_id,
    long long sample_idx,
    int out_t,
    const tcn_fixed_t dut[TCN_CHANNELS],
    long long& compared_rows,
    long long& diff_count,
    long long& warning_rows,
    double& max_abs_diff_global,
    double& sum_abs_diff,
    double& sum_sq_diff
) {
    double max_abs_diff_row = 0.0;
    double sum_sq_row = 0.0;
    out << row_id << "," << sample_idx << "," << stage << "," << out_t;
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        const double ref = parse_double_cell(cells, base_col + out_t * TCN_CHANNELS + c);
        const double dut_value = static_cast<double>(dut[c]);
        const double diff = std::fabs(dut_value - ref);
        if (diff > max_abs_diff_row) {
            max_abs_diff_row = diff;
        }
        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;
        sum_sq_row += diff * diff;
        diff_count++;
        out << "," << ref << "," << dut_value << "," << diff;
    }
    const double rmse_row = std::sqrt(sum_sq_row / static_cast<double>(TCN_CHANNELS));
    out << "," << max_abs_diff_row << "," << rmse_row << "\n";
    if (max_abs_diff_row > max_abs_diff_global) {
        max_abs_diff_global = max_abs_diff_row;
    }
    if (max_abs_diff_row > TCN_STAGE2_WARN_ABS_TOL) {
        warning_rows++;
    }
    compared_rows++;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tcn_stage2_streaming_testbench <stage2_vectors.csv> <block_output.csv> <readout_output.csv>\n";
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open input: " << argv[1] << "\n";
        return 2;
    }
    std::ofstream block_out(argv[2]);
    if (!block_out) {
        std::cerr << "failed to open block output: " << argv[2] << "\n";
        return 2;
    }
    std::ofstream readout_out(argv[3]);
    if (!readout_out) {
        std::cerr << "failed to open readout output: " << argv[3] << "\n";
        return 2;
    }

    std::string header;
    std::getline(in, header);

    block_out << std::setprecision(10);
    readout_out << std::setprecision(10);

    block_out << "row_id,sample_idx,stage,out_t";
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        block_out << ",ref_c" << c << ",dut_c" << c << ",abs_diff_c" << c;
    }
    block_out << ",max_abs_diff,rmse_timestep\n";

    readout_out << "row_id,sample_idx";
    for (int i = 0; i < TCN_READOUT_DIM; ++i) {
        readout_out << ",ref_readout_" << i << ",dut_readout_" << i << ",abs_diff_readout_" << i;
    }
    readout_out << ",max_abs_diff,rmse,sign_agree_0,sign_agree_1,sign_agree_joint,valid_seen\n";

    const int input_cols = TCN_SEQ_LEN * TCN_FEATURE_DIM;
    BlockRefOffsets offsets;
    offsets.block0 = 2 + input_cols;
    offsets.block1 = offsets.block0 + TCN_STAGE2_BLOCK0_OUT * TCN_CHANNELS;
    offsets.block2 = offsets.block1 + TCN_STAGE2_BLOCK1_OUT * TCN_CHANNELS;
    offsets.block3 = offsets.block2 + TCN_STAGE2_BLOCK2_OUT * TCN_CHANNELS;
    offsets.block4 = offsets.block3 + TCN_STAGE2_BLOCK3_OUT * TCN_CHANNELS;
    offsets.readout = offsets.block4 + TCN_STAGE2_BLOCK4_OUT * TCN_CHANNELS;
    const int expected_cols = offsets.readout + TCN_READOUT_DIM;

    long long samples = 0;
    long long block_rows = 0;
    long long readout_rows = 0;
    long long malformed_valid = 0;
    long long warning_rows = 0;
    long long diff_count = 0;
    long long sign_agree0 = 0;
    long long sign_agree1 = 0;
    long long sign_agree_joint = 0;
    double max_abs_diff_global = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    double readout_max_abs_diff_global = 0.0;
    double readout_sum_sq_diff = 0.0;
    double readout_sum_abs_diff = 0.0;
    long long readout_diff_count = 0;

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

        const long long row_id = parse_ll_cell(cells, 0);
        const long long sample_idx = parse_ll_cell(cells, 1);
        tcn_fixed_t in_vec[TCN_CHANNELS];
        tcn_fixed_t readout[TCN_READOUT_DIM];
        bool readout_valid[1];
        tcn_fixed_t b0[TCN_CHANNELS];
        tcn_fixed_t b1[TCN_CHANNELS];
        tcn_fixed_t b2[TCN_CHANNELS];
        tcn_fixed_t b3[TCN_CHANNELS];
        tcn_fixed_t b4[TCN_CHANNELS];
        bool v0[1];
        bool v1[1];
        bool v2[1];
        bool v3[1];
        bool v4[1];

        zero_vec(in_vec);
        tcn_streaming_step(in_vec, true, readout, readout_valid, b0, v0, b1, v1, b2, v2, b3, v3, b4, v4);

        int out_idx[5] = {0, 0, 0, 0, 0};
        int readout_valid_count = 0;
        for (int event_t = 0; event_t < TCN_SEQ_LEN_IN; ++event_t) {
            if (event_t < TCN_PREPAD) {
                zero_vec(in_vec);
            } else {
                const int src_t = event_t - TCN_PREPAD;
                int col = 2 + src_t * TCN_FEATURE_DIM;
                for (int c = 0; c < TCN_FEATURE_DIM; ++c) {
                    in_vec[c] = (tcn_fixed_t)parse_double_cell(cells, col + c);
                }
            }

            tcn_streaming_step(in_vec, false, readout, readout_valid, b0, v0, b1, v1, b2, v2, b3, v3, b4, v4);

            const bool expected_v0 = event_t >= block_start_event(0);
            const bool expected_v1 = event_t >= block_start_event(1);
            const bool expected_v2 = event_t >= block_start_event(2);
            const bool expected_v3 = event_t >= block_start_event(3);
            const bool expected_v4 = event_t >= block_start_event(4);
            if (v0[0] != expected_v0 || v1[0] != expected_v1 || v2[0] != expected_v2 ||
                v3[0] != expected_v3 || v4[0] != expected_v4) {
                malformed_valid++;
            }

            if (v0[0]) {
                compare_block_row(block_out, cells, offsets.block0, "block0", row_id, sample_idx, out_idx[0]++, b0,
                                  block_rows, diff_count, warning_rows, max_abs_diff_global, sum_abs_diff, sum_sq_diff);
            }
            if (v1[0]) {
                compare_block_row(block_out, cells, offsets.block1, "block1", row_id, sample_idx, out_idx[1]++, b1,
                                  block_rows, diff_count, warning_rows, max_abs_diff_global, sum_abs_diff, sum_sq_diff);
            }
            if (v2[0]) {
                compare_block_row(block_out, cells, offsets.block2, "block2", row_id, sample_idx, out_idx[2]++, b2,
                                  block_rows, diff_count, warning_rows, max_abs_diff_global, sum_abs_diff, sum_sq_diff);
            }
            if (v3[0]) {
                compare_block_row(block_out, cells, offsets.block3, "block3", row_id, sample_idx, out_idx[3]++, b3,
                                  block_rows, diff_count, warning_rows, max_abs_diff_global, sum_abs_diff, sum_sq_diff);
            }
            if (v4[0]) {
                compare_block_row(block_out, cells, offsets.block4, "block4", row_id, sample_idx, out_idx[4]++, b4,
                                  block_rows, diff_count, warning_rows, max_abs_diff_global, sum_abs_diff, sum_sq_diff);
            }

            if (readout_valid[0]) {
                readout_valid_count++;
                double max_abs_diff_readout = 0.0;
                double sum_sq_readout_row = 0.0;
                readout_out << row_id << "," << sample_idx;
                for (int i = 0; i < TCN_READOUT_DIM; ++i) {
                    const double ref = parse_double_cell(cells, offsets.readout + i);
                    const double dut = static_cast<double>(readout[i]);
                    const double diff = std::fabs(dut - ref);
                    if (diff > max_abs_diff_readout) {
                        max_abs_diff_readout = diff;
                    }
                    readout_sum_abs_diff += diff;
                    readout_sum_sq_diff += diff * diff;
                    sum_sq_readout_row += diff * diff;
                    readout_diff_count++;
                    readout_out << "," << ref << "," << dut << "," << diff;
                }
                const double ref0 = parse_double_cell(cells, offsets.readout);
                const double ref1 = parse_double_cell(cells, offsets.readout + 1);
                const double dut0 = static_cast<double>(readout[0]);
                const double dut1 = static_cast<double>(readout[1]);
                const int agree0 = sign_with_zero_positive(ref0) == sign_with_zero_positive(dut0) ? 1 : 0;
                const int agree1 = sign_with_zero_positive(ref1) == sign_with_zero_positive(dut1) ? 1 : 0;
                const int agree_joint = agree0 && agree1 ? 1 : 0;
                sign_agree0 += agree0;
                sign_agree1 += agree1;
                sign_agree_joint += agree_joint;
                if (max_abs_diff_readout > readout_max_abs_diff_global) {
                    readout_max_abs_diff_global = max_abs_diff_readout;
                }
                readout_out << "," << max_abs_diff_readout << ","
                            << std::sqrt(sum_sq_readout_row / static_cast<double>(TCN_READOUT_DIM)) << ","
                            << agree0 << "," << agree1 << "," << agree_joint << ",1\n";
                readout_rows++;
            }
        }

        const int expected_counts[5] = {TCN_STAGE2_BLOCK0_OUT, TCN_STAGE2_BLOCK1_OUT, TCN_STAGE2_BLOCK2_OUT,
                                        TCN_STAGE2_BLOCK3_OUT, TCN_STAGE2_BLOCK4_OUT};
        for (int b = 0; b < 5; ++b) {
            if (out_idx[b] != expected_counts[b]) {
                std::cerr << "unexpected block" << b << " valid count for sample " << sample_idx
                          << ": got " << out_idx[b] << " expected " << expected_counts[b] << "\n";
                return 2;
            }
        }
        if (readout_valid_count != 1) {
            std::cerr << "unexpected readout valid count for sample " << sample_idx << ": "
                      << readout_valid_count << " expected 1\n";
            return 2;
        }
        samples++;
    }

    const double mean_abs_diff = diff_count > 0 ? sum_abs_diff / static_cast<double>(diff_count) : 0.0;
    const double rmse = diff_count > 0 ? std::sqrt(sum_sq_diff / static_cast<double>(diff_count)) : 0.0;
    const double readout_mean_abs_diff = readout_diff_count > 0 ? readout_sum_abs_diff / static_cast<double>(readout_diff_count) : 0.0;
    const double readout_rmse = readout_diff_count > 0 ? std::sqrt(readout_sum_sq_diff / static_cast<double>(readout_diff_count)) : 0.0;

    std::cout << "samples=" << samples << " block_rows=" << block_rows
              << " block_compared_values=" << diff_count << "\n";
    std::cout << "block_max_abs_diff=" << max_abs_diff_global << " block_mean_abs_diff=" << mean_abs_diff
              << " block_rmse=" << rmse << " warning_abs_tol=" << TCN_STAGE2_WARN_ABS_TOL
              << " warning_rows=" << warning_rows << "\n";
    std::cout << "readout_rows=" << readout_rows << " readout_max_abs_diff=" << readout_max_abs_diff_global
              << " readout_mean_abs_diff=" << readout_mean_abs_diff << " readout_rmse=" << readout_rmse << "\n";
    std::cout << "sign_agreement_readout_0=" << (readout_rows ? (double)sign_agree0 / readout_rows : 0.0)
              << " sign_agreement_readout_1=" << (readout_rows ? (double)sign_agree1 / readout_rows : 0.0)
              << " joint=" << (readout_rows ? (double)sign_agree_joint / readout_rows : 0.0)
              << " zero_sign_convention=positive\n";
    std::cout << "valid_alignment_errors=" << malformed_valid << "\n";

    return (malformed_valid == 0 && warning_rows == 0) ? 0 : 1;
}
