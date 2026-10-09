#include "tcn_prepad_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static const double TCN_ABS_TOL = 1.0;

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

int sign_with_zero_positive(double x) {
    return x < 0.0 ? -1 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tcn_prepad_testbench <input_vectors.csv> <output_results.csv>\n";
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
    out << "row_id,sample_idx";
    for (int i = 0; i < 2; ++i) {
        out << ",ref_readout_" << i << ",dut_readout_" << i << ",abs_diff_readout_" << i;
    }
    out << ",max_abs_diff,pass\n";

    long long total = 0;
    long long mismatches = 0;
    long long sign_agree0 = 0;
    long long sign_agree1 = 0;
    long long sign_agree_joint = 0;
    double max_abs_diff_global = 0.0;
    double sum_abs_diff = 0.0;
    long long diff_count = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        const int expected_cols = 2 + TCN_SEQ_LEN * TCN_FEATURE_DIM + TCN_READOUT_DIM;
        if (static_cast<int>(cells.size()) < expected_cols) {
            std::cerr << "malformed row with " << cells.size() << " columns at total=" << total
                      << " expected at least " << expected_cols << "\n";
            return 2;
        }
        const long long row_id = parse_ll_cell(cells, 0);
        const long long sample_idx = parse_ll_cell(cells, 1);
        tcn_fixed_t x[TCN_SEQ_LEN][TCN_FEATURE_DIM];
        tcn_fixed_t readout[TCN_READOUT_DIM];
        int col = 2;
        for (int t = 0; t < TCN_SEQ_LEN; ++t) {
            for (int f = 0; f < TCN_FEATURE_DIM; ++f) {
                x[t][f] = (tcn_fixed_t)parse_double_cell(cells, col++);
            }
        }
        double ref[TCN_READOUT_DIM];
        for (int i = 0; i < TCN_READOUT_DIM; ++i) {
            ref[i] = parse_double_cell(cells, col++);
        }

        tcn_prepad_infer(x, readout);

        double dut[TCN_READOUT_DIM];
        double diff[TCN_READOUT_DIM];
        double max_abs_diff_row = 0.0;
        for (int i = 0; i < TCN_READOUT_DIM; ++i) {
            dut[i] = static_cast<double>(readout[i]);
            diff[i] = std::fabs(dut[i] - ref[i]);
            if (diff[i] > max_abs_diff_row) {
                max_abs_diff_row = diff[i];
            }
            sum_abs_diff += diff[i];
            diff_count++;
        }
        const bool agree0 = sign_with_zero_positive(ref[0]) == sign_with_zero_positive(dut[0]);
        const bool agree1 = sign_with_zero_positive(ref[1]) == sign_with_zero_positive(dut[1]);
        sign_agree0 += agree0 ? 1 : 0;
        sign_agree1 += agree1 ? 1 : 0;
        sign_agree_joint += (agree0 && agree1) ? 1 : 0;
        const int ok = max_abs_diff_row <= TCN_ABS_TOL ? 1 : 0;
        mismatches += ok ? 0 : 1;
        if (max_abs_diff_row > max_abs_diff_global) {
            max_abs_diff_global = max_abs_diff_row;
        }
        out << row_id << "," << sample_idx;
        for (int i = 0; i < TCN_READOUT_DIM; ++i) {
            out << "," << ref[i] << "," << dut[i] << "," << diff[i];
        }
        out << "," << max_abs_diff_row << "," << ok << "\n";
        total++;
    }

    const double mean_abs_diff = diff_count > 0 ? sum_abs_diff / static_cast<double>(diff_count) : 0.0;
    std::cout << "total=" << total << " mismatches=" << mismatches << "\n";
    std::cout << "max_abs_diff=" << max_abs_diff_global << " mean_abs_diff=" << mean_abs_diff
              << " tolerance=" << TCN_ABS_TOL << "\n";
    std::cout << "sign_agreement_readout_0=" << (total ? (double)sign_agree0 / total : 0.0)
              << " sign_agreement_readout_1=" << (total ? (double)sign_agree1 / total : 0.0)
              << " joint=" << (total ? (double)sign_agree_joint / total : 0.0)
              << " zero_sign_convention=positive\n";
    return mismatches == 0 ? 0 : 1;
}

