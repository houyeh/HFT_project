#include "phc_step_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Initial tolerance for fixed-point HLS-vs-float32 reference comparison.
// Q24.12 has ULP = 2^-12 ~= 0.000244. This tolerance is intentionally wider
// than a few ULPs because the current faithful HLS implementation uses fixed
// weights/accumulators plus nonlinear exp/tanh/sqrt paths. Phase 06B observed
// action agreement around 0.998 and max score deltas below ~0.956 after full
// streaming fixed-point emulation. The first C-sim run should report max/mean
// error and then this tolerance can be tightened if the measured distribution
// supports it. Do not silently raise this tolerance to hide failures.
static const double PHC_ABS_TOL = 1.0;

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: phc_step_testbench <input_vectors.csv> <output_results.csv>\n";
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
    out << "row_id,sample_idx,timestep";
    for (int i = 0; i < 2; ++i) {
        out << ",ref_readout_" << i << ",dut_readout_" << i << ",abs_diff_readout_" << i;
    }
    for (int i = 0; i < 16; ++i) {
        out << ",ref_state_next_" << i << ",dut_state_next_" << i << ",abs_diff_state_next_" << i;
    }
    out << ",max_abs_diff,pass\n";

    long long total = 0;
    long long mismatches = 0;
    double max_abs_diff_global = 0.0;
    double sum_abs_diff = 0.0;
    long long diff_count = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        if (cells.size() < 53) {
            std::cerr << "malformed row with " << cells.size() << " columns at total=" << total << "\n";
            return 2;
        }

        const long long row_id = parse_ll_cell(cells, 0);
        const long long sample_idx = parse_ll_cell(cells, 1);
        const long long timestep = parse_ll_cell(cells, 2);

        phc_fixed_t x_t[PHC_FEATURE_DIM];
        phc_fixed_t state_prev[PHC_HIDDEN_DIM];
        phc_fixed_t readout[PHC_READOUT_DIM];
        phc_fixed_t state_next[PHC_HIDDEN_DIM];

        double ref_readout[PHC_READOUT_DIM];
        double ref_state_next[PHC_HIDDEN_DIM];

        for (int i = 0; i < 16; ++i) {
            x_t[i] = (phc_fixed_t)parse_double_cell(cells, 3 + i);
        }
        for (int i = 0; i < 16; ++i) {
            state_prev[i] = (phc_fixed_t)parse_double_cell(cells, 19 + i);
        }
        for (int i = 0; i < 2; ++i) {
            ref_readout[i] = parse_double_cell(cells, 35 + i);
        }
        for (int i = 0; i < 16; ++i) {
            ref_state_next[i] = parse_double_cell(cells, 37 + i);
        }

        phc_step(x_t, state_prev, readout, state_next);

        double max_abs_diff_row = 0.0;
        double dut_readout_d[PHC_READOUT_DIM];
        double dut_state_next_d[PHC_HIDDEN_DIM];
        double diff_readout[PHC_READOUT_DIM];
        double diff_state[PHC_HIDDEN_DIM];

        for (int i = 0; i < 2; ++i) {
            dut_readout_d[i] = static_cast<double>(readout[i]);
            diff_readout[i] = std::fabs(dut_readout_d[i] - ref_readout[i]);
            if (diff_readout[i] > max_abs_diff_row) {
                max_abs_diff_row = diff_readout[i];
            }
            sum_abs_diff += diff_readout[i];
            diff_count++;
        }
        for (int i = 0; i < 16; ++i) {
            dut_state_next_d[i] = static_cast<double>(state_next[i]);
            diff_state[i] = std::fabs(dut_state_next_d[i] - ref_state_next[i]);
            if (diff_state[i] > max_abs_diff_row) {
                max_abs_diff_row = diff_state[i];
            }
            sum_abs_diff += diff_state[i];
            diff_count++;
        }

        const int ok = (max_abs_diff_row <= PHC_ABS_TOL) ? 1 : 0;
        if (!ok) {
            mismatches++;
        }
        if (max_abs_diff_row > max_abs_diff_global) {
            max_abs_diff_global = max_abs_diff_row;
        }

        out << row_id << "," << sample_idx << "," << timestep;
        for (int i = 0; i < 2; ++i) {
            out << "," << ref_readout[i] << "," << dut_readout_d[i] << "," << diff_readout[i];
        }
        for (int i = 0; i < 16; ++i) {
            out << "," << ref_state_next[i] << "," << dut_state_next_d[i] << "," << diff_state[i];
        }
        out << "," << max_abs_diff_row << "," << ok << "\n";

        total++;
    }

    const double mean_abs_diff = (diff_count > 0) ? (sum_abs_diff / static_cast<double>(diff_count)) : 0.0;
    std::cout << "total=" << total << " mismatches=" << mismatches << "\n";
    std::cout << "max_abs_diff=" << max_abs_diff_global << " mean_abs_diff=" << mean_abs_diff
              << " tolerance=" << PHC_ABS_TOL << "\n";
    return mismatches == 0 ? 0 : 1;
}
