#include "fi2010_phc_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static const double FI_ABS_TOL = 1.0;

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
        std::cerr << "usage: fi2010_phc_testbench <input_vectors.csv> <output_results.csv>\n";
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
    for (int i = 0; i < 3; ++i) {
        out << ",ref_logit_" << i << ",dut_logit_" << i << ",abs_diff_logit_" << i;
    }
    for (int i = 0; i < 16; ++i) {
        out << ",ref_state_next_" << i << ",dut_state_next_" << i << ",abs_diff_state_next_" << i;
    }
    out << ",max_abs_diff,pass\n";

    long long total = 0;
    long long mismatches = 0;
    long long sign_agree[3] = {0, 0, 0};
    long long joint_sign_agree = 0;
    double max_abs_diff_global = 0.0;
    double sum_abs_diff = 0.0;
    long long diff_count = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        if (cells.size() < 78) {
            std::cerr << "malformed row with " << cells.size() << " columns at total=" << total << "\n";
            return 2;
        }

        const long long row_id = parse_ll_cell(cells, 0);
        const long long sample_idx = parse_ll_cell(cells, 1);
        const long long timestep = parse_ll_cell(cells, 2);

        fi_fixed_t x_t[FI2010_FEATURE_DIM];
        fi_fixed_t state_prev[FI2010_HIDDEN_DIM];
        fi_fixed_t logits[FI2010_READOUT_DIM];
        fi_fixed_t state_next[FI2010_HIDDEN_DIM];

        double ref_logits[FI2010_READOUT_DIM];
        double ref_state_next[FI2010_HIDDEN_DIM];

        for (int i = 0; i < 40; ++i) {
            x_t[i] = (fi_fixed_t)parse_double_cell(cells, 3 + i);
        }
        for (int i = 0; i < 16; ++i) {
            state_prev[i] = (fi_fixed_t)parse_double_cell(cells, 43 + i);
        }
        for (int i = 0; i < 3; ++i) {
            ref_logits[i] = parse_double_cell(cells, 59 + i);
        }
        for (int i = 0; i < 16; ++i) {
            ref_state_next[i] = parse_double_cell(cells, 62 + i);
        }

        fi2010_phc_step(x_t, state_prev, logits, state_next);

        double dut_logits_d[FI2010_READOUT_DIM];
        double dut_state_next_d[FI2010_HIDDEN_DIM];
        double diff_logits[FI2010_READOUT_DIM];
        double diff_state[FI2010_HIDDEN_DIM];
        double max_abs_diff_row = 0.0;

        bool joint_ok = true;
        for (int i = 0; i < 3; ++i) {
            dut_logits_d[i] = static_cast<double>(logits[i]);
            diff_logits[i] = std::fabs(dut_logits_d[i] - ref_logits[i]);
            if (diff_logits[i] > max_abs_diff_row) {
                max_abs_diff_row = diff_logits[i];
            }
            sum_abs_diff += diff_logits[i];
            diff_count++;
            if (sign_with_zero_positive(dut_logits_d[i]) == sign_with_zero_positive(ref_logits[i])) {
                sign_agree[i]++;
            } else {
                joint_ok = false;
            }
        }
        if (joint_ok) {
            joint_sign_agree++;
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

        const int ok = (max_abs_diff_row <= FI_ABS_TOL) ? 1 : 0;
        if (!ok) {
            mismatches++;
        }
        if (max_abs_diff_row > max_abs_diff_global) {
            max_abs_diff_global = max_abs_diff_row;
        }

        out << row_id << "," << sample_idx << "," << timestep;
        for (int i = 0; i < 3; ++i) {
            out << "," << ref_logits[i] << "," << dut_logits_d[i] << "," << diff_logits[i];
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
              << " tolerance=" << FI_ABS_TOL << "\n";
    for (int i = 0; i < 3; ++i) {
        double rate = total > 0 ? (double)sign_agree[i] / (double)total : 0.0;
        std::cout << "sign_agreement_logit_" << i << "=" << rate << "\n";
    }
    std::cout << "joint_sign_agreement=" << (total > 0 ? (double)joint_sign_agree / (double)total : 0.0)
              << " zero_sign_convention=positive\n";
    return mismatches == 0 ? 0 : 1;
}
