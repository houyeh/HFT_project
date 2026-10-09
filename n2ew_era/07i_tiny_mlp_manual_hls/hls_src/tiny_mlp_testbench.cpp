#include "tiny_mlp_hls.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) {
        cells.push_back(cell);
    }
    return cells;
}

long long parse_ll(const std::vector<std::string>& cells, int index) {
    return std::strtoll(cells[index].c_str(), 0, 10);
}

tiny_data_t data_from_raw(long long raw) {
    tiny_data_t value;
    value.range(27, 0) = (ap_uint<28>)raw;
    return value;
}

long long acc_raw(const tiny_acc_t& value) {
    ap_int<40> raw = value.range(39, 0);
    return raw.to_int64();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tiny_mlp_testbench <vectors.csv> <results.csv>\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) {
        std::cerr << "failed to open input or output file\n";
        return 2;
    }

    std::string line;
    std::getline(input, line);
    output << "row_id,ref_logit0_q,ref_logit1_q,ref_logit2_q,dut_logit0_q,dut_logit1_q,dut_logit2_q,ref_action,dut_action,pass\n";

    long long total = 0;
    long long mismatches = 0;
    long long max_abs_raw_diff = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> cells = split_csv(line);
        if (cells.size() != 21) {
            std::cerr << "malformed row " << total << ": columns=" << cells.size() << "\n";
            return 2;
        }
        tiny_data_t x[TINY_MLP_INPUT_DIM];
        tiny_acc_t logits[TINY_MLP_OUTPUT_DIM];
        tiny_action_t action;
        for (int i = 0; i < TINY_MLP_INPUT_DIM; ++i) {
            x[i] = data_from_raw(parse_ll(cells, 1 + i));
        }
        long long reference[TINY_MLP_OUTPUT_DIM];
        for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) {
            reference[o] = parse_ll(cells, 17 + o);
        }
        const int reference_action = (int)parse_ll(cells, 20);

        tiny_mlp_infer(x, logits, action);

        bool pass = ((int)action == reference_action);
        long long actual[TINY_MLP_OUTPUT_DIM];
        for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) {
            actual[o] = acc_raw(logits[o]);
            long long difference = actual[o] - reference[o];
            if (difference < 0) difference = -difference;
            if (difference > max_abs_raw_diff) max_abs_raw_diff = difference;
            if (actual[o] != reference[o]) pass = false;
        }
        if (!pass) ++mismatches;
        output << total;
        for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) output << ',' << reference[o];
        for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) output << ',' << actual[o];
        output << ',' << reference_action << ',' << (int)action << ',' << (pass ? 1 : 0) << '\n';
        ++total;
    }

    std::cout << "total=" << total << " mismatches=" << mismatches
              << " max_abs_raw_diff=" << max_abs_raw_diff << "\n";
    return mismatches == 0 ? 0 : 1;
}
