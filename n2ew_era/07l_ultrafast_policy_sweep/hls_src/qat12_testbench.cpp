#include "qat12_hls.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) cells.push_back(cell);
    return cells;
}
long long parse(const std::vector<std::string>& cells, int index) {
    return std::strtoll(cells[index].c_str(), 0, 10);
}
qat_data_t data_from_raw(long long raw) {
    qat_data_t value;
    value.range(11, 0) = (ap_uint<12>)raw;
    return value;
}
long long acc_raw(const qat_acc_t& value) {
    ap_int<32> raw = value.range(31, 0);
    return raw.to_int64();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) return 2;
    std::string line;
    std::getline(input, line);
    output << "row_id,ref0,ref1,ref2,dut0,dut1,dut2,ref_action,dut_action,pass\n";
    long long total = 0, mismatches = 0, max_diff = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto cells = split(line);
        if (cells.size() != 21) return 2;
        qat_data_t x[QAT_INPUT_DIM];
        qat_acc_t logits[QAT_OUTPUT_DIM];
        qat_action_t action;
        for (int i = 0; i < QAT_INPUT_DIM; ++i) x[i] = data_from_raw(parse(cells, 1 + i));
        long long reference[QAT_OUTPUT_DIM];
        for (int o = 0; o < QAT_OUTPUT_DIM; ++o) reference[o] = parse(cells, 17 + o);
        const int reference_action = (int)parse(cells, 20);
        qat12_tiny_mlp(x, logits, action);
        bool pass = ((int)action == reference_action);
        long long actual[QAT_OUTPUT_DIM];
        for (int o = 0; o < QAT_OUTPUT_DIM; ++o) {
            actual[o] = acc_raw(logits[o]);
            long long diff = actual[o] - reference[o];
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff != 0) pass = false;
        }
        if (!pass) ++mismatches;
        output << total;
        for (int o = 0; o < QAT_OUTPUT_DIM; ++o) output << ',' << reference[o];
        for (int o = 0; o < QAT_OUTPUT_DIM; ++o) output << ',' << actual[o];
        output << ',' << reference_action << ',' << (int)action << ',' << (pass ? 1 : 0) << '\n';
        ++total;
    }
    std::cout << "total=" << total << " mismatches=" << mismatches << " max_abs_raw_diff=" << max_diff << "\n";
    return mismatches == 0 ? 0 : 1;
}
