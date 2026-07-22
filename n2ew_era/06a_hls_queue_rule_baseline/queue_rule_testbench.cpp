#include "queue_rule_hls.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: queue_rule_testbench <input_vectors.csv> <output_results.csv>\n";
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

    std::string line;
    std::getline(in, line);  // header
    out << "row_id,split,bid_sz_1,ask_sz_1,python_action,cpp_action,match\n";

    long long total = 0;
    long long mismatches = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string row_id_s, split, bid_s, ask_s, py_s;
        std::getline(ss, row_id_s, ',');
        std::getline(ss, split, ',');
        std::getline(ss, bid_s, ',');
        std::getline(ss, ask_s, ',');
        std::getline(ss, py_s, ',');

        const int32_t bid = static_cast<int32_t>(std::strtol(bid_s.c_str(), nullptr, 10));
        const int32_t ask = static_cast<int32_t>(std::strtol(ask_s.c_str(), nullptr, 10));
        const int py = static_cast<int>(std::strtol(py_s.c_str(), nullptr, 10));
        const int cpp = static_cast<int>(queue_rule_action(bid, ask));
        const int ok = (py == cpp) ? 1 : 0;
        if (!ok) {
            mismatches++;
        }
        total++;
        out << row_id_s << "," << split << "," << bid << "," << ask << "," << py << "," << cpp << "," << ok << "\n";
    }
    std::cout << "total=" << total << " mismatches=" << mismatches << "\n";
    return mismatches == 0 ? 0 : 1;
}
