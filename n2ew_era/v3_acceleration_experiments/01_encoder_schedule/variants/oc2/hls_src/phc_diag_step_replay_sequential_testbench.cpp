#include "phc_diag_step_hls.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cells.push_back(item);
    }
    return cells;
}

std::map<std::string, int> header_index(const std::string& header) {
    const std::vector<std::string> cells = split_csv_line(header);
    std::map<std::string, int> result;
    for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
        result[cells[i]] = i;
    }
    return result;
}

int require_column(const std::map<std::string, int>& columns, const std::string& name) {
    const std::map<std::string, int>::const_iterator it = columns.find(name);
    if (it == columns.end()) {
        std::cerr << "missing required column: " << name << "\n";
        std::exit(2);
    }
    return it->second;
}

double parse_double(const std::vector<std::string>& cells, int index) {
    if (index < 0 || index >= static_cast<int>(cells.size())) {
        std::cerr << "column index out of range: " << index << "\n";
        std::exit(2);
    }
    return std::strtod(cells[index].c_str(), nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: phc_diag_step_replay_sequential_testbench <features.csv> <output.csv>\n";
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
    if (!std::getline(in, header)) {
        std::cerr << "input CSV is empty\n";
        return 2;
    }
    const std::map<std::string, int> columns = header_index(header);
    const int replay_event_id_col = require_column(columns, "replay_event_id");
    const int event_index_col = require_column(columns, "event_index");
    const int timestamp_us_col = require_column(columns, "timestamp_us");
    int x_cols[PHC_FEATURE_DIM];
    for (int i = 0; i < PHC_FEATURE_DIM; ++i) {
        x_cols[i] = require_column(columns, "x_t_" + std::to_string(i));
    }

    out << std::setprecision(17);
    out << "replay_event_id,event_index,timestamp_us";
    for (int i = 0; i < PHC_READOUT_DIM; ++i) {
        out << ",dut_readout_" << i;
    }
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
        out << ",dut_state_next_" << i;
    }
    out << ",feedback_mode\n";

    phc_fixed_t state_prev[PHC_HIDDEN_DIM];
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
        state_prev[i] = (phc_fixed_t)0.0;
    }

    long long total = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        phc_fixed_t x_t[PHC_FEATURE_DIM];
        phc_fixed_t readout[PHC_READOUT_DIM];
        phc_fixed_t state_next[PHC_HIDDEN_DIM];
        for (int i = 0; i < PHC_FEATURE_DIM; ++i) {
            x_t[i] = (phc_fixed_t)parse_double(cells, x_cols[i]);
        }

        phc_diag_step(x_t, state_prev, readout, state_next);

        out << cells[replay_event_id_col] << "," << cells[event_index_col] << ","
            << cells[timestamp_us_col];
        for (int i = 0; i < PHC_READOUT_DIM; ++i) {
            out << "," << static_cast<double>(readout[i]);
        }
        for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
            out << "," << static_cast<double>(state_next[i]);
            state_prev[i] = state_next[i];
        }
        out << ",hls_v3diag_dut_state_feedback\n";
        ++total;
    }

    std::cout << "total=" << total << "\n";
    std::cout << "feedback_mode=hls_v3diag_dut_state_feedback\n";
    if (total != 4096) {
        std::cerr << "expected 4096 replay rows, got " << total << "\n";
        return 1;
    }
    return 0;
}
