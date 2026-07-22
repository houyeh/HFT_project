from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06m_phc_lite_error_decomposition"
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"
COMMON_SRC = HLS_SRC / "common"
LUT_SRC = HLS_SRC / "lut_only"
QUANT_SRC = HLS_SRC / "quant_only"
LUT_COMPONENT = ROOT / "hls_component_lut_only"
QUANT_COMPONENT = ROOT / "hls_component_quant_only"

PHASE06F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
TEST_VECTOR_CSV = PHASE06F / "results" / "phc_step_test_vectors.csv"
STEP_CKPT_PATH = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml" / "models" / "phc_stp_step_hls_seed2.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phc_stp_step_hls import PHCSTPStepHLS  # noqa: E402

EXPORT_MAP = {
    "enc_linear.weight": "ENC_LINEAR_W",
    "enc_linear.bias": "ENC_LINEAR_B",
    "enc_norm.weight": "ENC_NORM_GAMMA",
    "enc_norm.bias": "ENC_NORM_BETA",
    "gate.weight": "GATE_W",
    "gate.bias": "GATE_B",
    "candidate.weight": "CANDIDATE_W",
    "candidate.bias": "CANDIDATE_B",
    "gain_from_encoded.weight": "GAIN_FROM_ENCODED_W",
    "gain_from_encoded.bias": "GAIN_FROM_ENCODED_B",
    "gain_from_prev_state.weight": "GAIN_FROM_PREV_STATE_W",
    "synapse.weight": "SYNAPSE_W",
    "synapse.bias": "SYNAPSE_B",
    "loop_norm.weight": "LOOP_NORM_GAMMA",
    "loop_norm.bias": "LOOP_NORM_BETA",
    "readout.weight": "READOUT_W",
    "readout.bias": "READOUT_B",
    "decay": "DECAY",
    "alpha_scalar": "ALPHA_SCALAR",
}


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, COMMON_SRC, LUT_SRC, QUANT_SRC, LUT_COMPONENT, QUANT_COMPONENT]:
        path.mkdir(parents=True, exist_ok=True)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_step_state() -> dict[str, np.ndarray]:
    payload = torch_load(STEP_CKPT_PATH)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step checkpoint is marked new5_used=True.")
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    state = {}
    for key, value in model.state_dict().items():
        state[key] = value.detach().cpu().numpy().astype(np.float64)
    alpha = float(np.ravel(state["alpha_scalar"])[0])
    if not (0.58 <= alpha <= 0.61):
        raise RuntimeError(f"Unexpected raw alpha_scalar={alpha:.9f}; refusing to prepare source.")
    return state


def double_literal(value: float) -> str:
    return f"{float(value):.17g}"


def cpp_array_initializer(arr: np.ndarray, indent: int = 0) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    spaces = " " * indent
    if arr.ndim == 0:
        return double_literal(float(arr.item()))
    if arr.ndim == 1:
        return "{ " + ", ".join(double_literal(float(v)) for v in arr.tolist()) + " }"
    if arr.ndim == 2:
        rows = []
        for row in arr:
            rows.append(spaces + "    { " + ", ".join(double_literal(float(v)) for v in row.tolist()) + " }")
        return "{\n" + ",\n".join(rows) + "\n" + spaces + "}"
    raise ValueError(f"Unsupported tensor rank: {arr.ndim}")


def cpp_decl(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return f"static const double {name}[1] = {{ {double_literal(float(arr.item()))} }};"
    if arr.ndim == 1:
        return f"static const double {name}[{arr.shape[0]}] = {cpp_array_initializer(arr)};"
    if arr.ndim == 2:
        return f"static const double {name}[{arr.shape[0]}][{arr.shape[1]}] = {cpp_array_initializer(arr)};"
    raise ValueError(f"Unsupported tensor rank for {name}: {arr.ndim}")


def write_double_weights() -> None:
    state = load_step_state()
    blocks = []
    for tensor_name, cpp_name in EXPORT_MAP.items():
        if tensor_name not in state:
            raise KeyError(f"Missing expected tensor: {tensor_name}")
        blocks.append(f"// {tensor_name}")
        blocks.append(cpp_decl(cpp_name, state[tensor_name]))
        blocks.append("")
    header = [
        "#pragma once",
        "",
        "// Generated for Phase 06M only-LUT-error C-sim control.",
        f"// Source step checkpoint: {STEP_CKPT_PATH}",
        "// Values are unquantized float64 constants derived from the float32 checkpoint tensors.",
        "",
        "namespace phc_step_weights {",
        "",
        "static const int FEATURE_DIM = 16;",
        "static const int HIDDEN_DIM = 16;",
        "static const int INPUT_DIM = 32;",
        "static const int OUTPUT_DIM = 18;",
        "",
        *blocks,
        "}  // namespace phc_step_weights",
        "",
    ]
    (LUT_SRC / "phc_weights_double.hpp").write_text("\n".join(header), encoding="utf-8")


def write_common_testbench() -> None:
    text = r'''#include "phc_step_hls.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same tolerance as Phase 06F testbench. Do not change it for this decomposition.
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
'''
    (COMMON_SRC / "phc_step_testbench.cpp").write_text(text, encoding="utf-8")


def write_lut_only_source() -> None:
    hpp = r'''#pragma once

typedef double phc_fixed_t;
typedef double phc_acc_t;
typedef double phc_var_t;

static const int PHC_FEATURE_DIM = 16;
static const int PHC_HIDDEN_DIM = 16;
static const int PHC_READOUT_DIM = 2;

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
);
'''
    cpp = r'''#include "phc_step_hls.hpp"
#include "phc_weights_double.hpp"

#include <cmath>

namespace {

static const phc_acc_t LN_EPS = 1.0e-5;

phc_acc_t tanh_lut_scalar(phc_acc_t x) {
    static const phc_acc_t TANH_LUT[65] = {
        -0.9993292997, -0.9991388858, -0.9988944427, -0.9985806592,
        -0.9981778976, -0.9976609795, -0.9969976355, -0.9961465307,
        -0.9950547537, -0.9936546343, -0.9918597246, -0.9895597486,
        -0.9866142982, -0.9828450292, -0.9780261147, -0.9718727459,
        -0.9640275801, -0.9540452602, -0.9413755385, -0.9253462253,
        -0.9051482536, -0.8798266997, -0.8482836400, -0.8093010702,
        -0.7615941560, -0.7039056039, -0.6351489524, -0.5545997223,
        -0.4621171573, -0.3583573984, -0.2449186624, -0.1243530018,
        0.0000000000,  0.1243530018,  0.2449186624,  0.3583573984,
        0.4621171573,  0.5545997223,  0.6351489524,  0.7039056039,
        0.7615941560,  0.8093010702,  0.8482836400,  0.8798266997,
        0.9051482536,  0.9253462253,  0.9413755385,  0.9540452602,
        0.9640275801,  0.9718727459,  0.9780261147,  0.9828450292,
        0.9866142982,  0.9895597486,  0.9918597246,  0.9936546343,
        0.9950547537,  0.9961465307,  0.9969976355,  0.9976609795,
        0.9981778976,  0.9985806592,  0.9988944427,  0.9991388858,
        0.9993292997
    };
    if (x <= -4.0) {
        return -1.0;
    }
    if (x >= 4.0) {
        return 1.0;
    }
    phc_acc_t scaled = (x + 4.0) * 8.0;
    int idx = (int)scaled;
    if (idx < 0) {
        idx = 0;
    }
    if (idx > 63) {
        idx = 63;
    }
    phc_acc_t frac = scaled - (phc_acc_t)idx;
    phc_acc_t y0 = TANH_LUT[idx];
    phc_acc_t y1 = TANH_LUT[idx + 1];
    return y0 + frac * (y1 - y0);
}

phc_acc_t sigmoid_scalar(phc_acc_t x) {
    return 0.5 * (tanh_lut_scalar(0.5 * x) + 1.0);
}

void sigmoid_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = tanh_lut_scalar(in[i]);
    }
}

void silu_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_16_16_fixed(const phc_fixed_t in[16], const double w[16][16], const double b[16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const phc_acc_t in[16], const double w[16][16], const double b[16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const phc_fixed_t in[16], const double w[16][16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = 0.0;
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_2_acc(const phc_acc_t in[16], const double w[2][16], const double b[2], phc_acc_t out[2]) {
    for (int j = 0; j < 2; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const phc_acc_t in[16], const double gamma[16], const double beta[16], phc_acc_t out[16]) {
    phc_acc_t mean = 0.0;
    for (int i = 0; i < 16; ++i) {
        mean += in[i];
    }
    mean = mean / 16.0;
    phc_var_t var = 0.0;
    for (int i = 0; i < 16; ++i) {
        phc_acc_t d = in[i] - mean;
        var += d * d;
    }
    var = var / 16.0;
    phc_acc_t inv_std = 1.0 / std::sqrt(var + LN_EPS);
    for (int i = 0; i < 16; ++i) {
        out[i] = gamma[i] * (in[i] - mean) * inv_std + beta[i];
    }
}

}  // namespace

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
) {
    phc_acc_t enc_linear_out[16];
    phc_acc_t encoded_pre[16];
    phc_acc_t encoded[16];
    phc_acc_t gate_pre[16];
    phc_acc_t gate_t[16];
    phc_acc_t candidate_pre[16];
    phc_acc_t candidate_t[16];
    phc_acc_t state_temporal[16];
    phc_acc_t gain_encoded[16];
    phc_acc_t gain_prev_state[16];
    phc_acc_t gain_pre[16];
    phc_acc_t gain_t[16];
    phc_acc_t message_pre[16];
    phc_acc_t message[16];
    phc_acc_t loop_input[16];
    phc_acc_t z[16];
    phc_acc_t readout_acc[2];

    using namespace phc_step_weights;

    linear_16_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);

    linear_16_16_acc(encoded, GATE_W, GATE_B, gate_pre);
    sigmoid_16(gate_pre, gate_t);

    linear_16_16_acc(encoded, CANDIDATE_W, CANDIDATE_B, candidate_pre);
    tanh_16(candidate_pre, candidate_t);

    for (int i = 0; i < 16; ++i) {
        state_temporal[i] = DECAY[i] * state_prev[i] + gate_t[i] * candidate_t[i];
    }

    linear_16_16_acc(encoded, GAIN_FROM_ENCODED_W, GAIN_FROM_ENCODED_B, gain_encoded);
    linear_16_16_fixed_no_bias(state_prev, GAIN_FROM_PREV_STATE_W, gain_prev_state);
    for (int i = 0; i < 16; ++i) {
        gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
    }
    sigmoid_16(gain_pre, gain_t);

    linear_16_16_acc(state_temporal, SYNAPSE_W, SYNAPSE_B, message_pre);
    silu_16(message_pre, message);

    for (int i = 0; i < 16; ++i) {
        loop_input[i] = state_temporal[i] + ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_2_acc(z, READOUT_W, READOUT_B, readout_acc);

    for (int i = 0; i < 2; ++i) {
        readout[i] = readout_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
        state_next[i] = z[i];
    }
}
'''
    (LUT_SRC / "phc_step_hls.hpp").write_text(hpp, encoding="utf-8")
    (LUT_SRC / "phc_step_hls.cpp").write_text(cpp, encoding="utf-8")


def write_quant_only_source() -> None:
    hpp = (PHASE06F / "hls_src" / "phc_step_hls.hpp").read_text(encoding="utf-8")
    cpp = r'''#include "phc_step_hls.hpp"
#include "phc_weights.hpp"

#include <cmath>
#include <hls_math.h>

namespace {

static const phc_acc_t LN_EPS = (phc_acc_t)1.0e-5;

// Quant-only control: keep Q24.12/ap_fixed inputs, weights, accumulators, and
// final casts, but replace the 65-point LUT nonlinearities with real floating
// sigmoid/tanh/SiLU evaluated in double and cast back to phc_acc_t.
phc_acc_t sigmoid_scalar(phc_acc_t x) {
#pragma HLS INLINE
    const double xd = static_cast<double>(x);
    return (phc_acc_t)(1.0 / (1.0 + std::exp(-xd)));
}

phc_acc_t tanh_scalar(phc_acc_t x) {
#pragma HLS INLINE
    return (phc_acc_t)std::tanh(static_cast<double>(x));
}

void sigmoid_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = tanh_scalar(in[i]);
    }
}

void silu_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_16_16_fixed(const phc_fixed_t in[16], const phc_fixed_t w[16][16], const phc_fixed_t b[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const phc_acc_t in[16], const phc_fixed_t w[16][16], const phc_fixed_t b[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const phc_fixed_t in[16], const phc_fixed_t w[16][16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = 0;
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_2_acc(const phc_acc_t in[16], const phc_fixed_t w[2][16], const phc_fixed_t b[2], phc_acc_t out[2]) {
#pragma HLS INLINE
    for (int j = 0; j < 2; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const phc_acc_t in[16], const phc_fixed_t gamma[16], const phc_fixed_t beta[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    phc_acc_t mean = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        mean += in[i];
    }
    mean = mean / (phc_acc_t)16.0;
    phc_var_t var = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        phc_acc_t d = in[i] - mean;
        var += (phc_var_t)d * (phc_var_t)d;
    }
    var = var / (phc_var_t)16.0;
    phc_acc_t inv_std = (phc_acc_t)1.0 / (phc_acc_t)hls::sqrt(var + (phc_var_t)LN_EPS);
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = (phc_acc_t)gamma[i] * (in[i] - mean) * inv_std + (phc_acc_t)beta[i];
    }
}

}  // namespace

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
) {
    phc_acc_t enc_linear_out[16];
    phc_acc_t encoded_pre[16];
    phc_acc_t encoded[16];
    phc_acc_t gate_pre[16];
    phc_acc_t gate_t[16];
    phc_acc_t candidate_pre[16];
    phc_acc_t candidate_t[16];
    phc_acc_t state_temporal[16];
    phc_acc_t gain_encoded[16];
    phc_acc_t gain_prev_state[16];
    phc_acc_t gain_pre[16];
    phc_acc_t gain_t[16];
    phc_acc_t message_pre[16];
    phc_acc_t message[16];
    phc_acc_t loop_input[16];
    phc_acc_t z[16];
    phc_acc_t readout_acc[2];

    using namespace phc_step_weights;

    linear_16_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);

    linear_16_16_acc(encoded, GATE_W, GATE_B, gate_pre);
    sigmoid_16(gate_pre, gate_t);

    linear_16_16_acc(encoded, CANDIDATE_W, CANDIDATE_B, candidate_pre);
    tanh_16(candidate_pre, candidate_t);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_temporal[i] = (phc_acc_t)DECAY[i] * (phc_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
    }

    linear_16_16_acc(encoded, GAIN_FROM_ENCODED_W, GAIN_FROM_ENCODED_B, gain_encoded);
    linear_16_16_fixed_no_bias(state_prev, GAIN_FROM_PREV_STATE_W, gain_prev_state);
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
    }
    sigmoid_16(gain_pre, gain_t);

    linear_16_16_acc(state_temporal, SYNAPSE_W, SYNAPSE_B, message_pre);
    silu_16(message_pre, message);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        loop_input[i] = state_temporal[i] + (phc_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_2_acc(z, READOUT_W, READOUT_B, readout_acc);

    for (int i = 0; i < 2; ++i) {
#pragma HLS PIPELINE off
        readout[i] = (phc_fixed_t)readout_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (phc_fixed_t)z[i];
    }
}
'''
    (QUANT_SRC / "phc_step_hls.hpp").write_text(hpp, encoding="utf-8")
    (QUANT_SRC / "phc_step_hls.cpp").write_text(cpp, encoding="utf-8")
    shutil.copy2(PHASE06F / "hls_src" / "phc_weights.hpp", QUANT_SRC / "phc_weights.hpp")


def cfg_text(variant: str) -> str:
    out_csv = RESULTS / f"phc_step_{variant}_csim_out.csv"
    variant_dir = "lut_only" if variant == "only_lut_error" else "quant_only"
    files = [
        "part=xc7z020clg400-1",
        "",
        "[hls]",
        "flow_target=vivado",
        "syn.top=phc_step",
        f"syn.file=../hls_src/{variant_dir}/phc_step_hls.cpp",
        f"syn.file=../hls_src/{variant_dir}/phc_step_hls.hpp",
    ]
    if variant == "only_lut_error":
        files.append("syn.file=../hls_src/lut_only/phc_weights_double.hpp")
    else:
        files.append("syn.file=../hls_src/quant_only/phc_weights.hpp")
    files.extend(
        [
            "tb.file=../hls_src/common/phc_step_testbench.cpp",
            "clock=10ns",
            f"csim.argv={TEST_VECTOR_CSV.as_posix()} {out_csv.as_posix()}",
        ]
    )
    return "\n".join(files) + "\n"


def write_configs() -> None:
    (LUT_COMPONENT / "hls_config.cfg").write_text(cfg_text("only_lut_error"), encoding="utf-8")
    (QUANT_COMPONENT / "hls_config.cfg").write_text(cfg_text("only_quant_error"), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not TEST_VECTOR_CSV.exists():
        raise FileNotFoundError(f"Missing test vector CSV: {TEST_VECTOR_CSV}")
    write_double_weights()
    write_common_testbench()
    write_lut_only_source()
    write_quant_only_source()
    write_configs()
    print(f"Prepared Phase 06M sources under {ROOT}")
    print(f"Test vectors: {TEST_VECTOR_CSV}")
    print(f"LUT-only config: {LUT_COMPONENT / 'hls_config.cfg'}")
    print(f"Quant-only config: {QUANT_COMPONENT / 'hls_config.cfg'}")


if __name__ == "__main__":
    main()
