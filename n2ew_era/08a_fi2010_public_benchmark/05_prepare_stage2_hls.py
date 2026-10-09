from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fi2010_phc_lite import PHCLiteFI2010Classifier, count_trainable_parameters


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
HLS_SRC = ROOT / "hls_src"
HLS_COMPONENT = ROOT / "hls_component"

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZON = 50
SAMPLE_COUNT = 25_600
TOTAL_BITS = 24
FRAC_BITS = 12
INT_MIN = -(2 ** (TOTAL_BITS - 1))
INT_MAX = 2 ** (TOTAL_BITS - 1) - 1
AP_TYPE = "ap_fixed<24,12>"

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
    for path in [RESULTS, MODELS, HLS_SRC, HLS_COMPONENT]:
        path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_checkpoint() -> tuple[Path, dict[str, Any], list[dict[str, object]]]:
    metrics = read_rows(RESULTS / "stage1_metrics_v2.csv")
    candidates = [
        row
        for row in metrics
        if int(row["horizon"]) == HORIZON and row["loss_mode"] == "raw" and row["phase"] == "08a_stage1_v2"
    ]
    if len(candidates) != 3:
        raise RuntimeError(f"Expected 3 raw candidate rows for k={HORIZON}, got {len(candidates)}")
    # NOTE (fixed per VALIDATION_METHODOLOGY_SOUNDNESS_AUDIT.md 3.x "real bug"):
    # Selecting the Stage2 representative checkpoint by sorting on test_macro_f1
    # uses the held-out test set to make a model-selection decision, which violates
    # the test-isolation discipline Stage1 itself claims. Ranking must use
    # val_macro_f1 only; test_macro_f1 is recorded for reporting but never used to
    # choose the checkpoint.
    ranked = sorted(candidates, key=lambda r: float(r["val_macro_f1"]))
    selected = ranked[len(ranked) // 2]
    checkpoint = ROOT / selected["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch_load(checkpoint)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: selected FI-2010 checkpoint is marked new5_used=True")
    rows = []
    for rank, row in enumerate(ranked):
        rows.append(
            {
                "rank_by_val_macro_f1_for_k50_raw": rank,
                "selected_for_stage2": row is selected,
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "loss_mode": row["loss_mode"],
                "val_macro_f1": float(row["val_macro_f1"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "checkpoint": row["checkpoint"],
                "selection_rule": "k=50 raw primary horizon; median seed by Stage 1 v2 VAL macro F1 (test-isolated; test_macro_f1 recorded for reporting only, fixed 2026-08-16 per red-team audit)",
            }
        )
    return checkpoint, payload, rows


def q24_12(values: np.ndarray) -> tuple[np.ndarray, int]:
    scaled = np.rint(values.astype(np.float64) * (2**FRAC_BITS))
    clipped = np.clip(scaled, INT_MIN, INT_MAX)
    saturated = int(np.count_nonzero(scaled != clipped))
    return clipped.astype(np.int64), saturated


def fixed_literal(int_value: int) -> str:
    quantized_value = float(int(int_value)) / float(2**FRAC_BITS)
    return f"({AP_TYPE})({quantized_value:.12g})"


def cpp_array_initializer(q: np.ndarray, indent: int = 0) -> str:
    spaces = " " * indent
    if q.ndim == 0:
        return fixed_literal(int(q.item()))
    if q.ndim == 1:
        return "{ " + ", ".join(fixed_literal(int(v)) for v in q.tolist()) + " }"
    if q.ndim == 2:
        rows = []
        for row in q:
            rows.append(spaces + "    { " + ", ".join(fixed_literal(int(v)) for v in row.tolist()) + " }")
        return "{\n" + ",\n".join(rows) + "\n" + spaces + "}"
    raise ValueError(f"Unsupported tensor rank: {q.ndim}")


def cpp_decl(name: str, q: np.ndarray) -> str:
    if q.ndim == 0:
        return f"static const {AP_TYPE} {name} = {fixed_literal(int(q.item()))};"
    if q.ndim == 1:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}] = {cpp_array_initializer(q)};"
    if q.ndim == 2:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}][{q.shape[1]}] = {cpp_array_initializer(q)};"
    raise ValueError(f"Unsupported tensor rank for {name}: {q.ndim}")


def export_weights(model: PHCLiteFI2010Classifier, checkpoint: Path, payload: dict[str, Any]) -> pd.DataFrame:
    state = model.state_dict()
    audit_rows: list[dict[str, Any]] = []
    header_blocks: list[str] = []
    quantized: dict[str, np.ndarray] = {}
    for tensor_name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        q, saturated = q24_12(arr)
        quantized[tensor_name] = q
        audit_rows.append(
            {
                "name": tensor_name,
                "cpp_name": EXPORT_MAP.get(tensor_name, ""),
                "shape": "x".join(str(x) for x in arr.shape) if arr.shape else "scalar",
                "num_values": int(arr.size),
                "float_min": float(np.nanmin(arr)) if arr.size else np.nan,
                "float_max": float(np.nanmax(arr)) if arr.size else np.nan,
                "quantized_min": int(np.min(q)) if q.size else 0,
                "quantized_max": int(np.max(q)) if q.size else 0,
                "num_saturated": int(saturated),
                "total_bits": TOTAL_BITS,
                "frac_bits": FRAC_BITS,
                "exported_to_header": tensor_name in EXPORT_MAP,
            }
        )

    for tensor_name, cpp_name in EXPORT_MAP.items():
        if tensor_name not in quantized:
            raise KeyError(f"Missing expected tensor in state_dict: {tensor_name}")
        header_blocks.append(f"// {tensor_name}")
        header_blocks.append(cpp_decl(cpp_name, quantized[tensor_name]))
        header_blocks.append("")

    header = [
        "#pragma once",
        "",
        "#include <ap_fixed.h>",
        "",
        "// Generated by n2ew_era/08a_fi2010_public_benchmark/05_prepare_stage2_hls.py",
        f"// Source checkpoint: {checkpoint}",
        f"// Horizon: {payload.get('horizon')}",
        f"// Seed: {payload.get('seed')}",
        f"// Loss mode: {payload.get('loss_mode')}",
        f"// Best epoch: {payload.get('best_epoch')}",
        f"// Quantization: Q{TOTAL_BITS}.{FRAC_BITS} == ap_fixed<{TOTAL_BITS},{TOTAL_BITS - FRAC_BITS}>",
        "",
        "namespace fi2010_phc_weights {",
        "",
        f"using weight_t = {AP_TYPE};",
        "",
        "static const int FEATURE_DIM = 40;",
        "static const int HIDDEN_DIM = 16;",
        "static const int READOUT_DIM = 3;",
        "static const int Q_TOTAL_BITS = 24;",
        "static const int Q_FRAC_BITS = 12;",
        "",
        *header_blocks,
        "}  // namespace fi2010_phc_weights",
        "",
    ]
    (HLS_SRC / "fi2010_phc_weights.hpp").write_text("\n".join(header), encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / "stage2_weight_export_audit.csv", index=False)
    return audit


def load_val_features() -> np.ndarray:
    with np.load(CACHE / "train_cf7.npz") as z:
        features = z["features"].astype(np.float32, copy=False)
    cut = int(np.floor(features.shape[0] * TRAIN_FRACTION))
    return features[cut:]


def make_windows(features: np.ndarray, starts: np.ndarray) -> torch.Tensor:
    idx = starts[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
    return torch.from_numpy(np.ascontiguousarray(features[idx])).to(dtype=torch.float32)


@torch.no_grad()
def generate_test_vectors(model: PHCLiteFI2010Classifier) -> pd.DataFrame:
    features = load_val_features()
    n_windows = int(features.shape[0] - WINDOW + 1)
    n = min(SAMPLE_COUNT, n_windows)
    starts = np.arange(n, dtype=np.int64)
    x_window = make_windows(features, starts)
    x_t = torch.from_numpy(np.ascontiguousarray(features[starts + WINDOW - 1])).to(dtype=torch.float32)
    state_prev = torch.zeros((n, 16), dtype=torch.float32)
    model.eval()
    logits_full = model(x_window)
    logits_step, state_next = model.forward_step(x_t, state_prev)
    fast_diff = float((logits_full - logits_step).detach().abs().max().cpu())
    if fast_diff > 1e-7:
        raise RuntimeError(f"Full-window vs final-step mismatch: max_abs_diff={fast_diff}")

    arr = np.concatenate(
        [
            np.arange(n, dtype=np.float64).reshape(-1, 1),
            starts.astype(np.float64).reshape(-1, 1),
            np.full((n, 1), WINDOW - 1, dtype=np.float64),
            x_t.detach().cpu().numpy().astype(np.float32),
            state_prev.detach().cpu().numpy().astype(np.float32),
            logits_full.detach().cpu().numpy().astype(np.float32),
            state_next.detach().cpu().numpy().astype(np.float32),
        ],
        axis=1,
    )
    columns = (
        ["row_id", "sample_idx", "timestep"]
        + [f"x_t_{i}" for i in range(40)]
        + [f"state_prev_{i}" for i in range(16)]
        + [f"ref_logit_{i}" for i in range(3)]
        + [f"ref_state_next_{i}" for i in range(16)]
    )
    df = pd.DataFrame(arr, columns=columns)
    for col in ["row_id", "sample_idx", "timestep"]:
        df[col] = df[col].astype(np.int64)
    float_cols = [c for c in df.columns if c not in {"row_id", "sample_idx", "timestep"}]
    df[float_cols] = df[float_cols].astype(np.float32)
    out_path = RESULTS / "fi2010_phc_stage2_test_vectors.csv"
    df.to_csv(out_path, index=False, float_format="%.9g")
    write_rows(
        RESULTS / "stage2_test_vector_summary.csv",
        [
            {
                "split_source": "validation_tail_of_train_cf7",
                "row_count": len(df),
                "window_length": WINDOW,
                "feature_dim": 40,
                "readout_dim": 3,
                "fast_full_window_max_abs_diff": fast_diff,
                "csv_path": str(out_path.relative_to(ROOT)),
                "csv_size_bytes": out_path.stat().st_size,
                "new5_used": False,
            }
        ],
    )
    return df


def write_hls_header() -> None:
    (HLS_SRC / "fi2010_phc_hls.hpp").write_text(
        """#pragma once

#if defined(__has_include)
#  if __has_include(<ap_fixed.h>)
#    include <ap_fixed.h>
#  else
#    ifndef FI2010_AP_FIXED_FALLBACK_DEFINED
#    define FI2010_AP_FIXED_FALLBACK_DEFINED
template <int W, int I>
using ap_fixed = double;
#    endif
#  endif
#else
#  include <ap_fixed.h>
#endif

typedef ap_fixed<24, 12> fi_fixed_t;
typedef ap_fixed<40, 16> fi_acc_t;
typedef ap_fixed<48, 24> fi_var_t;

static const int FI2010_FEATURE_DIM = 40;
static const int FI2010_HIDDEN_DIM = 16;
static const int FI2010_READOUT_DIM = 3;

void fi2010_phc_step(
    const fi_fixed_t x_t[FI2010_FEATURE_DIM],
    const fi_fixed_t state_prev[FI2010_HIDDEN_DIM],
    fi_fixed_t logits[FI2010_READOUT_DIM],
    fi_fixed_t state_next[FI2010_HIDDEN_DIM]
);
""",
        encoding="utf-8",
    )


def write_hls_cpp() -> None:
    (HLS_SRC / "fi2010_phc_hls.cpp").write_text(
        r'''#include "fi2010_phc_hls.hpp"
#include "fi2010_phc_weights.hpp"

#include <hls_math.h>

namespace {

static const fi_acc_t LN_EPS = (fi_acc_t)1.0e-5;

fi_acc_t tanh_lut_scalar(fi_acc_t x) {
#pragma HLS INLINE
    static const fi_acc_t TANH_LUT[65] = {
        (fi_acc_t)-0.9993292997, (fi_acc_t)-0.9991388858, (fi_acc_t)-0.9988944427, (fi_acc_t)-0.9985806592,
        (fi_acc_t)-0.9981778976, (fi_acc_t)-0.9976609795, (fi_acc_t)-0.9969976355, (fi_acc_t)-0.9961465307,
        (fi_acc_t)-0.9950547537, (fi_acc_t)-0.9936546343, (fi_acc_t)-0.9918597246, (fi_acc_t)-0.9895597486,
        (fi_acc_t)-0.9866142982, (fi_acc_t)-0.9828450292, (fi_acc_t)-0.9780261147, (fi_acc_t)-0.9718727459,
        (fi_acc_t)-0.9640275801, (fi_acc_t)-0.9540452602, (fi_acc_t)-0.9413755385, (fi_acc_t)-0.9253462253,
        (fi_acc_t)-0.9051482536, (fi_acc_t)-0.8798266997, (fi_acc_t)-0.8482836400, (fi_acc_t)-0.8093010702,
        (fi_acc_t)-0.7615941560, (fi_acc_t)-0.7039056039, (fi_acc_t)-0.6351489524, (fi_acc_t)-0.5545997223,
        (fi_acc_t)-0.4621171573, (fi_acc_t)-0.3583573984, (fi_acc_t)-0.2449186624, (fi_acc_t)-0.1243530018,
        (fi_acc_t)0.0000000000,  (fi_acc_t)0.1243530018,  (fi_acc_t)0.2449186624,  (fi_acc_t)0.3583573984,
        (fi_acc_t)0.4621171573,  (fi_acc_t)0.5545997223,  (fi_acc_t)0.6351489524,  (fi_acc_t)0.7039056039,
        (fi_acc_t)0.7615941560,  (fi_acc_t)0.8093010702,  (fi_acc_t)0.8482836400,  (fi_acc_t)0.8798266997,
        (fi_acc_t)0.9051482536,  (fi_acc_t)0.9253462253,  (fi_acc_t)0.9413755385,  (fi_acc_t)0.9540452602,
        (fi_acc_t)0.9640275801,  (fi_acc_t)0.9718727459,  (fi_acc_t)0.9780261147,  (fi_acc_t)0.9828450292,
        (fi_acc_t)0.9866142982,  (fi_acc_t)0.9895597486,  (fi_acc_t)0.9918597246,  (fi_acc_t)0.9936546343,
        (fi_acc_t)0.9950547537,  (fi_acc_t)0.9961465307,  (fi_acc_t)0.9969976355,  (fi_acc_t)0.9976609795,
        (fi_acc_t)0.9981778976,  (fi_acc_t)0.9985806592,  (fi_acc_t)0.9988944427,  (fi_acc_t)0.9991388858,
        (fi_acc_t)0.9993292997
    };
#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1
    if (x <= (fi_acc_t)-4.0) {
        return (fi_acc_t)-1.0;
    }
    if (x >= (fi_acc_t)4.0) {
        return (fi_acc_t)1.0;
    }
    fi_acc_t scaled = (x + (fi_acc_t)4.0) * (fi_acc_t)8.0;
    int idx = (int)scaled;
    if (idx < 0) idx = 0;
    if (idx > 63) idx = 63;
    fi_acc_t frac = scaled - (fi_acc_t)idx;
    fi_acc_t y0 = TANH_LUT[idx];
    fi_acc_t y1 = TANH_LUT[idx + 1];
    return y0 + frac * (y1 - y0);
}

fi_acc_t sigmoid_scalar(fi_acc_t x) {
#pragma HLS INLINE
    return (fi_acc_t)0.5 * (tanh_lut_scalar((fi_acc_t)0.5 * x) + (fi_acc_t)1.0);
}

void sigmoid_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = tanh_lut_scalar(in[i]);
    }
}

void silu_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_40_16_fixed(const fi_fixed_t in[40], const fi_fixed_t w[16][40], const fi_fixed_t b[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 40; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * (fi_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const fi_acc_t in[16], const fi_fixed_t w[16][16], const fi_fixed_t b[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const fi_fixed_t in[16], const fi_fixed_t w[16][16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = 0;
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * (fi_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_3_acc(const fi_acc_t in[16], const fi_fixed_t w[3][16], const fi_fixed_t b[3], fi_acc_t out[3]) {
#pragma HLS INLINE
    for (int j = 0; j < 3; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const fi_acc_t in[16], const fi_fixed_t gamma[16], const fi_fixed_t beta[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    fi_acc_t mean = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        mean += in[i];
    }
    mean = mean / (fi_acc_t)16.0;

    fi_var_t var = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        fi_acc_t d = in[i] - mean;
        var += (fi_var_t)d * (fi_var_t)d;
    }
    var = var / (fi_var_t)16.0;
    fi_acc_t inv_std = (fi_acc_t)1.0 / (fi_acc_t)hls::sqrt(var + (fi_var_t)LN_EPS);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = (fi_acc_t)gamma[i] * (in[i] - mean) * inv_std + (fi_acc_t)beta[i];
    }
}

}  // namespace

void fi2010_phc_step(
    const fi_fixed_t x_t[FI2010_FEATURE_DIM],
    const fi_fixed_t state_prev[FI2010_HIDDEN_DIM],
    fi_fixed_t logits[FI2010_READOUT_DIM],
    fi_fixed_t state_next[FI2010_HIDDEN_DIM]
) {
    fi_acc_t enc_linear_out[16];
    fi_acc_t encoded_pre[16];
    fi_acc_t encoded[16];
    fi_acc_t gate_pre[16];
    fi_acc_t gate_t[16];
    fi_acc_t candidate_pre[16];
    fi_acc_t candidate_t[16];
    fi_acc_t state_temporal[16];
    fi_acc_t gain_encoded[16];
    fi_acc_t gain_prev_state[16];
    fi_acc_t gain_pre[16];
    fi_acc_t gain_t[16];
    fi_acc_t message_pre[16];
    fi_acc_t message[16];
    fi_acc_t loop_input[16];
    fi_acc_t z[16];
    fi_acc_t logits_acc[3];

    using namespace fi2010_phc_weights;

    linear_40_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);

    linear_16_16_acc(encoded, GATE_W, GATE_B, gate_pre);
    sigmoid_16(gate_pre, gate_t);

    linear_16_16_acc(encoded, CANDIDATE_W, CANDIDATE_B, candidate_pre);
    tanh_16(candidate_pre, candidate_t);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_temporal[i] = (fi_acc_t)DECAY[i] * (fi_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
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
        loop_input[i] = state_temporal[i] + (fi_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_3_acc(z, READOUT_W, READOUT_B, logits_acc);

    for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE off
        logits[i] = (fi_fixed_t)logits_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (fi_fixed_t)z[i];
    }
}
''',
        encoding="utf-8",
    )


def write_testbench() -> None:
    (HLS_SRC / "fi2010_phc_testbench.cpp").write_text(
        r'''#include "fi2010_phc_hls.hpp"

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
''',
        encoding="utf-8",
    )


def write_hls_config() -> None:
    input_csv = (RESULTS / "fi2010_phc_stage2_test_vectors.csv").resolve().as_posix()
    out_csv = (RESULTS / "fi2010_phc_stage2_csim_out.csv").resolve().as_posix()
    cfg = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=fi2010_phc_step
syn.file=../hls_src/fi2010_phc_hls.cpp
syn.file=../hls_src/fi2010_phc_hls.hpp
syn.file=../hls_src/fi2010_phc_weights.hpp
tb.file=../hls_src/fi2010_phc_testbench.cpp
clock=10ns
csim.argv={input_csv} {out_csv}
"""
    (HLS_COMPONENT / "hls_config.cfg").write_text(cfg, encoding="utf-8")
    tcl = """open_project hls_component
set_top fi2010_phc_step
add_files ../hls_src/fi2010_phc_hls.cpp
add_files ../hls_src/fi2010_phc_hls.hpp
add_files ../hls_src/fi2010_phc_weights.hpp
add_files -tb ../hls_src/fi2010_phc_testbench.cpp
open_solution -flow_target vivado hls
set_part xc7z020-clg400-1
create_clock -period 10 -name default
csynth_design
exit
"""
    (HLS_COMPONENT / "run_csynth_only.tcl").write_text(tcl, encoding="utf-8")


def write_stage2_manifest(
    checkpoint: Path,
    payload: dict[str, Any],
    selection_rows: list[dict[str, object]],
    audit: pd.DataFrame,
) -> None:
    copied_ckpt = MODELS / checkpoint.name
    if checkpoint.resolve() != copied_ckpt.resolve() and not copied_ckpt.exists():
        shutil.copy2(checkpoint, copied_ckpt)
    manifest = {
        "phase": "08a_stage2_hls_feasibility",
        "status": "prepared_for_csim_csynth",
        "selected_checkpoint": str(checkpoint.relative_to(ROOT)),
        "selected_checkpoint_copy": str(copied_ckpt.relative_to(ROOT)),
        "selection_rule": selection_rows[0]["selection_rule"],
        "horizon": int(payload["horizon"]),
        "seed": int(payload["seed"]),
        "loss_mode": payload["loss_mode"],
        "best_epoch": int(payload["best_epoch"]),
        "feature_dim": int(payload["feature_dim"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "num_classes": int(payload["num_classes"]),
        "trainable_parameters": int(payload["trainable_parameters"]),
        "new5_used": bool(payload.get("new5_used", False)),
        "quantization": f"Q{TOTAL_BITS}.{FRAC_BITS}",
        "any_weight_saturated": bool(int(audit["num_saturated"].sum()) > 0),
        "num_weight_saturated": int(audit["num_saturated"].sum()),
        "stage2_scope": "csim_and_csynth_only; no cosim, no P&R, no bitstream, no board work",
    }
    (RESULTS / "stage2_hls_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    checkpoint, payload, selection_rows = select_checkpoint()
    write_rows(RESULTS / "stage2_checkpoint_selection.csv", selection_rows)
    model = PHCLiteFI2010Classifier(
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_classes=int(payload["num_classes"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    audit = export_weights(model, checkpoint, payload)
    vectors = generate_test_vectors(model)
    write_hls_header()
    write_hls_cpp()
    write_testbench()
    write_hls_config()
    write_stage2_manifest(checkpoint, payload, selection_rows, audit)
    print(
        json.dumps(
            {
                "selected_checkpoint": str(checkpoint.relative_to(ROOT)),
                "horizon": int(payload["horizon"]),
                "seed": int(payload["seed"]),
                "loss_mode": payload["loss_mode"],
                "test_vector_rows": int(len(vectors)),
                "weight_saturations": int(audit["num_saturated"].sum()),
                "hls_config": str((HLS_COMPONENT / "hls_config.cfg").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
