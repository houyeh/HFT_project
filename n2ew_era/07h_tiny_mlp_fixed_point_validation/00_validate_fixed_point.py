from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07h_tiny_mlp_fixed_point_validation"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PHASE07G = PROJECT_ROOT / "n2ew_era" / "07g_tiny_mlp_policy"
PHASE07G_MAIN = PHASE07G / "00_run_tiny_mlp_policy.py"
PHASE07G_CORRECTION = PHASE07G / "02_audit_terminal_settlement_correction.py"

ACC_TOTAL_BITS = 40
ACC_INTEGER_BITS = 16
ACC_FRAC_BITS = ACC_TOTAL_BITS - ACC_INTEGER_BITS
FORMATS = ((24, 12), (20, 10), (18, 8), (16, 6), (14, 6), (12, 5))
VAL_MIN_ACTION_AGREEMENT = 0.999
VAL_MIN_PNL_RETENTION = 0.99


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase = load_module("tiny_mlp_phase_07h", PHASE07G_MAIN)
correction = load_module("tiny_mlp_terminal_correction_07h", PHASE07G_CORRECTION)
helpers = phase.helpers
p05a = phase.p05a


@dataclass(frozen=True)
class FixedFormat:
    total_bits: int
    integer_bits: int

    @property
    def frac_bits(self) -> int:
        return self.total_bits - self.integer_bits

    @property
    def name(self) -> str:
        return f"ap_fixed_{self.total_bits}_{self.integer_bits}"


def wrap_signed(values: np.ndarray, bits: int) -> np.ndarray:
    modulus = np.int64(1 << bits)
    sign = np.int64(1 << (bits - 1))
    unsigned = np.mod(values, modulus)
    return np.where(unsigned >= sign, unsigned - modulus, unsigned).astype(np.int64)


def quantize(values: np.ndarray, fmt: FixedFormat) -> tuple[np.ndarray, int]:
    scaled = np.floor(np.asarray(values, dtype=np.float64) * float(1 << fmt.frac_bits))
    minimum = -(1 << (fmt.total_bits - 1))
    maximum = (1 << (fmt.total_bits - 1)) - 1
    overflow = int(np.count_nonzero((scaled < minimum) | (scaled > maximum)))
    return wrap_signed(scaled.astype(np.int64), fmt.total_bits), overflow


def align_fraction(values: np.ndarray, source_frac: int, target_frac: int) -> np.ndarray:
    if source_frac < target_frac:
        return values << (target_frac - source_frac)
    if source_frac > target_frac:
        return values >> (source_frac - target_frac)
    return values


def accumulator_overflow_count(values: np.ndarray) -> int:
    minimum = -(1 << (ACC_TOTAL_BITS - 1))
    maximum = (1 << (ACC_TOTAL_BITS - 1)) - 1
    return int(np.count_nonzero((values < minimum) | (values > maximum)))


def dense_fixed(
    x_int: np.ndarray, weight_int: np.ndarray, bias_int: np.ndarray,
    fmt: FixedFormat,
) -> tuple[np.ndarray, int]:
    product_frac = 2 * fmt.frac_bits
    if product_frac <= ACC_FRAC_BITS:
        accum = (x_int @ weight_int.T) << (ACC_FRAC_BITS - product_frac)
    else:
        # This branch preserves per-product AP_TRN semantics if a future format
        # has more fractional product bits than the accumulator.
        accum = np.zeros((len(x_int), len(weight_int)), dtype=np.int64)
        shift = product_frac - ACC_FRAC_BITS
        for input_index in range(x_int.shape[1]):
            product = x_int[:, input_index : input_index + 1] * weight_int[:, input_index]
            accum += product >> shift
    accum += align_fraction(bias_int, fmt.frac_bits, ACC_FRAC_BITS)
    overflow = accumulator_overflow_count(accum)
    return wrap_signed(accum, ACC_TOTAL_BITS), overflow


def cast_accumulator_to_data(
    accum: np.ndarray, fmt: FixedFormat,
) -> tuple[np.ndarray, int]:
    aligned = align_fraction(accum, ACC_FRAC_BITS, fmt.frac_bits)
    minimum = -(1 << (fmt.total_bits - 1))
    maximum = (1 << (fmt.total_bits - 1)) - 1
    overflow = int(np.count_nonzero((aligned < minimum) | (aligned > maximum)))
    return wrap_signed(aligned, fmt.total_bits), overflow


def fixed_forward(
    raw_x: np.ndarray, w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray, fmt: FixedFormat,
) -> tuple[np.ndarray, dict[str, int]]:
    x_int, input_overflow = quantize(raw_x, fmt)
    w1_int, w1_overflow = quantize(w1, fmt)
    b1_int, b1_overflow = quantize(b1, fmt)
    w2_int, w2_overflow = quantize(w2, fmt)
    b2_int, b2_overflow = quantize(b2, fmt)

    hidden_acc, hidden_acc_overflow = dense_fixed(x_int, w1_int, b1_int, fmt)
    hidden_acc = np.maximum(hidden_acc, 0)
    hidden_int, hidden_cast_overflow = cast_accumulator_to_data(hidden_acc, fmt)
    output_acc, output_acc_overflow = dense_fixed(hidden_int, w2_int, b2_int, fmt)
    output_int, output_cast_overflow = cast_accumulator_to_data(output_acc, fmt)
    logits = output_int.astype(np.float64) / float(1 << fmt.frac_bits)
    audit = {
        "input_overflow_count": input_overflow,
        "w1_overflow_count": w1_overflow,
        "b1_overflow_count": b1_overflow,
        "w2_overflow_count": w2_overflow,
        "b2_overflow_count": b2_overflow,
        "hidden_acc_overflow_count": hidden_acc_overflow,
        "hidden_cast_overflow_count": hidden_cast_overflow,
        "output_acc_overflow_count": output_acc_overflow,
        "output_cast_overflow_count": output_cast_overflow,
    }
    audit["total_overflow_count"] = int(sum(audit.values()))
    return logits, audit


def float_forward(raw_x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden = np.maximum(raw_x @ w1.T + b1, 0.0)
    return hidden @ w2.T + b2


def export_header(
    path: Path, fmt: FixedFormat, w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> None:
    arrays = {
        "TINY_MLP_W1": quantize(w1, fmt)[0],
        "TINY_MLP_B1": quantize(b1, fmt)[0],
        "TINY_MLP_W2": quantize(w2, fmt)[0],
        "TINY_MLP_B2": quantize(b2, fmt)[0],
    }
    lines = [
        "#pragma once",
        "",
        f"// Quantized with AP_TRN/AP_WRAP semantics for ap_fixed<{fmt.total_bits},{fmt.integer_bits}>.",
        f"static const int TINY_MLP_DATA_TOTAL_BITS = {fmt.total_bits};",
        f"static const int TINY_MLP_DATA_INTEGER_BITS = {fmt.integer_bits};",
        "static const int TINY_MLP_INPUT_DIM = 16;",
        "static const int TINY_MLP_HIDDEN_DIM = 8;",
        "static const int TINY_MLP_OUTPUT_DIM = 3;",
        "",
    ]
    for name, values in arrays.items():
        if values.ndim == 1:
            body = ", ".join(str(int(value)) for value in values)
            lines.append(f"static const long long {name}[{len(values)}] = {{{body}}};")
        else:
            lines.append(f"static const long long {name}[{values.shape[0]}][{values.shape[1]}] = {{")
            for row in values:
                body = ", ".join(str(int(value)) for value in row)
                lines.append(f"    {{{body}}},")
            lines.append("};")
        lines.append("")
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    selected = json.loads(
        (PHASE07G / "results" / "tiny_mlp_selected.json").read_text(encoding="utf-8")
    )
    if selected["candidate_id"] != "tiny_mlp_direct_action_h8_seed1":
        raise RuntimeError(f"Unexpected frozen candidate: {selected['candidate_id']}")
    state = selected["state_dict"]
    standardized_w1 = np.asarray(state["net.0.weight"], dtype=np.float64)
    standardized_b1 = np.asarray(state["net.0.bias"], dtype=np.float64)
    w2 = np.asarray(state["net.2.weight"], dtype=np.float64)
    b2 = np.asarray(state["net.2.bias"], dtype=np.float64)
    mean = np.asarray(selected["feature_mean"], dtype=np.float64)
    scale = np.asarray(selected["feature_scale"], dtype=np.float64)
    w1 = standardized_w1 / scale[np.newaxis, :]
    b1 = standardized_b1 - np.sum(standardized_w1 * mean[np.newaxis, :] / scale[np.newaxis, :], axis=1)

    np.savez(
        MODELS / "tiny_mlp_h8_seed1_folded_float_weights.npz",
        w1=w1.astype(np.float32), b1=b1.astype(np.float32),
        w2=w2.astype(np.float32), b2=b2.astype(np.float32),
        feature_names=np.asarray(selected["feature_names"]),
    )

    norms = p05a.compute_fill_norms_from_train_sequences()
    equivalence_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], np.ndarray] = {}
    float_decisions: dict[str, np.ndarray] = {}

    checkpoint = torch.load(
        PHASE07G / "models" / f"{selected['candidate_id']}.pt",
        map_location="cpu", weights_only=False,
    )
    torch_model = phase.TinyMLP(16, 8, 3)
    torch_model.load_state_dict(checkpoint["state_dict"])
    torch_model.eval()

    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        raw_x, _ = helpers.base_features(payload)
        numpy_logits = float_forward(raw_x, w1, b1, w2, b2)
        standardized = ((raw_x - mean) / scale).astype(np.float32)
        with torch.no_grad():
            torch_logits = torch_model(torch.from_numpy(standardized)).numpy().astype(np.float64)
        difference = numpy_logits - torch_logits
        equivalence_rows.append({
            "split": split,
            "row_count": len(raw_x),
            "max_abs_logit_diff": float(np.max(np.abs(difference))),
            "mean_abs_logit_diff": float(np.mean(np.abs(difference))),
            "argmax_agreement": float(np.mean(np.argmax(numpy_logits, axis=1) == np.argmax(torch_logits, axis=1))),
        })
        float_decision = np.argmax(numpy_logits, axis=1).astype(np.int8)
        float_decisions[split] = float_decision
        float_metrics = correction.evaluate_policy(
            split, "tiny_mlp_float", payload, float_decision, norms
        )
        prediction_cache[(split, "float")] = float_decision
        for total_bits, integer_bits in FORMATS:
            fmt = FixedFormat(total_bits, integer_bits)
            fixed_logits, audit = fixed_forward(raw_x, w1, b1, w2, b2, fmt)
            fixed_decision = np.argmax(fixed_logits, axis=1).astype(np.int8)
            fixed_metrics = correction.evaluate_policy(
                split, f"tiny_mlp_{fmt.name}", payload, fixed_decision, norms
            )
            disagreement = fixed_decision != float_decision
            float_sorted = np.sort(numpy_logits, axis=1)
            float_margin = float_sorted[:, -1] - float_sorted[:, -2]
            logit_error = fixed_logits - numpy_logits
            row = {
                "split": split,
                "format": fmt.name,
                "total_bits": total_bits,
                "integer_bits": integer_bits,
                "fractional_bits": fmt.frac_bits,
                "action_agreement_rate": float(np.mean(~disagreement)),
                "action_flip_count": int(disagreement.sum()),
                "max_abs_logit_diff": float(np.max(np.abs(logit_error))),
                "mean_abs_logit_diff": float(np.mean(np.abs(logit_error))),
                "logit_rmse": float(np.sqrt(np.mean(logit_error ** 2))),
                "minimum_float_margin_among_flips": float(np.min(float_margin[disagreement])) if disagreement.any() else np.nan,
                "float_sequential_excess_pnl": float(float_metrics["corrected_sequential_excess_pnl"]),
                "fixed_sequential_excess_pnl": float(fixed_metrics["corrected_sequential_excess_pnl"]),
                "fixed_pnl_retention_pct": (
                    100.0 * float(fixed_metrics["corrected_sequential_excess_pnl"])
                    / float(float_metrics["corrected_sequential_excess_pnl"])
                ),
                "pnl_delta_fixed_minus_float": (
                    float(fixed_metrics["corrected_sequential_excess_pnl"])
                    - float(float_metrics["corrected_sequential_excess_pnl"])
                ),
                "fixed_fill_count": int(fixed_metrics["corrected_fill_count"]),
                "fixed_forced_no_quote_count": int(fixed_metrics["corrected_forced_no_quote_count"]),
                "fixed_final_inventory": int(fixed_metrics["corrected_final_inventory"]),
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used": False,
            }
            sweep_rows.append(row)
            range_rows.append({"split": split, "format": fmt.name, **audit})
            prediction_cache[(split, fmt.name)] = fixed_decision
            if disagreement.any():
                indices = np.flatnonzero(disagreement)
                for index in indices[:100]:
                    flip_rows.append({
                        "split": split,
                        "format": fmt.name,
                        "row_index": int(index),
                        "float_action": int(float_decision[index]),
                        "fixed_action": int(fixed_decision[index]),
                        "float_margin": float(float_margin[index]),
                        "max_abs_row_logit_diff": float(np.max(np.abs(logit_error[index]))),
                    })

    equivalence = pd.DataFrame(equivalence_rows)
    sweep = pd.DataFrame(sweep_rows)
    ranges = pd.DataFrame(range_rows)
    equivalence.to_csv(RESULTS / "float_export_equivalence.csv", index=False)
    sweep.to_csv(RESULTS / "fixed_point_sweep.csv", index=False)
    ranges.to_csv(RESULTS / "fixed_point_range_audit.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(RESULTS / "fixed_point_action_flip_examples.csv", index=False)

    val = sweep[sweep["split"] == "val"].copy()
    val_ranges = ranges[ranges["split"] == "val"][["format", "total_overflow_count"]]
    val = val.merge(val_ranges, on="format")
    eligible = val[
        (val["action_agreement_rate"] >= VAL_MIN_ACTION_AGREEMENT)
        & (val["fixed_pnl_retention_pct"] >= 100.0 * VAL_MIN_PNL_RETENTION)
        & (val["total_overflow_count"] == 0)
    ].sort_values(["total_bits", "integer_bits"])
    if eligible.empty:
        selected_format = val.sort_values(
            ["action_agreement_rate", "fixed_pnl_retention_pct", "total_bits"],
            ascending=[False, False, True],
        ).iloc[0]
        selection_status = "no_format_cleared_gate"
    else:
        selected_format = eligible.iloc[0]
        selection_status = "gate_pass"
    format_name = str(selected_format["format"])
    selected_rows = sweep[sweep["format"] == format_name].copy()
    selected_rows.to_csv(RESULTS / "selected_fixed_point_results.csv", index=False)

    fmt = FixedFormat(int(selected_format["total_bits"]), int(selected_format["integer_bits"]))
    np.savez(
        MODELS / f"tiny_mlp_h8_seed1_{fmt.name}_integer_weights.npz",
        w1=quantize(w1, fmt)[0], b1=quantize(b1, fmt)[0],
        w2=quantize(w2, fmt)[0], b2=quantize(b2, fmt)[0],
        total_bits=fmt.total_bits, integer_bits=fmt.integer_bits,
        accumulator_total_bits=ACC_TOTAL_BITS, accumulator_integer_bits=ACC_INTEGER_BITS,
    )
    export_header(MODELS / f"tiny_mlp_h8_seed1_{fmt.name}_weights.hpp", fmt, w1, b1, w2, b2)
    selection = {
        "status": selection_status,
        "format": format_name,
        "total_bits": fmt.total_bits,
        "integer_bits": fmt.integer_bits,
        "fractional_bits": fmt.frac_bits,
        "accumulator": f"ap_fixed<{ACC_TOTAL_BITS},{ACC_INTEGER_BITS}>",
        "rounding": "AP_TRN",
        "overflow": "AP_WRAP",
        "selection_split": "val",
        "val_min_action_agreement": VAL_MIN_ACTION_AGREEMENT,
        "val_min_pnl_retention": VAL_MIN_PNL_RETENTION,
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used": False,
    }
    (RESULTS / "selected_fixed_point.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    selected_range = ranges[ranges["format"] == format_name]
    gate_pass = selection_status == "gate_pass"
    status = "resolved_positive" if gate_pass else "resolved_negative"
    lines = [
        "# Tiny MLP Fixed-Point Equivalence Audit\n",
        f"status: `{status}`\n",
        "## Scope and frozen candidate\n",
        "The frozen candidate is `tiny_mlp_direct_action_h8_seed1`: 16 current-event causal features, one 8-unit ReLU hidden layer, and three buy/sell/hold logits (163 trainable parameters). The PHC teacher and tiny MLP are not retrained. Train-derived standardization is algebraically folded into the first-layer weights. Val selects the minimum passing precision; NEW3/NEW4 are diagnostics only. NEW5 was not loaded. This is integer software emulation, not HLS/csim/csynth evidence.\n",
        "## Arithmetic contract\n",
        f"Data and weights use the swept `ap_fixed<W,I>` format with `AP_TRN` and `AP_WRAP`. Both dense accumulators use `ap_fixed<{ACC_TOTAL_BITS},{ACC_INTEGER_BITS}>`. The emulator quantizes inputs, folded weights, and biases, performs integer multiply-accumulate with explicit binary-point alignment, casts the hidden activation after ReLU, and casts the three output logits before argmax.\n",
        "## Float export equivalence\n",
        equivalence.to_markdown(index=False, floatfmt=".12g") + "\n",
        "## Precision sweep\n",
        sweep.to_markdown(index=False, floatfmt=".9g") + "\n",
        "## Range and overflow audit\n",
        ranges.to_markdown(index=False) + "\n",
        "## Val-selected precision\n",
        f"Selection status: `{selection_status}`. The minimum val-passing format is `{format_name}` under action-agreement >= `{VAL_MIN_ACTION_AGREEMENT:.4f}`, fixed-PnL retention >= `{VAL_MIN_PNL_RETENTION:.4f}`, and zero observed overflow.\n",
        selected_rows.to_markdown(index=False, floatfmt=".9g") + "\n",
        "Selected-format overflow rows:\n",
        selected_range.to_markdown(index=False) + "\n",
        "## Decision\n",
        (
            "The selected fixed-point format clears the predeclared software-equivalence gate and can proceed to an independent manual-HLS implementation. The next phase must still establish CSim agreement, csynth latency/resources, RTL cosim behavior, routed timing, and on-board behavior before making hardware claims.\n"
            if gate_pass
            else "No swept format clears the software-equivalence gate. Do not begin HLS until the arithmetic contract or precision grid is reviewed.\n"
        ),
    ]
    (ROOT / "AUDIT_tiny_mlp_fixed_point.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] selected_format={format_name} status={selection_status}", flush=True)
    print(selected_rows[[
        "split", "action_agreement_rate", "action_flip_count",
        "fixed_pnl_retention_pct", "pnl_delta_fixed_minus_float",
    ]].to_string(index=False), flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_tiny_mlp_fixed_point.md'}", flush=True)


if __name__ == "__main__":
    main()
