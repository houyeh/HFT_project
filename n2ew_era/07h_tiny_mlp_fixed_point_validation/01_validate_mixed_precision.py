from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07h_tiny_mlp_fixed_point_validation"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
BASE_SCRIPT = ROOT / "00_validate_fixed_point.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("tiny_mlp_fixed_base_07h", BASE_SCRIPT)
helpers = base.helpers
p05a = base.p05a
correction = base.correction


@dataclass(frozen=True)
class MixedConfig:
    name: str
    data: Any
    weight: Any
    bias: Any
    output_compare: str

    @property
    def one_dsp_shape_candidate(self) -> bool:
        return max(self.data.total_bits, self.weight.total_bits) <= 25 and min(
            self.data.total_bits, self.weight.total_bits
        ) <= 18


CONFIGS = (
    MixedConfig("data24_12_weight18_5_accargmax", base.FixedFormat(24, 12), base.FixedFormat(18, 5), base.FixedFormat(24, 12), "accumulator"),
    MixedConfig("data24_12_weight20_6_accargmax", base.FixedFormat(24, 12), base.FixedFormat(20, 6), base.FixedFormat(24, 12), "accumulator"),
    MixedConfig("uniform24_12_accargmax", base.FixedFormat(24, 12), base.FixedFormat(24, 12), base.FixedFormat(24, 12), "accumulator"),
    MixedConfig("uniform28_14_accargmax", base.FixedFormat(28, 14), base.FixedFormat(28, 14), base.FixedFormat(28, 14), "accumulator"),
    MixedConfig("uniform32_16_accargmax", base.FixedFormat(32, 16), base.FixedFormat(32, 16), base.FixedFormat(32, 16), "accumulator"),
)


def dense_mixed(
    x_int: np.ndarray, weight_int: np.ndarray, bias_int: np.ndarray,
    data_fmt: Any, weight_fmt: Any, bias_fmt: Any,
) -> tuple[np.ndarray, int]:
    product_frac = data_fmt.frac_bits + weight_fmt.frac_bits
    if product_frac <= base.ACC_FRAC_BITS:
        accum = (x_int @ weight_int.T) << (base.ACC_FRAC_BITS - product_frac)
    else:
        accum = np.zeros((len(x_int), len(weight_int)), dtype=np.int64)
        shift = product_frac - base.ACC_FRAC_BITS
        for input_index in range(x_int.shape[1]):
            product = x_int[:, input_index : input_index + 1] * weight_int[:, input_index]
            accum += product >> shift
    accum += base.align_fraction(bias_int, bias_fmt.frac_bits, base.ACC_FRAC_BITS)
    overflow = base.accumulator_overflow_count(accum)
    return base.wrap_signed(accum, base.ACC_TOTAL_BITS), overflow


def mixed_forward(
    raw_x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    config: MixedConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    x_int, input_overflow = base.quantize(raw_x, config.data)
    w1_int, w1_overflow = base.quantize(w1, config.weight)
    b1_int, b1_overflow = base.quantize(b1, config.bias)
    w2_int, w2_overflow = base.quantize(w2, config.weight)
    b2_int, b2_overflow = base.quantize(b2, config.bias)
    hidden_acc, hidden_acc_overflow = dense_mixed(
        x_int, w1_int, b1_int, config.data, config.weight, config.bias
    )
    hidden_acc = np.maximum(hidden_acc, 0)
    hidden_int, hidden_cast_overflow = base.cast_accumulator_to_data(hidden_acc, config.data)
    output_acc, output_acc_overflow = dense_mixed(
        hidden_int, w2_int, b2_int, config.data, config.weight, config.bias
    )
    if config.output_compare == "accumulator":
        logits = output_acc.astype(np.float64) / float(1 << base.ACC_FRAC_BITS)
        output_cast_overflow = 0
    else:
        output_int, output_cast_overflow = base.cast_accumulator_to_data(output_acc, config.data)
        logits = output_int.astype(np.float64) / float(1 << config.data.frac_bits)
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


def export_mixed_header(
    path: Path, config: MixedConfig, w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> None:
    arrays = {
        "TINY_MLP_W1": base.quantize(w1, config.weight)[0],
        "TINY_MLP_B1": base.quantize(b1, config.bias)[0],
        "TINY_MLP_W2": base.quantize(w2, config.weight)[0],
        "TINY_MLP_B2": base.quantize(b2, config.bias)[0],
    }
    lines = [
        "#pragma once",
        "",
        "// AP_TRN/AP_WRAP integer constants. Final action compares accumulator logits.",
        f"static const int TINY_MLP_DATA_TOTAL_BITS = {config.data.total_bits};",
        f"static const int TINY_MLP_DATA_INTEGER_BITS = {config.data.integer_bits};",
        f"static const int TINY_MLP_WEIGHT_TOTAL_BITS = {config.weight.total_bits};",
        f"static const int TINY_MLP_WEIGHT_INTEGER_BITS = {config.weight.integer_bits};",
        f"static const int TINY_MLP_BIAS_TOTAL_BITS = {config.bias.total_bits};",
        f"static const int TINY_MLP_BIAS_INTEGER_BITS = {config.bias.integer_bits};",
        f"static const int TINY_MLP_ACC_TOTAL_BITS = {base.ACC_TOTAL_BITS};",
        f"static const int TINY_MLP_ACC_INTEGER_BITS = {base.ACC_INTEGER_BITS};",
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
    weights = np.load(MODELS / "tiny_mlp_h8_seed1_folded_float_weights.npz")
    w1 = weights["w1"].astype(np.float64)
    b1 = weights["b1"].astype(np.float64)
    w2 = weights["w2"].astype(np.float64)
    b2 = weights["b2"].astype(np.float64)
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        raw_x, _ = helpers.base_features(payload)
        float_logits = base.float_forward(raw_x, w1, b1, w2, b2)
        float_decision = np.argmax(float_logits, axis=1).astype(np.int8)
        float_metrics = correction.evaluate_policy(split, "tiny_mlp_float", payload, float_decision, norms)
        margins = np.sort(float_logits, axis=1)[:, -1] - np.sort(float_logits, axis=1)[:, -2]
        for config in CONFIGS:
            fixed_logits, audit = mixed_forward(raw_x, w1, b1, w2, b2, config)
            fixed_decision = np.argmax(fixed_logits, axis=1).astype(np.int8)
            disagreement = fixed_decision != float_decision
            fixed_metrics = correction.evaluate_policy(
                split, f"tiny_mlp_{config.name}", payload, fixed_decision, norms
            )
            error = fixed_logits - float_logits
            rows.append({
                "split": split,
                "config": config.name,
                "data_format": f"ap_fixed<{config.data.total_bits},{config.data.integer_bits}>",
                "weight_format": f"ap_fixed<{config.weight.total_bits},{config.weight.integer_bits}>",
                "bias_format": f"ap_fixed<{config.bias.total_bits},{config.bias.integer_bits}>",
                "accumulator_format": f"ap_fixed<{base.ACC_TOTAL_BITS},{base.ACC_INTEGER_BITS}>",
                "output_compare": config.output_compare,
                "one_dsp_shape_candidate": config.one_dsp_shape_candidate,
                "action_agreement_rate": float(np.mean(~disagreement)),
                "action_flip_count": int(disagreement.sum()),
                "minimum_float_margin_among_flips": float(np.min(margins[disagreement])) if disagreement.any() else np.nan,
                "max_abs_logit_diff": float(np.max(np.abs(error))),
                "mean_abs_logit_diff": float(np.mean(np.abs(error))),
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
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used": False,
            })
            range_rows.append({"split": split, "config": config.name, **audit})
    metrics = pd.DataFrame(rows)
    ranges = pd.DataFrame(range_rows)
    metrics.to_csv(RESULTS / "mixed_precision_sweep.csv", index=False)
    ranges.to_csv(RESULTS / "mixed_precision_range_audit.csv", index=False)

    val = metrics[metrics["split"] == "val"].merge(
        ranges[ranges["split"] == "val"][["config", "total_overflow_count"]], on="config"
    )
    eligible = val[
        (val["action_agreement_rate"] >= base.VAL_MIN_ACTION_AGREEMENT)
        & (val["fixed_pnl_retention_pct"] >= 100.0 * base.VAL_MIN_PNL_RETENTION)
        & (val["total_overflow_count"] == 0)
    ].copy()
    eligible["hardware_rank"] = (~eligible["one_dsp_shape_candidate"].astype(bool)).astype(int)
    eligible["total_width"] = eligible["data_format"].str.extract(r"<(\d+)")[0].astype(int) + eligible["weight_format"].str.extract(r"<(\d+)")[0].astype(int)
    if eligible.empty:
        selected = val.sort_values(
            ["action_agreement_rate", "fixed_pnl_retention_pct"], ascending=[False, False]
        ).iloc[0]
        status = "no_config_cleared_gate"
    else:
        selected = eligible.sort_values(
            ["hardware_rank", "total_width", "action_agreement_rate"], ascending=[True, True, False]
        ).iloc[0]
        status = "gate_pass"
    selected_name = str(selected["config"])
    selected_rows = metrics[metrics["config"] == selected_name]
    selected_ranges = ranges[ranges["config"] == selected_name]
    selected_rows.to_csv(RESULTS / "selected_mixed_precision_results.csv", index=False)
    selected_config = next(config for config in CONFIGS if config.name == selected_name)
    export_mixed_header(
        MODELS / f"tiny_mlp_h8_seed1_{selected_name}_weights.hpp",
        selected_config, w1, b1, w2, b2,
    )
    np.savez(
        MODELS / f"tiny_mlp_h8_seed1_{selected_name}_integer_weights.npz",
        w1=base.quantize(w1, selected_config.weight)[0],
        b1=base.quantize(b1, selected_config.bias)[0],
        w2=base.quantize(w2, selected_config.weight)[0],
        b2=base.quantize(b2, selected_config.bias)[0],
    )
    selection = {
        "status": status,
        "config": selected_name,
        "data_format": str(selected["data_format"]),
        "weight_format": str(selected["weight_format"]),
        "bias_format": str(selected["bias_format"]),
        "accumulator_format": str(selected["accumulator_format"]),
        "output_compare": str(selected["output_compare"]),
        "one_dsp_shape_candidate": bool(selected["one_dsp_shape_candidate"]),
        "selection_split": "val",
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used": False,
    }
    (RESULTS / "selected_mixed_precision.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    report_status = "resolved_positive" if status == "gate_pass" else "resolved_negative"
    lines = [
        "# Tiny MLP Mixed-Precision Fixed-Point Audit\n",
        f"status: `{report_status}`\n",
        "## Purpose\n",
        "This follow-up isolates two hardware-relevant changes without retraining: weights may use a narrower type than activations, and the action argmax compares the existing 40-bit accumulator logits directly instead of introducing an unnecessary output cast. Val selects the minimum passing hardware candidate; NEW3/NEW4 remain diagnostics. NEW5 was not loaded.\n",
        "## Candidate sweep\n",
        metrics.to_markdown(index=False, floatfmt=".9g") + "\n",
        "## Range audit\n",
        ranges.to_markdown(index=False) + "\n",
        "## Selected configuration\n",
        f"Selection status: `{status}`. Selected `{selected_name}`. `one_dsp_shape_candidate` only means the operand widths fit the nominal DSP48E1 25x18 multiplier shape; actual DSP inference and count still require csynth.\n",
        selected_rows.to_markdown(index=False, floatfmt=".9g") + "\n",
        selected_ranges.to_markdown(index=False) + "\n",
        "## Decision\n",
        (
            "The selected mixed-precision contract clears the software fixed-point gate and is ready for an independent manual-HLS CSim implementation. This is not yet evidence that Vitis maps every multiply to one DSP, meets timing, or preserves the emulator exactly; those claims require csim/csynth/cosim/P&R.\n"
            if status == "gate_pass"
            else "No tested mixed-precision contract clears the fixed-point gate. HLS should remain blocked pending arithmetic review.\n"
        ),
    ]
    (ROOT / "AUDIT_tiny_mlp_mixed_precision.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] selected_config={selected_name} status={status}", flush=True)
    print(selected_rows[[
        "split", "action_agreement_rate", "action_flip_count",
        "fixed_pnl_retention_pct", "pnl_delta_fixed_minus_float",
    ]].to_string(index=False), flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_tiny_mlp_mixed_precision.md'}", flush=True)


if __name__ == "__main__":
    main()
