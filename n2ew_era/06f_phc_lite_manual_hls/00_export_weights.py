from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"

STEP_CKPT_PATH = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml" / "models" / "phc_stp_step_hls_seed2.pt"
SOURCE_CKPT_PATH = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr" / "models" / "phc_stp_v2_seed2_best_val.pt"
STEP_REPORT_PATH = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml" / "PHC_STP_STEP_HLS4ML_FINAL_REPORT.md"
FIXED_POINT_REPORT_PATH = (
    PROJECT_ROOT
    / "n2ew_era"
    / "06b_phc_lite_streaming_fixed_point_equivalence"
    / "PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phc_stp_step_hls import PHCSTPStepHLS  # noqa: E402

TOTAL_BITS = 24
FRAC_BITS = 12
INT_MIN = -(2 ** (TOTAL_BITS - 1))
INT_MAX = 2 ** (TOTAL_BITS - 1) - 1
AP_TYPE = "ap_fixed<24,12>"
PHASE = "06f"

OMITTED_ROUTING_TENSORS = {
    "input_x.weight",
    "input_state.weight",
    "output_pack.weight",
    "output_pack.bias",
}

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
    for path in [ROOT, MODELS, RESULTS, HLS_SRC]:
        path.mkdir(parents=True, exist_ok=True)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_step_model() -> tuple[PHCSTPStepHLS, dict[str, Any]]:
    if not STEP_CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 06E step checkpoint: {STEP_CKPT_PATH}")
    payload = torch_load(STEP_CKPT_PATH)
    if payload.get("phase") != "06e":
        raise RuntimeError(f"Expected Phase 06E checkpoint, got phase={payload.get('phase')}")
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step checkpoint is marked new5_used=True.")
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


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
    raise ValueError(f"Only scalar/1D/2D tensors are supported, got shape {q.shape}")


def cpp_decl(name: str, q: np.ndarray) -> str:
    if q.ndim == 0:
        return f"static const {AP_TYPE} {name} = {fixed_literal(int(q.item()))};"
    if q.ndim == 1:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}] = {cpp_array_initializer(q)};"
    if q.ndim == 2:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}][{q.shape[1]}] = {cpp_array_initializer(q)};"
    raise ValueError(f"Unsupported tensor rank for {name}: {q.ndim}")


def tensor_float_stats(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def tensor_quant_stats(q: np.ndarray) -> tuple[int, int]:
    if q.size == 0:
        return 0, 0
    return int(np.min(q)), int(np.max(q))


def export_weights() -> pd.DataFrame:
    ensure_dirs()
    model, payload = load_step_model()
    state = model.state_dict()
    alpha_raw = float(state["alpha_scalar"].view(-1)[0].detach().cpu())
    if not (0.58 <= alpha_raw <= 0.61):
        raise RuntimeError(
            f"alpha_scalar={alpha_raw:.9f} is not the expected raw-alpha value around 0.5917. "
            "Do not export a sigmoid-transformed alpha."
        )

    header_blocks: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    quantized_cache: dict[str, np.ndarray] = {}

    for tensor_name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        q, saturated = q24_12(arr)
        quantized_cache[tensor_name] = q
        fmin, fmax = tensor_float_stats(arr)
        qmin, qmax = tensor_quant_stats(q)
        exported = tensor_name in EXPORT_MAP
        omitted_routing = tensor_name in OMITTED_ROUTING_TENSORS
        audit_rows.append(
            {
                "phase": PHASE,
                "name": tensor_name,
                "cpp_name": EXPORT_MAP.get(tensor_name, ""),
                "shape": "x".join(str(x) for x in arr.shape) if arr.shape else "scalar",
                "num_values": int(arr.size),
                "float_min": fmin,
                "float_max": fmax,
                "quantized_min": qmin,
                "quantized_max": qmax,
                "num_saturated": saturated,
                "total_bits": TOTAL_BITS,
                "frac_bits": FRAC_BITS,
                "exported_to_header": bool(exported),
                "omitted_direct_hls_plumbing": bool(omitted_routing),
                "note": "Implemented directly as slicing/state/output packing in C++ datapath." if omitted_routing else "",
            }
        )

    for tensor_name, cpp_name in EXPORT_MAP.items():
        if tensor_name not in quantized_cache:
            raise KeyError(f"Missing expected tensor in state_dict: {tensor_name}")
        header_blocks.append(f"// {tensor_name}")
        header_blocks.append(cpp_decl(cpp_name, quantized_cache[tensor_name]))
        header_blocks.append("")

    header = [
        "#pragma once",
        "",
        "#include <ap_fixed.h>",
        "",
        "// Generated by n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py",
        f"// Source PHC-STP checkpoint: {SOURCE_CKPT_PATH}",
        f"// Step checkpoint: {STEP_CKPT_PATH}",
        f"// Source seed: {payload.get('source_seed', 2)}",
        f"// Source best epoch: {payload.get('source_best_epoch', 'unknown')}",
        f"// Quantization: Q{TOTAL_BITS}.{FRAC_BITS} == ap_fixed<{TOTAL_BITS},{TOTAL_BITS - FRAC_BITS}>",
        "// Fixed routing layers input_x, input_state, and output_pack are intentionally",
        "// omitted from this header and should be implemented as direct slicing and",
        "// output packing in the HLS datapath.",
        "// alpha_scalar is the raw PHCLiteSTP alpha value, not sigmoid(alpha).",
        "",
        "namespace phc_step_weights {",
        "",
        f"using weight_t = {AP_TYPE};",
        "",
        "static const int FEATURE_DIM = 16;",
        "static const int HIDDEN_DIM = 16;",
        "static const int INPUT_DIM = 32;",
        "static const int OUTPUT_DIM = 18;",
        "static const int Q_TOTAL_BITS = 24;",
        "static const int Q_FRAC_BITS = 12;",
        "",
        *header_blocks,
        "}  // namespace phc_step_weights",
        "",
    ]
    (HLS_SRC / "phc_weights.hpp").write_text("\n".join(header), encoding="utf-8")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / "weight_export_audit.csv", index=False)

    exported_payload = {
        "phase": PHASE,
        "source_checkpoint": str(SOURCE_CKPT_PATH),
        "step_checkpoint": str(STEP_CKPT_PATH),
        "exported_header": str(HLS_SRC / "phc_weights.hpp"),
        "total_bits": TOTAL_BITS,
        "frac_bits": FRAC_BITS,
        "alpha_scalar_raw": alpha_raw,
        "any_saturated": bool(audit["num_saturated"].sum() > 0),
        "num_saturated_total": int(audit["num_saturated"].sum()),
        "omitted_routing_tensors": sorted(OMITTED_ROUTING_TENSORS),
    }
    (RESULTS / "weight_export_manifest.json").write_text(json.dumps(exported_payload, indent=2), encoding="utf-8")

    # Keep a copy of the exact step checkpoint in this phase for a self-contained package.
    phase_ckpt = MODELS / STEP_CKPT_PATH.name
    if not phase_ckpt.exists():
        phase_ckpt.write_bytes(STEP_CKPT_PATH.read_bytes())

    write_report(audit, exported_payload)
    return audit


def write_report(audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    total_saturated = int(audit["num_saturated"].sum()) if not audit.empty else -1
    saturation_status = "PASS" if total_saturated == 0 else "FAIL"
    exported = audit[audit["exported_to_header"]].copy()
    omitted = audit[audit["omitted_direct_hls_plumbing"]].copy()
    lines = [
        "# PHC Weight Export Report",
        "",
        f"Final quantization saturation status: `{saturation_status}`.",
        "",
        "## Source",
        "",
        f"- Source checkpoint: `{SOURCE_CKPT_PATH}`.",
        f"- Step checkpoint loaded directly: `{STEP_CKPT_PATH}`.",
        f"- Exported header: `{HLS_SRC / 'phc_weights.hpp'}`.",
        "- No training, no threshold selection, and no NEW5 access were performed.",
        f"- `alpha_scalar` confirmed as raw alpha: `{manifest['alpha_scalar_raw']:.9f}`. It is not `sigmoid(alpha)`.",
        "",
        "## Quantization Scheme",
        "",
        "`Q24.12 == ap_fixed<24,12>` is reused exactly from Phase 06B. The 06B fixed-point equivalence report showed action agreement around `0.998`, selected-action agreement around `0.998`, and max absolute causal-fill PnL delta `2,455` versus multi-million FP32 baseline PnL; material degradation was `False`.",
        "",
        "Quantization rule:",
        "",
        "```text",
        "q_int = round(float_value * 2^12)",
        "q_int = clip(q_int, -2^23, 2^23 - 1)",
        "ap_fixed_value = q_int / 2^12",
        "```",
        "",
        "## Routing Tensor Decision",
        "",
        "`input_x`, `input_state`, and `output_pack` are fixed routing/identity matrices introduced only to make the 06E step graph frontend-friendly. They are audited below but intentionally omitted from `phc_weights.hpp`; the HLS datapath should implement them as direct input slicing, state passthrough, and output packing. This avoids wasting multipliers or storage on identity matrices.",
        "",
        "## Exported Tensor Audit",
        "",
        exported.to_markdown(index=False),
        "",
        "## Omitted Routing Tensor Audit",
        "",
        omitted.to_markdown(index=False),
        "",
        "## Full Audit Table",
        "",
        audit.to_markdown(index=False),
        "",
        "## Saturation Decision",
        "",
        f"- Total saturated values: `{total_saturated}`.",
        f"- PASS/FAIL: `{saturation_status}`.",
    ]
    if total_saturated:
        lines.extend(
            [
                "",
                "**Saturation was detected. This materially weakens the fixed-point equivalence claim and must be investigated before manual HLS implementation proceeds.**",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No tensor values saturated under Q24.12. The manual HLS weight export is ready for the next C++ datapath implementation step.",
            ]
        )
    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
            f"- Phase 06E step equivalence report: `{STEP_REPORT_PATH}`.",
            f"- Phase 06B fixed-point evidence: `{FIXED_POINT_REPORT_PATH}`.",
            "",
            "This phase exports constants only. It does not prove FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.",
        ]
    )
    (ROOT / "PHC_WEIGHT_EXPORT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = export_weights()
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
