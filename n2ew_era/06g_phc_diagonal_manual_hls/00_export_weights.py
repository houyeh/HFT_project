from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06g_phc_diagonal_manual_hls"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"

PHASE05F = PROJECT_ROOT / "n2ew_era" / "05f_phc_diagonal_ssm"
SELECTED_POLICY_PATH = PHASE05F / "results" / "phc_diagonal_ssm_selected_policy.json"
MODEL_SRC_PATH = PHASE05F / "models"
PHASE05F_REPORT = PHASE05F / "PHC_DIAGONAL_SSM_ABLATION_REPORT.md"
FIXED_POINT_REPORT = (
    PROJECT_ROOT
    / "n2ew_era"
    / "06b_phc_lite_streaming_fixed_point_equivalence"
    / "PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md"
)

if str(PHASE05F) not in sys.path:
    sys.path.insert(0, str(PHASE05F))

from models.phc_diagonal_ssm import PHCDiagonalSSM  # noqa: E402

TOTAL_BITS = 24
FRAC_BITS = 12
INT_MIN = -(2 ** (TOTAL_BITS - 1))
INT_MAX = 2 ** (TOTAL_BITS - 1) - 1
AP_TYPE = "ap_fixed<24,12>"
PHASE = "06g"

EXPORT_MAP = {
    "encoder.0.weight": "ENC_LINEAR_W",
    "encoder.0.bias": "ENC_LINEAR_B",
    "encoder.1.weight": "ENC_NORM_GAMMA",
    "encoder.1.bias": "ENC_NORM_BETA",
    "diag_decay_baked_sigmoid": "DIAG_DECAY",
    "diag_gain_baked_sigmoid": "DIAG_GAIN",
    "readout.weight": "READOUT_W",
    "readout.bias": "READOUT_B",
}

RAW_PARAM_NAMES = {
    "diag_decay_param": "raw parameter; sigmoid is baked into DIAG_DECAY for HLS",
    "diag_gain_param": "raw parameter; sigmoid is baked into DIAG_GAIN for HLS",
}


def ensure_dirs() -> None:
    for path in [ROOT, MODELS, RESULTS, HLS_SRC]:
        path.mkdir(parents=True, exist_ok=True)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def selected_seed() -> tuple[int, dict[str, Any]]:
    payload = json.loads(SELECTED_POLICY_PATH.read_text(encoding="utf-8"))
    if payload.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: Phase 05F selected policy is marked new5_used_for_selection=True")
    seed = int(payload["seed"])
    if payload.get("model") != "phc_diagonal_ssm":
        raise RuntimeError(f"Unexpected selected model: {payload.get('model')}")
    return seed, payload


def load_selected_model() -> tuple[PHCDiagonalSSM, dict[str, Any], dict[str, Any], Path]:
    seed, selected = selected_seed()
    ckpt_path = MODEL_SRC_PATH / f"phc_diagonal_ssm_seed{seed}_best_val.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing selected 05F checkpoint: {ckpt_path}")
    ckpt = torch_load(ckpt_path)
    if bool(ckpt.get("new5_used_for_selection", False)):
        raise RuntimeError("Hard rule violation: checkpoint marked new5_used_for_selection=True")
    if ckpt.get("phase") != "05f":
        raise RuntimeError(f"Expected phase 05f checkpoint, got {ckpt.get('phase')}")
    feature_dim = int(ckpt["feature_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    model = PHCDiagonalSSM(feature_dim=feature_dim, hidden_dim=hidden_dim, mtl_steps=int(ckpt.get("mtl_steps", 1)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, selected, ckpt_path


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
    if q.ndim == 1:
        return "{ " + ", ".join(fixed_literal(int(v)) for v in q.tolist()) + " }"
    if q.ndim == 2:
        rows = []
        for row in q:
            rows.append(spaces + "    { " + ", ".join(fixed_literal(int(v)) for v in row.tolist()) + " }")
        return "{\n" + ",\n".join(rows) + "\n" + spaces + "}"
    raise ValueError(f"Only 1D/2D tensors are supported, got shape {q.shape}")


def cpp_decl(name: str, q: np.ndarray) -> str:
    if q.ndim == 1:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}] = {cpp_array_initializer(q)};"
    if q.ndim == 2:
        return f"static const {AP_TYPE} {name}[{q.shape[0]}][{q.shape[1]}] = {cpp_array_initializer(q)};"
    raise ValueError(f"Unsupported tensor rank for {name}: {q.ndim}")


def tensor_stats(arr: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {
        "shape": "x".join(str(x) for x in arr.shape),
        "num_values": int(arr.size),
        "float_min": float(np.nanmin(arr)),
        "float_max": float(np.nanmax(arr)),
        "quantized_min": int(np.min(q)),
        "quantized_max": int(np.max(q)),
    }


def export_weights() -> pd.DataFrame:
    ensure_dirs()
    model, ckpt, selected, source_ckpt_path = load_selected_model()
    state = model.state_dict()
    tensors: dict[str, np.ndarray] = {
        "encoder.0.weight": state["encoder.0.weight"].detach().cpu().numpy().astype(np.float32),
        "encoder.0.bias": state["encoder.0.bias"].detach().cpu().numpy().astype(np.float32),
        "encoder.1.weight": state["encoder.1.weight"].detach().cpu().numpy().astype(np.float32),
        "encoder.1.bias": state["encoder.1.bias"].detach().cpu().numpy().astype(np.float32),
        "diag_decay_baked_sigmoid": torch.sigmoid(state["diag_decay_param"]).detach().cpu().numpy().astype(np.float32),
        "diag_gain_baked_sigmoid": torch.sigmoid(state["diag_gain_param"]).detach().cpu().numpy().astype(np.float32),
        "readout.weight": state["readout.weight"].detach().cpu().numpy().astype(np.float32),
        "readout.bias": state["readout.bias"].detach().cpu().numpy().astype(np.float32),
    }
    raw_audit = {
        "diag_decay_param": state["diag_decay_param"].detach().cpu().numpy().astype(np.float32),
        "diag_gain_param": state["diag_gain_param"].detach().cpu().numpy().astype(np.float32),
    }

    rows: list[dict[str, Any]] = []
    header_blocks: list[str] = []
    for name, arr in tensors.items():
        q, saturated = q24_12(arr)
        cpp_name = EXPORT_MAP[name]
        rows.append(
            {
                "phase": PHASE,
                "name": name,
                "cpp_name": cpp_name,
                **tensor_stats(arr, q),
                "num_saturated": saturated,
                "total_bits": TOTAL_BITS,
                "frac_bits": FRAC_BITS,
                "exported_to_header": True,
                "note": "sigmoid baked into exported constant" if "baked_sigmoid" in name else "",
            }
        )
        header_blocks.append(f"// {name}")
        header_blocks.append(cpp_decl(cpp_name, q))
        header_blocks.append("")

    for name, arr in raw_audit.items():
        q, saturated = q24_12(arr)
        rows.append(
            {
                "phase": PHASE,
                "name": name,
                "cpp_name": "",
                **tensor_stats(arr, q),
                "num_saturated": saturated,
                "total_bits": TOTAL_BITS,
                "frac_bits": FRAC_BITS,
                "exported_to_header": False,
                "note": RAW_PARAM_NAMES[name],
            }
        )

    header_lines = [
        "#pragma once",
        "",
        "#include <ap_fixed.h>",
        "",
        "// Generated by n2ew_era/06g_phc_diagonal_manual_hls/00_export_weights.py",
        f"// Source 05F checkpoint: {source_ckpt_path}",
        f"// Selected seed: {selected['seed']}",
        f"// Selected best epoch: {selected['validation_metrics']['best_epoch']}",
        f"// Quantization: Q{TOTAL_BITS}.{FRAC_BITS} == ap_fixed<{TOTAL_BITS},{TOTAL_BITS - FRAC_BITS}>",
        "// DIAG_DECAY and DIAG_GAIN are sigmoid(raw_param) values baked into constants.",
        "// There are no dense gate/candidate/synapse/gain matrices in PHCDiagonalSSM.",
        "",
        "namespace phc_diag_weights {",
        "",
        f"using weight_t = {AP_TYPE};",
        "static const int FEATURE_DIM = 16;",
        "static const int HIDDEN_DIM = 16;",
        "static const int READOUT_DIM = 2;",
        "static const int Q_TOTAL_BITS = 24;",
        "static const int Q_FRAC_BITS = 12;",
        "",
        *header_blocks,
        "}  // namespace phc_diag_weights",
        "",
    ]
    (HLS_SRC / "phc_diag_weights.hpp").write_text("\n".join(header_lines), encoding="utf-8")

    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "weight_export_audit.csv", index=False)
    phase_ckpt = MODELS / source_ckpt_path.name
    if not phase_ckpt.exists():
        phase_ckpt.write_bytes(source_ckpt_path.read_bytes())

    manifest = {
        "phase": PHASE,
        "source_checkpoint": str(source_ckpt_path),
        "selected_policy": str(SELECTED_POLICY_PATH),
        "selected_seed": int(selected["seed"]),
        "selected_best_epoch": int(selected["validation_metrics"]["best_epoch"]),
        "exported_header": str(HLS_SRC / "phc_diag_weights.hpp"),
        "total_bits": TOTAL_BITS,
        "frac_bits": FRAC_BITS,
        "diag_decay_export": "sigmoid(diag_decay_param) baked into DIAG_DECAY",
        "diag_gain_export": "sigmoid(diag_gain_param) baked into DIAG_GAIN",
        "num_saturated_total": int(audit["num_saturated"].sum()),
        "any_saturated": bool(audit["num_saturated"].sum() > 0),
        "new5_used": False,
    }
    (RESULTS / "weight_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(audit, manifest)
    return audit


def write_report(audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    total_saturated = int(audit["num_saturated"].sum())
    saturation_status = "PASS" if total_saturated == 0 else "FAIL"
    exported = audit[audit["exported_to_header"]].copy()
    raw = audit[~audit["exported_to_header"]].copy()
    lines = [
        "# PHC Diagonal SSM Weight Export Report",
        "",
        f"Final quantization saturation status: `{saturation_status}`.",
        "",
        "## Source",
        "",
        f"- Source checkpoint: `{manifest['source_checkpoint']}`.",
        f"- Selected policy JSON: `{manifest['selected_policy']}`.",
        f"- Selected seed: `{manifest['selected_seed']}`.",
        f"- Selected best epoch: `{manifest['selected_best_epoch']}`.",
        f"- Exported header: `{manifest['exported_header']}`.",
        "- No training, no threshold selection, and no NEW5 access were performed.",
        "",
        "## Quantization Scheme",
        "",
        "`Q24.12 == ap_fixed<24,12>` is reused exactly from Phase 06B/06F for comparability.",
        "",
        "```text",
        "q_int = round(float_value * 2^12)",
        "q_int = clip(q_int, -2^23, 2^23 - 1)",
        "ap_fixed_value = q_int / 2^12",
        "```",
        "",
        "## Diagonal Recurrence Export Decision",
        "",
        "`DIAG_DECAY` and `DIAG_GAIN` are exported as `sigmoid(raw_param)` constants. This removes runtime sigmoid operations from the HLS recurrence stage. The raw parameters are still audited below but are intentionally not exported to the HLS header.",
        "",
        "There are no gate/candidate/synapse/gain-from-encoded/gain-from-state tensors in PHCDiagonalSSM, so none are exported or invented.",
        "",
        "## Exported Tensor Audit",
        "",
        exported.to_markdown(index=False),
        "",
        "## Raw Parameter Audit",
        "",
        raw.to_markdown(index=False),
        "",
        "## Saturation Decision",
        "",
        f"- Total saturated values: `{total_saturated}`.",
        f"- PASS/FAIL: `{saturation_status}`.",
    ]
    if total_saturated:
        lines.append("**Saturation was detected and must be investigated before proceeding.**")
    else:
        lines.append("No tensor values saturated under Q24.12.")
    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
            f"- Phase 05F report: `{PHASE05F_REPORT}`.",
            f"- Phase 06B fixed-point evidence: `{FIXED_POINT_REPORT}`.",
            "",
            "This phase exports constants only. It does not prove FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.",
        ]
    )
    (ROOT / "PHC_DIAGONAL_WEIGHT_EXPORT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = export_weights()
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
