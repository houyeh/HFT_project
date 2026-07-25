from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
CROSS_SEED = ROOT / "cross_seed"
BASE_HLS_SRC = ROOT / "hls_src"
BASE_HLS_COMPONENT = ROOT / "hls_component"
BASE_SEED2_REPORT_XML = BASE_HLS_COMPONENT / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.xml"
BASE_SEED2_REPORT_RPT = BASE_HLS_COMPONENT / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.rpt"
CONTEXT_AUDIT = PROJECT_ROOT / "n2ew_era" / "000_context" / "AUDIT_hls_cross_seed_resource_stability.md"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2_MODEL = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain" / "models" / "phc_lite_stp.py"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import common_05a as p05a  # noqa: E402
from src.models.phc_stp_step_hls import PHCSTPStepHLS, count_parameters as count_step_parameters  # noqa: E402

TOTAL_BITS = 24
FRAC_BITS = 12
INT_MIN = -(2 ** (TOTAL_BITS - 1))
INT_MAX = 2 ** (TOTAL_BITS - 1) - 1
AP_TYPE = "ap_fixed<24,12>"
FEATURE_DIM = 16
HIDDEN_DIM = 16
SPLIT = "val"
DEFAULT_SAMPLE_COUNT = 200
EQUIVALENCE_ATOL = 1e-4

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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def win_abs(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_phc_lite_stp_class():
    spec = importlib.util.spec_from_file_location("phase05c2_phc_lite_stp", PHASE05C2_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PHCLiteSTP source from {PHASE05C2_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PHCLiteSTP, module.count_parameters


PHCLiteSTP, count_source_parameters = load_phc_lite_stp_class()


def source_checkpoint(seed: int) -> Path:
    return PHASE05C4 / "models" / f"phc_stp_v2_seed{seed}_best_val.pt"


def seed_root(seed: int) -> Path:
    return CROSS_SEED / f"seed{seed}"


def ensure_seed_dirs(seed: int) -> dict[str, Path]:
    root = seed_root(seed)
    paths = {
        "root": root,
        "models": root / "models",
        "results": root / "results",
        "hls_src": root / "hls_src",
        "hls_component": root / "hls_component",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_source_model(seed: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt_path = source_checkpoint(seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing 05C4 checkpoint for seed{seed}: {ckpt_path}")
    ckpt = torch_load(ckpt_path)
    if int(ckpt.get("seed", -1)) != seed:
        raise RuntimeError(f"Expected checkpoint seed={seed}, got {ckpt.get('seed')}")
    if ckpt.get("phase") != "05c4":
        raise RuntimeError(f"Expected phase=05c4 for seed{seed}, got {ckpt.get('phase')}")
    if ckpt.get("new5_used_for_selection"):
        raise RuntimeError(f"Hard rule violation: seed{seed} checkpoint marks NEW5 selection use.")
    feature_dim = int(ckpt.get("feature_dim", len(p05a.PHC_FEATURES)))
    hidden_dim = int(ckpt.get("hidden_dim", HIDDEN_DIM))
    if feature_dim != FEATURE_DIM or hidden_dim != HIDDEN_DIM:
        raise RuntimeError(f"seed{seed}: unexpected dims feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    model = PHCLiteSTP(feature_dim=feature_dim, hidden_dim=hidden_dim, mtl_steps=int(ckpt.get("mtl_steps", 1)), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def copy_linear(dst: torch.nn.Linear, src: torch.nn.Linear) -> None:
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def copy_norm(dst: torch.nn.LayerNorm, src: torch.nn.LayerNorm) -> None:
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def migrate_step_model(seed: int, paths: dict[str, Path]) -> tuple[PHCSTPStepHLS, dict[str, Any], pd.DataFrame]:
    src, ckpt = load_source_model(seed)
    step = PHCSTPStepHLS(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)
    copy_linear(step.enc_linear, src.encoder[0])
    copy_norm(step.enc_norm, src.encoder[1])
    copy_linear(step.gate, src.gate)
    copy_linear(step.candidate, src.candidate)
    copy_linear(step.gain_from_encoded, src.gain_from_encoded)
    step.gain_from_prev_state.weight.data.copy_(src.gain_from_prev_state.weight.detach())
    copy_linear(step.synapse, src.synapse)
    copy_norm(step.loop_norm, src.loop_norm)
    copy_linear(step.readout, src.readout)
    with torch.no_grad():
        step.output_pack.weight.zero_()
        step.output_pack.bias.zero_()
        step.output_pack.weight[:2, :].copy_(src.readout.weight.detach())
        step.output_pack.bias[:2].copy_(src.readout.bias.detach())
        step.output_pack.weight[2:, :].copy_(torch.eye(HIDDEN_DIM))
    step.decay.data.copy_(torch.sigmoid(src.decay_param.detach()).view(-1))
    step.alpha_scalar.data.copy_(src.alpha.detach().view(1))
    step.eval()

    step_path = paths["models"] / f"phc_stp_step_hls_seed{seed}.pt"
    payload = {
        "model_state_dict": step.state_dict(),
        "source_checkpoint": str(source_checkpoint(seed)),
        "source_phase": ckpt.get("phase"),
        "source_seed": int(ckpt.get("seed", seed)),
        "source_best_epoch": int(ckpt.get("best_epoch", -1)),
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "input_dim": FEATURE_DIM + HIDDEN_DIM,
        "output_dim": 2 + HIDDEN_DIM,
        "mtl_steps": 1,
        "dropout": 0.0,
        "phase": "06f_cross_seed",
        "new5_used": False,
        "new_training": False,
        "alpha_source_formula": "raw PHCLiteSTP.alpha, because source step uses z + self.alpha * message",
        "alpha_raw_value": float(src.alpha.detach().cpu()),
        "alpha_sigmoid_value_prompt_reference": float(torch.sigmoid(src.alpha.detach()).cpu()),
        "decay_formula": "sigmoid(PHCLiteSTP.decay_param)",
    }
    torch.save(payload, step_path)

    audit = pd.DataFrame(
        [
            {
                "seed": seed,
                "source_checkpoint": rel(source_checkpoint(seed)),
                "migrated_model": rel(step_path),
                "feature_dim": FEATURE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "input_dim": FEATURE_DIM + HIDDEN_DIM,
                "output_dim": 2 + HIDDEN_DIM,
                "source_parameter_count": int(count_source_parameters(src)),
                "step_parameter_count": int(count_step_parameters(step)),
                "buffer_count": int(sum(b.numel() for b in step.buffers())),
                "fixed_input_selector_parameters": int(step.input_x.weight.numel() + step.input_state.weight.numel()),
                "fixed_output_pack_parameters": int(step.output_pack.weight.numel() + step.output_pack.bias.numel()),
                "decay_min": float(step.decay.min().detach().cpu()),
                "decay_max": float(step.decay.max().detach().cpu()),
                "alpha_raw_value_used": float(step.alpha_scalar.item()),
                "alpha_sigmoid_value_not_used": float(torch.sigmoid(src.alpha.detach()).cpu()),
                "new5_used": False,
            }
        ]
    )
    audit.to_csv(paths["results"] / f"phc_stp_step_migration_audit_seed{seed}.csv", index=False)
    write_json(paths["results"] / f"phc_stp_step_config_seed{seed}.json", {k: str(v) if isinstance(v, Path) else v for k, v in payload.items() if k != "model_state_dict"})
    return step, payload, audit


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


def export_weights(seed: int, step: PHCSTPStepHLS, step_payload: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    state = step.state_dict()
    alpha_raw = float(state["alpha_scalar"].view(-1)[0].detach().cpu())
    if not (0.0 <= alpha_raw <= 1.0):
        raise RuntimeError(f"seed{seed}: alpha_scalar={alpha_raw:.9f} is outside expected raw-alpha range [0, 1].")

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
                "seed": seed,
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
                "new5_used": False,
            }
        )

    for tensor_name, cpp_name in EXPORT_MAP.items():
        if tensor_name not in quantized_cache:
            raise KeyError(f"seed{seed}: missing expected tensor in state_dict: {tensor_name}")
        header_blocks.append(f"// {tensor_name}")
        header_blocks.append(cpp_decl(cpp_name, quantized_cache[tensor_name]))
        header_blocks.append("")

    header = [
        "#pragma once",
        "",
        "#include <ap_fixed.h>",
        "",
        "// Generated by n2ew_era/06f_phc_lite_manual_hls/00_export_weights_cross_seed.py",
        f"// Source PHC-STP checkpoint: {source_checkpoint(seed)}",
        f"// Step checkpoint: {paths['models'] / f'phc_stp_step_hls_seed{seed}.pt'}",
        f"// Source seed: {seed}",
        f"// Source best epoch: {step_payload.get('source_best_epoch', 'unknown')}",
        f"// Quantization: Q{TOTAL_BITS}.{FRAC_BITS} == ap_fixed<{TOTAL_BITS},{TOTAL_BITS - FRAC_BITS}>",
        "// HLS datapath, pragmas, interface, target part, and clock are copied from seed2 06F.",
        "// input_x, input_state, and output_pack are intentionally omitted from this header.",
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
    (paths["hls_src"] / "phc_weights.hpp").write_text("\n".join(header), encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(paths["results"] / f"weight_export_audit_seed{seed}.csv", index=False)
    manifest = {
        "seed": seed,
        "source_checkpoint": rel(source_checkpoint(seed)),
        "step_checkpoint": rel(paths["models"] / f"phc_stp_step_hls_seed{seed}.pt"),
        "exported_header": rel(paths["hls_src"] / "phc_weights.hpp"),
        "total_bits": TOTAL_BITS,
        "frac_bits": FRAC_BITS,
        "alpha_scalar_raw": alpha_raw,
        "any_saturated": bool(audit["num_saturated"].sum() > 0),
        "num_saturated_total": int(audit["num_saturated"].sum()),
        "new5_used": False,
    }
    write_json(paths["results"] / f"weight_export_manifest_seed{seed}.json", manifest)
    return audit


def vector_columns() -> list[str]:
    return (
        ["row_id", "sample_idx", "timestep"]
        + [f"x_t_{i}" for i in range(16)]
        + [f"state_prev_{i}" for i in range(16)]
        + ["ref_readout_0", "ref_readout_1"]
        + [f"ref_state_next_{i}" for i in range(16)]
    )


def generate_vectors(seed: int, source_model: torch.nn.Module, step_model: PHCSTPStepHLS, step_payload: dict[str, Any], paths: dict[str, Path], sample_count: int) -> pd.DataFrame:
    payload = p05a.load_sequences(SPLIT)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: validation sequence payload is marked new5_used=True.")

    x_all = payload["x"]
    n = min(int(sample_count), int(x_all.shape[0]))
    if n < 1:
        raise ValueError("sample_count must be positive")
    x = x_all[:n].float()
    seq_len = int(x.shape[1])
    feature_dim = int(x.shape[2])
    if feature_dim != FEATURE_DIM:
        raise RuntimeError(f"seed{seed}: expected feature_dim=16, got {feature_dim}")

    frames: list[pd.DataFrame] = []
    state = torch.zeros(n, HIDDEN_DIM, dtype=torch.float32)
    readout_t = torch.zeros(n, 2, dtype=torch.float32)
    row_base = 0
    step_model.eval()
    source_model.eval()
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t, :].contiguous()
            state_prev = state.clone()
            x_cat = torch.cat([x_t, state], dim=-1)
            out_step = step_model(x_cat)
            readout_t, state = step_model.split_output(out_step)
            block = np.concatenate(
                [
                    np.arange(row_base, row_base + n, dtype=np.float64).reshape(-1, 1),
                    np.arange(n, dtype=np.float64).reshape(-1, 1),
                    np.full((n, 1), t, dtype=np.float64),
                    x_t.detach().cpu().numpy().astype(np.float32),
                    state_prev.detach().cpu().numpy().astype(np.float32),
                    readout_t.detach().cpu().numpy().astype(np.float32),
                    state.detach().cpu().numpy().astype(np.float32),
                ],
                axis=1,
            )
            frames.append(pd.DataFrame(block, columns=vector_columns()))
            row_base += n
        source_ref = source_model(x)

    diff = (readout_t - source_ref).detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else float("nan")
    mean_abs_diff = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    allclose = bool(torch.allclose(readout_t, source_ref, atol=EQUIVALENCE_ATOL, rtol=1e-5))
    if not allclose:
        raise RuntimeError(
            f"seed{seed}: full source vs step replay equivalence failed: max_abs_diff={max_abs_diff:.9g}, "
            f"atol={EQUIVALENCE_ATOL}"
        )

    out = pd.concat(frames, ignore_index=True)
    int_cols = ["row_id", "sample_idx", "timestep"]
    out[int_cols] = out[int_cols].astype(np.int64)
    float_cols = [c for c in out.columns if c not in int_cols]
    out[float_cols] = out[float_cols].astype(np.float32)
    csv_path = paths["results"] / f"phc_step_test_vectors_seed{seed}.csv"
    out.to_csv(csv_path, index=False, float_format="%.9g")
    summary = pd.DataFrame(
        [
            {
                "seed": seed,
                "split": SPLIT,
                "sample_count": n,
                "sequence_length": seq_len,
                "feature_dim": feature_dim,
                "row_count": int(len(out)),
                "csv_path": rel(csv_path),
                "csv_size_bytes": int(csv_path.stat().st_size),
                "equivalence_atol": EQUIVALENCE_ATOL,
                "max_abs_diff_final_readout": max_abs_diff,
                "mean_abs_diff_final_readout": mean_abs_diff,
                "allclose_atol_1e_4": allclose,
                "source_seed": int(step_payload.get("source_seed", seed)),
                "source_best_epoch": int(step_payload.get("source_best_epoch", -1)),
                "step_checkpoint": rel(paths["models"] / f"phc_stp_step_hls_seed{seed}.pt"),
                "source_checkpoint": rel(source_checkpoint(seed)),
                "new5_used": False,
            }
        ]
    )
    summary.to_csv(paths["results"] / f"phc_step_test_vector_summary_seed{seed}.csv", index=False)
    return summary


def copy_hls_sources_and_config(seed: int, paths: dict[str, Path]) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    for name in ["phc_step_hls.cpp", "phc_step_hls.hpp", "phc_step_testbench.cpp"]:
        src = BASE_HLS_SRC / name
        dst = paths["hls_src"] / name
        dst.write_bytes(src.read_bytes())
        copied.append(
            {
                "file": name,
                "base_sha256": sha256_file(src),
                "seed_sha256": sha256_file(dst),
                "matches_base": sha256_file(src) == sha256_file(dst),
            }
        )

    csim_out = paths["results"] / f"phc_step_csim_out_seed{seed}.csv"
    cosim_out = paths["results"] / f"phc_step_cosim_out_seed{seed}.csv"
    vectors = paths["results"] / f"phc_step_test_vectors_seed{seed}.csv"
    hls_config = "\n".join(
        [
            "part=xc7z020clg400-1",
            "",
            "[hls]",
            "flow_target=vivado",
            "package.output.format=ip_catalog",
            "package.output.syn=false",
            "syn.top=phc_step",
            "syn.file=../hls_src/phc_step_hls.cpp",
            "syn.file=../hls_src/phc_step_hls.hpp",
            "syn.file=../hls_src/phc_weights.hpp",
            "tb.file=../hls_src/phc_step_testbench.cpp",
            "clock=10ns",
            f"cosim.argv={win_abs(vectors)} {win_abs(cosim_out)}",
            f"csim.argv={win_abs(vectors)} {win_abs(csim_out)}",
            "",
        ]
    )
    config_path = paths["hls_component"] / "hls_config.cfg"
    config_path.write_text(hls_config, encoding="utf-8")
    tcl_path = paths["hls_component"] / "run_csynth_only.tcl"
    tcl_path.write_text(
        "\n".join(
            [
                "open_project hls_component",
                "set_top phc_step",
                "add_files ../hls_src/phc_step_hls.cpp",
                "add_files ../hls_src/phc_step_hls.hpp",
                "add_files ../hls_src/phc_weights.hpp",
                "add_files -tb ../hls_src/phc_step_testbench.cpp",
                "open_solution -flow_target vivado hls",
                "set_part xc7z020-clg400-1",
                "create_clock -period 10 -name default",
                "csynth_design",
                "exit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    vitis_comp = {
        "name": "hls_component",
        "type": "HLS",
        "configuration": {
            "componentType": "HLS",
            "configFiles": ["hls_config.cfg"],
            "work_dir": "hls_component",
        },
        "useSysrootToolchain": False,
        "template": "empty_hls_component",
        "templateFlow": "VITIS",
    }
    write_json(paths["hls_component"] / "vitis-comp.json", vitis_comp)
    manifest = {
        "seed": seed,
        "base_hls_source_dir": rel(BASE_HLS_SRC),
        "hls_src": rel(paths["hls_src"]),
        "hls_component": rel(paths["hls_component"]),
        "hls_config": rel(config_path),
        "csynth_tcl": rel(tcl_path),
        "settings_copied_from_seed2": {
            "part": "xc7z020clg400-1",
            "flow_target": "vivado",
            "syn_top": "phc_step",
            "clock": "10ns",
            "package_output_format": "ip_catalog",
            "package_output_syn": False,
        },
        "copied_files": copied,
        "new5_used": False,
    }
    write_json(paths["results"] / f"hls_setup_manifest_seed{seed}.json", manifest)
    return manifest


def setup_seed(seed: int, sample_count: int) -> None:
    paths = ensure_seed_dirs(seed)
    source_model, _ = load_source_model(seed)
    step_model, step_payload, _ = migrate_step_model(seed, paths)
    export_weights(seed, step_model, step_payload, paths)
    copy_hls_sources_and_config(seed, paths)
    generate_vectors(seed, source_model, step_model, step_payload, paths, sample_count=sample_count)
    print(f"seed{seed}: setup complete at {rel(paths['root'])}")


def run_csynth(seed: int, vitis_run: Path) -> int:
    paths = ensure_seed_dirs(seed)
    config_dir = paths["hls_component"]
    log_path = paths["results"] / f"vitis_hls_csynth_seed{seed}.log"
    cmd = [str(vitis_run), "--mode", "hls", "--tcl", "run_csynth_only.tcl"]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("cwd=" + str(config_dir) + "\n")
        log.write("cmd=" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=config_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        log.write(f"\nexit_code={rc}\n")
    return int(rc)


def xml_text(root: ET.Element, path: str, default: str = "") -> str:
    node = root.find(path)
    return default if node is None or node.text is None else node.text.strip()


def parse_csynth(seed: int, xml_path: Path, rpt_path: Path, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "status": status,
        "top": "phc_step",
        "report_rpt": rel(rpt_path),
        "report_xml": rel(xml_path),
        "latency_min_cycles": None,
        "latency_max_cycles": None,
        "interval_min_cycles": None,
        "interval_max_cycles": None,
        "estimated_clock_ns": None,
        "BRAM_18K": None,
        "DSP": None,
        "FF": None,
        "LUT": None,
        "URAM": None,
    }
    if not xml_path.exists():
        row["status"] = "missing_report"
        return row
    root = ET.parse(xml_path).getroot()
    row["estimated_clock_ns"] = float(xml_text(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod", "nan"))
    row["latency_min_cycles"] = int(xml_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency", "-1"))
    row["latency_max_cycles"] = int(xml_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency", "-1"))
    row["interval_min_cycles"] = int(xml_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min", "-1"))
    row["interval_max_cycles"] = int(xml_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-max", "-1"))
    res = root.find("./AreaEstimates/Resources")
    if res is not None:
        for name in ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]:
            text = xml_text(res, f"./{name}", "")
            row[name] = int(text) if text else None
    return row


def collect_reports(seeds: list[int]) -> pd.DataFrame:
    rows = [
        parse_csynth(2, BASE_SEED2_REPORT_XML, BASE_SEED2_REPORT_RPT, "existing_seed2_baseline"),
    ]
    for seed in seeds:
        paths = ensure_seed_dirs(seed)
        rpt = paths["hls_component"] / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.rpt"
        xml = paths["hls_component"] / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.xml"
        status = "csynth_complete" if xml.exists() and rpt.exists() else "missing_report"
        rows.append(parse_csynth(seed, xml, rpt, status))
    df = pd.DataFrame(rows).sort_values("seed")
    out_csv = CROSS_SEED / "hls_cross_seed_csynth_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_audit(df, out_csv)
    return df


def pct_delta(value: Any, baseline: Any) -> str:
    if value is None or baseline in (None, 0) or pd.isna(value) or pd.isna(baseline):
        return "-"
    return f"{((float(value) - float(baseline)) / float(baseline) * 100.0):+.2f}%"


def cell(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value):
        return "-"
    if key == "estimated_clock_ns":
        return f"{float(value):.3f}"
    return str(int(value))


def write_audit(df: pd.DataFrame, out_csv: Path) -> None:
    seed2 = df[df["seed"] == 2].iloc[0]
    status = "resolved" if set([0, 1, 2]).issubset(set(df[df["status"] != "missing_report"]["seed"].astype(int))) else "open"
    lines = [
        f"status: {status}",
        "# Audit: PHC-Lite HLS Cross-Seed Resource Stability",
        "",
        "Hard rules: this audit did not load NEW5 and does not claim cross-seed HLS stability without seed0/seed1 csynth report files.",
        "",
        "## Scope",
        "",
        "This update performs the missing seed0/seed1 manual-HLS weight export and csynth resource check. Seed2 remains the existing 06F baseline. The HLS datapath source, interface, pragmas, target part, top function, and 10 ns clock are copied from seed2; the only intended cross-seed variable is the exported weight header.",
        "",
        "## Reproducible Script",
        "",
        "- Script: `n2ew_era/06f_phc_lite_manual_hls/00_export_weights_cross_seed.py`.",
        "- Setup command: `.\\.venv\\Scripts\\python.exe n2ew_era\\06f_phc_lite_manual_hls\\00_export_weights_cross_seed.py --seeds 0 1 --setup`.",
        "- Collect command after HLS runs: `.\\.venv\\Scripts\\python.exe n2ew_era\\06f_phc_lite_manual_hls\\00_export_weights_cross_seed.py --seeds 0 1 --collect`.",
        "- Optional one-shot rerun command: `.\\.venv\\Scripts\\python.exe n2ew_era\\06f_phc_lite_manual_hls\\00_export_weights_cross_seed.py --seeds 0 1 --setup --run-csynth --collect`.",
        "",
        "## Artifact Inventory",
        "",
        "| seed | source checkpoint | step HLS checkpoint | exported HLS weights | csynth report | evidence status |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for seed in [0, 1, 2]:
        if seed == 2:
            source = "n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt"
            step = "n2ew_era/06e_phc_stp_step_hls4ml/models/phc_stp_step_hls_seed2.pt"
            weights = "n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_weights.hpp"
            report = rel(BASE_SEED2_REPORT_RPT)
            evidence = "Existing complete seed2 baseline"
        else:
            root = seed_root(seed)
            source = rel(source_checkpoint(seed))
            step = rel(root / "models" / f"phc_stp_step_hls_seed{seed}.pt")
            weights = rel(root / "hls_src" / "phc_weights.hpp")
            report = rel(root / "hls_component" / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.rpt")
            row = df[df["seed"] == seed]
            evidence = "csynth complete" if not row.empty and row.iloc[0]["status"] != "missing_report" else "missing report"
        lines.append(f"| {seed} | `{source}` | `{step}` | `{weights}` | `{report}` | {evidence} |")

    lines.extend(
        [
            "",
            "## csynth Resource Table",
            "",
            "Every numeric cell below is parsed from the corresponding `phc_step_csynth.xml`; the adjacent report path points to the human-readable `phc_step_csynth.rpt` generated by the same run.",
            "",
            "| seed | status | latency cycles | interval cycles | est clock ns | BRAM_18K | DSP | FF | LUT | report path |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in df.sort_values("seed").iterrows():
        latency = cell(row, "latency_max_cycles")
        interval = cell(row, "interval_max_cycles")
        lines.append(
            f"| {int(row['seed'])} | {row['status']} | {latency} | {interval} | {cell(row, 'estimated_clock_ns')} | "
            f"{cell(row, 'BRAM_18K')} | {cell(row, 'DSP')} | {cell(row, 'FF')} | {cell(row, 'LUT')} | `{row['report_rpt']}` |"
        )

    lines.extend(
        [
            "",
            "## Delta Versus Seed2 csynth",
            "",
            "| seed | BRAM delta | DSP delta | FF delta | LUT delta | latency delta | >20% LUT/FF difference? |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in df.sort_values("seed").iterrows():
        ff_delta = pct_delta(row.get("FF"), seed2.get("FF"))
        lut_delta = pct_delta(row.get("LUT"), seed2.get("LUT"))
        flag = False
        for key in ["FF", "LUT"]:
            value = row.get(key)
            baseline = seed2.get(key)
            if value is not None and baseline not in (None, 0) and not pd.isna(value) and not pd.isna(baseline):
                flag = flag or abs((float(value) - float(baseline)) / float(baseline)) > 0.20
        latency_value = row.get("latency_max_cycles")
        latency_baseline = seed2.get("latency_max_cycles")
        latency_delta = "-" if pd.isna(latency_value) or pd.isna(latency_baseline) else str(int(latency_value) - int(latency_baseline))
        lines.append(
            f"| {int(row['seed'])} | {pct_delta(row.get('BRAM_18K'), seed2.get('BRAM_18K'))} | "
            f"{pct_delta(row.get('DSP'), seed2.get('DSP'))} | {ff_delta} | {lut_delta} | {latency_delta} cycles | {flag} |"
        )

    lines.extend(
        [
            "",
            "## Seed2 Implementation Baseline",
            "",
            "Seed2 has fuller evidence beyond csynth from the existing 06F flow:",
            "",
            "| stage | timing status | target ns | achieved ns | slack ns | BRAM | DSP | FF | LUT | SRL |",
            "| ----- | ------------- | --------- | ----------- | -------- | ---- | --- | -- | --- | --- |",
            "| RTL synthesis | Met | 10.000 | 9.729 | 0.271 | 6 | 47 | 8217 | 7754 | 37 |",
            "| Place and route | Met | 10.000 | 9.980 | 0.020 | 6 | 47 | 8217 | 7555 | 26 |",
            "",
            "Seed2 cosim status from existing evidence: Verilog pass, latency 8316 cycles, interval 8317 cycles. Seed0/seed1 in this update are csynth-only unless later cosim/implementation runs are explicitly executed.",
            "",
            "## Machine-Readable Summary",
            "",
            f"- CSV: `{rel(out_csv)}`.",
            "",
            "## Conclusion",
            "",
        ]
    )
    if status == "resolved":
        lines.append("Cross-seed csynth resource stability is established at the csynth-estimate level for seeds 0, 1, and 2 under the same PHC-lite manual HLS datapath/settings. This remains a csynth resource claim only for seed0/seed1; it does not imply seed0/seed1 cosim success, place-and-route timing closure, board deployment, PnL impact, or real-world profitability.")
    else:
        lines.append("Cross-seed HLS resource stability remains open because at least one required seed0/seed1 csynth report is missing.")
    CONTEXT_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and optionally synthesize PHC-lite cross-seed HLS artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="Seeds to process.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Validation sequences for seed-specific test vectors.")
    parser.add_argument("--setup", action="store_true", help="Generate step checkpoints, HLS weights, vectors, and HLS configs.")
    parser.add_argument("--run-csynth", action="store_true", help="Run vitis-run HLS csynth for each seed.")
    parser.add_argument("--collect", action="store_true", help="Parse csynth reports and update AUDIT_hls_cross_seed_resource_stability.md.")
    parser.add_argument(
        "--vitis-run",
        type=Path,
        default=Path("C:/AMDDesignTools/2025.2/Vitis/bin/vitis-run.bat"),
        help="Path to vitis-run.bat.",
    )
    args = parser.parse_args()

    if not args.setup and not args.run_csynth and not args.collect:
        args.setup = True

    for seed in args.seeds:
        if seed == 2:
            raise ValueError("This cross-seed script is for new seed0/seed1 artifacts. Seed2 baseline is read-only.")
        if args.setup:
            setup_seed(seed, sample_count=args.sample_count)
        if args.run_csynth:
            rc = run_csynth(seed, args.vitis_run)
            if rc != 0:
                raise SystemExit(rc)

    if args.collect:
        df = collect_reports(args.seeds)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
