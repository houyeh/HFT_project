from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06u_phc_latency_breakthrough"
VARIANT = ROOT / "variants" / "v2_hidden8_unroll8_100mhz"
RESULTS = ROOT / "results" / "v2_hidden8_unroll8_100mhz"
MODELS = VARIANT / "models"
HLS_SRC = VARIANT / "hls_src"
HLS_COMPONENT = VARIANT / "hls_component"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2_MODEL = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain" / "models" / "phc_lite_stp.py"
PHASE05E = PROJECT_ROOT / "n2ew_era" / "05e_phc_hidden8_ablation"

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
HIDDEN_DIM = 8
READOUT_DIM = 2
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
        return str(path.resolve()).replace("\\", "/")


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


def ensure_dirs() -> None:
    for path in [RESULTS, MODELS, HLS_SRC, HLS_COMPONENT]:
        path.mkdir(parents=True, exist_ok=True)


def choose_median_val_seed() -> int:
    path = PHASE05E / "results" / "phc_stp_v2_seed_stability.csv"
    df = pd.read_csv(path)
    val = df[df["split"] == "val"].copy()
    if val.empty:
        raise RuntimeError(f"No val rows found in {path}")
    val = val.sort_values(["action_pnl", "seed"], ascending=[True, True]).reset_index(drop=True)
    seed = int(val.iloc[len(val) // 2]["seed"])
    audit = val[["seed", "split", "action_pnl", "selected_action_count", "threshold_selected_on_val_for_seed"]]
    audit.to_csv(RESULTS / "hidden8_median_seed_selection_val.csv", index=False)
    return seed


def source_checkpoint(seed: int) -> Path:
    return PHASE05E / "models" / f"phc_stp_v2_seed{seed}_best_val.pt"


def load_source_model(seed: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt_path = source_checkpoint(seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing 05E checkpoint for seed{seed}: {ckpt_path}")
    ckpt = torch_load(ckpt_path)
    if int(ckpt.get("seed", -1)) != seed:
        raise RuntimeError(f"Expected checkpoint seed={seed}, got {ckpt.get('seed')}")
    if ckpt.get("phase") != "05e":
        raise RuntimeError(f"Expected phase=05e for seed{seed}, got {ckpt.get('phase')}")
    if ckpt.get("new5_used_for_selection") or ckpt.get("new5_used"):
        raise RuntimeError(f"Hard rule violation: seed{seed} checkpoint marks NEW5 use.")

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


def migrate_step_model(seed: int) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
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
        step.output_pack.weight[:READOUT_DIM, :].copy_(src.readout.weight.detach())
        step.output_pack.bias[:READOUT_DIM].copy_(src.readout.bias.detach())
        step.output_pack.weight[READOUT_DIM:, :].copy_(torch.eye(HIDDEN_DIM))
    step.decay.data.copy_(torch.sigmoid(src.decay_param.detach()).view(-1))
    step.alpha_scalar.data.copy_(src.alpha.detach().view(1))
    step.eval()

    step_path = MODELS / f"phc_stp_step_hls_hidden8_seed{seed}.pt"
    payload = {
        "model_state_dict": step.state_dict(),
        "source_checkpoint": rel(source_checkpoint(seed)),
        "source_phase": ckpt.get("phase"),
        "source_seed": int(ckpt.get("seed", seed)),
        "source_best_epoch": int(ckpt.get("best_epoch", -1)),
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "input_dim": FEATURE_DIM + HIDDEN_DIM,
        "output_dim": READOUT_DIM + HIDDEN_DIM,
        "mtl_steps": int(ckpt.get("mtl_steps", 1)),
        "dropout": 0.0,
        "phase": "06u_v2_hidden8_unroll8",
        "new5_used": False,
        "new_training": False,
        "seed_selection_rule": "median val action_pnl among 05E hidden8 seeds",
        "alpha_source_formula": "raw PHCLiteSTP.alpha, because source step uses z + self.alpha * message",
        "alpha_raw_value": float(src.alpha.detach().cpu()),
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
                "output_dim": READOUT_DIM + HIDDEN_DIM,
                "source_parameter_count": int(count_source_parameters(src)),
                "step_parameter_count": int(count_step_parameters(step)),
                "buffer_count": int(sum(b.numel() for b in step.buffers())),
                "decay_min": float(step.decay.min().detach().cpu()),
                "decay_max": float(step.decay.max().detach().cpu()),
                "alpha_raw_value_used": float(step.alpha_scalar.item()),
                "new5_used": False,
            }
        ]
    )
    audit.to_csv(RESULTS / f"phc_hidden8_step_migration_audit_seed{seed}.csv", index=False)
    payload_for_json = {k: v for k, v in payload.items() if k != "model_state_dict"}
    payload_for_json["step_checkpoint"] = rel(step_path)
    write_json(RESULTS / f"phc_hidden8_step_config_seed{seed}.json", payload_for_json)
    return src, step, payload


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


def export_weights(seed: int, step: torch.nn.Module, step_payload: dict[str, Any]) -> pd.DataFrame:
    state = step.state_dict()
    header_blocks: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    quantized_cache: dict[str, np.ndarray] = {}

    for tensor_name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        q, saturated = q24_12(arr)
        quantized_cache[tensor_name] = q
        exported = tensor_name in EXPORT_MAP
        audit_rows.append(
            {
                "seed": seed,
                "name": tensor_name,
                "cpp_name": EXPORT_MAP.get(tensor_name, ""),
                "shape": "x".join(str(x) for x in arr.shape) if arr.shape else "scalar",
                "num_values": int(arr.size),
                "float_min": float(np.nanmin(arr)) if arr.size else float("nan"),
                "float_max": float(np.nanmax(arr)) if arr.size else float("nan"),
                "quantized_min": int(np.min(q)) if q.size else 0,
                "quantized_max": int(np.max(q)) if q.size else 0,
                "num_saturated": saturated,
                "total_bits": TOTAL_BITS,
                "frac_bits": FRAC_BITS,
                "exported_to_header": bool(exported),
                "omitted_direct_hls_plumbing": bool(tensor_name in OMITTED_ROUTING_TENSORS),
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
        "// Generated by n2ew_era/06u_phc_latency_breakthrough/00_prepare_v2_hidden8.py",
        f"// Source PHC-STP hidden8 checkpoint: {source_checkpoint(seed)}",
        f"// Source seed: {seed}",
        f"// Source best epoch: {step_payload.get('source_best_epoch', 'unknown')}",
        f"// Quantization: Q{TOTAL_BITS}.{FRAC_BITS} == ap_fixed<{TOTAL_BITS},{TOTAL_BITS - FRAC_BITS}>",
        "// V2 scheduling target: hidden8, unroll8, 10 ns clock.",
        "// alpha_scalar is the raw PHCLiteSTP alpha value, not sigmoid(alpha).",
        "",
        "namespace phc_step_weights {",
        "",
        f"using weight_t = {AP_TYPE};",
        "",
        f"static const int FEATURE_DIM = {FEATURE_DIM};",
        f"static const int HIDDEN_DIM = {HIDDEN_DIM};",
        f"static const int INPUT_DIM = {FEATURE_DIM + HIDDEN_DIM};",
        f"static const int OUTPUT_DIM = {READOUT_DIM + HIDDEN_DIM};",
        f"static const int Q_TOTAL_BITS = {TOTAL_BITS};",
        f"static const int Q_FRAC_BITS = {FRAC_BITS};",
        "",
        *header_blocks,
        "}  // namespace phc_step_weights",
        "",
    ]
    (HLS_SRC / "phc_weights.hpp").write_text("\n".join(header), encoding="utf-8")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / f"weight_export_audit_hidden8_seed{seed}.csv", index=False)
    write_json(
        RESULTS / f"weight_export_manifest_hidden8_seed{seed}.json",
        {
            "seed": seed,
            "source_checkpoint": rel(source_checkpoint(seed)),
            "step_checkpoint": rel(MODELS / f"phc_stp_step_hls_hidden8_seed{seed}.pt"),
            "exported_header": rel(HLS_SRC / "phc_weights.hpp"),
            "total_bits": TOTAL_BITS,
            "frac_bits": FRAC_BITS,
            "any_saturated": bool(audit["num_saturated"].sum() > 0),
            "num_saturated_total": int(audit["num_saturated"].sum()),
            "new5_used": False,
        },
    )
    return audit


def vector_columns() -> list[str]:
    return (
        ["row_id", "sample_idx", "timestep"]
        + [f"x_t_{i}" for i in range(FEATURE_DIM)]
        + [f"state_prev_{i}" for i in range(HIDDEN_DIM)]
        + [f"ref_readout_{i}" for i in range(READOUT_DIM)]
        + [f"ref_state_next_{i}" for i in range(HIDDEN_DIM)]
    )


def generate_vectors(seed: int, source_model: torch.nn.Module, step_model: torch.nn.Module, step_payload: dict[str, Any], sample_count: int) -> pd.DataFrame:
    payload = p05a.load_sequences(SPLIT)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: validation sequence payload is marked new5_used=True.")

    x_all = payload["x"]
    n = min(int(sample_count), int(x_all.shape[0]))
    x = x_all[:n].float()
    seq_len = int(x.shape[1])
    feature_dim = int(x.shape[2])
    if feature_dim != FEATURE_DIM:
        raise RuntimeError(f"seed{seed}: expected feature_dim={FEATURE_DIM}, got {feature_dim}")

    frames: list[pd.DataFrame] = []
    state = torch.zeros(n, HIDDEN_DIM, dtype=torch.float32)
    readout_t = torch.zeros(n, READOUT_DIM, dtype=torch.float32)
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
    csv_path = RESULTS / f"phc_step_test_vectors_hidden8_seed{seed}.csv"
    out.to_csv(csv_path, index=False, float_format="%.9g")

    first1000 = out.head(1000)
    first1000_path = RESULTS / f"phc_step_test_vectors_hidden8_seed{seed}_first1000.csv"
    first1000.to_csv(first1000_path, index=False, float_format="%.9g")

    summary = pd.DataFrame(
        [
            {
                "seed": seed,
                "split": SPLIT,
                "sample_count": n,
                "sequence_length": seq_len,
                "feature_dim": feature_dim,
                "hidden_dim": HIDDEN_DIM,
                "row_count": int(len(out)),
                "csv_path": rel(csv_path),
                "first1000_csv_path": rel(first1000_path),
                "equivalence_atol": EQUIVALENCE_ATOL,
                "max_abs_diff_final_readout": max_abs_diff,
                "mean_abs_diff_final_readout": mean_abs_diff,
                "allclose_atol_1e_4": allclose,
                "source_seed": int(step_payload.get("source_seed", seed)),
                "source_best_epoch": int(step_payload.get("source_best_epoch", -1)),
                "step_checkpoint": rel(MODELS / f"phc_stp_step_hls_hidden8_seed{seed}.pt"),
                "source_checkpoint": rel(source_checkpoint(seed)),
                "new5_used": False,
            }
        ]
    )
    summary.to_csv(RESULTS / f"phc_step_test_vector_summary_hidden8_seed{seed}.csv", index=False)
    return summary


def write_config(seed: int) -> None:
    vectors = RESULTS / f"phc_step_test_vectors_hidden8_seed{seed}.csv"
    vectors_first1000 = RESULTS / f"phc_step_test_vectors_hidden8_seed{seed}_first1000.csv"
    csim_out = RESULTS / f"phc_step_csim_out_hidden8_seed{seed}.csv"
    cosim_out = RESULTS / f"phc_step_cosim_out_hidden8_seed{seed}_first1000.csv"
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
            f"cosim.argv={win_abs(vectors_first1000)} {win_abs(cosim_out)}",
            f"csim.argv={win_abs(vectors)} {win_abs(csim_out)}",
            "",
        ]
    )
    (HLS_COMPONENT / "hls_config.cfg").write_text(hls_config, encoding="utf-8")
    (HLS_COMPONENT / "run_csynth_only.tcl").write_text(
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
    write_json(
        HLS_COMPONENT / "vitis-comp.json",
        {
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
        },
    )
    write_json(
        RESULTS / f"hls_setup_manifest_hidden8_seed{seed}.json",
        {
            "seed": seed,
            "variant": "v2_hidden8_unroll8_100mhz",
            "hls_src": rel(HLS_SRC),
            "hls_component": rel(HLS_COMPONENT),
            "hls_config": rel(HLS_COMPONENT / "hls_config.cfg"),
            "csynth_tcl": rel(HLS_COMPONENT / "run_csynth_only.tcl"),
            "settings": {
                "part": "xc7z020clg400-1",
                "clock": "10ns",
                "flow_target": "vivado",
                "syn_top": "phc_step",
                "feature_dim": FEATURE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "unroll_factor": 8,
            },
            "source_hashes": {
                "phc_step_hls.cpp": sha256_file(HLS_SRC / "phc_step_hls.cpp"),
                "phc_step_hls.hpp": sha256_file(HLS_SRC / "phc_step_hls.hpp"),
                "phc_step_testbench.cpp": sha256_file(HLS_SRC / "phc_step_testbench.cpp"),
                "phc_weights.hpp": sha256_file(HLS_SRC / "phc_weights.hpp"),
            },
            "new5_used": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 06U V2 hidden8 unroll8 HLS artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Override hidden8 seed. Default: median val seed from 05E.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    args = parser.parse_args()

    ensure_dirs()
    seed = choose_median_val_seed() if args.seed is None else int(args.seed)
    if seed not in {0, 1, 2}:
        raise ValueError(f"Unexpected 05E seed: {seed}")
    source_model, step_model, payload = migrate_step_model(seed)
    export_weights(seed, step_model, payload)
    generate_vectors(seed, source_model, step_model, payload, sample_count=args.sample_count)
    write_config(seed)
    print(f"06U V2 hidden8 setup complete: seed={seed}, variant={rel(VARIANT)}")


if __name__ == "__main__":
    main()
