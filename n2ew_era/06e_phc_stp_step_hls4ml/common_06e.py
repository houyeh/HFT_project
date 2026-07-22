from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
HLS4ML_PRJ = ROOT / "hls4ml_prj"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2_MODEL = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain" / "models" / "phc_lite_stp.py"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import common_05a as p05a  # noqa: E402
from src.models.phc_stp_step_hls import PHCSTPStepHLS, count_parameters as count_step_parameters  # noqa: E402

PHASE = "06e"
FEATURE_DIM = 16
HIDDEN_DIM = 16
SELECTED_SEED = 2
MTL_STEPS = 1
DROPOUT = 0.0
SAMPLE_COUNT = 500
EQUIVALENCE_ATOL = 1e-4

CKPT_PATH = PHASE05C4 / "models" / "phc_stp_v2_seed2_best_val.pt"
STEP_MODEL_PATH = MODELS / "phc_stp_step_hls_seed2.pt"
HLS_INPUT_SHAPE = (1, FEATURE_DIM + HIDDEN_DIM)
HLS_OUTPUT_DIM = 2 + HIDDEN_DIM


def ensure_dirs() -> None:
    for path in [ROOT, MODELS, RESULTS, HLS4ML_PRJ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_phc_lite_stp_class():
    spec = importlib.util.spec_from_file_location("phase05c2_phc_lite_stp", PHASE05C2_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PHCLiteSTP source from {PHASE05C2_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PHCLiteSTP, module.count_parameters


PHCLiteSTP, count_source_parameters = _load_phc_lite_stp_class()


def load_source_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing adopted 05C4 checkpoint: {CKPT_PATH}")
    ckpt = torch_load(CKPT_PATH)
    if int(ckpt.get("seed", -1)) != SELECTED_SEED:
        raise RuntimeError(f"Expected seed={SELECTED_SEED}, got {ckpt.get('seed')}")
    if ckpt.get("phase") != "05c4":
        raise RuntimeError(f"Expected 05C4 checkpoint, got phase={ckpt.get('phase')}")
    if ckpt.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: checkpoint says NEW5 was used for selection.")
    feature_dim = int(ckpt.get("feature_dim", len(p05a.PHC_FEATURES)))
    hidden_dim = int(ckpt.get("hidden_dim", HIDDEN_DIM))
    model = PHCLiteSTP(feature_dim=feature_dim, hidden_dim=hidden_dim, mtl_steps=int(ckpt.get("mtl_steps", 1)), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _copy_linear(dst: torch.nn.Linear, src: torch.nn.Linear) -> None:
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def _copy_norm(dst: torch.nn.LayerNorm, src: torch.nn.LayerNorm) -> None:
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def migrate_weights() -> tuple[PHCSTPStepHLS, pd.DataFrame]:
    """Migrate the adopted 05C4 seed=2 PHCLiteSTP checkpoint into a single-step graph."""
    ensure_dirs()
    src, ckpt = load_source_model()
    feature_dim = int(ckpt.get("feature_dim", len(p05a.PHC_FEATURES)))
    hidden_dim = int(ckpt.get("hidden_dim", HIDDEN_DIM))
    if feature_dim != FEATURE_DIM or hidden_dim != HIDDEN_DIM:
        raise RuntimeError(f"Unexpected dimensions: feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    step = PHCSTPStepHLS(feature_dim=feature_dim, hidden_dim=hidden_dim)
    _copy_linear(step.enc_linear, src.encoder[0])
    _copy_norm(step.enc_norm, src.encoder[1])
    _copy_linear(step.gate, src.gate)
    _copy_linear(step.candidate, src.candidate)
    _copy_linear(step.gain_from_encoded, src.gain_from_encoded)
    step.gain_from_prev_state.weight.data.copy_(src.gain_from_prev_state.weight.detach())
    _copy_linear(step.synapse, src.synapse)
    _copy_norm(step.loop_norm, src.loop_norm)
    _copy_linear(step.readout, src.readout)
    with torch.no_grad():
        step.output_pack.weight.zero_()
        step.output_pack.bias.zero_()
        step.output_pack.weight[:2, :].copy_(src.readout.weight.detach())
        step.output_pack.bias[:2].copy_(src.readout.bias.detach())
        step.output_pack.weight[2:, :].copy_(torch.eye(hidden_dim))
    step.decay.data.copy_(torch.sigmoid(src.decay_param.detach()).view(-1))

    # The 05C2/05C4 source model applies `self.alpha` directly. The prompt's
    # sigmoid(alpha) wording would change the adopted model and fail equivalence.
    # Store the source-true raw multiplier as a fixed buffer.
    step.alpha_scalar.data.copy_(src.alpha.detach().view(1))
    step.eval()
    torch.save(
        {
            "model_state_dict": step.state_dict(),
            "source_checkpoint": str(CKPT_PATH),
            "source_phase": ckpt.get("phase"),
            "source_seed": int(ckpt.get("seed", SELECTED_SEED)),
            "source_best_epoch": int(ckpt.get("best_epoch", -1)),
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "input_dim": feature_dim + hidden_dim,
            "output_dim": 2 + hidden_dim,
            "mtl_steps": 1,
            "dropout": 0.0,
            "phase": PHASE,
            "new5_used": False,
            "new_training": False,
            "alpha_source_formula": "raw PHCLiteSTP.alpha, because source step uses z + self.alpha * message",
            "alpha_raw_value": float(src.alpha.detach().cpu()),
            "alpha_sigmoid_value_prompt_reference": float(torch.sigmoid(src.alpha.detach()).cpu()),
            "decay_formula": "sigmoid(PHCLiteSTP.decay_param)",
        },
        STEP_MODEL_PATH,
    )
    audit = pd.DataFrame(
        [
            {
                "phase": PHASE,
                "source_checkpoint": str(CKPT_PATH),
                "migrated_model": str(STEP_MODEL_PATH),
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "input_dim": feature_dim + hidden_dim,
                "output_dim": 2 + hidden_dim,
                "source_parameter_count": int(count_source_parameters(src)),
                "step_parameter_count": int(count_step_parameters(step)),
                "buffer_count": int(sum(b.numel() for b in step.buffers())),
                "fixed_input_selector_parameters": int(step.input_x.weight.numel() + step.input_state.weight.numel()),
                "fixed_output_pack_parameters": int(step.output_pack.weight.numel() + step.output_pack.bias.numel()),
                "decay_min": float(step.decay.min().detach().cpu()),
                "decay_max": float(step.decay.max().detach().cpu()),
                "alpha_raw_value_used_for_equivalence": float(step.alpha_scalar.item()),
                "alpha_sigmoid_value_not_used": float(torch.sigmoid(src.alpha.detach()).cpu()),
                "alpha_note": "05C2 source applies raw alpha directly; sigmoid(alpha) is intentionally not used.",
                "new5_used": False,
            }
        ]
    )
    audit.to_csv(RESULTS / "phc_stp_step_migration_audit.csv", index=False)
    write_json(RESULTS / "phc_stp_step_config.json", audit.iloc[0].to_dict())
    return step, audit


def load_step_model() -> tuple[PHCSTPStepHLS, dict[str, Any]]:
    if not STEP_MODEL_PATH.exists():
        migrate_weights()
    payload = torch_load(STEP_MODEL_PATH)
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def verify_equivalence(sample_count: int = SAMPLE_COUNT) -> pd.DataFrame:
    """Compare full PHCLiteSTP sequence inference with repeated PHCSTPStepHLS calls."""
    ensure_dirs()
    src, ckpt = load_source_model()
    step, step_payload = load_step_model()
    payload = p05a.load_sequences("val")
    x = payload["x"][:sample_count].float()
    n = int(x.shape[0])
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: sequence payload is marked NEW5-used.")
    with torch.no_grad():
        ref = src(x)
        state = torch.zeros(n, HIDDEN_DIM, dtype=torch.float32)
        readout_t = torch.zeros(n, 2, dtype=torch.float32)
        for t in range(int(x.shape[1])):
            x_cat = torch.cat([x[:, t, :], state], dim=-1)
            out_step = step(x_cat)
            readout_t, state = step.split_output(out_step)
    diff = (readout_t - ref).detach().cpu().numpy()
    out = pd.DataFrame(
        [
            {
                "phase": PHASE,
                "sample_count": n,
                "split": "val",
                "seq_len": int(x.shape[1]),
                "feature_dim": int(x.shape[2]),
                "source_checkpoint": str(CKPT_PATH),
                "step_checkpoint": str(STEP_MODEL_PATH),
                "source_seed": int(ckpt.get("seed", SELECTED_SEED)),
                "source_best_epoch": int(ckpt.get("best_epoch", -1)),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else float("nan"),
                "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
                "allclose_atol_1e_4": bool(torch.allclose(readout_t, ref, atol=EQUIVALENCE_ATOL, rtol=1e-5)),
                "equivalence_atol": EQUIVALENCE_ATOL,
                "alpha_source_formula": step_payload.get("alpha_source_formula", ""),
                "new5_used": False,
            }
        ]
    )
    out.to_csv(RESULTS / "phc_stp_step_equivalence.csv", index=False)
    return out


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    root = HLS4ML_PRJ.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to delete path outside HLS4ML_PRJ: {path}")
    if path.exists():
        shutil.rmtree(path)


def _layer_summary(hls_model: Any) -> tuple[int, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    try:
        layers = list(getattr(hls_model, "get_layers", lambda: [])())
    except Exception:
        layers = []
    for idx, layer in enumerate(layers):
        cls_name = layer.__class__.__name__
        layer_type = getattr(layer, "class_name", None) or getattr(layer, "name", None) or cls_name
        try:
            attrs = getattr(layer, "attributes", {})
            name = attrs.get("name", "") if isinstance(attrs, dict) else ""
        except Exception:
            name = ""
        rows.append({"idx": idx, "layer_class": cls_name, "layer_type": str(layer_type), "layer_name": str(name)})
    layer_types = ",".join(sorted({r["layer_class"] for r in rows}))
    return len(rows), layer_types, rows


def _base_status() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "model_name": "PHCSTPStepHLS",
        "source_model": "05c4 PHCLiteSTP seed=2",
        "hls_input_shape": str(HLS_INPUT_SHAPE),
        "hls_output_dim": HLS_OUTPUT_DIM,
        "torch_fx_attempted": False,
        "torch_fx_convertible": False,
        "torch_fx_compile_success": False,
        "torch_fx_cpp_project_generated": False,
        "torch_fx_parsed_layer_count": 0,
        "torch_fx_layer_types": "",
        "torch_fx_error": "",
        "torch_fx_3d_attempted": False,
        "torch_fx_3d_input_shape": "(1, 1, 32)",
        "torch_fx_3d_convertible": False,
        "torch_fx_3d_compile_success": False,
        "torch_fx_3d_cpp_project_generated": False,
        "torch_fx_3d_parsed_layer_count": 0,
        "torch_fx_3d_layer_types": "",
        "torch_fx_3d_error": "",
        "onnx_attempted": False,
        "onnx_export_success": False,
        "onnx_convertible": False,
        "onnx_compile_success": False,
        "onnx_cpp_project_generated": False,
        "onnx_parsed_layer_count": 0,
        "onnx_layer_types": "",
        "onnx_error": "",
        "selected_conversion_path": "neither",
        "vivado_available": bool(shutil.which("vivado_hls") or shutil.which("vivado")),
        "vivado_build_run": False,
        "vivado_build_success": False,
        "vivado_build_error": "",
        "hls4ml_version": "",
        "constantpad1d_layers_present": False,
        "precision": "ap_fixed<16,6>",
        "reuse_factor": 1,
        "strategy": "Latency",
        "new5_used": False,
    }


def hls4ml_conversion() -> pd.DataFrame:
    ensure_dirs()
    migrate_weights()
    verify_equivalence()
    step_model, _ = load_step_model()
    step_model.eval()
    status = _base_status()
    status["step_parameter_count"] = int(count_step_parameters(step_model))
    status["buffer_count"] = int(sum(b.numel() for b in step_model.buffers()))
    status["constantpad1d_layers_present"] = bool(any(m.__class__.__name__ == "ConstantPad1d" for m in step_model.modules()))
    layer_rows: list[dict[str, Any]] = []
    try:
        import hls4ml  # type: ignore

        status["hls4ml_version"] = str(getattr(hls4ml, "__version__", "unknown"))
        status["torch_fx_attempted"] = True
        out_dir = HLS4ML_PRJ / "phc_stp_step_hls_prj_torchfx"
        try:
            _safe_rmtree(out_dir)
            cfg = hls4ml.utils.config_from_pytorch_model(
                step_model,
                input_shape=HLS_INPUT_SHAPE,
                default_precision="ap_fixed<16,6>",
                granularity="name",
            )
            cfg.setdefault("Model", {})
            cfg["Model"]["Precision"] = "ap_fixed<16,6>"
            cfg["Model"]["ReuseFactor"] = 1
            cfg["Model"]["Strategy"] = "Latency"
            for layer_cfg in cfg.get("LayerName", {}).values():
                layer_cfg["ReuseFactor"] = 1
                layer_cfg.setdefault("Precision", "ap_fixed<16,6>")
            hls_model = hls4ml.converters.convert_from_pytorch_model(
                step_model,
                input_shape=HLS_INPUT_SHAPE,
                hls_config=cfg,
                output_dir=str(out_dir),
                project_name="phc_stp_step_hls_torchfx",
                backend="Vivado",
            )
            parsed, types, rows = _layer_summary(hls_model)
            for row in rows:
                row["path"] = "torch_fx"
            layer_rows.extend(rows)
            status["torch_fx_parsed_layer_count"] = parsed
            status["torch_fx_layer_types"] = types
            try:
                hls_model.write()
            except Exception:
                pass
            status["torch_fx_cpp_project_generated"] = bool(out_dir.exists())
            status["torch_fx_convertible"] = True
            hls_model.compile()
            status["torch_fx_compile_success"] = True
            status["selected_conversion_path"] = "torch_fx"
        except Exception as exc:  # noqa: BLE001
            status["torch_fx_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"

        status["torch_fx_3d_attempted"] = True
        out_dir_3d = HLS4ML_PRJ / "phc_stp_step_hls_prj_torchfx_3d"
        try:
            _safe_rmtree(out_dir_3d)
            cfg_3d = hls4ml.utils.config_from_pytorch_model(
                step_model,
                input_shape=(1, 1, FEATURE_DIM + HIDDEN_DIM),
                default_precision="ap_fixed<16,6>",
                granularity="name",
            )
            cfg_3d.setdefault("Model", {})
            cfg_3d["Model"]["Precision"] = "ap_fixed<16,6>"
            cfg_3d["Model"]["ReuseFactor"] = 1
            cfg_3d["Model"]["Strategy"] = "Latency"
            for layer_cfg in cfg_3d.get("LayerName", {}).values():
                layer_cfg["ReuseFactor"] = 1
                layer_cfg.setdefault("Precision", "ap_fixed<16,6>")
            hls_model_3d = hls4ml.converters.convert_from_pytorch_model(
                step_model,
                input_shape=(1, 1, FEATURE_DIM + HIDDEN_DIM),
                hls_config=cfg_3d,
                output_dir=str(out_dir_3d),
                project_name="phc_stp_step_hls_torchfx_3d",
                backend="Vivado",
            )
            parsed, types, rows = _layer_summary(hls_model_3d)
            for row in rows:
                row["path"] = "torch_fx_3d"
            layer_rows.extend(rows)
            status["torch_fx_3d_parsed_layer_count"] = parsed
            status["torch_fx_3d_layer_types"] = types
            try:
                hls_model_3d.write()
            except Exception:
                pass
            status["torch_fx_3d_cpp_project_generated"] = bool(out_dir_3d.exists())
            status["torch_fx_3d_convertible"] = True
            hls_model_3d.compile()
            status["torch_fx_3d_compile_success"] = True
            if status["selected_conversion_path"] == "neither":
                status["selected_conversion_path"] = "torch_fx_3d"
        except Exception as exc:  # noqa: BLE001
            status["torch_fx_3d_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"

        status["onnx_attempted"] = True
        onnx_dir = HLS4ML_PRJ / "phc_stp_step_hls_prj_onnx"
        onnx_path = HLS4ML_PRJ / "phc_stp_step_hls_seed2.onnx"
        try:
            _safe_rmtree(onnx_dir)
            dummy = torch.zeros(HLS_INPUT_SHAPE, dtype=torch.float32)
            torch.onnx.export(
                step_model,
                dummy,
                str(onnx_path),
                input_names=["x_cat"],
                output_names=["readout_and_state"],
                opset_version=13,
                do_constant_folding=True,
                dynamo=False,
            )
            status["onnx_export_success"] = True
            try:
                cfg_onnx = hls4ml.utils.config_from_onnx_model(str(onnx_path), default_precision="ap_fixed<16,6>", granularity="name")
            except Exception:
                cfg_onnx = {"Model": {"Precision": "ap_fixed<16,6>", "ReuseFactor": 1, "Strategy": "Latency"}}
            try:
                import onnx  # type: ignore

                onnx_model = onnx.load(str(onnx_path))
                try:
                    inferred = onnx.shape_inference.infer_shapes(onnx_model)
                    inferred_path = HLS4ML_PRJ / "phc_stp_step_hls_seed2_inferred.onnx"
                    onnx.save(inferred, str(inferred_path))
                    onnx_model = onnx.load(str(inferred_path))
                except Exception:
                    pass
            except Exception:
                onnx_model = str(onnx_path)
            hls_model_onnx = hls4ml.converters.convert_from_onnx_model(
                onnx_model,
                hls_config=cfg_onnx,
                output_dir=str(onnx_dir),
                project_name="phc_stp_step_hls_onnx",
                backend="Vivado",
            )
            parsed, types, rows = _layer_summary(hls_model_onnx)
            for row in rows:
                row["path"] = "onnx"
            layer_rows.extend(rows)
            status["onnx_parsed_layer_count"] = parsed
            status["onnx_layer_types"] = types
            try:
                hls_model_onnx.write()
            except Exception:
                pass
            status["onnx_cpp_project_generated"] = bool(onnx_dir.exists())
            status["onnx_convertible"] = True
            hls_model_onnx.compile()
            status["onnx_compile_success"] = True
            if status["selected_conversion_path"] == "neither":
                status["selected_conversion_path"] = "onnx"
        except Exception as exc:  # noqa: BLE001
            status["onnx_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    except Exception as exc:  # noqa: BLE001
        status["torch_fx_attempted"] = False
        status["onnx_attempted"] = False
        status["torch_fx_error"] = f"hls4ml import/setup failed: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        status["torch_fx_3d_error"] = status["torch_fx_error"]
        status["onnx_error"] = status["torch_fx_error"]
    out = pd.DataFrame([status])
    out.to_csv(RESULTS / "phc_stp_step_hls4ml_conversion_status.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(RESULTS / "phc_stp_step_hls4ml_layer_types.csv", index=False)
    write_hls_report(out, pd.DataFrame(layer_rows))
    return out


def write_hls_report(status: pd.DataFrame, layers: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "PHC_STP_STEP_HLS4ML_CONVERSION_REPORT.md",
        [
            "# PHC-STP Step hls4ml Conversion Report",
            "",
            "This phase maps the adopted 05C4 PHCLiteSTP seed=2 model into a single-step feed-forward graph. It does not retrain, retune, or touch NEW5.",
            "",
            "## Status",
            "",
            status.to_markdown(index=False),
            "",
            "## Parsed Layers",
            "",
            layers.to_markdown(index=False) if not layers.empty else "No hls4ml layers were parsed successfully.",
            "",
            "If conversion succeeds, the generated project is a C++ project-generation artifact only. It is not a PYNQ-Z2 deployment and does not prove FPGA PnL improvement.",
        ],
    )


def write_final_report() -> str:
    ensure_dirs()
    migration = safe_read_csv(RESULTS / "phc_stp_step_migration_audit.csv")
    equivalence = safe_read_csv(RESULTS / "phc_stp_step_equivalence.csv")
    hls = safe_read_csv(RESULTS / "phc_stp_step_hls4ml_conversion_status.csv")
    layers = safe_read_csv(RESULTS / "phc_stp_step_hls4ml_layer_types.csv")
    if not migration.empty and not equivalence.empty and not hls.empty:
        eq_pass = bool(equivalence["allclose_atol_1e_4"].iloc[0])
        convertible = bool(hls["torch_fx_convertible"].iloc[0]) or bool(hls.get("torch_fx_3d_convertible", pd.Series([False])).iloc[0]) or bool(hls["onnx_convertible"].iloc[0])
        compile_ok = bool(hls["torch_fx_compile_success"].iloc[0]) or bool(hls.get("torch_fx_3d_compile_success", pd.Series([False])).iloc[0]) or bool(hls["onnx_compile_success"].iloc[0])
        if eq_pass and compile_ok:
            status = "phc_stp_step_hls4ml_compile_success"
        elif eq_pass and convertible:
            status = "phc_stp_step_hls4ml_convertible_compile_failed"
        elif eq_pass:
            status = "phc_stp_step_equivalent_but_hls4ml_not_convertible"
        else:
            status = "phc_stp_step_equivalence_failed"
    else:
        status = "phc_stp_step_hls4ml_incomplete_missing_artifacts"
    write_markdown(
        ROOT / "PHC_STP_STEP_HLS4ML_FINAL_REPORT.md",
        [
            "# PHC-STP Step hls4ml Final Report",
            "",
            f"Final status: `{status}`.",
            "",
            "## 1. Model Summary",
            "",
            "- Source model: adopted Phase 05C4 `PHCLiteSTP`, seed 2.",
            "- Step model: `PHCSTPStepHLS`.",
            "- Input: `(1, 32)` = `[x_t 16 causal features | previous_state 16]`.",
            "- Output: `(1, 18)` = `[bid_score, ask_score | next_state 16]`.",
            "- hls4ml fallback may view the same vector as `(1, 1, 32)` to satisfy the current frontend's LayerNorm shape constraint.",
            "- This is a hardware-mapping phase only: no training, no threshold selection, no NEW5 usage.",
            "- The adopted PnL evidence remains Phase 05C4; this phase only builds a conversion pathway.",
            "",
            "## 2. Migration Audit",
            "",
            migration.to_markdown(index=False) if not migration.empty else "Migration audit missing.",
            "",
            "Important source-code closure: the 05C2/05C4 `PHCLiteSTP.step()` source applies raw `self.alpha` directly. Therefore migration uses raw alpha as a fixed buffer. Using `sigmoid(alpha)` would alter the adopted model and fail numerical equivalence.",
            "",
            "## 3. Equivalence Test",
            "",
            equivalence.to_markdown(index=False) if not equivalence.empty else "Equivalence audit missing.",
            "",
            "The verification loop repeatedly calls the step model for 128 timesteps outside the model graph. The step model itself contains no recurrent Python loop.",
            "",
            "## 4. hls4ml Conversion",
            "",
            hls.to_markdown(index=False) if not hls.empty else "hls4ml conversion status missing.",
            "",
            "## 5. Parsed Layer Types",
            "",
            layers.to_markdown(index=False) if not layers.empty else "No hls4ml parsed layer table was produced.",
            "",
            "## 6. Comparison vs 06C/06D",
            "",
            "Phase 06C/06D tested TCN hardware candidates; those TCNs were not adopted. This 06E model is different: it is the single-step hardware form of the adopted 05C4 PHC-STP sequence model. It does not change the strategy or the trained weights.",
            "",
            "## 7. Protocol Boundaries",
            "",
            "- NEW5 is locked and not touched.",
            "- No model training is performed.",
            "- No new evaluation pipeline or threshold selection is performed.",
            "- hls4ml conversion, if successful, means C++ project generation and Python-side compilation only.",
            "- Do not claim this is a PYNQ-Z2 deployment.",
            "- Do not claim FPGA improves PnL.",
            "- Do not claim true exchange replay or real-world profitability.",
        ],
    )
    return status


def run_all() -> None:
    migrate_weights()
    verify_equivalence()
    hls4ml_conversion()
    print(write_final_report())


if __name__ == "__main__":
    run_all()
