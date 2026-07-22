from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06g_phc_hls4ml_a3d3_retry"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
HLS4ML_PRJ = ROOT / "hls4ml_prj"
PHASE06E = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"
PHASE06F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"

for path in [PROJECT_ROOT, PHASE06E]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_06e as p06e  # noqa: E402


FEATURE_DIM = 16
HIDDEN_DIM = 16
INPUT_SHAPE = (1, FEATURE_DIM + HIDDEN_DIM)
DEFAULT_PRECISION = "ap_fixed<16,6>"
REUSE_FACTOR = 1
STRATEGY = "Latency"


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, HLS4ML_PRJ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


class PHCSTPStepManualLayerNorm(nn.Module):
    """Step model equivalent to PHCSTPStepHLS with LayerNorm expanded into ops.

    This wrapper is intentionally local to Phase 06G. It does not modify the
    deployed checkpoint, the source step class, or the hand-written HLS project.
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.feature_dim = int(src.feature_dim)
        self.hidden_dim = int(src.hidden_dim)
        self.input_x = src.input_x
        self.input_state = src.input_state
        self.enc_linear = src.enc_linear
        self.gate = src.gate
        self.candidate = src.candidate
        self.gain_from_encoded = src.gain_from_encoded
        self.gain_from_prev_state = src.gain_from_prev_state
        self.synapse = src.synapse
        self.readout = src.readout
        self.output_pack = src.output_pack

        self.register_buffer("enc_norm_weight", src.enc_norm.weight.detach().clone())
        self.register_buffer("enc_norm_bias", src.enc_norm.bias.detach().clone())
        self.register_buffer("loop_norm_weight", src.loop_norm.weight.detach().clone())
        self.register_buffer("loop_norm_bias", src.loop_norm.bias.detach().clone())
        self.register_buffer("decay", src.decay.detach().clone())
        self.register_buffer("alpha_scalar", src.alpha_scalar.detach().clone())
        self.enc_norm_eps = float(src.enc_norm.eps)
        self.loop_norm_eps = float(src.loop_norm.eps)

    @staticmethod
    def manual_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + eps)
        return centered * inv_std * weight + bias

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        x_t = self.input_x(x_cat)
        state_prev = self.input_state(x_cat)

        encoded_pre_linear = self.enc_linear(x_t)
        encoded_pre = self.manual_layer_norm(
            encoded_pre_linear,
            self.enc_norm_weight,
            self.enc_norm_bias,
            self.enc_norm_eps,
        )
        encoded = encoded_pre * torch.sigmoid(encoded_pre)
        gate_t = torch.sigmoid(self.gate(encoded))
        candidate_t = torch.tanh(self.candidate(encoded))
        state_temporal = self.decay * state_prev + gate_t * candidate_t

        gain_t = torch.sigmoid(self.gain_from_encoded(encoded) + self.gain_from_prev_state(state_prev))
        message_pre = self.synapse(state_temporal)
        message = message_pre * torch.sigmoid(message_pre)
        msg_gated = gain_t * message
        loop_pre = state_temporal + self.alpha_scalar * msg_gated
        z = self.manual_layer_norm(loop_pre, self.loop_norm_weight, self.loop_norm_bias, self.loop_norm_eps)

        return self.output_pack(z)


class PHCSTPStepUnrolledLayerNorm(PHCSTPStepManualLayerNorm):
    """Manual LayerNorm variant that avoids torch.mean in the FX graph."""

    @staticmethod
    def manual_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        cols = [x[:, i : i + 1] for i in range(16)]
        mean = cols[0]
        for col in cols[1:]:
            mean = mean + col
        mean = mean * (1.0 / 16.0)

        centered = [col - mean for col in cols]
        var = centered[0] * centered[0]
        for col in centered[1:]:
            var = var + col * col
        var = var * (1.0 / 16.0)
        inv_std = torch.rsqrt(var + eps)

        out_cols = []
        for i, col in enumerate(centered):
            out_cols.append(col * inv_std * weight[i] + bias[i])
        return torch.cat(out_cols, dim=-1)


def load_models() -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    step_model, payload = p06e.load_step_model()
    if payload.get("new5_used") or payload.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: loaded step payload is marked NEW5-used.")
    step_model.eval()
    manual = PHCSTPStepManualLayerNorm(step_model)
    manual.eval()
    return step_model, manual, payload


def make_unrolled_model(step_model: nn.Module) -> nn.Module:
    model = PHCSTPStepUnrolledLayerNorm(step_model)
    model.eval()
    return model


def verify_model_equivalence(step_model: nn.Module, candidate_model: nn.Module, label: str) -> dict[str, Any]:
    torch.manual_seed(123)
    x = torch.randn(64, FEATURE_DIM + HIDDEN_DIM, dtype=torch.float32)
    with torch.no_grad():
        ref = step_model(x)
        got = candidate_model(x)
    diff = (ref - got).detach().cpu()
    row = {
        "model": label,
        "sample_count": int(x.shape[0]),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "allclose_atol_1e_6": bool(torch.allclose(ref, got, atol=1e-6, rtol=1e-6)),
    }
    return row


def configure_hls4ml(hls4ml: Any, model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        default_precision=DEFAULT_PRECISION,
        granularity="name",
    )
    cfg.setdefault("Model", {})
    cfg["Model"]["Precision"] = DEFAULT_PRECISION
    cfg["Model"]["ReuseFactor"] = REUSE_FACTOR
    cfg["Model"]["Strategy"] = STRATEGY
    for layer_cfg in cfg.get("LayerName", {}).values():
        layer_cfg["ReuseFactor"] = REUSE_FACTOR
        layer_cfg.setdefault("Precision", DEFAULT_PRECISION)
    return cfg


def layer_summary(hls_model: Any) -> tuple[int, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    try:
        graph = getattr(hls_model, "graph", None)
        layers = list(graph.get_layers()) if graph is not None and hasattr(graph, "get_layers") else []
    except Exception:
        layers = []
    for layer in layers:
        row = {
            "name": str(getattr(layer, "name", "")),
            "class_name": layer.__class__.__name__,
            "type": str(getattr(layer, "type", "")),
        }
        rows.append(row)
    types = ",".join(sorted({r["class_name"] for r in rows}))
    return len(rows), types, rows


def find_csynth_reports(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.rglob("*csynth.rpt"))


def parse_csynth_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {"csynth_report_path": str(path)}
    # hls4ml/Vivado reports vary; keep parsing conservative and auditable.
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 5 and parts[0] in {"BRAM_18K", "DSP", "FF", "LUT"}:
            try:
                out[parts[0]] = int(float(parts[1]))
            except Exception:
                pass
        if "Latency" in line and "cycles" in line and "min" not in out:
            out.setdefault("latency_line", line.strip())
    return out


def try_compile_and_build(hls_model: Any, status: dict[str, Any], prefix: str, out_dir: Path) -> None:
    try:
        hls_model.compile()
        status[f"{prefix}_compile_success"] = True
    except Exception as exc:  # noqa: BLE001
        status[f"{prefix}_compile_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        return

    try:
        hls_model.build(csim=True, synth=True)
        status[f"{prefix}_build_run"] = True
        status[f"{prefix}_build_success"] = True
    except Exception as exc:  # noqa: BLE001
        status[f"{prefix}_build_run"] = True
        status[f"{prefix}_build_success"] = False
        status[f"{prefix}_build_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=10)}"

    reports = find_csynth_reports(out_dir)
    status[f"{prefix}_csynth_report_count"] = len(reports)
    if reports:
        parsed = parse_csynth_report(reports[0])
        for key, value in parsed.items():
            status[f"{prefix}_{key}"] = value


def run_torch_fx_manual(hls4ml: Any, manual_model: nn.Module, method_label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix = f"torch_fx_{method_label}"
    out_dir = HLS4ML_PRJ / f"phc_stp_step_{method_label}_torchfx"
    status: dict[str, Any] = {
        "method": prefix,
        "attempted": True,
        "manual_layernorm": True,
        "manual_layernorm_variant": method_label,
        "fixed_batch_size": True,
        "input_shape": str(INPUT_SHAPE),
        "convertible": False,
        "compile_success": False,
        "build_run": False,
        "build_success": False,
        "project_dir": str(out_dir),
    }
    layer_rows: list[dict[str, Any]] = []
    try:
        safe_rmtree(out_dir)
        cfg = configure_hls4ml(hls4ml, manual_model, INPUT_SHAPE)
        write_json(RESULTS / f"{prefix}_config.json", cfg)
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            manual_model,
            input_shape=INPUT_SHAPE,
            hls_config=cfg,
            output_dir=str(out_dir),
            project_name=f"phc_stp_step_{method_label}_torchfx",
            backend="Vivado",
        )
        parsed, types, rows = layer_summary(hls_model)
        for row in rows:
            row["method"] = prefix
        layer_rows.extend(rows)
        status["parsed_layer_count"] = parsed
        status["layer_types"] = types
        try:
            hls_model.write()
        except Exception:
            pass
        status["cpp_project_generated"] = bool(out_dir.exists())
        status["convertible"] = True
        try_compile_and_build(hls_model, status, prefix, out_dir)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
    return status, layer_rows


def export_onnx(model: nn.Module, out_path: Path) -> None:
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["x_cat"],
        output_names=["readout_and_state"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )


def simplify_or_infer_onnx(onnx_path: Path) -> tuple[Any, dict[str, Any]]:
    info: dict[str, Any] = {
        "onnx_shape_inference_attempted": False,
        "onnx_shape_inference_success": False,
        "onnxsim_attempted": False,
        "onnxsim_success": False,
    }
    import onnx  # type: ignore

    model = onnx.load(str(onnx_path))
    info["onnx_shape_inference_attempted"] = True
    try:
        model = onnx.shape_inference.infer_shapes(model)
        inferred_path = onnx_path.with_name(onnx_path.stem + "_inferred.onnx")
        onnx.save(model, str(inferred_path))
        model = onnx.load(str(inferred_path))
        info["onnx_shape_inference_success"] = True
        info["onnx_inferred_path"] = str(inferred_path)
    except Exception as exc:  # noqa: BLE001
        info["onnx_shape_inference_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import onnxsim  # type: ignore

        info["onnxsim_attempted"] = True
        simplified, ok = onnxsim.simplify(model, overwrite_input_shapes={"x_cat": list(INPUT_SHAPE)})
        info["onnxsim_success"] = bool(ok)
        if ok:
            simplified_path = onnx_path.with_name(onnx_path.stem + "_inferred_simplified.onnx")
            onnx.save(simplified, str(simplified_path))
            model = onnx.load(str(simplified_path))
            info["onnxsim_path"] = str(simplified_path)
    except Exception as exc:  # noqa: BLE001
        info["onnxsim_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    try:
        from onnx import TensorProto, helper  # type: ignore

        existing = {vi.name for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
        added = 0
        for init in model.graph.initializer:
            if init.name in existing:
                continue
            vi = helper.make_tensor_value_info(init.name, init.data_type or TensorProto.FLOAT, list(init.dims))
            model.graph.value_info.append(vi)
            added += 1
        annotated_path = onnx_path.with_name(onnx_path.stem + "_shape_annotated.onnx")
        onnx.save(model, str(annotated_path))
        model = onnx.load(str(annotated_path))
        info["initializer_shape_annotation_attempted"] = True
        info["initializer_shape_annotation_added"] = int(added)
        info["initializer_shape_annotation_path"] = str(annotated_path)
    except Exception as exc:  # noqa: BLE001
        info["initializer_shape_annotation_attempted"] = True
        info["initializer_shape_annotation_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    return model, info


def run_onnx_shape_inference(hls4ml: Any, model: nn.Module, model_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix = f"onnx_shape_inference_{model_name}"
    out_dir = HLS4ML_PRJ / f"phc_stp_step_{model_name}_onnx"
    onnx_path = MODELS / f"phc_stp_step_{model_name}.onnx"
    status: dict[str, Any] = {
        "method": prefix,
        "attempted": True,
        "model_source": model_name,
        "fixed_batch_size": True,
        "input_shape": str(INPUT_SHAPE),
        "onnx_export_success": False,
        "convertible": False,
        "compile_success": False,
        "build_run": False,
        "build_success": False,
        "project_dir": str(out_dir),
        "onnx_path": str(onnx_path),
    }
    layer_rows: list[dict[str, Any]] = []
    try:
        safe_rmtree(out_dir)
        export_onnx(model, onnx_path)
        status["onnx_export_success"] = True
        onnx_model, infer_info = simplify_or_infer_onnx(onnx_path)
        status.update(infer_info)
        try:
            cfg = hls4ml.utils.config_from_onnx_model(str(onnx_path), default_precision=DEFAULT_PRECISION, granularity="name")
        except Exception as exc:  # noqa: BLE001
            status["config_from_onnx_error"] = f"{type(exc).__name__}: {exc}"
            cfg = {"Model": {"Precision": DEFAULT_PRECISION, "ReuseFactor": REUSE_FACTOR, "Strategy": STRATEGY}}
        write_json(RESULTS / f"{prefix}_config.json", cfg)
        hls_model = hls4ml.converters.convert_from_onnx_model(
            onnx_model,
            hls_config=cfg,
            output_dir=str(out_dir),
            project_name=f"phc_stp_step_{model_name}_onnx",
            backend="Vivado",
        )
        parsed, types, rows = layer_summary(hls_model)
        for row in rows:
            row["method"] = prefix
        layer_rows.extend(rows)
        status["parsed_layer_count"] = parsed
        status["layer_types"] = types
        try:
            hls_model.write()
        except Exception:
            pass
        status["cpp_project_generated"] = bool(out_dir.exists())
        status["convertible"] = True
        try_compile_and_build(hls_model, status, prefix, out_dir)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
    return status, layer_rows


def load_06f_reference() -> dict[str, Any]:
    # Fixed values are drawn from the frozen 06F latency breakdown report. We
    # keep them here only for comparison in the 06G report; no 06F files are
    # modified.
    return {
        "implementation": "06f_manual_hls",
        "latency_cycles": 8316,
        "latency_us_at_100mhz": 83.16,
        "interval": 8317,
        "DSP": 46,
        "DSP_pct": 20.9,
        "LUT": 24836,
        "LUT_pct_note": "06f routed report gives 14.20%; csynth utilization varies by report context",
        "FF": 14518,
        "FF_pct": 13.6,
        "BRAM_18K": 6,
        "BRAM_pct": 2.1,
        "source": str(PHASE06F / "PHC_HLS_LATENCY_BREAKDOWN_REPORT.md"),
    }


def write_report(status_rows: list[dict[str, Any]], layers: pd.DataFrame, equivalence: dict[str, Any]) -> None:
    status_df = pd.DataFrame(status_rows)
    success_rows = status_df[(status_df.get("convertible", False) == True) | (status_df.get("build_success", False) == True)] if not status_df.empty else pd.DataFrame()
    build_success = bool((status_df.get("build_success", pd.Series(dtype=bool)) == True).any()) if not status_df.empty else False
    convertible = bool((status_df.get("convertible", pd.Series(dtype=bool)) == True).any()) if not status_df.empty else False
    final_status = (
        "hls4ml_vivado_synthesis_report_available"
        if build_success
        else ("hls4ml_conversion_reached_model_object_but_build_failed" if convertible else "hls4ml_still_blocked_at_conversion")
    )

    comparison = pd.DataFrame([load_06f_reference()])
    comparison.to_csv(RESULTS / "manual_06f_reference_comparison.csv", index=False)
    concise_rows = []
    for row in status_rows:
        err = str(row.get("error", "") or row.get("compile_error", "") or row.get("build_error", ""))
        first_line = err.splitlines()[0] if err else ""
        if row.get("method") == "torch_fx_manual_layernorm":
            prior = "06E PyTorch FX failed on nn.LayerNorm shape [None,16]; 06G moved past nn.LayerNorm but stopped at unsupported torch.mean."
        elif row.get("method") == "torch_fx_unrolled_layernorm":
            prior = "Second FX attempt avoids torch.mean by explicit 16-way reduction; compare error line for next unsupported primitive."
        elif row.get("method") == "onnx_shape_inference_original_step":
            prior = "Initializer shape annotation moved past 06E's MatMul_72 shape blocker; hls4ml now stops at unsupported ONNX Gemm."
        elif row.get("method") == "onnx_shape_inference_manual_layernorm":
            prior = "Initializer shape annotation moved past MatMul_74 shape lookup; hls4ml now stops at unsupported ONNX Gemm."
        else:
            prior = ""
        concise_rows.append(
            {
                "method": row.get("method"),
                "convertible": row.get("convertible"),
                "compile_success": row.get("compile_success"),
                "build_run": row.get("build_run"),
                "build_success": row.get("build_success"),
                "first_error_line": first_line,
                "difference_vs_06e": prior,
            }
        )
    concise_df = pd.DataFrame(concise_rows)
    concise_df.to_csv(RESULTS / "hls4ml_retry_concise_status.csv", index=False)

    lines = [
        "# PHC hls4ml A3D3 Retry Report",
        "",
        f"Final status: `{final_status}`.",
        "",
        "## Scope",
        "",
        "- New isolated phase: `n2ew_era/06g_phc_hls4ml_a3d3_retry/`.",
        "- Source weights: adopted Phase 05C4 seed=2 step checkpoint produced in Phase 06E.",
        "- No training, no threshold selection, and no NEW5 usage.",
        "- The frozen manual HLS project under `06f_phc_lite_manual_hls` was not modified.",
        "",
        "## Pre-Conversion Equivalence",
        "",
        pd.DataFrame(equivalence if isinstance(equivalence, list) else [equivalence]).to_markdown(index=False),
        "",
        "The manual-LayerNorm wrapper must match the original step model before it is handed to hls4ml.",
        "",
        "## Retry Status",
        "",
        concise_df.to_markdown(index=False) if not concise_df.empty else "No retry rows were produced.",
        "",
        "Full tracebacks are stored in `results/hls4ml_retry_errors.txt`; the raw wide status table is stored in `results/hls4ml_retry_status.csv`.",
        "",
        "## Parsed hls4ml Layers",
        "",
        layers.to_markdown(index=False) if not layers.empty else "No hls4ml layer table was produced.",
        "",
        "## Comparison Endpoint",
        "",
        "The frozen hand-written 06F endpoint remains the reference if hls4ml cannot produce a csynth report:",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if build_success:
        lines += [
            "At least one hls4ml route reached `build(csim=True, synth=True)` and produced a synthesis artifact. Use the status table above for the exact method and report path.",
            "Resource numbers from hls4ml are exploratory and do not replace the validated 06F hand-written HLS implementation unless separately verified.",
        ]
    elif convertible:
        lines += [
            "At least one route produced an hls4ml model object, but compile/build did not complete. This is progress beyond the original 06E parsing failures, but it is still not a Vivado synthesis report.",
        ]
    else:
        lines += [
            "Both bounded retry routes remain blocked before a usable hls4ml synthesis report is produced.",
            "This is still useful evidence: the error messages below show whether the failure moved past the original 06E LayerNorm and ONNX MatMul shape blockers.",
        ]
    lines += [
        "",
        "## Protocol Boundaries",
        "",
        "- This phase does not claim PYNQ-Z2 deployment.",
        "- This phase does not claim FPGA PnL improvement.",
        "- This phase does not prove real-world profitability or true exchange replay.",
        "- If hls4ml succeeds, the result is a hardware-conversion artifact only; strategy evidence remains Phase 05C4.",
    ]
    write_text(ROOT / "RETRY_REPORT.md", "\n".join(lines) + "\n")

    # Keep full errors readable outside wide CSV cells.
    with (RESULTS / "hls4ml_retry_errors.txt").open("w", encoding="utf-8") as f:
        for row in status_rows:
            f.write(f"=== {row.get('method')} ===\n")
            for key, value in row.items():
                if key.endswith("error") or key == "error":
                    f.write(f"[{key}]\n{value}\n\n")

    print(f"Final status: {final_status}")
    print(status_df[["method", "convertible", "compile_success", "build_run", "build_success"]].to_string(index=False) if not status_df.empty else "No status rows")


def main() -> None:
    ensure_dirs()
    step_model, manual_model, payload = load_models()
    unrolled_model = make_unrolled_model(step_model)
    payload_summary = {
        "source_step_checkpoint": str(p06e.STEP_MODEL_PATH),
        "source_05c4_checkpoint": str(p06e.CKPT_PATH),
        "seed": payload.get("seed"),
        "phase": payload.get("phase"),
        "new5_used": bool(payload.get("new5_used") or payload.get("new5_used_for_selection")),
        "feature_dim": payload.get("feature_dim"),
        "hidden_dim": payload.get("hidden_dim"),
    }
    write_json(RESULTS / "source_payload_summary.json", payload_summary)
    equivalence_rows = [
        verify_model_equivalence(step_model, manual_model, "manual_layernorm_vectorized_mean"),
        verify_model_equivalence(step_model, unrolled_model, "manual_layernorm_unrolled_reduction"),
    ]
    pd.DataFrame(equivalence_rows).to_csv(RESULTS / "manual_layernorm_equivalence.csv", index=False)
    for row in equivalence_rows:
        if not row["allclose_atol_1e_6"]:
            raise RuntimeError(f"Manual LayerNorm wrapper is not equivalent to the step model: {row}")

    status_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    try:
        import hls4ml  # type: ignore

        hls4ml_version = str(getattr(hls4ml, "__version__", "unknown"))
        for label, model in [("manual_layernorm", manual_model), ("unrolled_layernorm", unrolled_model)]:
            torch_fx_status, torch_fx_layers = run_torch_fx_manual(hls4ml, model, label)
            torch_fx_status["hls4ml_version"] = hls4ml_version
            status_rows.append(torch_fx_status)
            layer_rows.extend(torch_fx_layers)

        # Method 2 is the ONNX shape-inference route. We first try the
        # original step graph (matching 06E), then the manual-LayerNorm graph
        # if the original still fails. This stays within the requested bounded
        # 2-3 correction attempts and records both outcomes explicitly.
        for model_name, model in [("original_step", step_model), ("manual_layernorm", manual_model)]:
            onnx_status, onnx_layers = run_onnx_shape_inference(hls4ml, model, model_name)
            onnx_status["hls4ml_version"] = hls4ml_version
            status_rows.append(onnx_status)
            layer_rows.extend(onnx_layers)
            if onnx_status.get("build_success"):
                break
    except Exception as exc:  # noqa: BLE001
        status_rows.append(
            {
                "method": "hls4ml_import_or_global_setup",
                "attempted": True,
                "convertible": False,
                "compile_success": False,
                "build_run": False,
                "build_success": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}",
            }
        )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(RESULTS / "hls4ml_retry_status.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    layers_df = pd.DataFrame(layer_rows)
    layers_df.to_csv(RESULTS / "hls4ml_retry_layer_types.csv", index=False)
    write_report(status_rows, layers_df, equivalence_rows)


if __name__ == "__main__":
    main()
