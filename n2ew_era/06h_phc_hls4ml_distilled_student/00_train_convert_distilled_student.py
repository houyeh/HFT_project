from __future__ import annotations

import csv
import copy
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06h_phc_hls4ml_distilled_student"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
HLS4ML_PRJ = ROOT / "hls4ml_prj"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE06E = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"
PHASE06F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"

for path in [PROJECT_ROOT, PHASE05A, PHASE06E]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_05a as p05a  # noqa: E402
import common_06e as p06e  # noqa: E402


FEATURE_DIM = 16
HIDDEN_DIM = 16
OUTPUT_DIM = 2 + HIDDEN_DIM
DEFAULT_PRECISION = "ap_fixed<16,6>"
REUSE_FACTOR = 1
STRATEGY = "Latency"
HLS4ML_BACKEND = "Vitis"
HLS_PART = "xc7z020clg400-1"
HLS_CLOCK_PERIOD_NS = 10
CLAMPED_PRECISION = "fixed<32,16,RND_CONV,SAT,0>"
CLAMPED_PLAIN_PRECISION = "fixed<32,16>"
CLAMPED_METHOD = "keras_distilled_student_clamped32"


@dataclass(frozen=True)
class DistillConfig:
    train_sequences: int = 4096
    val_sequences: int = 1024
    batch_size: int = 4096
    max_epochs: int = 50
    patience: int = 7
    learning_rate: float = 1e-3
    seed: int = 2


def ensure_dirs() -> None:
    for path in [ROOT, MODELS, RESULTS, HLS4ML_PRJ]:
        path.mkdir(parents=True, exist_ok=True)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def resolve_bash_for_hls4ml() -> dict[str, Any]:
    """Ensure hls4ml sees Git Bash instead of the WSL System32 shim.

    hls4ml's compile step invokes shell tooling. On Windows, PATH can resolve
    ``bash`` to the WSL launcher in ``C:\\Windows\\System32``; that produces
    opaque downstream compile failures. This check records the original
    resolution and, only when the System32 shim is detected, prepends Git for
    Windows' bash directory.
    """

    original_bash = shutil.which("bash")
    original_path = os.environ.get("PATH", "")
    row: dict[str, Any] = {
        "original_bash": original_bash or "",
        "detected_windows_system32_wsl_bash": False,
        "detected_windowsapps_wsl_bash": False,
        "detected_wsl_bash_shim": False,
        "path_modified": False,
        "selected_bash": original_bash or "",
        "selected_reason": "original PATH bash retained",
    }
    if original_bash:
        lower = original_bash.lower()
        row["detected_windows_system32_wsl_bash"] = "system32" in lower and "windows" in lower
        row["detected_windowsapps_wsl_bash"] = "windowsapps" in lower and lower.endswith("bash.exe")
        row["detected_wsl_bash_shim"] = bool(row["detected_windows_system32_wsl_bash"] or row["detected_windowsapps_wsl_bash"])
    if row["detected_wsl_bash_shim"]:
        candidates = [
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ]
        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            row["selected_reason"] = "Git for Windows bash not found"
            write_json(RESULTS / "bash_resolution_check.json", row)
            raise RuntimeError(
                "hls4ml compile would use the WSL System32 bash shim, but Git for Windows bash was not found. "
                "Install Git for Windows or add its bin directory to PATH before running this phase."
            )
        os.environ["PATH"] = str(selected.parent) + os.pathsep + original_path
        row["path_modified"] = True
        row["selected_bash"] = str(selected)
        row["selected_reason"] = "prepended Git for Windows bash because a Windows WSL bash shim was detected"
    row["post_patch_bash"] = shutil.which("bash") or ""
    write_json(RESULTS / "bash_resolution_check.json", row)
    return row


def resolve_vitis_for_hls4ml() -> dict[str, Any]:
    """Ensure the hls4ml Vitis backend inherits a sourced Vitis environment."""

    original_vitis_run = shutil.which("vitis-run") or shutil.which("vitis-run.bat")
    original_path = os.environ.get("PATH", "")
    row: dict[str, Any] = {
        "original_vitis_run": original_vitis_run or "",
        "path_modified": False,
        "selected_vitis_run": original_vitis_run or "",
        "selected_reason": "original PATH vitis-run retained",
        "settings64_path": "",
        "settings64_found": False,
        "settings64_applied": False,
        "settings64_error": "",
    }
    candidates = [
        Path(r"C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat"),
        Path(r"C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.exe"),
        Path(r"C:\Xilinx\Vitis\2025.2\bin\vitis-run.bat"),
        Path(r"C:\Xilinx\Vitis\2025.2\bin\vitis-run.exe"),
    ]
    if not original_vitis_run:
        selected = next((path for path in candidates if path.exists()), None)
        if selected is not None:
            os.environ["PATH"] = str(selected.parent) + os.pathsep + original_path
            row["path_modified"] = True
            row["selected_vitis_run"] = str(selected)
            row["selected_reason"] = "prepended detected Vitis bin directory for hls4ml Vitis backend"
        else:
            row["selected_reason"] = "vitis-run not found in PATH or known install locations"
    selected_vitis = Path(row["selected_vitis_run"]) if row["selected_vitis_run"] else None
    settings_candidates: list[Path] = []
    if selected_vitis is not None:
        settings_candidates.append(selected_vitis.parent.parent / "settings64.bat")
    settings_candidates += [
        Path(r"C:\AMDDesignTools\2025.2\Vitis\settings64.bat"),
        Path(r"C:\AMDDesignTools\2025.2\settings64.bat"),
        Path(r"C:\Xilinx\Vitis\2025.2\settings64.bat"),
        Path(r"C:\Xilinx\2025.2\settings64.bat"),
    ]
    settings = next((path for path in settings_candidates if path.exists()), None)
    if settings is not None:
        row["settings64_path"] = str(settings)
        row["settings64_found"] = True
        env_bat = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="ascii") as f:
                env_bat = Path(f.name)
                f.write("@echo off\n")
                f.write(f'call "{settings}" >nul\n')
                f.write("set\n")
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(env_bat)],
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
            )
            applied = 0
            for line in completed.stdout.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key:
                    os.environ[key] = value
                    applied += 1
            row["settings64_applied"] = True
            row["settings64_env_keys_applied"] = applied
            row["path_modified"] = True
            row["selected_reason"] = "called Vitis settings64.bat and imported the resulting environment before hls4ml build"
        except Exception as exc:  # noqa: BLE001
            row["settings64_error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                row["settings64_error_stdout"] = (exc.stdout or "")[-2000:]
                row["settings64_error_stderr"] = (exc.stderr or "")[-2000:]
            row["selected_reason"] = "settings64.bat was found but could not be applied"
            write_json(RESULTS / "vitis_resolution_check.json", row)
            raise RuntimeError(
                f"Failed to apply Vitis settings64.bat before hls4ml build: {settings}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if env_bat is not None:
                try:
                    env_bat.unlink(missing_ok=True)
                except Exception:
                    pass
    row["post_patch_vitis_run"] = shutil.which("vitis-run") or shutil.which("vitis-run.bat") or ""
    row["post_patch_clang"] = shutil.which("clang") or shutil.which("clang.exe") or ""
    row["post_patch_clang_tidy"] = shutil.which("clang-tidy") or shutil.which("clang-tidy.exe") or ""
    write_json(RESULTS / "vitis_resolution_check.json", row)
    return row


class CaptureProcessOutput:
    """Capture Python and child-process stdout/stderr at file-descriptor level."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.stdout_fd = -1
        self.stderr_fd = -1
        self.saved_stdout = -1
        self.saved_stderr = -1

    def __enter__(self) -> "CaptureProcessOutput":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8", buffering=1)
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)
        os.dup2(self.file.fileno(), self.stdout_fd)
        os.dup2(self.file.fileno(), self.stderr_fd)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        if self.saved_stdout >= 0:
            os.dup2(self.saved_stdout, self.stdout_fd)
            os.close(self.saved_stdout)
        if self.saved_stderr >= 0:
            os.dup2(self.saved_stderr, self.stderr_fd)
            os.close(self.saved_stderr)
        if self.file is not None:
            self.file.close()


def dependency_audit() -> pd.DataFrame:
    rows = []
    for module_name in ["tensorflow", "keras", "hls4ml", "torch", "numpy", "pandas"]:
        row = {"module": module_name, "available": False, "version": "", "error": ""}
        try:
            module = importlib.import_module(module_name)
            row["available"] = True
            row["version"] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dependency_audit.csv", index=False)
    return out


def require_keras_stack(dep: pd.DataFrame) -> tuple[Any, Any, Any]:
    missing = dep.loc[dep["module"].isin(["tensorflow", "hls4ml"]) & ~dep["available"], "module"].tolist()
    if missing:
        raise RuntimeError(f"Missing required Keras/hls4ml stack modules: {missing}")
    tf = importlib.import_module("tensorflow")
    hls4ml = importlib.import_module("hls4ml")
    keras = tf.keras
    return tf, keras, hls4ml


def load_teacher() -> tuple[torch.nn.Module, dict[str, Any]]:
    teacher, payload = p06e.load_step_model()
    if payload.get("new5_used") or payload.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: teacher step payload is marked NEW5-used.")
    teacher.eval()
    return teacher, payload


def make_distillation_pairs(split: str, n_sequences: int, teacher: torch.nn.Module) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = p05a.load_sequences(split)
    if payload.get("new5_used"):
        raise RuntimeError(f"Hard rule violation: {split} payload is marked new5_used=True")
    x = payload["x"][:n_sequences].float()
    n = int(x.shape[0])
    seq_len = int(x.shape[1])
    xs = np.empty((n * seq_len, FEATURE_DIM + HIDDEN_DIM), dtype=np.float32)
    ys = np.empty((n * seq_len, OUTPUT_DIM), dtype=np.float32)
    with torch.no_grad():
        state = torch.zeros(n, HIDDEN_DIM, dtype=torch.float32)
        row = 0
        for t in range(seq_len):
            x_cat = torch.cat([x[:, t, :], state], dim=-1)
            out = teacher(x_cat)
            next_readout, next_state = teacher.split_output(out)
            batch = int(x_cat.shape[0])
            xs[row : row + batch] = x_cat.cpu().numpy()
            ys[row : row + batch] = torch.cat([next_readout, next_state], dim=-1).cpu().numpy()
            state = next_state
            row += batch
    meta = {
        "split": split,
        "sequence_count": n,
        "seq_len": seq_len,
        "row_count": int(xs.shape[0]),
        "new5_used": False,
    }
    return xs, ys, meta


class _DiagonalConstraintPlaceholder:
    """Documentation placeholder; real Keras constraint is built inside TF scope."""


def build_student_model(tf: Any, keras: Any, teacher_payload: dict[str, Any]) -> Any:
    class DiagonalConstraint(keras.constraints.Constraint):
        def __call__(self, w):
            eye = tf.eye(HIDDEN_DIM, dtype=w.dtype)
            return w * eye

    x_in = keras.Input(shape=(FEATURE_DIM + HIDDEN_DIM,), name="x_cat")
    x_t = keras.layers.Dense(
        FEATURE_DIM,
        use_bias=False,
        trainable=False,
        name="input_x",
    )(x_in)
    state_prev = keras.layers.Dense(
        HIDDEN_DIM,
        use_bias=False,
        trainable=False,
        name="input_state",
    )(x_in)

    encoded_pre = keras.layers.Dense(HIDDEN_DIM, name="enc_linear")(x_t)
    encoded_sig = keras.layers.Activation("sigmoid", name="encoded_sigmoid")(encoded_pre)
    encoded = keras.layers.Multiply(name="encoded_silu")([encoded_pre, encoded_sig])

    gate_pre = keras.layers.Dense(HIDDEN_DIM, name="gate")(encoded)
    gate_t = keras.layers.Activation("sigmoid", name="gate_sigmoid")(gate_pre)
    cand_pre = keras.layers.Dense(HIDDEN_DIM, name="candidate")(encoded)
    candidate_t = keras.layers.Activation("tanh", name="candidate_tanh")(cand_pre)

    decay_state = keras.layers.Dense(
        HIDDEN_DIM,
        use_bias=False,
        kernel_constraint=DiagonalConstraint(),
        name="decay_diag",
    )(state_prev)
    gated_candidate = keras.layers.Multiply(name="gate_candidate_mul")([gate_t, candidate_t])
    state_temporal = keras.layers.Add(name="state_temporal")([decay_state, gated_candidate])

    gain_encoded = keras.layers.Dense(HIDDEN_DIM, name="gain_from_encoded")(encoded)
    gain_prev = keras.layers.Dense(HIDDEN_DIM, use_bias=False, name="gain_from_prev_state")(state_prev)
    gain_pre = keras.layers.Add(name="gain_pre")([gain_encoded, gain_prev])
    gain_t = keras.layers.Activation("sigmoid", name="gain_sigmoid")(gain_pre)

    message_pre = keras.layers.Dense(HIDDEN_DIM, name="synapse")(state_temporal)
    message_sig = keras.layers.Activation("sigmoid", name="message_sigmoid")(message_pre)
    message = keras.layers.Multiply(name="message_silu")([message_pre, message_sig])
    gain_message = keras.layers.Multiply(name="gain_message_mul")([gain_t, message])
    alpha_msg = keras.layers.Dense(
        HIDDEN_DIM,
        use_bias=False,
        trainable=False,
        name="alpha_diag_const",
    )(gain_message)
    loop_input = keras.layers.Add(name="loop_input_without_layernorm")([state_temporal, alpha_msg])
    readout = keras.layers.Dense(2, name="readout")(loop_input)
    out = keras.layers.Concatenate(name="readout_and_state")([readout, state_temporal])

    model = keras.Model(inputs=x_in, outputs=out, name="phc_distilled_student_no_layernorm")

    # Fixed routing layers and scalar constants. The student deliberately
    # removes both teacher LayerNorms, but keeps the same step interface.
    input_x_w = np.zeros((FEATURE_DIM + HIDDEN_DIM, FEATURE_DIM), dtype=np.float32)
    input_x_w[:FEATURE_DIM, :] = np.eye(FEATURE_DIM, dtype=np.float32)
    input_state_w = np.zeros((FEATURE_DIM + HIDDEN_DIM, HIDDEN_DIM), dtype=np.float32)
    input_state_w[FEATURE_DIM:, :] = np.eye(HIDDEN_DIM, dtype=np.float32)
    model.get_layer("input_x").set_weights([input_x_w])
    model.get_layer("input_state").set_weights([input_state_w])

    decay = np.asarray(teacher_payload.get("decay", np.ones(HIDDEN_DIM)), dtype=np.float32)
    if decay.shape != (HIDDEN_DIM,):
        decay = np.ones(HIDDEN_DIM, dtype=np.float32)
    model.get_layer("decay_diag").set_weights([np.diag(decay).astype(np.float32)])
    alpha_value = float(np.asarray(teacher_payload.get("alpha_scalar", [1.0]), dtype=np.float32).reshape(-1)[0])
    model.get_layer("alpha_diag_const").set_weights([np.eye(HIDDEN_DIM, dtype=np.float32) * alpha_value])
    return model


def train_student(tf: Any, keras: Any, config: DistillConfig) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    teacher, teacher_payload = load_teacher()
    x_train, y_train, train_meta = make_distillation_pairs("train", config.train_sequences, teacher)
    x_val, y_val, val_meta = make_distillation_pairs("val", config.val_sequences, teacher)

    model = build_student_model(tf, keras, teacher_payload)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss="mse")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
            min_delta=1e-6,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    model_path = MODELS / "phc_distilled_student.keras"
    model.save(model_path)
    pd.DataFrame(history.history).to_csv(RESULTS / "distillation_training_curve.csv", index_label="epoch_zero_based")
    meta = {
        "model_path": str(model_path),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "teacher_checkpoint": str(p06e.STEP_MODEL_PATH),
        "new5_used": False,
    }
    write_json(RESULTS / "distillation_dataset_meta.json", meta)
    return model, meta


def distillation_quality(model: Any, config: DistillConfig) -> pd.DataFrame:
    teacher, _ = load_teacher()
    rows = []
    for split, nseq in [("val", min(config.val_sequences, 1024))]:
        x_eval, y_eval, meta = make_distillation_pairs(split, nseq, teacher)
        pred = model.predict(x_eval, batch_size=config.batch_size, verbose=0).astype(np.float32)
        diff = pred - y_eval
        sign_ref = np.where(y_eval[:, :2] >= 0, 1, -1)
        sign_pred = np.where(pred[:, :2] >= 0, 1, -1)
        rows.append(
            {
                "split": split,
                "row_count": int(x_eval.shape[0]),
                "rmse_all": float(np.sqrt(np.mean(diff * diff))),
                "rmse_readout": float(np.sqrt(np.mean(diff[:, :2] * diff[:, :2]))),
                "rmse_state": float(np.sqrt(np.mean(diff[:, 2:] * diff[:, 2:]))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "sign_agreement_readout_0": float(np.mean(sign_ref[:, 0] == sign_pred[:, 0])),
                "sign_agreement_readout_1": float(np.mean(sign_ref[:, 1] == sign_pred[:, 1])),
                "joint_sign_agreement": float(np.mean(np.all(sign_ref == sign_pred, axis=1))),
                "new5_used": bool(meta.get("new5_used", False)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "distillation_quality.csv", index=False)
    return out


def configure_hls4ml(hls4ml: Any, model: Any) -> dict[str, Any]:
    cfg = hls4ml.utils.config_from_keras_model(
        model,
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=REUSE_FACTOR,
        granularity="name",
        backend=HLS4ML_BACKEND,
    )
    cfg.setdefault("Model", {})
    cfg["Model"]["Precision"] = DEFAULT_PRECISION
    cfg["Model"]["ReuseFactor"] = REUSE_FACTOR
    cfg["Model"]["Strategy"] = STRATEGY
    for layer_cfg in cfg.get("LayerName", {}).values():
        layer_cfg["ReuseFactor"] = REUSE_FACTOR
        layer_cfg.setdefault("Precision", DEFAULT_PRECISION)
    write_json(RESULTS / "hls4ml_keras_config.json", cfg)
    return cfg


PRECISION_RE = re.compile(r"(?:ap_)?(?:u?fixed|u?int|int)<\s*(\d+)\s*(?:,|>)")
TYPEDEF_RE = re.compile(r"typedef\s+(ap_[a-zA-Z_]+<[^;]+>)\s+([A-Za-z0-9_]+);")


def precision_total_bits(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = PRECISION_RE.search(value)
    if not match:
        return None
    return int(match.group(1))


def precision_rows_from_config(cfg: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, layer_cfg in cfg.get("LayerName", {}).items():
        precision = layer_cfg.get("Precision", {})
        if isinstance(precision, str):
            precision = {"result": precision}
        if not isinstance(precision, dict):
            continue
        for field, value in precision.items():
            rows.append(
                {
                    "source": source,
                    "layer_name": layer_name,
                    "field": field,
                    "ap_fixed_string": value,
                    "total_bits": precision_total_bits(value),
                    "clamped": False,
                }
            )
    return rows


def precision_rows_from_defines(defines_path: Path, source: str) -> list[dict[str, Any]]:
    if not defines_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in defines_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TYPEDEF_RE.search(line)
        if not match:
            continue
        fixed_type, type_name = match.groups()
        field = "typedef"
        layer_name = type_name
        for suffix in ["_accum_t", "_result_t", "_weight_t", "_bias_t", "_table_t"]:
            if type_name.endswith(suffix):
                layer_name = type_name[: -len(suffix)]
                field = suffix.strip("_")
                break
        rows.append(
            {
                "source": source,
                "layer_name": layer_name,
                "field": field,
                "ap_fixed_string": fixed_type,
                "total_bits": precision_total_bits(fixed_type),
                "clamped": False,
            }
        )
    return rows


def write_precision_audit(rows: list[dict[str, Any]]) -> None:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "06h_precision_audit.csv", index=False)


def clamp_config_precision(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return clamp_config_precision_with(cfg, CLAMPED_PRECISION, "clamped_config")


def clamp_config_precision_plain(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Clamp precision width only, leaving ap_fixed rounding/overflow defaults implicit."""

    return clamp_config_precision_with(cfg, CLAMPED_PLAIN_PRECISION, "clamped32_plain_config")


def clamp_config_precision_with(
    cfg: dict[str, Any],
    clamp_precision: str,
    source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clamped = copy.deepcopy(cfg)
    rows: list[dict[str, Any]] = []
    for layer_name, layer_cfg in clamped.get("LayerName", {}).items():
        precision = layer_cfg.get("Precision", {})
        if isinstance(precision, str):
            precision = {"result": precision}
            layer_cfg["Precision"] = precision
        if not isinstance(precision, dict):
            continue
        for field, value in list(precision.items()):
            if field not in {"accum", "result", "accum_t", "result_t"}:
                continue
            old_bits = precision_total_bits(value)
            should_clamp = value == "auto" or (old_bits is not None and old_bits > 32)
            if layer_name in {"x_cat", "readout_and_state"} and field in {"result", "result_t"}:
                new_value = DEFAULT_PRECISION
            elif should_clamp:
                new_value = clamp_precision
            else:
                new_value = value
            precision[field] = new_value
            rows.append(
                {
                    "source": source,
                    "layer_name": layer_name,
                    "field": field,
                    "ap_fixed_string": new_value,
                    "total_bits": precision_total_bits(new_value),
                    "clamped": new_value != value,
                    "original_ap_fixed_string": value,
                    "original_total_bits": old_bits,
                }
            )
    return clamped, rows


def find_csynth_reports(path: Path) -> list[Path]:
    return sorted(path.rglob("*csynth.rpt")) if path.exists() else []


def find_design_size_reports(path: Path) -> list[Path]:
    return sorted(path.rglob("*csynth_design_size.rpt")) if path.exists() else []


def parse_csynth(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {"csynth_report_path": str(path)}
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 5 and parts[0] in {"BRAM_18K", "DSP", "FF", "LUT"}:
            try:
                out[parts[0]] = int(float(parts[1]))
            except Exception:
                pass
    latency_match = re.search(r"\|\s*Latency \(cycles\)\s*\|\s*min\s*\|\s*max\s*\|", text)
    if latency_match:
        lines = text[latency_match.end() :].splitlines()
        for line in lines:
            cells = [p.strip() for p in line.strip().strip("|").split("|")]
            ints = [int(x) for x in cells if x.isdigit()]
            if len(ints) >= 2:
                out["latency_cycles_min"] = ints[0]
                out["latency_cycles_max"] = ints[1]
                break
    clock_match = re.search(r"Estimated\s+Clock\s+Period\s*\|\s*([\d.]+)\s*ns", text, re.IGNORECASE)
    if clock_match:
        out["estimated_clock_period_ns"] = float(clock_match.group(1))
    return out


def build_hls_model(hls_model: Any, out_dir: Path, row: dict[str, Any], compile_log: Path, build_log: Path) -> dict[str, Any]:
    row["bash_resolution_check_path"] = str(RESULTS / "bash_resolution_check.json")
    row["vitis_resolution_check_path"] = str(RESULTS / "vitis_resolution_check.json")
    row["compile_full_log_path"] = str(compile_log)
    row["build_full_log_path"] = str(build_log)
    try:
        bash_info = resolve_bash_for_hls4ml()
        row["compile_bash_original"] = bash_info.get("original_bash", "")
        row["compile_bash_selected"] = bash_info.get("selected_bash", "")
        row["compile_bash_path_modified"] = bool(bash_info.get("path_modified", False))
        with CaptureProcessOutput(compile_log):
            print(f"[06H compile] {row['method']} hls_model.compile() start")
            print(f"[06H compile] selected_bash={bash_info.get('selected_bash', '')}")
            hls_model.compile()
            print(f"[06H compile] {row['method']} hls_model.compile() success")
        row["compile_success"] = True
    except Exception as exc:  # noqa: BLE001
        row["compile_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        with compile_log.open("a", encoding="utf-8") as f:
            f.write("\n[06H compile] exception propagated to Python\n")
            f.write(row["compile_error"])
            f.write("\n")
    try:
        with CaptureProcessOutput(build_log):
            print(f"[06H build] {row['method']} hls_model.build(csim=True, synth=True) start")
            print(f"[06H build] compile_success={row.get('compile_success', False)}")
            vitis_info = resolve_vitis_for_hls4ml()
            row["build_vitis_original"] = vitis_info.get("original_vitis_run", "")
            row["build_vitis_selected"] = vitis_info.get("selected_vitis_run", "")
            row["build_vitis_path_modified"] = bool(vitis_info.get("path_modified", False))
            print(f"[06H build] selected_vitis_run={vitis_info.get('selected_vitis_run', '')}")
            hls_model.build(csim=True, synth=True)
            print(f"[06H build] {row['method']} hls_model.build(csim=True, synth=True) success")
        row["build_run"] = True
        row["build_success"] = True
    except Exception as exc:  # noqa: BLE001
        row["build_run"] = True
        row["build_success"] = False
        row["build_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=10)}"
        with build_log.open("a", encoding="utf-8") as f:
            f.write("\n[06H build] exception propagated to Python\n")
            f.write(row["build_error"])
            f.write("\n")
    reports = find_csynth_reports(out_dir)
    row["csynth_report_count"] = len(reports)
    if reports:
        row.update(parse_csynth(reports[0]))
    else:
        size_reports = find_design_size_reports(out_dir)
        row["csynth_design_size_report_count"] = len(size_reports)
        if size_reports:
            row["csynth_design_size_report_path"] = str(size_reports[0])
        build_text = build_log.read_text(encoding="utf-8", errors="ignore") if build_log.exists() else ""
        indicators = [
            line.strip()
            for line in build_text.splitlines()
            if "not recognized" in line
            or "does not exist" in line
            or "error" in line.lower()
            or "failed" in line.lower()
            or "config_array_partition" in line
            or "CSim done" in line
        ]
        row["build_success"] = False
        row["build_error"] = "hls_model.build(csim=True, synth=True) did not produce a full csynth report."
        if indicators:
            row["build_error"] += " Build log indicators: " + " | ".join(indicators[:12])
    return row


def base_status_row(method: str, out_dir: Path) -> dict[str, Any]:
    return {
        "method": method,
        "convertible": False,
        "compile_success": False,
        "build_run": False,
        "build_success": False,
        "project_dir": str(out_dir),
        "precision": DEFAULT_PRECISION,
        "reuse_factor": REUSE_FACTOR,
        "strategy": STRATEGY,
        "backend": HLS4ML_BACKEND,
        "part": HLS_PART,
        "clock_period_ns": HLS_CLOCK_PERIOD_NS,
    }


def convert_and_build(hls4ml: Any, model: Any) -> pd.DataFrame:
    existing_status = RESULTS / "hls4ml_keras_conversion_status.csv"
    prior_rows = []
    if existing_status.exists():
        try:
            old = pd.read_csv(existing_status)
            prior_rows = old.loc[old["method"] != CLAMPED_METHOD].to_dict("records")
        except Exception:
            prior_rows = []
    out_dir = HLS4ML_PRJ / "phc_distilled_student_keras_clamped32"
    compile_log = RESULTS / "06h_compile_full_log_clamped.txt"
    build_log = RESULTS / "06h_build_full_log_clamped.txt"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    row: dict[str, Any] = base_status_row(CLAMPED_METHOD, out_dir)
    try:
        cfg = configure_hls4ml(hls4ml, model)
        write_json(RESULTS / "06h_hls4ml_config_dump.json", cfg)
        audit_rows = precision_rows_from_config(cfg, "config_from_keras_model_raw")
        old_defines = HLS4ML_PRJ / "phc_distilled_student_keras" / "firmware" / "defines.h"
        audit_rows += precision_rows_from_defines(old_defines, "unclamped_generated_defines")
        clamped_cfg, clamp_rows = clamp_config_precision(cfg)
        audit_rows += clamp_rows
        write_json(RESULTS / "06h_hls4ml_config_clamped32.json", clamped_cfg)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=clamped_cfg,
            output_dir=str(out_dir),
            project_name="phc_distilled_student_keras_clamped32",
            backend=HLS4ML_BACKEND,
            part=HLS_PART,
            clock_period=HLS_CLOCK_PERIOD_NS,
        )
        row["convertible"] = True
        row = build_hls_model(hls_model, out_dir, row, compile_log, build_log)
        clamped_defines = out_dir / "firmware" / "defines.h"
        audit_rows += precision_rows_from_defines(clamped_defines, "clamped32_generated_defines")
        write_precision_audit(audit_rows)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        if not compile_log.exists():
            write_text(
                compile_log,
                "hls_model.compile() was not reached. The pipeline failed before compile.\n\n"
                + row["error"],
            )
        if not build_log.exists():
            write_text(
                build_log,
                "hls_model.build(csim=True, synth=True) was not reached. The pipeline failed before build.\n\n"
                + row["error"],
            )
    out = pd.DataFrame(prior_rows + [row])
    out.to_csv(RESULTS / "hls4ml_keras_conversion_status.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return out


def write_report(status: str, dep: pd.DataFrame, quality: pd.DataFrame | None, hls_status: pd.DataFrame | None, error: str = "") -> None:
    manual_ref = pd.DataFrame(
        [
            {
                "implementation": "06f_manual_hls_phc_stp_step",
                "latency_cycles": 8316,
                "latency_us_at_100mhz": 83.16,
                "DSP": 46,
                "LUT": 24836,
                "FF": 14518,
                "BRAM_18K": 6,
                "verification_level": "manual HLS csynth + RTL cosim + P&R timing report available",
            }
        ]
    )
    lines = [
        "# PHC hls4ml Distilled Student Report",
        "",
        f"Final status: `{status}`.",
        "",
        "## Scope",
        "",
        "- This is an isolated Phase 06H experiment.",
        "- It does not modify `06f_phc_lite_manual_hls`, `06g_phc_hls4ml_a3d3_retry`, or `05c4_phc_stp_cosine_lr`.",
        "- Teacher checkpoint: adopted 05C4 seed=2 PHC-STP step model.",
        "- No NEW5 data is used.",
        "",
        "## Dependency Audit",
        "",
        dep.to_markdown(index=False),
        "",
        "## Student Architecture",
        "",
        "The intended student is a Keras-native step model using only standard frontend layers: `Dense`, `Activation`, `Multiply`, `Add`, and `Concatenate`.",
        "It follows the requested 9-step PHC-STP skeleton but **removes both teacher LayerNorm blocks entirely**. This is not PHC-lite itself and is not numerically equivalent to the 06F deployment model.",
        "",
        "Differences from teacher:",
        "",
        "- `enc_norm` removed: `encoded_pre = Dense(16)(x_t)` goes directly into SiLU.",
        "- `loop_norm` removed: `z = loop_input` before readout.",
        "- `decay` and `alpha` are represented by HLS-friendly Dense-style constant/diagonal operations.",
        "- The student is trained by distillation against teacher readout/state outputs; it is not a strategy retraining phase.",
        "",
    ]
    if quality is not None and not quality.empty:
        lines += ["## Distillation Quality", "", quality.to_markdown(index=False), ""]
    else:
        lines += ["## Distillation Quality", "", "Not available because the run did not reach Keras training.", ""]
    if hls_status is not None and not hls_status.empty:
        lines += ["## hls4ml / Vitis Status", "", hls_status.to_markdown(index=False), ""]
        summary_cols = [
            "method",
            "backend",
            "part",
            "clock_period_ns",
            "build_success",
            "csynth_report_count",
            "csynth_design_size_report_count",
            "csynth_report_path",
            "csynth_design_size_report_path",
            "LUT",
            "FF",
            "DSP",
            "BRAM_18K",
            "latency_cycles_min",
            "latency_cycles_max",
            "estimated_clock_period_ns",
        ]
        present_cols = [col for col in summary_cols if col in hls_status.columns]
        if present_cols:
            lines += ["### Synthesis Result Summary", "", hls_status[present_cols].to_markdown(index=False), ""]
        for _, status_row in hls_status.iterrows():
            build_log_value = status_row.get("build_full_log_path", "")
            if not isinstance(build_log_value, str) or not build_log_value:
                continue
            build_log = Path(build_log_value)
            if not build_log.exists():
                continue
            excerpt: list[str] = []
            for line in build_log.read_text(encoding="utf-8", errors="ignore").splitlines():
                if (
                    "vitis-run" in line
                    or "vivado_hls" in line
                    or "set_part" in line
                    or "Part '" in line
                    or "config_array_partition" in line
                    or "CSim done" in line
                    or "csynth_design" in line
                    or "Synthesis failed" in line
                ):
                    excerpt.append(line)
                if len(excerpt) >= 12:
                    break
            if excerpt:
                lines += [
                    "### Build Tool Evidence",
                    "",
                    f"Method: `{status_row.get('method', 'unknown')}`",
                    "",
                    "The build log confirms the Vitis path was exercised rather than the deprecated `vivado_hls` path:",
                    "",
                    "```text",
                    *excerpt,
                    "```",
                    "",
                ]
    else:
        lines += ["## hls4ml / Vitis Status", "", "No hls4ml conversion status is available.", ""]
    audit_path = RESULTS / "06h_precision_audit.csv"
    if audit_path.exists():
        try:
            audit = pd.read_csv(audit_path)
            summary = (
                audit.groupby("source", dropna=False)["total_bits"]
                .max()
                .reset_index()
                .rename(columns={"total_bits": "max_total_bits"})
            )
            over32 = audit.loc[
                audit["total_bits"].fillna(0) > 32,
                ["source", "layer_name", "field", "ap_fixed_string", "total_bits"],
            ].head(25)
            lines += ["## Precision Audit", "", summary.to_markdown(index=False), ""]
            if not over32.empty:
                lines += ["### Precision Entries Above 32 Bits", "", over32.to_markdown(index=False), ""]
        except Exception as exc:  # noqa: BLE001
            lines += ["## Precision Audit", "", f"Could not summarize `{audit_path}`: {type(exc).__name__}: {exc}", ""]
    lines += [
        "## 06F Reference Endpoint",
        "",
        manual_ref.to_markdown(index=False),
        "",
    ]
    if error:
        lines += ["## Blocking Error", "", "```text", error.strip(), "```", ""]
    lines += [
        "## Claim Boundary",
        "",
        "- This student model is not PHC-lite itself.",
        "- This result must not be mixed with 06F/06G PHC-lite verification results.",
        "- This phase does not prove FPGA PnL improvement.",
        "- This phase does not prove real-world profitability or true exchange replay.",
    ]
    write_text(ROOT / "DISTILLED_STUDENT_REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    config = DistillConfig()
    dep = dependency_audit()
    write_json(RESULTS / "distillation_config.json", config.__dict__)
    quality = None
    hls_status = None
    try:
        tf, keras, hls4ml = require_keras_stack(dep)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
        write_text(
            RESULTS / "06h_compile_full_log.txt",
            "hls_model.compile() was not reached because the TensorFlow/Keras frontend dependency gate failed.\n\n"
            + err,
        )
        write_report("blocked_missing_tensorflow_keras_frontend", dep, quality, hls_status, err)
        print("Final status: blocked_missing_tensorflow_keras_frontend")
        print(err)
        return
    try:
        model, _meta = train_student(tf, keras, config)
        quality = distillation_quality(model, config)
        hls_status = convert_and_build(hls4ml, model)
        if bool(hls_status.get("build_success", pd.Series([False])).fillna(False).astype(bool).any()):
            status = "keras_student_hls4ml_vitis_synthesis_report_available"
        elif bool(hls_status.get("convertible", pd.Series([False])).fillna(False).astype(bool).any()):
            status = "keras_student_convertible_but_build_failed"
        else:
            status = "keras_student_conversion_failed"
        write_report(status, dep, quality, hls_status)
        print(f"Final status: {status}")
        print(hls_status.to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        compile_log = RESULTS / "06h_compile_full_log.txt"
        if not compile_log.exists():
            write_text(
                compile_log,
                "hls_model.compile() was not reached because the pipeline failed before conversion/compile.\n\n"
                + err,
            )
        write_report("keras_student_pipeline_failed", dep, quality, hls_status, err)
        print("Final status: keras_student_pipeline_failed")
        print(err)


if __name__ == "__main__":
    main()
