from __future__ import annotations

import csv
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
HLS4ML_PRJ = ROOT / "hls4ml_prj"

METHOD = "keras_distilled_student_clamped32_plain_hls4ml_main_nocsim"
PROJECT_NAME = "phc_distilled_student_keras_clamped32_plain_hls4ml_main"

FEATURE_DIM = 16
HIDDEN_DIM = 16
OUTPUT_DIM = 18
DEFAULT_PRECISION = "ap_fixed<16,6>"
CLAMPED_PLAIN_PRECISION = "fixed<32,16>"
REUSE_FACTOR = 1
STRATEGY = "Latency"
HLS4ML_BACKEND = "Vitis"
HLS_PART = "xc7z020clg400-1"
HLS_CLOCK_PERIOD_NS = 10

STATUS_CSV = RESULTS / "hls4ml_keras_conversion_status.csv"
BUILD_LOG = RESULTS / "06h_build_full_log_clamped32_plain_hls4ml_main_nocsim.txt"
ENV_JSON = RESULTS / "06h_upgrade_test_environment.json"
CONFIG_JSON = RESULTS / "06h_hls4ml_config_clamped32_plain_hls4ml_main.json"
PRECISION_CSV = RESULTS / "06h_precision_audit_clamped32_plain_hls4ml_main.csv"
UPGRADE_REPORT = ROOT / "HLS4ML_UPGRADE_DIAGNOSTIC_REPORT.md"

PRECISION_RE = re.compile(r"(?:ap_)?(?:u?fixed|u?int|int)<\s*(\d+)\s*(?:,|>)")


def ensure_dirs() -> None:
    for path in [MODELS, RESULTS, HLS4ML_PRJ]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")


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


def resolve_vitis_environment() -> dict[str, Any]:
    original_path = os.environ.get("PATH", "")
    original_vitis_run = shutil.which("vitis-run") or shutil.which("vitis-run.bat") or ""
    vitis_bin_candidates = [
        Path(r"C:\AMDDesignTools\2025.2\Vitis\bin"),
        Path(r"C:\Xilinx\Vitis\2025.2\bin"),
    ]
    row: dict[str, Any] = {
        "original_vitis_run": original_vitis_run,
        "vitis_bin_prepended": False,
        "vitis_bin_path": "",
        "settings64_path": "",
        "settings64_found": False,
        "settings64_applied": False,
        "settings64_env_keys_applied": 0,
        "path_modified": False,
        "post_patch_vitis_run": "",
        "post_patch_clang": "",
        "post_patch_clang_tidy": "",
    }
    if not original_vitis_run:
        vitis_bin = next((path for path in vitis_bin_candidates if (path / "vitis-run.bat").exists()), None)
        if vitis_bin is not None:
            os.environ["PATH"] = str(vitis_bin) + os.pathsep + original_path
            row["vitis_bin_prepended"] = True
            row["vitis_bin_path"] = str(vitis_bin)

    candidates = [
        Path(r"C:\AMDDesignTools\2025.2\Vitis\settings64.bat"),
        Path(r"C:\AMDDesignTools\2025.2\settings64.bat"),
        Path(r"C:\Xilinx\Vitis\2025.2\settings64.bat"),
        Path(r"C:\Xilinx\2025.2\settings64.bat"),
    ]
    settings = next((path for path in candidates if path.exists()), None)
    if settings is None:
        row["post_patch_vitis_run"] = shutil.which("vitis-run") or shutil.which("vitis-run.bat") or ""
        return row

    row["settings64_path"] = str(settings)
    row["settings64_found"] = True
    env_bat: Path | None = None
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
        row["path_modified"] = os.environ.get("PATH", "") != original_path
    finally:
        if env_bat is not None:
            env_bat.unlink(missing_ok=True)

    if not (shutil.which("vitis-run") or shutil.which("vitis-run.bat")):
        vitis_bin = next((path for path in vitis_bin_candidates if (path / "vitis-run.bat").exists()), None)
        if vitis_bin is not None:
            os.environ["PATH"] = str(vitis_bin) + os.pathsep + os.environ.get("PATH", "")
            row["vitis_bin_prepended"] = True
            row["vitis_bin_path"] = str(vitis_bin)
    row["post_patch_vitis_run"] = shutil.which("vitis-run") or shutil.which("vitis-run.bat") or ""
    row["post_patch_clang"] = shutil.which("clang") or shutil.which("clang.exe") or ""
    row["post_patch_clang_tidy"] = shutil.which("clang-tidy") or shutil.which("clang-tidy.exe") or ""
    return row


def load_student_model(tf: Any, keras: Any) -> Any:
    class DiagonalConstraint(keras.constraints.Constraint):
        def __call__(self, w):
            eye = tf.eye(HIDDEN_DIM, dtype=w.dtype)
            return w * eye

    model_path = MODELS / "phc_distilled_student.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Existing distilled student model not found: {model_path}")
    return keras.models.load_model(
        model_path,
        custom_objects={"DiagonalConstraint": DiagonalConstraint},
        compile=False,
        safe_mode=False,
    )


def precision_total_bits(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = PRECISION_RE.search(value)
    return int(match.group(1)) if match else None


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
    return cfg


def clamp_config_precision_plain(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
                new_value = CLAMPED_PLAIN_PRECISION
            else:
                new_value = value
            precision[field] = new_value
            rows.append(
                {
                    "source": "hls4ml_main_clamped32_plain_config",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(col for row in rows for col in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_status(row: dict[str, Any]) -> None:
    prior_rows: list[dict[str, Any]] = []
    columns: list[str] = []
    if STATUS_CSV.exists():
        with STATUS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns.extend(reader.fieldnames or [])
            for item in reader:
                if item.get("method") != METHOD:
                    prior_rows.append(item)
    columns = list(dict.fromkeys([*columns, *row.keys()]))
    with STATUS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prior_rows)
        writer.writerow(row)


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
        for line in text[latency_match.end() :].splitlines():
            cells = [p.strip() for p in line.strip().strip("|").split("|")]
            ints = [int(x) for x in cells if x.isdigit()]
            if len(ints) >= 2:
                out["latency_cycles_min"] = ints[0]
                out["latency_cycles_max"] = ints[1]
                break
    clock_match = re.search(r"Estimated\s+Clock\s+Period\s*\|\s*([\d.]+)\s*ns", text, re.IGNORECASE)
    if clock_match:
        out["estimated_clock_period_ns"] = clock_match.group(1)
    return out


def grep_flow_signature(out_dir: Path) -> dict[str, Any]:
    logs = sorted(out_dir.rglob("autopilot.flow.log"))
    result: dict[str, Any] = {
        "autopilot_flow_log_count": len(logs),
        "autopilot_flow_log": str(logs[-1]) if logs else "",
        "loop_label_result_line": "",
        "loop_label_error_line": "",
        "clang_tidy_syntax_line": "",
    }
    if not logs:
        return result
    text = logs[-1].read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "run_clang loop-label exec" in line:
            for follow in lines[i + 1 : i + 8]:
                if "run_clang result=" in follow:
                    result["loop_label_result_line"] = follow.strip()
                    break
        if "Caught error in clang_tidy" in line:
            result["loop_label_error_line"] = line.strip()
        if "Syntax check failed in clang-tidy" in line:
            result["clang_tidy_syntax_line"] = line.strip()
    return result


def failure_excerpt(log_path: Path, out_dir: Path, limit: int = 40) -> list[str]:
    hits: list[str] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            lower = line.lower()
            if (
                "error" in lower
                or "failed" in lower
                or "exception" in lower
                or "csynth_design" in line
                or "clang-tidy" in line
                or "syntax check" in lower
                or "vitis-run" in line
                or "config_array_partition" in line
                or "synthesis" in lower
            ):
                hits.append(line)
    flow = grep_flow_signature(out_dir)
    for key in ["loop_label_result_line", "loop_label_error_line", "clang_tidy_syntax_line"]:
        if flow.get(key):
            hits.append(f"{key}: {flow[key]}")
    return hits[-limit:]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(dict.fromkeys(col for row in rows for col in row))
    normalized = [[str(row.get(col, "")) for col in columns] for row in rows]
    widths = [
        max(len(str(col)), *(len(r[i]) for r in normalized))
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for r in normalized
    ]
    return "\n".join([header, sep, *body])


def write_report(row: dict[str, Any], env: dict[str, Any], excerpt: list[str]) -> None:
    quality_path = RESULTS / "distillation_quality.csv"
    quality_text = quality_path.read_text(encoding="utf-8", errors="ignore").strip() if quality_path.exists() else ""
    comparison = [
        {
            "implementation": "06F manual HLS PHC-STP",
            "model_identity": "production PHC-lite/STP seed2",
            "verification": "csim+cosim+routed P&R in 06F",
            "latency_cycles": "8316",
            "latency_us": "83.16 @ 100MHz",
            "DSP": "46",
            "LUT": "24836",
            "FF": "14518",
            "BRAM_18K": "6",
        },
        {
            "implementation": METHOD,
            "model_identity": "06H distilled Keras student, not PHC-lite itself",
            "verification": "upgrade-test no-CSim csynth attempt",
            "latency_cycles": f"{row.get('latency_cycles_min', '')}/{row.get('latency_cycles_max', '')}",
            "latency_us": "",
            "DSP": row.get("DSP", ""),
            "LUT": row.get("LUT", ""),
            "FF": row.get("FF", ""),
            "BRAM_18K": row.get("BRAM_18K", ""),
        },
    ]
    lines = [
        "# hls4ml Upgrade Diagnostic Report",
        "",
        "Status: `resolved_negative` if `csynth_report_count` remains zero; `resolved_positive` only if a full csynth report exists.",
        "",
        "## Scope",
        "",
        "- This is an environment/tool-version diagnostic for 06H only.",
        "- It reuses `models/phc_distilled_student.keras`; it does not retrain or re-distill the student.",
        "- It does not modify `06f_phc_lite_manual_hls`, `06g_phc_hls4ml_a3d3_retry`, `05c4_phc_stp_cosine_lr`, or NEW5.",
        "- The tested model is a distilled Keras student, not PHC-lite itself.",
        "",
        "## Version Evidence",
        "",
        "- PyPI version check: `pip index versions hls4ml` reported latest stable `1.3.0`, so no newer PyPI stable release exists above the project baseline.",
        "- Upgrade-test version: GitHub main installed as `hls4ml 1.4.0.dev32+gb90fb067` from commit `b90fb06736baa0908a8995fc1cf4ac4a7d1c241f`.",
        "- Official issue/changelog search did not reveal a directly named `Vitis 2025.2 clang-tidy nnet_utils` fix; this test is therefore an empirical main-branch compatibility probe.",
        "- The requested 06H-local venv path hit Windows long-path installation failure while installing TensorFlow. The successful isolated venv is the shorter repo-root `.h6`; it still does not touch the main project `.venv`.",
        "",
        "## Environment",
        "",
        markdown_table([{k: env.get(k, "") for k in ["python_executable", "python_version", "tensorflow_version", "keras_version", "hls4ml_version", "hls4ml_file"]}]),
        "",
        "## Build Result",
        "",
        markdown_table([{k: row.get(k, "") for k in ["method", "convertible", "build_run", "build_success", "csim_skipped", "csynth_report_count", "csynth_report_path", "loop_label_result_line", "clang_tidy_syntax_line"]}]),
        "",
        "## 06F Reference Endpoint Comparison",
        "",
        markdown_table(comparison),
        "",
        "## Distillation Quality Boundary",
        "",
        "Existing 06H quality metrics are reproduced below only to prevent confusion with 06F bit-exact PHC HLS. The key val split number is `joint_sign_agreement=0.9503707885742188`.",
        "",
        "```csv",
        quality_text,
        "```",
        "",
        "## Build Termination Excerpt",
        "",
        "```text",
        "\n".join(excerpt) if excerpt else "(no failure excerpt)",
        "```",
        "",
        "## Conclusion",
        "",
    ]
    if int(row.get("csynth_report_count") or 0) > 0 and not row.get("clang_tidy_syntax_line"):
        lines.append("升版 hls4ml 到 `1.4.0.dev32+gb90fb067` 有解決 clang-tidy 卡點，因為 no-CSim csynth 產生了完整 csynth report。")
    else:
        lines.append("升版 hls4ml 到 `1.4.0.dev32+gb90fb067` 沒有完成驗收標準，因為這次仍未產生完整 csynth report。")
    UPGRADE_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    out_dir = HLS4ML_PRJ / PROJECT_NAME
    row: dict[str, Any] = {
        "method": METHOD,
        "convertible": False,
        "compile_success": False,
        "compile_skipped": True,
        "build_run": False,
        "build_success": False,
        "csim_skipped": True,
        "csim_requested": False,
        "synth_requested": True,
        "project_dir": str(out_dir),
        "precision": DEFAULT_PRECISION,
        "plain_precision": CLAMPED_PLAIN_PRECISION,
        "reuse_factor": REUSE_FACTOR,
        "strategy": STRATEGY,
        "backend": HLS4ML_BACKEND,
        "part": HLS_PART,
        "clock_period_ns": HLS_CLOCK_PERIOD_NS,
        "build_full_log_path": str(BUILD_LOG),
    }
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    try:
        import tensorflow as tf
        import keras
        import hls4ml

        env.update(
            {
                "tensorflow_version": getattr(tf, "__version__", ""),
                "keras_version": getattr(keras, "__version__", ""),
                "hls4ml_version": getattr(hls4ml, "__version__", ""),
                "hls4ml_file": getattr(hls4ml, "__file__", ""),
            }
        )
        model = load_student_model(tf, keras)
        cfg = configure_hls4ml(hls4ml, model)
        plain_cfg, precision_rows = clamp_config_precision_plain(cfg)
        write_json(CONFIG_JSON, plain_cfg)
        write_csv(PRECISION_CSV, precision_rows)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=plain_cfg,
            output_dir=str(out_dir),
            project_name=PROJECT_NAME,
            backend=HLS4ML_BACKEND,
            part=HLS_PART,
            clock_period=HLS_CLOCK_PERIOD_NS,
        )
        row["convertible"] = True
        with CaptureProcessOutput(BUILD_LOG):
            print(f"[06H upgrade build] {METHOD} hls_model.build(csim=False, synth=True) start")
            print(f"[06H upgrade build] python_executable={sys.executable}")
            print(f"[06H upgrade build] tensorflow={env['tensorflow_version']}")
            print(f"[06H upgrade build] keras={env['keras_version']}")
            print(f"[06H upgrade build] hls4ml={env['hls4ml_version']}")
            vitis_env = resolve_vitis_environment()
            env.update(vitis_env)
            print(f"[06H upgrade build] settings64_applied={vitis_env.get('settings64_applied')}")
            print(f"[06H upgrade build] post_patch_vitis_run={vitis_env.get('post_patch_vitis_run')}")
            print(f"[06H upgrade build] post_patch_clang_tidy={vitis_env.get('post_patch_clang_tidy')}")
            hls_model.build(csim=False, synth=True)
            print(f"[06H upgrade build] {METHOD} hls_model.build returned")
        row["build_run"] = True
        row["build_success"] = True
    except Exception as exc:  # noqa: BLE001
        row["build_run"] = True
        row["build_success"] = False
        row["build_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        with BUILD_LOG.open("a", encoding="utf-8") as f:
            f.write("\n[06H upgrade build] exception propagated to Python\n")
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
        row["build_success"] = False
        if not row.get("build_error"):
            row["build_error"] = "hls_model.build(csim=False, synth=True) did not produce a full csynth report."
    row.update(grep_flow_signature(out_dir))
    write_json(ENV_JSON, env)
    append_status(row)
    excerpt = failure_excerpt(BUILD_LOG, out_dir)
    write_report(row, env, excerpt)
    print(json.dumps(sanitize(row), indent=2))


if __name__ == "__main__":
    main()
