from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
SOURCE_SCRIPT = ROOT / "00_train_convert_distilled_student.py"
METHOD = "keras_distilled_student_clamped32_nocsim_settings64fix"


def load_phase_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase06h_distilled_student", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_existing_student_model(phase: Any, tf: Any, keras: Any, env: dict[str, Any]) -> Any:
    class DiagonalConstraint(keras.constraints.Constraint):
        def __call__(self, w):
            eye = tf.eye(phase.HIDDEN_DIM, dtype=w.dtype)
            return w * eye

    model_path = phase.MODELS / "phc_distilled_student.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Existing distilled student model not found: {model_path}")
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={"DiagonalConstraint": DiagonalConstraint},
            compile=False,
            safe_mode=False,
        )
        env["model_load_strategy"] = "keras.models.load_model"
        return model
    except Exception as exc:  # noqa: BLE001
        env["model_load_model_error"] = f"{type(exc).__name__}: {exc}"
        _teacher, teacher_payload = phase.load_teacher()
        model = phase.build_student_model(tf, keras, teacher_payload)
        model.load_weights(model_path)
        env["model_load_strategy"] = "rebuild_student_graph_then_load_existing_weights"
        return model


def update_status_csv(phase: Any, row: dict[str, Any]) -> pd.DataFrame:
    path = phase.RESULTS / "hls4ml_keras_conversion_status.csv"
    prior = pd.DataFrame()
    if path.exists():
        prior = pd.read_csv(path)
        if "method" in prior.columns:
            prior = prior.loc[prior["method"] != METHOD]
    out = pd.concat([prior, pd.DataFrame([row])], ignore_index=True, sort=False)
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return out


def report_failure_excerpt(log_path: Path, limit: int = 18) -> list[str]:
    if not log_path.exists():
        return [f"Build log missing: {log_path}"]
    hits: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        lower = line.lower()
        if (
            "error" in lower
            or "failed" in lower
            or "exception" in lower
            or "csynth_design" in line
            or "vitis-run" in line
            or "set_part" in line
            or "config_array_partition" in line
            or "synthesis" in lower
        ):
            hits.append(line)
    return hits[-limit:]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(dict.fromkeys(col for row in rows for col in row))
    normalized = [[str(row.get(col, "")) for col in columns] for row in rows]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in normalized))
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in normalized
    ]
    return "\n".join([header, sep, *body])


def append_or_replace_report_section(phase: Any, row: dict[str, Any], env: dict[str, Any], excerpt: list[str]) -> None:
    report_path = phase.ROOT / "DISTILLED_STUDENT_REPORT.md"
    old = report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else ""
    marker = "## Clamped32 no-CSim csynth Attempt With settings64.bat"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip() + "\n"

    summary_keys = [
        "method",
        "build_success",
        "csim_skipped",
        "csynth_report_count",
        "csynth_report_path",
        "LUT",
        "FF",
        "DSP",
        "BRAM_18K",
        "latency_cycles_min",
        "latency_cycles_max",
        "estimated_clock_period_ns",
    ]
    summary = {key: row.get(key, "") for key in summary_keys if key in row}
    summary_table = markdown_table([summary])
    excerpt_block = "\n".join(excerpt) if excerpt else "(no failure excerpt)"

    section = [
        "",
        marker,
        "",
        f"- Method row: `{METHOD}`.",
        "- `csim_skipped=True`: this run intentionally called `hls_model.build(csim=False, synth=True)` to isolate the Vitis csynth behavior from the known Windows CSim/mingw32-make no-rule failure.",
        "- `settings64fix=True`: this run uses the patched 06H `resolve_vitis_for_hls4ml()` path, which calls `settings64.bat` and imports the resulting Vitis/Vivado environment before invoking hls4ml's Vitis backend.",
        "- Claim boundary: this version has not passed CSim functional verification. If its csynth numbers are later quoted, they must be labeled as synthesis/resource/latency estimates only; Python-vs-C++ numerical consistency is not verified for this no-CSim run.",
        "- Scope boundary: this 06H run did not modify `06f_phc_lite_manual_hls`, `06g_phc_hls4ml_a3d3_retry`, or `05c4_phc_stp_cosine_lr`, and did not use NEW5.",
        "",
        "### Execution Environment",
        "",
        f"- Python executable: `{env.get('python_executable', '')}`",
        f"- hls4ml path: `{env.get('hls4ml_file', '')}`",
        f"- hls4ml version: `{env.get('hls4ml_version', '')}`",
        f"- TensorFlow version: `{env.get('tensorflow_version', '')}`",
        f"- Keras module/version: `{env.get('keras_module', '')}` / `{env.get('keras_version', '')}`",
        f"- Model load strategy: `{env.get('model_load_strategy', '')}`",
        f"- Vitis selected path: `{env.get('selected_vitis_run', '')}`",
        "",
        "### no-CSim Synthesis Summary",
        "",
        summary_table,
        "",
    ]
    if not row.get("build_success", False):
        first_error = next(
            (
                line
                for line in excerpt
                if "ERROR:" in line or "Exception:" in line
            ),
            str(row.get("build_error", "")).splitlines()[0] if row.get("build_error") else "",
        )
        terminal_stop = next((line for line in excerpt if "Syntax check failed" in line), "")
        section += [
            "### Failure Position",
            "",
            f"- First ERROR line before synthesis completion: `{first_error}`",
            f"- Terminal csynth stopping line: `{terminal_stop}`",
            "- Interpretation boundary: the `config_array_partition` ERROR appears before `set_part`, but Vitis continued into `csynth_design`; no root-cause fix was attempted in this run.",
            "- Full build log path: `results/06h_build_full_log_clamped32_nocsim_settings64fix.txt`",
            "",
            "### Failure Excerpt",
            "",
            "```text",
            excerpt_block,
            "```",
            "",
        ]
    report_path.write_text(old.rstrip() + "\n" + "\n".join(section).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    phase = load_phase_module()
    phase.ensure_dirs()
    build_log = phase.RESULTS / "06h_build_full_log_clamped32_nocsim_settings64fix.txt"
    env_path = phase.RESULTS / "06h_python_hls4ml_environment_nocsim.json"
    out_dir = phase.HLS4ML_PRJ / "phc_distilled_student_keras_clamped32"
    row: dict[str, Any] = phase.base_status_row(METHOD, out_dir)
    row.update(
        {
            "compile_skipped": True,
            "csim_skipped": True,
            "csim_requested": False,
            "synth_requested": True,
            "build_full_log_path": str(build_log),
            "bash_resolution_check_path": str(phase.RESULTS / "bash_resolution_check.json"),
            "vitis_resolution_check_path": str(phase.RESULTS / "vitis_resolution_check.json"),
            "precision_clamp_config": "clamped32",
            "settings64_fix_applied": True,
        }
    )
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    try:
        dep = phase.dependency_audit()
        tf, keras, hls4ml = phase.require_keras_stack(dep)
        env.update(
            {
                "tensorflow_version": getattr(tf, "__version__", ""),
                "keras_module": getattr(keras, "__name__", ""),
                "keras_version": getattr(keras, "__version__", ""),
                "hls4ml_file": getattr(hls4ml, "__file__", ""),
                "hls4ml_version": getattr(hls4ml, "__version__", ""),
            }
        )
        row["python_executable"] = env["python_executable"]
        row["python_version"] = env["python_version"]
        row["hls4ml_path"] = env["hls4ml_file"]
        row["hls4ml_version"] = env["hls4ml_version"]
        model = load_existing_student_model(phase, tf, keras, env)
        cfg = phase.configure_hls4ml(hls4ml, model)
        clamped_cfg, _clamp_rows = phase.clamp_config_precision(cfg)
        phase.write_json(phase.RESULTS / "06h_hls4ml_config_clamped32.json", clamped_cfg)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=clamped_cfg,
            output_dir=str(out_dir),
            project_name="phc_distilled_student_keras_clamped32",
            backend=phase.HLS4ML_BACKEND,
            part=phase.HLS_PART,
            clock_period=phase.HLS_CLOCK_PERIOD_NS,
        )
        row["convertible"] = True
        with phase.CaptureProcessOutput(build_log):
            print(f"[06H build] {METHOD} hls_model.build(csim=False, synth=True) start")
            print(f"[06H build] python_executable={sys.executable}")
            print(f"[06H build] hls4ml_file={env.get('hls4ml_file', '')}")
            print(f"[06H build] hls4ml_version={env.get('hls4ml_version', '')}")
            bash_info = phase.resolve_bash_for_hls4ml()
            vitis_info = phase.resolve_vitis_for_hls4ml()
            row["compile_bash_original"] = bash_info.get("original_bash", "")
            row["compile_bash_selected"] = bash_info.get("selected_bash", "")
            row["compile_bash_path_modified"] = bool(bash_info.get("path_modified", False))
            row["build_vitis_original"] = vitis_info.get("original_vitis_run", "")
            row["build_vitis_selected"] = vitis_info.get("selected_vitis_run", "")
            row["build_vitis_path_modified"] = bool(vitis_info.get("path_modified", False))
            env.update(vitis_info)
            print(f"[06H build] selected_vitis_run={vitis_info.get('selected_vitis_run', '')}")
            hls_model.build(csim=False, synth=True)
            print(f"[06H build] {METHOD} hls_model.build(csim=False, synth=True) returned")
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
    reports = phase.find_csynth_reports(out_dir)
    row["csynth_report_count"] = len(reports)
    if reports:
        row.update(phase.parse_csynth(reports[0]))
    else:
        size_reports = phase.find_design_size_reports(out_dir)
        row["csynth_design_size_report_count"] = len(size_reports)
        if size_reports:
            row["csynth_design_size_report_path"] = str(size_reports[0])
        row["build_success"] = False
        if not row.get("build_error"):
            row["build_error"] = "hls_model.build(csim=False, synth=True) did not produce a full csynth report."
    if row.get("build_success") and row.get("csynth_report_count", 0) < 1:
        row["build_success"] = False
    env["selected_vitis_run"] = row.get("build_vitis_selected", env.get("selected_vitis_run", ""))
    write_json(env_path, env)
    update_status_csv(phase, row)
    append_or_replace_report_section(phase, row, env, report_failure_excerpt(build_log))
    print(json.dumps({k: row.get(k) for k in sorted(row)}, indent=2, default=str))


if __name__ == "__main__":
    main()
