from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
STEP_MODEL_PATH = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml" / "models" / "phc_stp_step_hls_seed2.pt"
TORCHSCRIPT_PATH = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "models" / "phc_stp_step_hls_seed2_torchscript.pt"
CPP_BENCH_EXE = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results" / "libtorch_build" / "cpp_libtorch_benchmark.exe"
TORCH_LIB_DIR_CANDIDATES = [
    PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
    Path(r"C:\Users\holbo\00VS_projjects\02_Michael_score_extractor\.conda-ai\Lib\site-packages\torch\lib"),
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phc_stp_step_hls import PHCSTPStepHLS  # noqa: E402


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_eager_model() -> PHCSTPStepHLS:
    payload = torch_load(STEP_MODEL_PATH)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step payload is marked new5_used=true.")
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def check_checkpoint_equivalence() -> dict[str, Any]:
    eager = load_eager_model()
    scripted = torch.jit.load(str(TORCHSCRIPT_PATH), map_location="cpu")
    scripted.eval()

    torch.manual_seed(20260808)
    fixed_inputs = torch.stack(
        [
            torch.zeros(32, dtype=torch.float32),
            torch.ones(32, dtype=torch.float32) * 0.125,
            torch.linspace(-1.0, 1.0, 32, dtype=torch.float32),
            torch.randn(32, dtype=torch.float32) * 0.25,
            torch.randn(32, dtype=torch.float32),
        ],
        dim=0,
    )

    with torch.no_grad():
        eager_out = eager(fixed_inputs)
        scripted_out = scripted(fixed_inputs)
    diff = (eager_out - scripted_out).abs().cpu().numpy()
    row_path = RESULTS / "checkpoint_torchscript_equivalence.csv"
    with row_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "max_abs_diff", "mean_abs_diff"])
        writer.writeheader()
        for i in range(diff.shape[0]):
            writer.writerow(
                {
                    "sample_idx": i,
                    "max_abs_diff": f"{float(diff[i].max()):.12g}",
                    "mean_abs_diff": f"{float(diff[i].mean()):.12g}",
                }
            )

    payload = {
        "status": "pass" if float(diff.max()) <= 1e-6 else "fail",
        "eager_checkpoint": rel(STEP_MODEL_PATH),
        "torchscript_checkpoint": rel(TORCHSCRIPT_PATH),
        "fixed_input_count": int(fixed_inputs.shape[0]),
        "output_dim": int(eager_out.shape[-1]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": 1e-6,
        "per_sample_csv": rel(row_path),
        "new5_used": False,
    }
    write_json(RESULTS / "checkpoint_torchscript_equivalence.json", payload)
    return payload


def powershell_json(script: str) -> dict[str, Any]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        script,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    payload: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0 and result.stdout.strip():
        try:
            payload["parsed"] = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    return payload


def cpu_snapshot(label: str) -> dict[str, Any]:
    cpu_load = powershell_json(
        "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples | "
        "Select-Object -Property CookedValue | ConvertTo-Json -Compress"
    )
    top_processes = powershell_json(
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 12 "
        "ProcessName,Id,CPU,WorkingSet64 | ConvertTo-Json -Compress"
    )
    return {
        "label": label,
        "captured_at_unix": time.time(),
        "cpu_load_counter": cpu_load,
        "top_processes": top_processes,
    }


def run_cpp_baseline() -> dict[str, Any]:
    out_json = RESULTS / "cpu_baseline_timing_libtorch_controlled_20260808.json"
    raw_csv = RESULTS / "cpu_baseline_timing_libtorch_controlled_20260808.csv"
    log_path = RESULTS / "cpu_baseline_timing_libtorch_controlled_20260808.log"
    cmd = [str(CPP_BENCH_EXE), str(TORCHSCRIPT_PATH), str(out_json), str(raw_csv), "8"]
    env = os.environ.copy()
    torch_lib_dirs = [p for p in TORCH_LIB_DIR_CANDIDATES if (p / "torch_cpu.dll").exists()]
    prepended = torch_lib_dirs[0] if torch_lib_dirs else None
    if prepended is not None:
        env["PATH"] = str(prepended) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, env=env)
    log_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "prepended_torch_lib_dir": rel(prepended) if prepended is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    payload.update({"returncode": result.returncode, "log_path": rel(log_path), "raw_latency_csv": rel(raw_csv)})
    return payload


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_us": float(values.mean()),
        "median_us": float(np.percentile(values, 50)),
        "p99_us": float(np.percentile(values, 99)),
        "min_us": float(values.min()),
        "max_us": float(values.max()),
        "std_us": float(values.std(ddof=0)),
    }


def run_python_parity() -> dict[str, Any]:
    warmup_iters = 100
    timing_iters = 10000
    torch_threads = 8
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass
    model = load_eager_model()
    x_cat = torch.zeros(1, 32, dtype=torch.float32)
    with torch.no_grad():
        timed_model = torch.jit.trace(model, x_cat, strict=True)
        timed_model = torch.jit.optimize_for_inference(timed_model)
        timed_model.eval()
        for _ in range(warmup_iters):
            _ = timed_model(x_cat)
        samples = np.empty(timing_iters, dtype=np.float64)
        for i in range(timing_iters):
            t0 = time.perf_counter_ns()
            _ = timed_model(x_cat)
            t1 = time.perf_counter_ns()
            samples[i] = (t1 - t0) / 1000.0

    raw_csv = RESULTS / "cpu_trace_constant_forward_parity_controlled_20260808.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_us"])
        for i, latency_us in enumerate(samples):
            writer.writerow([i, f"{latency_us:.9f}"])

    payload = {
        "status": "cpu_trace_constant_forward_parity_controlled_completed",
        "device": "CPU",
        "backend": "python_torchscript_trace",
        "model": "PHC-lite step model (TorchScript float32)",
        "batch_size": 1,
        "input": "torch.zeros(1, 32)",
        "warmup_iters": warmup_iters,
        "timing_iters": timing_iters,
        "timed_region": "timed_model(x_cat) forward only",
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "torch_version": str(torch.__version__),
        "step_model_checkpoint": rel(STEP_MODEL_PATH),
        "raw_latency_csv": rel(raw_csv),
        "new5_used": False,
        **summarize(samples),
    }
    write_json(RESULTS / "cpu_trace_constant_forward_parity_controlled_20260808.json", payload)
    return payload


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "cpu_latency_root_cause_audit_completed",
        "new5_used": False,
        "checkpoint_equivalence": check_checkpoint_equivalence(),
        "snapshots": {},
    }

    manifest["snapshots"]["before_cpp"] = cpu_snapshot("before_cpp")
    manifest["cpp_baseline"] = run_cpp_baseline()
    manifest["snapshots"]["after_cpp_before_python"] = cpu_snapshot("after_cpp_before_python")
    manifest["python_parity"] = run_python_parity()
    manifest["snapshots"]["after_python"] = cpu_snapshot("after_python")

    write_json(RESULTS / "cpu_latency_root_cause_audit_20260808.json", manifest)
    print(json.dumps({
        "checkpoint_max_abs_diff": manifest["checkpoint_equivalence"]["max_abs_diff"],
        "cpp_mean_us": manifest["cpp_baseline"].get("mean_us"),
        "cpp_median_us": manifest["cpp_baseline"].get("median_us"),
        "cpp_p99_us": manifest["cpp_baseline"].get("p99_us"),
        "python_mean_us": manifest["python_parity"]["mean_us"],
        "python_median_us": manifest["python_parity"]["median_us"],
        "python_p99_us": manifest["python_parity"]["p99_us"],
    }, indent=2))


if __name__ == "__main__":
    main()
