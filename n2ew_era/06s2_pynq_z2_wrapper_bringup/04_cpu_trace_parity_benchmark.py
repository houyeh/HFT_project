from __future__ import annotations

import csv
import json
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phc_stp_step_hls import PHCSTPStepHLS  # noqa: E402


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model() -> PHCSTPStepHLS:
    payload = torch_load(STEP_MODEL_PATH)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step model payload is marked new5_used=true.")
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_us": float(values.mean()),
        "median_us": float(np.percentile(values, 50)),
        "p99_us": float(np.percentile(values, 99)),
        "min_us": float(values.min()),
        "max_us": float(values.max()),
        "std_us": float(values.std(ddof=0)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    warmup_iters = 100
    timing_iters = 10000
    torch_threads = 8

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass

    model = load_model()
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

    raw_csv = RESULTS / "cpu_trace_constant_forward_parity.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_us"])
        for i, latency_us in enumerate(samples):
            writer.writerow([i, f"{latency_us:.9f}"])

    payload = {
        "status": "cpu_trace_constant_forward_parity_completed",
        "purpose": "Python TorchScript trace parity check against cpp_libtorch_benchmark.cpp fixed-zero-input forward-only protocol.",
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
        "step_model_checkpoint": str(STEP_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "raw_latency_csv": str(raw_csv.relative_to(PROJECT_ROOT)),
        "new5_used": False,
        **summarize(samples),
    }
    out_json = RESULTS / "cpu_trace_constant_forward_parity.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"mean_us={payload['mean_us']:.3f} median_us={payload['median_us']:.3f} "
        f"p99_us={payload['p99_us']:.3f} min_us={payload['min_us']:.3f} max_us={payload['max_us']:.3f}"
    )


if __name__ == "__main__":
    main()
