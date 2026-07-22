"""Estimate CPU, C++ INT8, and FPGA latency from a saved hardware profile."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.hardware import profile_model_macs
from src.models.tcn import build_default_model


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def main() -> None:
    output_dir = Path("artifacts/04_hardware_handoff/latency_estimate")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_default_model(num_inputs=12)
    state_dict = torch.load("artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    example = torch.randn(1, 100, 12)
    profile = profile_model_macs(model, example)

    for _ in range(20):
        with torch.no_grad():
            _ = model(example)

    timings_ms = []
    for _ in range(200):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(example)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1_000.0)

    macs_per_sample = float(profile["total_macs"])
    estimated_cpp_int8_us = macs_per_sample / (8.0e9) * 1e6
    estimated_fpga_us = macs_per_sample / (50.0e9) * 1e6

    report = {
        "macs_per_sample": macs_per_sample,
        "params": int(profile["total_params"]),
        "weight_bytes_fp32": int(profile["weight_bytes_fp32"]),
        "weight_bytes_int8": int(profile["weight_bytes_int8"]),
        "measured_pytorch_cpu_latency_ms": {
            "mean": float(np.mean(timings_ms)),
            "p50": percentile(timings_ms, 50),
            "p90": percentile(timings_ms, 90),
            "p99": percentile(timings_ms, 99),
        },
        "estimated_cpp_int8_latency_us": estimated_cpp_int8_us,
        "estimated_fpga_latency_us": estimated_fpga_us,
        "assumptions": {
            "cpp_effective_int8_macs_per_second": 8.0e9,
            "fpga_effective_macs_per_second": 50.0e9,
            "note": "These are order-of-magnitude engineering estimates, not post-route measurements.",
        },
    }
    with (output_dir / "latency_estimate.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
