from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE06E = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"

if str(PHASE06E) not in sys.path:
    sys.path.insert(0, str(PHASE06E))
if str(PHASE05A) not in sys.path:
    sys.path.insert(1, str(PHASE05A))

import common_06e as p06e  # noqa: E402

WARMUP_ITERS = 100
TIMING_ITERS = 10_000
OUT_JSON = RESULTS / "cpu_baseline_timing.json"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def summarize_us(samples_us: list[float]) -> dict[str, float]:
    if not samples_us:
        raise ValueError("No timing samples collected.")
    ordered = sorted(samples_us)
    p99_idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return {
        "mean_us": float(statistics.fmean(samples_us)),
        "median_us": float(statistics.median(samples_us)),
        "p99_us": float(ordered[p99_idx]),
        "min_us": float(min(samples_us)),
    }


def time_call(fn: Callable[[torch.Tensor], torch.Tensor], x_cat: torch.Tensor) -> dict[str, float]:
    for _ in range(WARMUP_ITERS):
        fn(x_cat)

    samples_us: list[float] = []
    for _ in range(TIMING_ITERS):
        t0 = time.perf_counter()
        fn(x_cat)
        t1 = time.perf_counter()
        samples_us.append((t1 - t0) * 1_000_000.0)
    return summarize_us(samples_us)


@torch.no_grad()
def forward_no_grad(model: torch.nn.Module, x_cat: torch.Tensor) -> torch.Tensor:
    return model(x_cat)


def main() -> None:
    ensure_dirs()
    torch.set_grad_enabled(True)

    step_model, payload = p06e.load_step_model()
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step model payload is marked new5_used=True.")
    step_model.eval()
    step_model.cpu()

    x_cat = torch.zeros(1, 32, dtype=torch.float32, device="cpu")

    grad_enabled_stats = time_call(lambda x: step_model(x), x_cat)
    no_grad_stats = time_call(lambda x: forward_no_grad(step_model, x), x_cat)

    result = {
        "device": "CPU",
        "model": "PHC-lite step model (PyTorch float32)",
        "batch_size": 1,
        "warmup_iters": WARMUP_ITERS,
        "timing_iters": TIMING_ITERS,
        **no_grad_stats,
        "torch_no_grad": True,
        "new5_used": False,
        "step_checkpoint": str(p06e.STEP_MODEL_PATH),
        "grad_enabled_check": {
            **grad_enabled_stats,
            "torch_no_grad": False,
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "mean_us={mean_us:.3f} median_us={median_us:.3f} p99_us={p99_us:.3f} min_us={min_us:.3f}".format(
            **no_grad_stats
        )
    )


if __name__ == "__main__":
    main()
