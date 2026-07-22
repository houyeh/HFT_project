from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE06E = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"

if str(PHASE06E) not in sys.path:
    sys.path.insert(0, str(PHASE06E))
if str(PHASE05A) not in sys.path:
    sys.path.insert(1, str(PHASE05A))

import common_06e as p06e  # noqa: E402

OUT_MODEL = MODELS / "phc_stp_step_hls_seed2_torchscript.pt"
OUT_MANIFEST = RESULTS / "torchscript_export_manifest.json"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    model, payload = p06e.load_step_model()
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step model payload is marked new5_used=True.")
    model.eval()
    model.cpu()

    x_cat = torch.zeros(1, 32, dtype=torch.float32)
    with torch.no_grad():
        ref = model(x_cat)
        traced = torch.jit.trace(model, x_cat, strict=True)
        traced = torch.jit.freeze(traced.eval())
        out = traced(x_cat)
    max_abs_diff = float((ref - out).abs().max().item())
    if max_abs_diff > 1e-6:
        raise RuntimeError(f"TorchScript trace mismatch: max_abs_diff={max_abs_diff}")

    traced.save(str(OUT_MODEL))
    manifest = {
        "phase": "06f",
        "source_step_checkpoint": str(p06e.STEP_MODEL_PATH),
        "torchscript_model": str(OUT_MODEL),
        "input_shape": [1, 32],
        "output_shape": list(out.shape),
        "trace_method": "torch.jit.trace",
        "frozen": True,
        "max_abs_diff_vs_python_step": max_abs_diff,
        "new5_used": False,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
