from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07b_tcn_prepad_manual_hls"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE07A = PROJECT_ROOT / "n2ew_era" / "07a_tcn_feedforward_phc_tradeoff"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import common_05a as p05a  # noqa: E402
from src.models.tcn_prepad import TCNPrePadRegressor  # noqa: E402

SAMPLE_COUNT = 200
VARIANTS = {
    "scratch": PHASE07A / "models" / "tcn_prepad_scratch_16feat_seed2_best_val.pt",
    "distilled": PHASE07A / "models" / "tcn_prepad_distilled_16feat_seed1_best_val.pt",
}


def load_model(path: Path) -> tuple[TCNPrePadRegressor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCNPrePadRegressor(num_inputs=16, num_outputs=2, channels=16, kernel_size=3, dropout=0.1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict(model: TCNPrePadRegressor, x: torch.Tensor, scale: float) -> np.ndarray:
    with torch.no_grad():
        xb = x.float().permute(0, 2, 1).contiguous()
        xb = model.prepad(xb)
        out = model(xb).detach().cpu().numpy().astype(np.float32) * float(scale)
    return out


def write_vectors(variant: str, checkpoint: Path) -> dict[str, Any]:
    model, ckpt = load_model(checkpoint)
    payload = p05a.load_sequences("val")
    if payload.get("new5_used"):
        raise RuntimeError("Refusing payload marked new5_used=True")
    n = min(SAMPLE_COUNT, int(payload["x"].shape[0]))
    x = payload["x"][:n].float()
    scale = float(ckpt["target_scale"])
    ref = predict(model, x, scale)
    out_path = RESULTS / f"tcn_prepad_{variant}_test_vectors.csv"
    header = ["row_id", "sample_idx"]
    header += [f"x_{t}_{f}" for t in range(128) for f in range(16)]
    header += ["ref_readout_0", "ref_readout_1"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        x_np = x.numpy().astype(np.float32)
        for i in range(n):
            row = [i, i]
            row.extend(f"{float(v):.9g}" for v in x_np[i].reshape(-1))
            row.extend([f"{float(ref[i, 0]):.9g}", f"{float(ref[i, 1]):.9g}"])
            writer.writerow(row)
    return {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "test_vector_csv": str(out_path),
        "sample_count": n,
        "seq_len": 128,
        "feature_dim": 16,
        "target_scale": scale,
        "new5_used": False,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [write_vectors(variant, ckpt) for variant, ckpt in VARIANTS.items()]
    (RESULTS / "tcn_prepad_test_vector_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = [
        "# TCN PrePad Test Vector Report",
        "",
        f"- Sequence count per variant: `{SAMPLE_COUNT}` or fewer if validation payload is smaller.",
        "- Input columns are the unpadded `(128, 16)` PHC feature window. The HLS top function performs the 124-step causal prepad internally.",
        "- Reference readouts are PyTorch float32 model outputs after multiplying by each checkpoint's target scale.",
        "- NEW5 is not loaded.",
        "",
        "| variant | sample_count | csv |",
        "| --- | ---: | --- |",
    ]
    for row in rows:
        report.append(f"| {row['variant']} | {row['sample_count']} | {row['test_vector_csv']} |")
    (ROOT / "TCN_PREPAD_TEST_VECTOR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['variant']}: wrote {row['sample_count']} rows -> {row['test_vector_csv']}")


if __name__ == "__main__":
    main()
