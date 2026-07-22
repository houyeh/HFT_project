from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06g_phc_diagonal_manual_hls"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05F = PROJECT_ROOT / "n2ew_era" / "05f_phc_diagonal_ssm"
SELECTED_POLICY_PATH = PHASE05F / "results" / "phc_diagonal_ssm_selected_policy.json"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PHASE05F) not in sys.path:
    sys.path.insert(0, str(PHASE05F))

import common_05a as p05a  # noqa: E402
from models.phc_diagonal_ssm import PHCDiagonalSSM  # noqa: E402

SPLIT = "val"
DEFAULT_SAMPLE_COUNT = 200
EQUIVALENCE_ATOL = 1e-6
OUT_CSV = RESULTS / "phc_diag_step_test_vectors.csv"
OUT_REPORT = ROOT / "PHC_DIAGONAL_TEST_VECTOR_REPORT.md"
OUT_SUMMARY = RESULTS / "phc_diag_step_test_vector_summary.csv"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def selected_checkpoint() -> tuple[Path, dict, dict]:
    selected = json.loads(SELECTED_POLICY_PATH.read_text(encoding="utf-8"))
    if selected.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: selected policy is marked new5_used_for_selection=True")
    seed = int(selected["seed"])
    ckpt_path = PHASE05F / "models" / f"phc_diagonal_ssm_seed{seed}_best_val.pt"
    ckpt = torch_load(ckpt_path)
    if ckpt.get("phase") != "05f":
        raise RuntimeError(f"Expected phase 05f checkpoint, got {ckpt.get('phase')}")
    if ckpt.get("new5_used_for_selection"):
        raise RuntimeError("Hard rule violation: checkpoint is marked new5_used_for_selection=True")
    return ckpt_path, ckpt, selected


def load_model() -> tuple[PHCDiagonalSSM, Path, dict, dict]:
    ckpt_path, ckpt, selected = selected_checkpoint()
    model = PHCDiagonalSSM(
        feature_dim=int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt.get("mtl_steps", 1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_path, ckpt, selected


def columns() -> list[str]:
    return (
        ["row_id", "sample_idx", "timestep"]
        + [f"x_t_{i}" for i in range(16)]
        + [f"state_prev_{i}" for i in range(16)]
        + ["ref_readout_0", "ref_readout_1"]
        + [f"ref_state_next_{i}" for i in range(16)]
    )


def generate_vectors(sample_count: int = DEFAULT_SAMPLE_COUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    model, ckpt_path, ckpt, selected = load_model()
    payload = p05a.load_sequences(SPLIT)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: validation sequence payload is marked new5_used=True.")

    x_all = payload["x"]
    n = min(int(sample_count), int(x_all.shape[0]))
    if n < 1:
        raise ValueError("No validation sequences available")
    x = x_all[:n].float()
    seq_len = int(x.shape[1])
    feature_dim = int(x.shape[2])
    if feature_dim != 16:
        raise RuntimeError(f"Expected feature_dim=16, got {feature_dim}")

    frames: list[pd.DataFrame] = []
    state = torch.zeros(n, 16, dtype=torch.float32)
    readout_t = torch.zeros(n, 2, dtype=torch.float32)
    row_base = 0
    model.eval()
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t, :].contiguous()
            state_prev = state.clone()
            state = model.step(x_t, state)
            readout_t = model.readout(state)
            block = np.concatenate(
                [
                    np.arange(row_base, row_base + n, dtype=np.float64).reshape(-1, 1),
                    np.arange(n, dtype=np.float64).reshape(-1, 1),
                    np.full((n, 1), t, dtype=np.float64),
                    x_t.detach().cpu().numpy().astype(np.float32),
                    state_prev.detach().cpu().numpy().astype(np.float32),
                    readout_t.detach().cpu().numpy().astype(np.float32),
                    state.detach().cpu().numpy().astype(np.float32),
                ],
                axis=1,
            )
            frames.append(pd.DataFrame(block, columns=columns()))
            row_base += n
        full_ref = model(x)

    diff = (readout_t - full_ref).detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else float("nan")
    mean_abs_diff = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    allclose = bool(torch.allclose(readout_t, full_ref, atol=EQUIVALENCE_ATOL, rtol=1e-6))
    if not allclose:
        raise RuntimeError(
            f"Diagonal step replay vs full model equivalence failed: max_abs_diff={max_abs_diff:.9g}, "
            f"atol={EQUIVALENCE_ATOL}"
        )

    out = pd.concat(frames, ignore_index=True)
    int_cols = ["row_id", "sample_idx", "timestep"]
    out[int_cols] = out[int_cols].astype(np.int64)
    float_cols = [c for c in out.columns if c not in int_cols]
    out[float_cols] = out[float_cols].astype(np.float32)
    out.to_csv(OUT_CSV, index=False, float_format="%.9g")

    size_bytes = OUT_CSV.stat().st_size if OUT_CSV.exists() else 0
    summary = pd.DataFrame(
        [
            {
                "split": SPLIT,
                "sample_count": n,
                "sequence_length": seq_len,
                "feature_dim": feature_dim,
                "row_count": int(len(out)),
                "csv_path": str(OUT_CSV),
                "csv_size_bytes": int(size_bytes),
                "equivalence_atol": EQUIVALENCE_ATOL,
                "max_abs_diff_final_readout": max_abs_diff,
                "mean_abs_diff_final_readout": mean_abs_diff,
                "allclose": allclose,
                "selected_seed": int(selected["seed"]),
                "selected_best_epoch": int(selected["validation_metrics"]["best_epoch"]),
                "source_checkpoint": str(ckpt_path),
                "new5_used": False,
            }
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary.iloc[0].to_dict())
    return out, summary


def write_report(summary: dict) -> None:
    lines = [
        "# PHC Diagonal Step Test Vector Report",
        "",
        "This file documents the float32 input/output vectors used for the Phase 06G manual HLS C++ testbench. It reuses the Phase 05A validation sequence loader and the Phase 05F selected diagonal checkpoint. No training, no threshold selection, and no NEW5 access are performed.",
        "",
        "## Corpus Size",
        "",
        f"- Split: `{summary['split']}`.",
        f"- Sequences used: `{summary['sample_count']}`.",
        f"- Timesteps per sequence: `{summary['sequence_length']}`.",
        f"- Total recurrent-step rows: `{summary['row_count']}`.",
        "",
        "The corpus intentionally mirrors Phase 06F: 200 validation sequences x 128 timesteps = 25,600 rows.",
        "",
        "## Equivalence Cross-check",
        "",
        f"- Source checkpoint: `{summary['source_checkpoint']}`.",
        f"- Selected seed: `{summary['selected_seed']}`.",
        f"- Selected best epoch: `{summary['selected_best_epoch']}`.",
        f"- Max absolute final-readout diff: `{summary['max_abs_diff_final_readout']:.9g}`.",
        f"- Mean absolute final-readout diff: `{summary['mean_abs_diff_final_readout']:.9g}`.",
        f"- Allclose at `atol={summary['equivalence_atol']}`: `{summary['allclose']}`.",
        "",
        "## CSV Schema",
        "",
        "- `row_id`, `sample_idx`, `timestep`",
        "- `x_t_0..x_t_15`",
        "- `state_prev_0..state_prev_15`",
        "- `ref_readout_0`, `ref_readout_1`",
        "- `ref_state_next_0..ref_state_next_15`",
        "",
        "All floating-point values are written with `%.9g` formatting after conversion to float32.",
        "",
        "## Output",
        "",
        f"- CSV: `{summary['csv_path']}`.",
        f"- CSV size: `{summary['csv_size_bytes']}` bytes.",
        f"- Summary CSV: `{OUT_SUMMARY}`.",
        "",
        "This is a testbench reference artifact only. It does not claim FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _, summary = generate_vectors()
    row = summary.iloc[0].to_dict()
    print(
        "generated_rows={row_count} sample_count={sample_count} seq_len={sequence_length} "
        "max_abs_diff={max_abs_diff_final_readout:.9g} allclose={allclose} csv={csv_path}".format(**row)
    )


if __name__ == "__main__":
    main()
