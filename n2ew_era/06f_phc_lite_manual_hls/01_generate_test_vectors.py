from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

import common_05a as p05a  # noqa: E402
import common_06e as p06e  # noqa: E402

SPLIT = "val"
DEFAULT_SAMPLE_COUNT = 200
EQUIVALENCE_ATOL = 1e-4
OUT_CSV = RESULTS / "phc_step_test_vectors.csv"
OUT_REPORT = ROOT / "PHC_TEST_VECTOR_REPORT.md"
OUT_SUMMARY = RESULTS / "phc_step_test_vector_summary.csv"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    step_model, step_payload = p06e.load_step_model()
    source_model, source_ckpt = p06e.load_source_model()
    payload = p05a.load_sequences(SPLIT)
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: validation sequence payload is marked new5_used=True.")

    x_all = payload["x"]
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    n = min(int(sample_count), int(x_all.shape[0]))
    x = x_all[:n].float()
    seq_len = int(x.shape[1])
    feature_dim = int(x.shape[2])
    if feature_dim != 16:
        raise RuntimeError(f"Expected feature_dim=16, got {feature_dim}")

    frames: list[pd.DataFrame] = []
    state = torch.zeros(n, 16, dtype=torch.float32)
    readout_t = torch.zeros(n, 2, dtype=torch.float32)
    row_base = 0
    step_model.eval()
    source_model.eval()
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t, :].contiguous()
            state_prev = state.clone()
            x_cat = torch.cat([x_t, state], dim=-1)
            out_step = step_model(x_cat)
            readout_t, state = step_model.split_output(out_step)
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
        source_ref = source_model(x)

    diff = (readout_t - source_ref).detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else float("nan")
    mean_abs_diff = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    allclose = bool(torch.allclose(readout_t, source_ref, atol=EQUIVALENCE_ATOL, rtol=1e-5))
    if not allclose:
        raise RuntimeError(
            f"Full PHCLiteSTP source vs step replay equivalence failed: max_abs_diff={max_abs_diff:.9g}, "
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
                "allclose_atol_1e_4": allclose,
                "source_seed": int(source_ckpt.get("seed", -1)),
                "source_best_epoch": int(source_ckpt.get("best_epoch", -1)),
                "step_checkpoint": str(p06e.STEP_MODEL_PATH),
                "source_checkpoint": str(p06e.CKPT_PATH),
                "new5_used": False,
            }
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary.iloc[0].to_dict(), step_payload)
    return out, summary


def write_report(summary: dict, step_payload: dict) -> None:
    schema_lines = [
        "`row_id`, `sample_idx`, `timestep`",
        "`x_t_0..x_t_15`",
        "`state_prev_0..state_prev_15`",
        "`ref_readout_0`, `ref_readout_1`",
        "`ref_state_next_0..ref_state_next_15`",
    ]
    write_markdown(
        OUT_REPORT,
        [
            "# PHC Step Test Vector Report",
            "",
            "This phase generates float32 input/output reference vectors for a manual Vitis HLS C++ testbench. It reuses the frozen 06E step model and the Phase 05A validation sequence loader. No training, no threshold selection, and no NEW5 access are performed.",
            "",
            "## Corpus Size",
            "",
            f"- Split: `{summary['split']}`.",
            f"- Sequences used: `{summary['sample_count']}`.",
            f"- Timesteps per sequence: `{summary['sequence_length']}`.",
            f"- Total rows: `{summary['row_count']}`.",
            "",
            "The 200-sequence corpus gives 25,600 recurrent-step rows. This is large enough to cover many state transitions while staying practical for a recurrent HLS cosimulation with LayerNorm/sigmoid/tanh-style nonlinearities. It is a validation subsample, not the full split.",
            "",
            "## Equivalence Cross-check",
            "",
            f"- Source checkpoint: `{summary['source_checkpoint']}`.",
            f"- Step checkpoint: `{summary['step_checkpoint']}`.",
            f"- Source seed: `{summary['source_seed']}`.",
            f"- Source best epoch: `{summary['source_best_epoch']}`.",
            f"- Max absolute final-readout diff: `{summary['max_abs_diff_final_readout']:.9g}`.",
            f"- Mean absolute final-readout diff: `{summary['mean_abs_diff_final_readout']:.9g}`.",
            f"- Allclose at `atol=1e-4`: `{summary['allclose_atol_1e_4']}`.",
            f"- Step alpha source formula: `{step_payload.get('alpha_source_formula', '')}`.",
            "",
            "## CSV Schema",
            "",
            "\n".join(f"- {line}" for line in schema_lines),
            "",
            "All floating-point values are written with `%.9g` formatting after conversion to float32, to avoid CSV round-trip noise dominating the testbench tolerance.",
            "",
            "## Output",
            "",
            f"- CSV: `{summary['csv_path']}`.",
            f"- CSV size: `{summary['csv_size_bytes']}` bytes.",
            f"- Summary CSV: `{OUT_SUMMARY}`.",
            "",
            "This is a testbench reference artifact only. It does not claim FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.",
        ],
    )


def main() -> None:
    _, summary = generate_vectors()
    row = summary.iloc[0]
    print(
        "generated_rows={row_count} sample_count={sample_count} seq_len={sequence_length} "
        "max_abs_diff={max_abs_diff_final_readout:.9g} allclose={allclose_atol_1e_4} csv={csv_path}".format(
            **row.to_dict()
        )
    )


if __name__ == "__main__":
    main()
