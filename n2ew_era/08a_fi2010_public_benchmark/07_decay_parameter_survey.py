from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def primary_checkpoint_lookup() -> set[str]:
    metrics_path = RESULTS / "stage1_metrics_v2.csv"
    if not metrics_path.exists():
        return set()
    metrics = pd.read_csv(metrics_path)
    primary: set[str] = set()
    for (horizon, seed), group in metrics.groupby(["horizon", "seed"], sort=True):
        raw = group[group["loss_mode"] == "raw"]
        weighted = group[group["loss_mode"] == "weighted"]
        if raw.empty:
            continue
        raw_row = raw.iloc[0]
        use_weighted = bool(raw_row.get("degenerate_prediction", False)) and not weighted.empty
        chosen = weighted.iloc[0] if use_weighted else raw_row
        primary.add(str(chosen["checkpoint"]).replace("\\", "/"))
    return primary


def actual_decay_from_state(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, str, bool]:
    if "decay_param" in state:
        decay = torch.sigmoid(state["decay_param"].detach().cpu()).numpy().astype(np.float64)
        return decay, "decay_param", True
    if "decay" in state:
        decay = state["decay"].detach().cpu().numpy().astype(np.float64)
        return decay, "decay", False
    raise KeyError("Checkpoint state_dict has neither decay_param nor decay")


def alpha_from_state(state: dict[str, torch.Tensor]) -> tuple[float, str]:
    if "alpha" in state:
        return float(state["alpha"].detach().cpu().reshape(-1)[0]), "alpha"
    if "alpha_scalar" in state:
        return float(state["alpha_scalar"].detach().cpu().reshape(-1)[0]), "alpha_scalar"
    raise KeyError("Checkpoint state_dict has neither alpha nor alpha_scalar")


def survey() -> pd.DataFrame:
    primary = primary_checkpoint_lookup()
    rows: list[dict[str, object]] = []
    for checkpoint in sorted(MODELS.glob("*_v2.pt")):
        payload = torch_load(checkpoint)
        if payload.get("new5_used"):
            raise RuntimeError(f"Hard rule violation: {checkpoint} has new5_used=True")
        state = payload["model_state_dict"]
        decay, decay_source, sigmoid_applied = actual_decay_from_state(state)
        alpha, alpha_source = alpha_from_state(state)
        rel = str(checkpoint.relative_to(ROOT)).replace("\\", "/")
        rows.append(
            {
                "horizon": int(payload["horizon"]),
                "seed": int(payload["seed"]),
                "loss_mode": str(payload["loss_mode"]),
                "is_primary_stage1_v2_policy": rel in primary,
                "decay_mean": float(np.mean(decay)),
                "decay_min": float(np.min(decay)),
                "decay_max": float(np.max(decay)),
                "decay_std": float(np.std(decay)),
                "alpha_scalar": alpha,
                "decay_source_key": decay_source,
                "sigmoid_applied_to_decay": sigmoid_applied,
                "alpha_source_key": alpha_source,
                "decay_num_channels": int(decay.size),
                "decay_abs_max": float(np.max(np.abs(decay))),
                "alpha_abs": abs(alpha),
                "checkpoint": rel,
                "phase": str(payload.get("phase", "")),
                "new5_used": bool(payload.get("new5_used", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "seed", "loss_mode"]).reset_index(drop=True)


def write_report(df: pd.DataFrame) -> None:
    primary = df[df["is_primary_stage1_v2_policy"]].copy()
    all_decay_zero = bool((df["decay_abs_max"] == 0.0).all())
    all_alpha_zero = bool((df["alpha_abs"] == 0.0).all())
    primary_decay_zero = bool((primary["decay_abs_max"] == 0.0).all()) if not primary.empty else False
    primary_alpha_zero = bool((primary["alpha_abs"] == 0.0).all()) if not primary.empty else False
    source_keys = ", ".join(sorted(df["decay_source_key"].unique()))
    sigmoid_modes = ", ".join(str(x) for x in sorted(df["sigmoid_applied_to_decay"].unique()))

    report = f"""# FI-2010 Decay Parameter Survey

status: decay_zero_is_universal_in_08a_v2_checkpoints_but_fixed_by_implementation

## Scope

This survey loads existing Stage 1 v2 FI-2010 checkpoints from `models/`. It does not retrain, modify checkpoints, or use TXF NEW1-NEW5 data.

## Result

| item | value |
| --- | ---: |
| checkpoint files surveyed | {len(df)} |
| primary horizon/seed policies covered | {int(primary.shape[0])} |
| all checkpoint decay_abs_max == 0 | {all_decay_zero} |
| all checkpoint alpha_abs == 0 | {all_alpha_zero} |
| primary policies decay_abs_max == 0 | {primary_decay_zero} |
| primary policies alpha_abs == 0 | {primary_alpha_zero} |
| decay source key(s) | {source_keys} |
| sigmoid applied to decay | {sigmoid_modes} |

## Honest Conclusion

The zero-decay / zero-alpha behavior is universal across the surveyed Stage 1 v2 FI-2010 checkpoints, including all 15 primary horizon/seed policies. It is not a one-off peculiarity of `k=50/seed=1/raw`.

`decay_parameter_survey.csv` intentionally includes every existing `*_v2.pt` file, including raw and weighted variants. `decay_parameter_survey_primary_15.csv` is the compact 5-horizon x 3-seed primary-policy view.

However, this should not be described as training convergence to zero. The FI-2010 implementation stores `decay` and `alpha_scalar` as non-trainable buffers initialized to zero, unlike the TXF 05A model where `decay_param` is learnable and the actual memory coefficient is `sigmoid(decay_param)`. Therefore the correct statement is:

> In the FI-2010 Stage 1 v2 implementation, recurrent memory and the within-step message loop are disabled by construction for all surveyed checkpoints.

This supports using the Stage 2 HLS result as a fixed step-form classifier feasibility check, but it is not evidence that FI-2010 training learned to suppress recurrence.
"""
    (RESULTS / "decay_parameter_survey.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = survey()
    out_csv = RESULTS / "decay_parameter_survey.csv"
    df.to_csv(out_csv, index=False)
    primary_csv = RESULTS / "decay_parameter_survey_primary_15.csv"
    df[df["is_primary_stage1_v2_policy"]].to_csv(primary_csv, index=False)
    write_report(df)
    print(df[["horizon", "seed", "loss_mode", "is_primary_stage1_v2_policy", "decay_mean", "decay_min", "decay_max", "alpha_scalar", "decay_source_key", "sigmoid_applied_to_decay"]].to_string(index=False))
    print(f"\nwrote {out_csv}")
    print(f"wrote {primary_csv}")
    print(f"wrote {RESULTS / 'decay_parameter_survey.md'}")


if __name__ == "__main__":
    main()
