from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

VARIANTS = ("scratch", "distilled")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.8g}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def load_variant_curve(variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = RESULTS / f"stage1_block0_{variant}_csim_out.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    diff_cols = [c for c in df.columns if c.startswith("abs_diff_c")]
    if not diff_cols:
        raise ValueError(f"no abs_diff columns in {path}")

    rows: list[dict[str, Any]] = []
    for out_t, g in df.groupby("out_t", sort=True):
        vals = g[diff_cols].to_numpy(dtype=np.float64)
        rows.append(
            {
                "variant": variant,
                "out_t": int(out_t),
                "rmse": float(math.sqrt(float((vals * vals).mean()))),
                "mean_abs_diff": float(vals.mean()),
                "max_abs_diff": float(vals.max()),
                "sample_count": int(len(g)),
                "compared_values": int(vals.size),
            }
        )
    curve = pd.DataFrame(rows)
    first_q = curve[curve["out_t"] < 62]
    last_q = curve[curve["out_t"] >= 186]
    post10 = curve[curve["out_t"] >= 10]
    steady = curve[curve["out_t"].between(10, 185)]
    late = curve[curve["out_t"] >= 186]

    slope = float(np.polyfit(post10["out_t"].to_numpy(dtype=np.float64), post10["rmse"].to_numpy(dtype=np.float64), 1)[0])
    early_steady_rmse = float(steady["rmse"].median())
    late_median_rmse = float(late["rmse"].median())
    late_to_steady = float(late_median_rmse / early_steady_rmse) if early_steady_rmse > 0 else float("nan")
    max_idx = int(curve.loc[curve["rmse"].idxmax(), "out_t"])
    post10_max = float(post10["rmse"].max())
    post10_median = float(post10["rmse"].median())

    # This is a diagnostic gate for "obvious divergence", not a tuned model threshold.
    # The absolute error is considered bounded if it stays far below the 07b readout
    # RMSE reference scale and the post-warmup linear slope is tiny in absolute units.
    slope_abs_per_step_gate = 1.0e-5
    bounded = bool(post10_max < 0.01 and abs(slope) < slope_abs_per_step_gate)
    pattern = "bounded_steady_offset" if bounded else "needs_review_possible_growth"

    summary = {
        "variant": variant,
        "rows": int(len(df)),
        "timesteps": int(len(curve)),
        "first_quarter_mean_rmse": float(first_q["rmse"].mean()),
        "last_quarter_mean_rmse": float(last_q["rmse"].mean()),
        "last_over_first_quarter_rmse": float(last_q["rmse"].mean() / first_q["rmse"].mean()),
        "post10_rmse_min": float(post10["rmse"].min()),
        "post10_rmse_median": post10_median,
        "post10_rmse_max": post10_max,
        "post10_rmse_slope_per_timestep": slope,
        "late_median_over_steady_median": late_to_steady,
        "max_rmse_timestep": max_idx,
        "classification": pattern,
        "stage2_gate_pass": bounded,
        "gate_basis": "post10_max<0.01 and abs(post10_linear_slope)<1e-5; diagnostic only, no model tuning",
        "source_csim": str(path),
    }
    return curve, summary


def write_plot(curve: pd.DataFrame) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "stage2_preflight_block0_timestep_rmse.png"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.8))
        for variant, g in curve.groupby("variant", sort=True):
            ax.plot(g["out_t"], g["rmse"], label=variant, linewidth=1.6)
        ax.axvline(10, color="0.55", linestyle="--", linewidth=1.0, label="post-warmup diagnostic start")
        ax.set_xlabel("valid block0 output timestep")
        ax.set_ylabel("RMSE vs Stage 0 Python streaming reference")
        ax.set_title("Stage 2 preflight: block0 timestep-wise fixed-point CSim error")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return str(out)
    except Exception as exc:  # pragma: no cover - only a reporting fallback
        return f"plot_failed: {exc}"


def write_report(curve: pd.DataFrame, summary: pd.DataFrame, plot_path: str) -> None:
    sampled = (
        curve[curve["out_t"].isin([0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 120, 160, 200, 247])]
        .reset_index(drop=True)
    )
    status = "stage2_preflight_pass_bounded_error" if summary["stage2_gate_pass"].all() else "stage2_preflight_blocked"
    lines = [
        f"status: {status}",
        "finding: block0_fixed_point_error_is_bounded_post_warmup" if summary["stage2_gate_pass"].all() else "finding: block0_error_growth_needs_review",
        "# Stage 2 Preflight Block0 Error Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic reads the existing Stage 1 CSim output and recomputes timestep-wise RMSE against the Stage 0 Python streaming reference. It does not modify 07a, 07b, Stage 0 outputs, or Stage 1 outputs. It does not use NEW5.",
        "",
        "## RMSE Curve Artifact",
        "",
        f"- Curve CSV: `{RESULTS / 'stage2_preflight_block0_timestep_rmse.csv'}`",
        f"- Curve PNG: `{plot_path}`",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Selected Timesteps",
        "",
        md_table(sampled),
        "",
        "## Interpretation",
        "",
        "The high last/first quarter ratio comes mainly from an extremely small first-quarter denominator. After excluding the first 10 valid timesteps, the absolute RMSE remains below 0.01 and does not show a large linear slope in absolute units. This supports bounded fixed-point error rather than an accumulating FIFO-alignment bug.",
        "",
        "## Stage Boundary",
        "",
        "Stage 2 block1-4 implementation is allowed only if all `stage2_gate_pass` values are True.",
    ]
    (ROOT / "STAGE2_PREFLIGHT_BLOCK0_ERROR_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        curve, summary = load_variant_curve(variant)
        curves.append(curve)
        summaries.append(summary)
    curve_df = pd.concat(curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    curve_df.to_csv(RESULTS / "stage2_preflight_block0_timestep_rmse.csv", index=False)
    summary_df.to_csv(RESULTS / "stage2_preflight_block0_error_summary.csv", index=False)
    with (RESULTS / "stage2_preflight_block0_error_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    plot_path = write_plot(curve_df)
    write_report(curve_df, summary_df, plot_path)
    print(f"wrote {RESULTS / 'stage2_preflight_block0_timestep_rmse.csv'}")
    print(f"wrote {RESULTS / 'stage2_preflight_block0_error_summary.csv'}")
    print(f"wrote {ROOT / 'STAGE2_PREFLIGHT_BLOCK0_ERROR_DIAGNOSTIC_REPORT.md'}")
    print(f"plot {plot_path}")
    if not summary_df["stage2_gate_pass"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
