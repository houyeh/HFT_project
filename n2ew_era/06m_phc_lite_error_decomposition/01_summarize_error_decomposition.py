from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06m_phc_lite_error_decomposition"
RESULTS = ROOT / "results"
PHASE06F_RESULTS = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results"

ONLY_LUT_CSV = RESULTS / "phc_step_only_lut_error_csim_out.csv"
ONLY_QUANT_CSV = RESULTS / "phc_step_only_quant_error_csim_out.csv"
ORIGINAL_COSIM_CSV = PHASE06F_RESULTS / "phc_step_cosim_out.csv"
ORIGINAL_CSIM_CSV = PHASE06F_RESULTS / "phc_step_csim_out.csv"


def summarize_csv(path: Path, label: str, source: str) -> dict[str, float | int | str | bool]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV for {label}: {path}")
    total_rows = 0
    mismatches = 0
    max_abs = 0.0
    sum_abs = 0.0
    count_abs = 0
    abs_cols: list[str] | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        abs_cols = [c for c in reader.fieldnames if c.startswith("abs_diff_")]
        for row in reader:
            total_rows += 1
            if int(float(row.get("pass", "1"))) == 0:
                mismatches += 1
            row_max = float(row.get("max_abs_diff", "0") or 0.0)
            if row_max > max_abs:
                max_abs = row_max
            for col in abs_cols:
                sum_abs += float(row.get(col, "0") or 0.0)
                count_abs += 1
    return {
        "label": label,
        "source": source,
        "path": str(path),
        "total_rows": total_rows,
        "diff_dimensions_per_row": len(abs_cols or []),
        "mismatches": mismatches,
        "max_abs_diff": max_abs,
        "mean_abs_diff": (sum_abs / count_abs) if count_abs else 0.0,
        "real_csim_run": bool("csim" in label and "06f_original" not in label),
    }


def maybe_summarize_original_csim() -> dict[str, float | int | str | bool] | None:
    if ORIGINAL_CSIM_CSV.exists():
        return summarize_csv(ORIGINAL_CSIM_CSV, "06f_original_csim_combined", "existing_06f_csim_csv")
    return None


def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else 0.0


def write_outputs(rows: list[dict[str, float | int | str | bool]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "error_decomposition_summary.csv"
    fieldnames = [
        "label",
        "source",
        "path",
        "total_rows",
        "diff_dimensions_per_row",
        "mismatches",
        "max_abs_diff",
        "mean_abs_diff",
        "real_csim_run",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    row_by_label = {str(row["label"]): row for row in rows}
    original = row_by_label["06f_original_cosim_combined"]
    lut = row_by_label["only_lut_error_csim"]
    quant = row_by_label["only_quant_error_csim"]

    original_max = float(original["max_abs_diff"])
    lut_max = float(lut["max_abs_diff"])
    quant_max = float(quant["max_abs_diff"])
    component_sum_max = lut_max + quant_max
    interaction_max = original_max - component_sum_max

    original_mean = float(original["mean_abs_diff"])
    lut_mean = float(lut["mean_abs_diff"])
    quant_mean = float(quant["mean_abs_diff"])
    component_sum_mean = lut_mean + quant_mean
    interaction_mean = original_mean - component_sum_mean

    if component_sum_max:
        dominant_max = "quantization_noise" if quant_max > lut_max else "lut_interpolation"
    else:
        dominant_max = "undetermined"
    if component_sum_mean:
        dominant_mean = "quantization_noise" if quant_mean > lut_mean else "lut_interpolation"
    else:
        dominant_mean = "undetermined"

    decomposition = {
        "baseline_original_source": "06F existing cosim CSV",
        "original_max_abs_diff": original_max,
        "original_mean_abs_diff": original_mean,
        "only_lut_max_abs_diff": lut_max,
        "only_lut_mean_abs_diff": lut_mean,
        "only_quant_max_abs_diff": quant_max,
        "only_quant_mean_abs_diff": quant_mean,
        "component_sum_max_abs_diff": component_sum_max,
        "component_sum_mean_abs_diff": component_sum_mean,
        "component_sum_to_original_max_ratio": component_sum_max / original_max if original_max else 0.0,
        "component_sum_to_original_mean_ratio": component_sum_mean / original_mean if original_mean else 0.0,
        "max_interaction_residual": interaction_max,
        "mean_interaction_residual": interaction_mean,
        "lut_share_of_component_max_pct": pct(lut_max, component_sum_max),
        "quant_share_of_component_max_pct": pct(quant_max, component_sum_max),
        "lut_share_of_component_mean_pct": pct(lut_mean, component_sum_mean),
        "quant_share_of_component_mean_pct": pct(quant_mean, component_sum_mean),
        "dominant_component_by_max": dominant_max,
        "dominant_component_by_mean": dominant_mean,
    }
    (RESULTS / "error_decomposition_metrics.json").write_text(json.dumps(decomposition, indent=2), encoding="utf-8")

    interaction_note = (
        "The component sum is within roughly 20% of the original max error, so a first-order additive explanation is reasonably consistent."
        if original_max and abs(component_sum_max / original_max - 1.0) <= 0.20
        else "The component sum is not within roughly 20% of the original max error, so interaction effects or non-identical worst-case rows are material."
    )
    rows_md = [
        "# PHC-lite HLS Error Decomposition Report",
        "",
        "This phase does not modify Phase 06F source code, reports, test vectors, or tolerance. It adds two isolated Vitis HLS C-sim control runs using the same 25,600 test-vector rows and the same `PHC_ABS_TOL=1.0` testbench tolerance.",
        "",
        "## Control Definitions",
        "",
        "- `only_lut_error_csim`: inputs, previous state, weights, layernorm, dot products, and outputs are evaluated as unquantized `double`; sigmoid/tanh/SiLU still use the same 65-point piecewise-linear tanh LUT family as the 06F HLS implementation. This isolates nonlinear LUT interpolation error up to floating/double-vs-float32 roundoff.",
        "- `only_quant_error_csim`: inputs, previous state, weights, accumulators, layernorm, and final outputs retain the 06F `ap_fixed<24,12>`/fixed-point datapath; sigmoid/tanh/SiLU are replaced by true floating `exp`/`tanh` evaluations cast back to the fixed accumulator type. This isolates fixed-point quantization/noise with LUT interpolation removed.",
        "- `06f_original_cosim_combined`: existing 06F C/RTL co-sim output, where fixed-point quantization and LUT nonlinearities are both present. This is the cited `0.00725275` max-absolute-error baseline.",
        "",
        "## Raw Metrics",
        "",
        "| label | source | total_rows | dimensions/row | mismatches | max_abs_diff | mean_abs_diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rows_md.append(
            f"| {row['label']} | {row['source']} | {row['total_rows']} | {row['diff_dimensions_per_row']} | "
            f"{row['mismatches']} | {float(row['max_abs_diff']):.10g} | {float(row['mean_abs_diff']):.10g} |"
        )
    rows_md.extend(
        [
            "",
            "## Component Split Against 06F Original",
            "",
            f"- 06F original combined reference used here: `06f_original_cosim_combined`, max_abs_diff `{original_max:.10g}`, mean_abs_diff `{original_mean:.10g}`.",
            f"- only-LUT-error max_abs_diff: `{lut_max:.10g}`; mean_abs_diff: `{lut_mean:.10g}`.",
            f"- only-quant-error max_abs_diff: `{quant_max:.10g}`; mean_abs_diff: `{quant_mean:.10g}`.",
            f"- Component max sum: `{component_sum_max:.10g}` ({component_sum_max / original_max:.3f}x original max).",
            f"- Component mean sum: `{component_sum_mean:.10g}` ({component_sum_mean / original_mean:.3f}x original mean).",
            f"- Max-error component shares, normalized by component sum: LUT `{pct(lut_max, component_sum_max):.2f}%`, quantization `{pct(quant_max, component_sum_max):.2f}%`.",
            f"- Mean-error component shares, normalized by component sum: LUT `{pct(lut_mean, component_sum_mean):.2f}%`, quantization `{pct(quant_mean, component_sum_mean):.2f}%`.",
            f"- Dominant component by max_abs_diff: `{dominant_max}`.",
            f"- Dominant component by mean_abs_diff: `{dominant_mean}`.",
            f"- Interaction residual by max_abs_diff (`original - component_sum`): `{interaction_max:.10g}`.",
            f"- Interaction residual by mean_abs_diff (`original - component_sum`): `{interaction_mean:.10g}`.",
            "",
            "## Interpretation",
            "",
            interaction_note,
            "",
            "The percentages above are component-share diagnostics, not a proof that both error sources combine linearly on the exact same row. If the component sum differs from the combined 06F result, that residual is reported as interaction/non-aligned-worst-case behavior rather than hidden.",
            "",
            "## Claim Boundary",
            "",
            "- These are C-sim numerical controls only, not P&R.",
            "- The controls use the same test vectors as 06F and do not alter the 06F implementation.",
            "- This decomposition does not make claims about real exchange replay, real-world profitability, or FPGA-caused PnL improvement.",
        ]
    )
    (ROOT / "PHC_HLS_ERROR_DECOMPOSITION_REPORT.md").write_text("\n".join(rows_md) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, float | int | str | bool]] = [
        summarize_csv(ORIGINAL_COSIM_CSV, "06f_original_cosim_combined", "existing_06f_cosim_csv"),
        summarize_csv(ONLY_LUT_CSV, "only_lut_error_csim", "phase_06m_vitis_csim"),
        summarize_csv(ONLY_QUANT_CSV, "only_quant_error_csim", "phase_06m_vitis_csim"),
    ]
    original_csim = maybe_summarize_original_csim()
    if original_csim is not None:
        rows.insert(1, original_csim)
    write_outputs(rows)
    print(json.dumps(json.loads((RESULTS / "error_decomposition_metrics.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
