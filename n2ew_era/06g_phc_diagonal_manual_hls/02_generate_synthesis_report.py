from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06g_phc_diagonal_manual_hls"
RESULTS = ROOT / "results"
HLS_COMPONENT = ROOT / "hls_component"
HLS_REPORTS = HLS_COMPONENT / "hls_component" / "reports"
TECH_REPORT = PROJECT_ROOT / "n2ew_era" / "PHC_HLS_Technical_Report.md"
PHASE05F_REPORT = PROJECT_ROOT / "n2ew_era" / "05f_phc_diagonal_ssm" / "PHC_DIAGONAL_SSM_ABLATION_REPORT.md"
OUT_REPORT = ROOT / "PHC_DIAGONAL_SSM_MANUAL_HLS_SYNTHESIS_REPORT.md"
OUT_SUMMARY = RESULTS / "hls_run_summary.csv"

PYNQ_BUDGET = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}

BASELINE_06F = {
    "model": "phc_stp_v2_cosine_lr_dense16_06f",
    "latency_cycles": 8316,
    "latency_us_100mhz": 83.16,
    "interval_cycles": 8317,
    "DSP": 46,
    "LUT": 24836,
    "FF": 14518,
    "BRAM_18K": 6,
    "csim_cosim_total": 25600,
    "csim_cosim_mismatches": 0,
    "max_abs_diff": 0.00725275,
    "source": str(TECH_REPORT),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def summarize_output_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_run_output_missing",
            "path": str(path),
            "total": None,
            "mismatches": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    df = pd.read_csv(path)
    pass_col = df["pass"] if "pass" in df.columns else pd.Series(dtype=int)
    mismatch = int((pass_col == 0).sum()) if len(pass_col) else None
    max_abs = float(df["max_abs_diff"].max()) if "max_abs_diff" in df.columns and len(df) else None
    diff_cols = [c for c in df.columns if c.startswith("abs_diff_")]
    mean_abs = float(df[diff_cols].to_numpy().mean()) if diff_cols and len(df) else None
    return {
        "status": "PASS" if mismatch == 0 else "FAIL",
        "path": str(path),
        "total": int(len(df)),
        "mismatches": mismatch,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }


def parse_hls_compile_report() -> dict[str, Any]:
    candidates = [
        HLS_REPORTS / "hls_compile.rpt",
        HLS_COMPONENT / "hls_component" / "hls" / "syn" / "report" / "phc_diag_step_csynth.rpt",
    ]
    report_path = next((p for p in candidates if p.exists()), None)
    if report_path is None:
        return {"status": "not_run_report_missing", "path": "", "latency_cycles": None, "interval_cycles": None}
    text = read_text(report_path)
    out: dict[str, Any] = {"status": "found", "path": str(report_path)}
    # Vitis 2025 reports often include pipe-separated tables. Use permissive
    # regexes because report formatting differs between compile and csynth rpt.
    latency_match = re.search(r"Latency.*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", text, re.IGNORECASE | re.DOTALL)
    interval_match = re.search(r"Interval.*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", text, re.IGNORECASE | re.DOTALL)
    if latency_match:
        out["latency_cycles"] = int(latency_match.group(2))
    else:
        m = re.search(r"Latency\s*:\s*(\d+)", text, re.IGNORECASE)
        out["latency_cycles"] = int(m.group(1)) if m else None
    if interval_match:
        out["interval_cycles"] = int(interval_match.group(2))
    else:
        m = re.search(r"Interval\s*:\s*(\d+)", text, re.IGNORECASE)
        out["interval_cycles"] = int(m.group(1)) if m else None
    for resource in ["DSP", "LUT", "FF", "BRAM_18K", "BRAM"]:
        m = re.search(rf"{resource}.*?\|\s*([0-9,]+)\s*\|\s*([0-9,]+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            out[resource] = int(m.group(1).replace(",", ""))
    return out


def pct(value: Any, denom: int) -> Any:
    if value is None:
        return None
    return 100.0 * float(value) / float(denom)


def row_for_resource(model: str, metrics: dict[str, Any]) -> dict[str, Any]:
    bram = metrics.get("BRAM_18K", metrics.get("BRAM"))
    return {
        "model": model,
        "latency_cycles": metrics.get("latency_cycles"),
        "latency_us_100mhz": None
        if metrics.get("latency_cycles") is None
        else float(metrics["latency_cycles"]) * 0.01,
        "interval_cycles": metrics.get("interval_cycles"),
        "DSP": metrics.get("DSP"),
        "DSP_pct": pct(metrics.get("DSP"), PYNQ_BUDGET["DSP"]),
        "LUT": metrics.get("LUT"),
        "LUT_pct": pct(metrics.get("LUT"), PYNQ_BUDGET["LUT"]),
        "FF": metrics.get("FF"),
        "FF_pct": pct(metrics.get("FF"), PYNQ_BUDGET["FF"]),
        "BRAM_18K": bram,
        "BRAM_18K_pct": pct(bram, PYNQ_BUDGET["BRAM_18K"]),
    }


def make_report() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    weight_manifest_path = RESULTS / "weight_export_manifest.json"
    weight_manifest = json.loads(read_text(weight_manifest_path)) if weight_manifest_path.exists() else {}
    weight_audit = pd.read_csv(RESULTS / "weight_export_audit.csv") if (RESULTS / "weight_export_audit.csv").exists() else pd.DataFrame()
    vector_summary = (
        pd.read_csv(RESULTS / "phc_diag_step_test_vector_summary.csv")
        if (RESULTS / "phc_diag_step_test_vector_summary.csv").exists()
        else pd.DataFrame()
    )
    csim = summarize_output_csv(RESULTS / "phc_diag_step_csim_out.csv")
    cosim = summarize_output_csv(RESULTS / "phc_diag_step_cosim_out.csv")
    synth = parse_hls_compile_report()

    rows = [
        row_for_resource("06f_dense16_final", BASELINE_06F),
        row_for_resource("06g_diagonal16", synth),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    saturated = int(weight_audit["num_saturated"].sum()) if not weight_audit.empty else None
    saturation_status = "PASS" if saturated == 0 else ("not_available" if saturated is None else "FAIL")
    vector_rows = int(vector_summary.iloc[0]["row_count"]) if not vector_summary.empty else None
    vector_equiv = bool(vector_summary.iloc[0]["allclose"]) if not vector_summary.empty else None

    lines = [
        "# PHC Diagonal SSM Manual HLS Synthesis Report",
        "",
        "Status: `hls_package_ready_vitis_cli_not_available_in_current_shell`.",
        "",
        "This phase builds a manual HLS package for the Phase 05F PHCDiagonalSSM selected model. In the current Codex shell, `vitis-run` / `vitis-run.bat` was not found on PATH, so C-sim, C-synthesis, and C/RTL cosim were not executed here. The generated package is ready for the user to run in a Vitis-enabled shell.",
        "",
        "## Source And Protocol",
        "",
        f"- Source 05F selected checkpoint: `{weight_manifest.get('source_checkpoint', '')}`.",
        f"- Selected seed: `{weight_manifest.get('selected_seed', '')}`.",
        f"- Selected best epoch: `{weight_manifest.get('selected_best_epoch', '')}`.",
        "- Product/model data is inherited from Phase 05F; this phase does not retrain, reselect thresholds, or touch NEW5.",
        "- Diagonal recurrence is exactly `state_t = DIAG_DECAY * state_prev + DIAG_GAIN * encoded_t`.",
        "- `DIAG_DECAY` and `DIAG_GAIN` are baked sigmoid constants; no runtime sigmoid is used in recurrence.",
        "",
        "## Weight Export Audit",
        "",
        f"- Q24.12 saturation status: `{saturation_status}`.",
        f"- Total saturated values: `{saturated}`.",
        f"- Weight audit CSV: `{RESULTS / 'weight_export_audit.csv'}`.",
    ]
    if not weight_audit.empty:
        lines.extend(["", weight_audit.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Test Vector Audit",
            "",
            f"- Test-vector rows: `{vector_rows}`.",
            f"- Full-model vs step-replay allclose: `{vector_equiv}`.",
            f"- Test vector CSV: `{RESULTS / 'phc_diag_step_test_vectors.csv'}`.",
            "",
            "## C-Sim / C-Synthesis / Cosim Status",
            "",
            "| stage | status | total | mismatches | max_abs_diff | mean_abs_diff | path |",
            "|---|---|---:|---:|---:|---:|---|",
            f"| csim | {csim['status']} | {csim['total']} | {csim['mismatches']} | {csim['max_abs_diff']} | {csim['mean_abs_diff']} | `{csim['path']}` |",
            f"| cosim | {cosim['status']} | {cosim['total']} | {cosim['mismatches']} | {cosim['max_abs_diff']} | {cosim['mean_abs_diff']} | `{cosim['path']}` |",
            f"| csynth | {synth['status']} |  |  |  |  | `{synth.get('path', '')}` |",
            "",
            "## HLS Resource And Latency Comparison",
            "",
            summary.to_markdown(index=False),
            "",
            "06F dense16 baseline values are pulled from `n2ew_era/PHC_HLS_Technical_Report.md`: 8316 cycles / 83.16 us, DSP 46, LUT 24836, FF 14518, BRAM_18K 6.",
            "",
            "## Stage Pipeline Intent",
            "",
            "- Encoder stage: 16x16 matmul + LayerNorm + SiLU is kept sequential with explicit `PIPELINE off` on inner loops to avoid the known 06F auto-pipeline/auto-unroll DSP blowup.",
            "- Recurrence stage: diagonal elementwise recurrence has explicit `#pragma HLS PIPELINE II=1` on the 16-channel loop.",
            "- Readout stage: 2x16 matmul is kept sequential with `PIPELINE off`.",
            "",
            "Because Vitis was not available in this shell, the achieved encoder/recurrence II and final resource usage are not measured yet. Run the commands below from a Vitis-enabled shell to fill this report with actual numbers:",
            "",
            "```powershell",
            "cd C:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\n2ew_era\\06g_phc_diagonal_manual_hls\\hls_component",
            "vitis-run.bat --mode hls --csim --config hls_config.cfg --work_dir hls_component",
            "vitis-run.bat --mode hls --syn --config hls_config.cfg --work_dir hls_component",
            "vitis-run.bat --mode hls --cosim --config hls_config.cfg --work_dir hls_component",
            "```",
            "",
            "## PnL Framing Caveat From Phase 05F",
            "",
            "Diagonal16 is not broken and is not negative-PnL. It still beats HGB / queue / random baselines in Phase 05F, but regresses versus stronger PHC-family variants. The clean, artifact-safe cost to cite is the action PnL / oracle capture regression of roughly 8% to 13.5% versus dense16/dense8. The raw NEW3 causal-fill regression in 05F is not re-quoted as a clean capability-loss estimate here because it has not received the same simulator-state-carryover diagnostic that Phase 05E received.",
            "",
            "## Forbidden Claims",
            "",
            "- This phase does not retrain.",
            "- This phase does not touch NEW5.",
            "- This phase does not change the 06F deployed weights or claim to replace production.",
            "- This phase does not prove FPGA PnL improvement.",
            "- This phase does not prove true exchange replay or real-world profitability.",
            "- Until Vitis HLS is run, this phase does not prove diagonal16 latency, resource fit, csim, csynth, or cosim success.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = make_report()
    print(summary.to_string(index=False))
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
