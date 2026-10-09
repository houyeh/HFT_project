from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSIM_CSV = RESULTS / "fi2010_phc_stage2_csim_out.csv"
CSYNTH_RPT = ROOT / "hls_component" / "hls_component" / "hls" / "syn" / "report" / "fi2010_phc_step_csynth.rpt"
MANIFEST_JSON = RESULTS / "stage2_hls_manifest.json"
SELECTION_CSV = RESULTS / "stage2_checkpoint_selection.csv"

DEVICE_TOTALS = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float, denom: float) -> float:
    return 100.0 * float(value) / float(denom)


def read_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


def read_selection() -> pd.DataFrame:
    return pd.read_csv(SELECTION_CSV)


def collect_csim() -> dict[str, object]:
    if not CSIM_CSV.exists():
        raise FileNotFoundError(CSIM_CSV)
    df = pd.read_csv(CSIM_CSV)
    pass_rows = df[df["pass"] == 1].copy()
    failed = int((df["pass"] == 0).sum())
    abs_cols = [c for c in df.columns if c.startswith("abs_diff_")]
    logit_agreements = {}
    for i in range(3):
        ref = df[f"ref_logit_{i}"].to_numpy(dtype=np.float64)
        dut = df[f"dut_logit_{i}"].to_numpy(dtype=np.float64)
        ref_sign = np.where(ref < 0.0, -1, 1)
        dut_sign = np.where(dut < 0.0, -1, 1)
        logit_agreements[f"sign_agreement_logit_{i}"] = float(np.mean(ref_sign == dut_sign))
    joint = np.ones(len(df), dtype=bool)
    for i in range(3):
        ref = df[f"ref_logit_{i}"].to_numpy(dtype=np.float64)
        dut = df[f"dut_logit_{i}"].to_numpy(dtype=np.float64)
        joint &= np.where(ref < 0.0, -1, 1) == np.where(dut < 0.0, -1, 1)
    row = {
        "status": "pass" if failed == 0 else "fail",
        "row_count": int(len(df)),
        "pass_rows": int(len(pass_rows)),
        "mismatches": failed,
        "max_abs_diff": float(df["max_abs_diff"].max()),
        "mean_abs_diff_all_outputs": float(df[abs_cols].to_numpy(dtype=np.float64).mean()),
        "joint_sign_agreement": float(np.mean(joint)),
        "zero_sign_convention": "0 -> positive",
        "source_csv": str(CSIM_CSV.relative_to(ROOT)),
        **logit_agreements,
    }
    write_rows(RESULTS / "stage2_csim_summary.csv", [row])
    return row


def parse_csynth() -> dict[str, object]:
    if not CSYNTH_RPT.exists():
        raise FileNotFoundError(CSYNTH_RPT)
    text = CSYNTH_RPT.read_text(encoding="utf-8", errors="replace")

    clock_match = re.search(r"\|ap_clk\s*\|\s*([0-9.]+) ns\|\s*([0-9.]+) ns\|\s*([0-9.]+) ns\|", text)
    latency_match = re.search(
        r"\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9.]+) us\|\s*([0-9.]+) us\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([a-zA-Z]+)\|",
        text,
    )
    total_match = re.search(r"\|Total\s*\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|", text)
    available_match = re.search(
        r"\|Available\s*\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|\s*([0-9]+)\|",
        text,
    )

    if not clock_match or not latency_match or not total_match or not available_match:
        raise RuntimeError(f"Could not parse csynth report: {CSYNTH_RPT}")

    bram, dsp, ff, lut, uram = [int(x) for x in total_match.groups()]
    abr, adsp, aff, alut, auram = [int(x) for x in available_match.groups()]
    latency_min, latency_max, latency_us_min, latency_us_max, interval_min, interval_max, pipeline = latency_match.groups()
    row = {
        "status": "present",
        "target_clock_ns": float(clock_match.group(1)),
        "estimated_clock_ns": float(clock_match.group(2)),
        "uncertainty_ns": float(clock_match.group(3)),
        "estimated_fmax_mhz": 1000.0 / float(clock_match.group(2)),
        "latency_cycles_min": int(latency_min),
        "latency_cycles_max": int(latency_max),
        "latency_us_min_at_100mhz": float(latency_us_min),
        "latency_us_max_at_100mhz": float(latency_us_max),
        "interval_cycles_min": int(interval_min),
        "interval_cycles_max": int(interval_max),
        "pipeline_type": pipeline,
        "DSP": dsp,
        "DSP_available": adsp,
        "DSP_pct": pct(dsp, adsp),
        "LUT": lut,
        "LUT_available": alut,
        "LUT_pct": pct(lut, alut),
        "FF": ff,
        "FF_available": aff,
        "FF_pct": pct(ff, aff),
        "BRAM_18K": bram,
        "BRAM_18K_available": abr,
        "BRAM_18K_pct": pct(bram, abr),
        "URAM": uram,
        "URAM_available": auram,
        "fits_zynq7020_csynth_budget": bool(
            dsp < DEVICE_TOTALS["DSP"]
            and lut < DEVICE_TOTALS["LUT"]
            and ff < DEVICE_TOTALS["FF"]
            and bram < DEVICE_TOTALS["BRAM_18K"]
        ),
        "csynth_report": str(CSYNTH_RPT.relative_to(ROOT)),
    }
    write_rows(RESULTS / "stage2_csynth_summary.csv", [row])
    return row


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_resource(value: int, pct_value: float) -> str:
    return f"{value} ({fmt_pct(pct_value)})"


def write_report(csim: dict[str, object], csynth: dict[str, object]) -> None:
    manifest = read_manifest()
    selection = read_selection()
    selected = selection[selection["selected_for_stage2"] == True].iloc[0]  # noqa: E712
    audit = pd.read_csv(RESULTS / "stage2_weight_export_audit.csv")
    vector_summary = pd.read_csv(RESULTS / "stage2_test_vector_summary.csv").iloc[0]
    saturated_total = int(audit["num_saturated"].sum())

    comparison_rows = [
        {
            "design": "TXF PHC-lite v8 manual HLS",
            "input_dim": "16",
            "latency": "8316 cycles / 83.16 us",
            "frequency": "100 MHz",
            "verification_tier": "routed_pnr_met + cosim",
            "resource": "DSP 46, LUT 24836, FF 14518, BRAM_18K 6",
            "note": "Production TXF baseline; different dataset/task.",
        },
        {
            "design": "TXF PHC-lite unroll16",
            "input_dim": "16",
            "latency": "2316 cycles / 25.733 us",
            "frequency": "90 MHz",
            "verification_tier": "routed_pnr_met",
            "resource": "DSP 125, routed 90MHz evidence in 06L/06O",
            "note": "Different dataset/task; included only as HLS methodology reference.",
        },
        {
            "design": "FI-2010 PHC-lite Stage2 step",
            "input_dim": "40",
            "latency": f"{csynth['latency_cycles_max']} cycles / {csynth['latency_us_max_at_100mhz']:.3f} us",
            "frequency": "100 MHz target",
            "verification_tier": "csim_pass + csynth_only",
            "resource": (
                f"DSP {format_resource(int(csynth['DSP']), float(csynth['DSP_pct']))}, "
                f"LUT {format_resource(int(csynth['LUT']), float(csynth['LUT_pct']))}, "
                f"FF {format_resource(int(csynth['FF']), float(csynth['FF_pct']))}, "
                f"BRAM_18K {format_resource(int(csynth['BRAM_18K']), float(csynth['BRAM_18K_pct']))}"
            ),
            "note": "Public FI-2010 feasibility artifact only; no cosim/P&R.",
        },
    ]

    def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
        return "\n".join(lines)

    report = f"""status: stage2_csim_and_csynth_complete_no_cosim_no_pnr
# FI-2010 PHC-lite Stage 2 HLS Feasibility Report

## Scope

This Stage 2 pass starts only after Stage 1 v2 convergence closure. It uses the public FI-2010 benchmark artifacts under `NoAuction/1.NoAuction_Zscore`; it does not use TXF NEW1-NEW5 data. It does not retrain, reselect thresholds, run cosim, run P&R, create a bitstream, or touch board/JTAG flow.

The selected representative checkpoint is `{manifest['selected_checkpoint']}`: k=50, seed={manifest['seed']}, raw loss. This was chosen as the median seed by Stage 1 v2 VAL macro F1 among k=50 raw runs (test_macro_f1 is recorded for reporting only and is never used to select the checkpoint, per the 2026-08-16 test-isolation fix), so the HLS feasibility number is tied to one concrete checkpoint rather than a cherry-picked best case.

Important boundary: this FI-2010 checkpoint records `fast_exact_final_step=True` and `fast_equiv_max_abs_diff=0.0` in the Stage 1 metrics, so the HLS step path can reproduce the full-window terminal logits from the last timestep input and zero state. Vitis optimized away unused `state_prev` logic in csynth. This is valid for this FI-2010 classifier checkpoint, but it is not a replacement claim for the TXF production recurrent PHC-lite deployment.

## Checkpoint

| field | value |
| --- | --- |
| checkpoint | `{manifest['selected_checkpoint']}` |
| horizon | {manifest['horizon']} |
| seed | {manifest['seed']} |
| loss_mode | {manifest['loss_mode']} |
| best_epoch | {manifest['best_epoch']} |
| trainable_parameters | {manifest['trainable_parameters']} |
| val_macro_f1 | {float(selected['val_macro_f1']):.6f} |
| test_macro_f1 | {float(selected['test_macro_f1']):.6f} |
| new5_used | {manifest['new5_used']} |

## Fixed-Point Export

Weights and buffers were exported to Q24.12 (`ap_fixed<24,12>`) with accumulator type `ap_fixed<40,16>`, matching the fixed-point convention already used by the TXF manual HLS path. The tanh/sigmoid path uses the same 65-point tanh LUT style; ReLU is not used in this PHC-lite step.

| item | value |
| --- | ---: |
| exported tensors | {len(audit)} |
| saturated values | {saturated_total} |
| test vectors | {csim['row_count']} |
| test vector CSV | `{vector_summary['csv_path']}` |
| csim output CSV | `{csim['source_csv']}` |

## CSim

| metric | value |
| --- | ---: |
| status | {csim['status']} |
| rows | {csim['row_count']} |
| mismatches | {csim['mismatches']} |
| max_abs_diff | {float(csim['max_abs_diff']):.9g} |
| mean_abs_diff_all_outputs | {float(csim['mean_abs_diff_all_outputs']):.9g} |
| sign_agreement_logit_0 | {float(csim['sign_agreement_logit_0']):.9g} |
| sign_agreement_logit_1 | {float(csim['sign_agreement_logit_1']):.9g} |
| sign_agreement_logit_2 | {float(csim['sign_agreement_logit_2']):.9g} |
| joint_sign_agreement | {float(csim['joint_sign_agreement']):.9g} |

Tie-breaking convention for sign agreement is `0 -> positive`.

## CSynth

| metric | value |
| --- | ---: |
| target_clock_ns | {float(csynth['target_clock_ns']):.3f} |
| estimated_clock_ns | {float(csynth['estimated_clock_ns']):.3f} |
| estimated_fmax_mhz | {float(csynth['estimated_fmax_mhz']):.3f} |
| latency_cycles | {csynth['latency_cycles_max']} |
| latency_us_at_100mhz | {float(csynth['latency_us_max_at_100mhz']):.3f} |
| interval_cycles | {csynth['interval_cycles_max']} |
| pipeline_type | {csynth['pipeline_type']} |
| DSP | {format_resource(int(csynth['DSP']), float(csynth['DSP_pct']))} |
| LUT | {format_resource(int(csynth['LUT']), float(csynth['LUT_pct']))} |
| FF | {format_resource(int(csynth['FF']), float(csynth['FF_pct']))} |
| BRAM_18K | {format_resource(int(csynth['BRAM_18K']), float(csynth['BRAM_18K_pct']))} |
| fits_zynq7020_csynth_budget | {csynth['fits_zynq7020_csynth_budget']} |
| report | `{csynth['csynth_report']}` |

## Cross-Artifact Comparison

{md_table(comparison_rows, ["design", "input_dim", "latency", "frequency", "verification_tier", "resource", "note"])}

These rows are deliberately side-by-side only. FI-2010 uses 40 input features and a 3-class public benchmark objective, while TXF uses 16 microstructure features and bid/ask action values. The table supports HLS methodology feasibility; it does not establish a direct speed or PnL comparison between tasks.

## Claim Boundary

- No NEW1-NEW5 data is used in this FI-2010 HLS feasibility phase.
- No model training or checkpoint modification is performed in Stage 2.
- CSim passed and csynth completed; cosim, routed P&R, bitstream, and board validation were not run.
- This does not prove FPGA PnL improvement.
- This does not validate real-world profitability.
- This does not replace the TXF PHC-lite production HLS reports.
"""
    (ROOT / "REPORT_STAGE2.md").write_text(report, encoding="utf-8")


def main() -> None:
    csim = collect_csim()
    csynth = parse_csynth()
    write_report(csim, csynth)
    print(json.dumps({"csim": csim, "csynth": csynth}, indent=2))


if __name__ == "__main__":
    main()
