from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06l_phc_lite_pipeline_budget_sweep"
VARIANTS = ROOT / "variants"
RESULTS = ROOT / "results"
FACTORS = [1, 2, 4, 8, 16]

BUDGETS = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}

V7_ENDPOINT = {
    "label": "v7_auto_pipeline_over_budget",
    "latency_cycles": 604,
    "latency_us": 6.04,
    "II": 605,
    "DSP": 268,
    "DSP_pct": 121.0,
    "LUT": 41433,
    "LUT_pct": 77.0,
    "FF": 55719,
    "FF_pct": 52.0,
    "BRAM_18K": 0,
    "BRAM_pct": 0.0,
    "fits_budget": False,
}

V8_ENDPOINT = {
    "label": "v8_pipeline_off_production_baseline",
    "latency_cycles": 8316,
    "latency_us": 83.16,
    "II": 8317,
    "DSP": 46,
    "DSP_pct": 20.9,
    "LUT": 24836,
    "LUT_pct": 46.7,
    "FF": 14518,
    "FF_pct": 13.6,
    "BRAM_18K": 6,
    "BRAM_pct": 2.1,
    "fits_budget": True,
}


def summarize_csim(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "csim_status": "missing",
            "csim_total_rows": "",
            "csim_mismatches": "",
            "csim_max_abs_diff": "",
            "csim_mean_abs_diff": "",
        }
    total = 0
    mismatches = 0
    max_abs = 0.0
    sum_abs = 0.0
    count_abs = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header: {path}")
        abs_cols = [c for c in reader.fieldnames if c.startswith("abs_diff_")]
        for row in reader:
            total += 1
            if int(float(row.get("pass", "1"))) == 0:
                mismatches += 1
            max_abs = max(max_abs, float(row.get("max_abs_diff", "0") or 0))
            for col in abs_cols:
                sum_abs += float(row.get(col, "0") or 0)
                count_abs += 1
    return {
        "csim_status": "PASS" if mismatches == 0 else "FAIL",
        "csim_total_rows": total,
        "csim_mismatches": mismatches,
        "csim_max_abs_diff": max_abs,
        "csim_mean_abs_diff": sum_abs / count_abs if count_abs else 0.0,
    }


def find_csynth_report(factor: int) -> Path | None:
    variant = VARIANTS / f"factor_{factor}"
    reports = list(variant.rglob("phc_step_csynth.rpt"))
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def parse_int_cell(value: str) -> int | None:
    cleaned = value.replace(",", "").replace("-", "").strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_float_cell(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("-", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_csynth(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "csynth_status": "missing",
            "DSP": "",
            "DSP_pct": "",
            "LUT": "",
            "LUT_pct": "",
            "FF": "",
            "FF_pct": "",
            "BRAM_18K": "",
            "BRAM_pct": "",
            "latency_cycles": "",
            "latency_us": "",
            "II": "",
            "fits_budget": False,
            "csynth_report": "",
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    latency_cycles = None
    latency_us = None
    ii = None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 9:
            nums = [parse_int_cell(c) for c in cells]
            if nums[1] is not None and nums[2] is not None and "us" in line and "Latency" not in line:
                # Latency table row: | min | max | min_abs | max_abs | min_ii | max_ii | type |
                latency_cycles = nums[1]
                ii = nums[5]
                m = re.search(r"\|\s*[\d,]+\s*\|\s*[\d,]+\s*\|\s*([\d.]+)\s*us\|", line)
                if m:
                    latency_us = float(m.group(1))
                break

    total = available = util = None
    in_utilization_summary = False
    for line in text.splitlines():
        if line.startswith("== Utilization Estimates"):
            in_utilization_summary = True
            continue
        if not in_utilization_summary:
            continue
        stripped = line.strip()
        if stripped.startswith("|Total") and total is None:
            total = [parse_int_cell(c) for c in line.split("|")]
        elif stripped.startswith("|Available") and available is None:
            available = [parse_int_cell(c) for c in line.split("|")]
        elif stripped.startswith("|Utilization (%)") and util is None:
            util = [parse_float_cell(c) for c in line.split("|")]
        if total is not None and available is not None and util is not None:
            break
    if total is None:
        raise RuntimeError(f"Could not parse Total utilization row from {path}")
    # Split cells include leading/trailing empty items. Resource table order:
    # | Name | BRAM_18K | DSP | FF | LUT | URAM |
    bram = total[2]
    dsp = total[3]
    ff = total[4]
    lut = total[5]
    bram_pct = util[2] if util else None
    dsp_pct = util[3] if util else None
    ff_pct = util[4] if util else None
    lut_pct = util[5] if util else None
    fits = (
        dsp is not None
        and lut is not None
        and ff is not None
        and bram is not None
        and dsp < BUDGETS["DSP"]
        and lut < BUDGETS["LUT"]
        and ff < BUDGETS["FF"]
        and bram < BUDGETS["BRAM_18K"]
    )
    return {
        "csynth_status": "present",
        "DSP": dsp,
        "DSP_pct": dsp_pct,
        "LUT": lut,
        "LUT_pct": lut_pct,
        "FF": ff,
        "FF_pct": ff_pct,
        "BRAM_18K": bram,
        "BRAM_pct": bram_pct,
        "latency_cycles": latency_cycles,
        "latency_us": latency_us,
        "II": ii,
        "fits_budget": fits,
        "csynth_report": str(path),
    }


def row_for_factor(factor: int) -> dict[str, Any]:
    result_dir = RESULTS / f"factor_{factor}"
    row: dict[str, Any] = {"unroll_factor": factor}
    row.update(summarize_csim(result_dir / "phc_step_csim_out.csv"))
    row.update(parse_csynth(find_csynth_report(factor)))
    cosim = summarize_csim(result_dir / "phc_step_cosim_out.csv")
    row["cosim_status"] = cosim["csim_status"]
    row["cosim_mismatches"] = cosim["csim_mismatches"]
    row["cosim_max_abs_diff"] = cosim["csim_max_abs_diff"]
    row["cosim_mean_abs_diff"] = cosim["csim_mean_abs_diff"]
    if row.get("latency_cycles") not in ("", None):
        row["speedup_vs_v8_latency"] = V8_ENDPOINT["latency_cycles"] / float(row["latency_cycles"])
    else:
        row["speedup_vs_v8_latency"] = ""
    return row


def write_csv(rows: list[dict[str, Any]]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "pipeline_budget_sweep_summary.csv"
    fields = [
        "unroll_factor",
        "csim_status",
        "csim_total_rows",
        "csim_mismatches",
        "csim_max_abs_diff",
        "csim_mean_abs_diff",
        "DSP",
        "DSP_pct",
        "LUT",
        "LUT_pct",
        "FF",
        "FF_pct",
        "BRAM_18K",
        "BRAM_pct",
        "latency_cycles",
        "latency_us",
        "II",
        "fits_budget",
        "speedup_vs_v8_latency",
        "cosim_status",
        "cosim_mismatches",
        "cosim_max_abs_diff",
        "cosim_mean_abs_diff",
        "csynth_status",
        "csynth_report",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return out


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(rows: list[dict[str, Any]]) -> Path:
    eligible = [
        row
        for row in rows
        if row.get("fits_budget") is True
        and row.get("latency_cycles") not in ("", None)
        and row.get("csim_status") == "PASS"
    ]
    best = min(eligible, key=lambda r: int(r["latency_cycles"])) if eligible else None
    lines = [
        "# PHC-lite Pipeline Budget Sweep Report",
        "",
        "This report is generated from user-run Vitis HLS artifacts in this independent 06L phase. It does not modify the frozen 06F production baseline.",
        "",
        "## Endpoint References",
        "",
        "| endpoint | latency_cycles | latency_us | II | DSP | DSP_pct | LUT | LUT_pct | FF | FF_pct | BRAM_18K | BRAM_pct | fits_budget |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        f"| {V7_ENDPOINT['label']} | {V7_ENDPOINT['latency_cycles']} | {V7_ENDPOINT['latency_us']} | {V7_ENDPOINT['II']} | {V7_ENDPOINT['DSP']} | {V7_ENDPOINT['DSP_pct']} | {V7_ENDPOINT['LUT']} | {V7_ENDPOINT['LUT_pct']} | {V7_ENDPOINT['FF']} | {V7_ENDPOINT['FF_pct']} | {V7_ENDPOINT['BRAM_18K']} | {V7_ENDPOINT['BRAM_pct']} | {V7_ENDPOINT['fits_budget']} |",
        f"| {V8_ENDPOINT['label']} | {V8_ENDPOINT['latency_cycles']} | {V8_ENDPOINT['latency_us']} | {V8_ENDPOINT['II']} | {V8_ENDPOINT['DSP']} | {V8_ENDPOINT['DSP_pct']} | {V8_ENDPOINT['LUT']} | {V8_ENDPOINT['LUT_pct']} | {V8_ENDPOINT['FF']} | {V8_ENDPOINT['FF_pct']} | {V8_ENDPOINT['BRAM_18K']} | {V8_ENDPOINT['BRAM_pct']} | {V8_ENDPOINT['fits_budget']} |",
        "",
        "## Sweep Table",
        "",
        "| factor | csim | max_abs_diff | DSP | LUT | FF | BRAM | latency_cycles | latency_us | II | fits_budget | speedup_vs_v8 | cosim |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['unroll_factor']} | {fmt(row.get('csim_status'))} | {fmt(row.get('csim_max_abs_diff'))} | "
            f"{fmt(row.get('DSP'))} | {fmt(row.get('LUT'))} | {fmt(row.get('FF'))} | {fmt(row.get('BRAM_18K'))} | "
            f"{fmt(row.get('latency_cycles'))} | {fmt(row.get('latency_us'))} | {fmt(row.get('II'))} | "
            f"{fmt(row.get('fits_budget'))} | {fmt(row.get('speedup_vs_v8_latency'))} | {fmt(row.get('cosim_status'))} |"
        )
    lines.extend(["", "## Boundary Decision", ""])
    if best:
        lines.append(
            f"Best budget-fitting factor so far: `{best['unroll_factor']}` with latency `{best['latency_cycles']}` cycles "
            f"({best['latency_us']} us), speedup `{float(best['speedup_vs_v8_latency']):.3f}x` versus v8."
        )
    else:
        lines.append(
            "No factor currently has complete artifacts proving simultaneous DSP/LUT/FF/BRAM budget fit with passing C-sim. "
            "This may mean the sweep has not been run yet, or no middle point fits all budgets."
        )
    lines.extend(
        [
            "",
            "Cosim is only expected for factors that fit all four resource budgets. A missing cosim entry is not a failure by itself unless the factor was budget-fitting and intended for RTL confirmation.",
            "",
            "## Claim Boundary",
            "",
            "- This is a synthesis-level resource/latency sweep only.",
            "- Do not claim speedup until a factor has passing csim and all four resource budgets fit.",
            "- Do not modify 06F based on this table without a separate freeze decision.",
            "- This does not involve retraining, NEW5, P&R, exchange replay, or PnL claims.",
        ]
    )
    out = ROOT / "PHC_PIPELINE_BUDGET_SWEEP_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    rows = [row_for_factor(factor) for factor in FACTORS]
    csv_path = write_csv(rows)
    report_path = write_report(rows)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
