from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BUDGETS = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}


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


def summarize_output_csv(path: Path, prefix: str) -> dict[str, Any]:
    if not path.exists():
        return {
            f"{prefix}_status": "missing",
            f"{prefix}_total_rows": "",
            f"{prefix}_mismatches": "",
            f"{prefix}_max_abs_diff": "",
            f"{prefix}_mean_abs_diff": "",
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
        f"{prefix}_status": "PASS" if mismatches == 0 else "FAIL",
        f"{prefix}_total_rows": total,
        f"{prefix}_mismatches": mismatches,
        f"{prefix}_max_abs_diff": max_abs,
        f"{prefix}_mean_abs_diff": sum_abs / count_abs if count_abs else 0.0,
    }


def newest_report(name: str) -> Path | None:
    reports = list((ROOT / "hls_component").rglob(name))
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


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
            "fits_budget": "",
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


def parse_cosim_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"cosim_report_status": "missing", "cosim_report": ""}
    text = path.read_text(encoding="utf-8", errors="ignore")
    status = "present"
    if re.search(r"\bPASS\b", text, re.IGNORECASE):
        status = "PASS"
    elif re.search(r"\bFAIL\b", text, re.IGNORECASE):
        status = "FAIL"
    return {"cosim_report_status": status, "cosim_report": str(path)}


def collect() -> dict[str, Any]:
    row: dict[str, Any] = {"phase": ROOT.name}
    row.update(summarize_output_csv(RESULTS / "phc_step_csim_out.csv", "csim"))
    row.update(parse_csynth(newest_report("phc_step_csynth.rpt")))
    row.update(summarize_output_csv(RESULTS / "phc_step_cosim_out.csv", "cosim"))
    row.update(parse_cosim_report(newest_report("phc_step_cosim.rpt")))
    return row


def write_outputs(row: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "csim_status",
        "csim_total_rows",
        "csim_mismatches",
        "csim_max_abs_diff",
        "csim_mean_abs_diff",
        "csynth_status",
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
        "cosim_status",
        "cosim_total_rows",
        "cosim_mismatches",
        "cosim_max_abs_diff",
        "cosim_mean_abs_diff",
        "cosim_report_status",
        "csynth_report",
        "cosim_report",
    ]
    out_csv = RESULTS / "hls_result_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        f"# {ROOT.name} HLS Result Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for field in fields[1:]:
        lines.append(f"| {field} | {row.get(field, '')} |")
    (RESULTS / "hls_result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_csv)


def main() -> None:
    write_outputs(collect())


if __name__ == "__main__":
    main()
