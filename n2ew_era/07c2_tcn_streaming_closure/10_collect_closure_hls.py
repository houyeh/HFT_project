from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

VARIANTS = ("scratch", "distilled")
SETTINGS = (
    {"mode": "unroll_only", "suffix": "u2", "unroll": 2, "shared_history": False},
    {"mode": "unroll_only", "suffix": "u1", "unroll": 1, "shared_history": False},
    {"mode": "shared_history", "suffix": "shared_u1", "unroll": 1, "shared_history": True},
)

RESOURCE_BUDGET = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}

PHC_LITE_ROUTED_US = 83.16
TCN_MAC_ESTIMATE_US = 32.656
TCN_NAIVE_CSYNTH_US = 13703.85


def component_dir(variant: str, suffix: str) -> Path:
    return ROOT / f"hls_component_stage2_resource_{variant}_{suffix}"


def parse_csim_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "csim_summary": str(path),
        "csim_status": "missing",
    }
    if not path.exists():
        return out
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        out["csim_status"] = "empty"
        return out
    out["csim_status"] = "present"
    readout = next((row for row in rows if row.get("scope") == "readout"), None)
    if readout:
        for key in (
            "rows",
            "compared_values",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse",
            "sign_agreement_readout_0",
            "sign_agreement_readout_1",
            "sign_agreement_joint",
            "valid_alignment_errors",
        ):
            value = readout.get(key, "")
            if value == "":
                continue
            try:
                out[f"readout_{key}"] = float(value)
            except ValueError:
                out[f"readout_{key}"] = value
    block_rows = [row for row in rows if row.get("scope") == "block"]
    if block_rows:
        max_abs = max(float(row["max_abs_diff"]) for row in block_rows)
        total_values = sum(float(row["compared_values"]) for row in block_rows)
        sum_sq = sum((float(row["rmse"]) ** 2) * float(row["compared_values"]) for row in block_rows)
        out["block_max_abs_diff"] = max_abs
        out["block_rmse_pooled"] = math.sqrt(sum_sq / total_values) if total_values else 0.0
        out["block_stage_count"] = len(block_rows)
    if "readout_valid_alignment_errors" in out:
        valid_ok = float(out["readout_valid_alignment_errors"]) == 0.0
    else:
        valid_ok = False
    sign_ok = float(out.get("readout_sign_agreement_joint", 0.0)) == 1.0
    abs_ok = float(out.get("readout_max_abs_diff", 999.0)) <= 1.0 and float(out.get("block_max_abs_diff", 999.0)) <= 1.0
    out["csim_gate_pass"] = bool(valid_ok and sign_ok and abs_ok)
    return out


def parse_hlsrun_csim(component: Path) -> str:
    summary = component / "hls_component" / "hls_component.hlsrun_csim_summary"
    if not summary.exists():
        return "missing"
    text = summary.read_text(errors="ignore")
    if "CS_PASSED" in text:
        return "CS_PASSED"
    if "CS_FAILED" in text:
        return "CS_FAILED"
    return "present_unknown"


def find_top_report(component: Path) -> Path | None:
    report_dir = component / "hls_component" / "hls" / "syn" / "report"
    top = report_dir / "tcn_streaming_step_csynth.rpt"
    if top.exists():
        return top
    matches = sorted(report_dir.glob("*_csynth.rpt")) if report_dir.exists() else []
    return matches[0] if matches else None


def parse_csynth(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"csynth_status": "missing"}
    text = path.read_text(errors="ignore")
    out: dict[str, Any] = {"csynth_status": "present", "csynth_report": str(path)}
    latency_section_match = re.search(
        r"\+ Latency:\s*\* Summary:\s*(.*?)(?:\+ Detail:|== Utilization Estimates)",
        text,
        re.S,
    )
    if latency_section_match:
        for line in latency_section_match.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 7 and cells[0].isdigit() and cells[1].isdigit():
                out["latency_min_cycles"] = int(cells[0])
                out["latency_max_cycles"] = int(cells[1])
                out["latency_absolute_min_reported"] = cells[2]
                out["latency_absolute_max_reported"] = cells[3]
                out["ii_min_cycles"] = int(cells[4])
                out["ii_max_cycles"] = int(cells[5])
                out["pipeline_type"] = cells[6]
                out["latency_us_100mhz"] = float(out["latency_max_cycles"]) * 0.01
                out["interval_us_100mhz"] = float(out["ii_max_cycles"]) * 0.01
                break
    util_section = text.split("== Utilization Estimates", 1)
    if len(util_section) > 1:
        for line in util_section[1].splitlines():
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5 and cells[0] == "Total":
                names = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
                for name, value in zip(names, cells[1:6]):
                    try:
                        out[name] = int(value)
                    except ValueError:
                        pass
                break
    for key, budget in RESOURCE_BUDGET.items():
        if key in out:
            out[f"{key}_pct"] = float(out[key]) / budget * 100.0
    if all(key in out for key in RESOURCE_BUDGET):
        out["resource_fit_zynq7020"] = bool(all(float(out[key]) <= budget for key, budget in RESOURCE_BUDGET.items()))
    return out


def collect_errors_and_warnings(component: Path) -> dict[str, Any]:
    logs = [
        component / "hls_component" / "logs" / "hls_compile.log",
        component / "hls_component" / "logs" / "csim.log",
    ]
    errors: list[str] = []
    complete_unroll_warnings = 0
    for log_path in logs:
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="ignore").splitlines():
            if "converted into complete unroll" in line:
                complete_unroll_warnings += 1
            if "ERROR:" in line or "CRITICAL WARNING:" in line:
                errors.append(f"{log_path}: {line.strip()}")
    return {
        "complete_unroll_warning_count": complete_unroll_warnings,
        "error_excerpt": " | ".join(errors[-6:]),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for setting in SETTINGS:
            suffix = str(setting["suffix"])
            comp = component_dir(variant, suffix)
            row: dict[str, Any] = {
                "variant": variant,
                "mode": setting["mode"],
                "suffix": suffix,
                "unroll": int(setting["unroll"]),
                "shared_history_storage": bool(setting["shared_history"]),
                "component": str(comp),
                "hls_config": str(comp / "hls_config.cfg"),
            }
            row["hlsrun_csim_state"] = parse_hlsrun_csim(comp)
            row.update(parse_csim_summary(RESULTS / f"stage2_resource_{variant}_{suffix}_csim_summary.csv"))
            row.update(parse_csynth(find_top_report(comp)))
            if row.get("csynth_status") == "missing":
                partial_dir = comp / "hls_component" / "hls" / "syn" / "report"
                partial = sorted(partial_dir.glob("*_csynth.rpt")) if partial_dir.exists() else []
                row["csynth_partial_reports_count"] = len(partial)
            row.update(collect_errors_and_warnings(comp))
            rows.append(row)

    csv_path = RESULTS / "stage2_resource_sweep_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = RESULTS / "stage2_resource_sweep_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
