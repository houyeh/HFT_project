from __future__ import annotations

import csv
from datetime import datetime
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIANTS = ["scratch", "distilled"]
V8_PHC_LATENCY_US = 83.16
MAC_ESTIMATE_US = 32.656

RESOURCE_BUDGET = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}
EXPECTED_CSIM_ROWS = 200


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
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


def find_first(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.rglob(pattern))
    return matches[0] if matches else None


def parse_csim(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"csim_status": "missing"}
    df = pd.read_csv(path)
    total = int(len(df))
    failed = int((df["pass"] == 0).sum()) if "pass" in df.columns else -1
    readout_cols = [c for c in df.columns if c.startswith("abs_diff_readout_")]
    if readout_cols:
        diff_values = df[readout_cols].to_numpy(dtype=float)
        max_abs = float(diff_values.max())
        mean_abs = float(diff_values.mean())
        rmse = float((diff_values ** 2).mean() ** 0.5)
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
        rmse = float("nan")
    sign_rows = []
    for idx in [0, 1]:
        ref = df[f"ref_readout_{idx}"].to_numpy(dtype=float)
        dut = df[f"dut_readout_{idx}"].to_numpy(dtype=float)
        agree = ((ref >= 0) == (dut >= 0)).mean()
        sign_rows.append(float(agree))
    joint = float((((df["ref_readout_0"] >= 0) == (df["dut_readout_0"] >= 0)) & ((df["ref_readout_1"] >= 0) == (df["dut_readout_1"] >= 0))).mean())
    complete = total == EXPECTED_CSIM_ROWS
    if failed == 0 and complete:
        status = "pass"
    elif failed == 0:
        status = "partial_pass"
    else:
        status = "fail"
    return {
        "csim_status": status,
        "csim_output_csv": str(path),
        "csim_output_mtime_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "csim_rows": total,
        "csim_expected_rows": EXPECTED_CSIM_ROWS,
        "csim_complete_expected_rows": complete,
        "csim_failed_rows": failed,
        "csim_max_abs_diff_readout": max_abs,
        "csim_mean_abs_diff_readout": mean_abs,
        "csim_rmse_readout": rmse,
        "sign_agreement_readout_0": sign_rows[0],
        "sign_agreement_readout_1": sign_rows[1],
        "sign_agreement_joint": joint,
    }


def parse_csynth(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"csynth_status": "missing"}
    text = path.read_text(errors="ignore")
    out: dict[str, object] = {
        "csynth_status": "present",
        "csynth_report": str(path),
        "csynth_report_mtime_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }
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
    out["fits_zynq7020_budget"] = all(float(out.get(f"{key}_pct", 1e9)) < 100.0 for key in RESOURCE_BUDGET)
    if "latency_us_100mhz" in out:
        out["speedup_vs_phc_83p16us"] = V8_PHC_LATENCY_US / float(out["latency_us_100mhz"])
        out["delta_vs_mac_estimate_us"] = float(out["latency_us_100mhz"]) - MAC_ESTIMATE_US
    return out


def parse_cosim(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"cosim_status": "missing"}
    text = path.read_text(errors="ignore")
    status = "pass" if re.search(r"\bPass\b", text, re.I) else "present"
    return {"cosim_status": status, "cosim_report": str(path)}


def main() -> None:
    rows = []
    for variant in VARIANTS:
        component = ROOT / f"hls_component_{variant}"
        csim_csv = RESULTS / f"tcn_prepad_{variant}_csim_out.csv"
        csynth = find_first(component, "tcn_prepad_infer_csynth.rpt")
        cosim = find_first(component, "tcn_prepad_infer_cosim.rpt")
        row = {
            "variant": variant,
            "mac_estimate_latency_us": MAC_ESTIMATE_US,
            "phc_pnr_latency_us": V8_PHC_LATENCY_US,
            "verification_level_mac_estimate": "pre_synthesis_macs_only",
            "verification_level_tcn_hls": "csim_and_csynth_reports_parsed_no_cosim_or_pnr",
            "verification_level_phc": "P&R_routed_timing_verified",
        }
        row.update(parse_csim(csim_csv))
        row.update(parse_csynth(csynth))
        row.update(parse_cosim(cosim))
        rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "tcn_prepad_manual_hls_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# TCN PrePad Manual HLS Summary",
        "",
        "This collector reads existing C-sim/C-synthesis/Cosim outputs. It does not run Vitis.",
        "",
        md_table(pd.DataFrame(rows)) if rows else "No rows.",
    ]
    (ROOT / "TCN_PREPAD_MANUAL_HLS_SUMMARY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
