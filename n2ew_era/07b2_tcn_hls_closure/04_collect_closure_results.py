from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

ZYNQ7020 = {"lut": 53200, "ff": 106400, "dsp": 220, "bram_18k": 280}
EXPECTED_ROWS = 1000
VARIANT = "distilled"
PHC_V8_US = 83.160
PHC_UNROLL16_90MHZ_US = 25.733


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def pct(value: float | int | None, denom: int) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value) / float(denom)


def sign_value(value: float) -> int:
    return -1 if value < 0 else 1


def summarize_output_csv(path: Path, expected_rows: int) -> dict[str, object]:
    out: dict[str, object] = {
        "path": rel(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "rows": 0,
        "mismatches_or_bad_rows": None,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "sign_agreement_readout_0": None,
        "sign_agreement_readout_1": None,
        "joint_sign_agreement": None,
        "complete_expected_rows": False,
        "status": "missing",
    }
    if not path.exists():
        return out

    total = 0
    bad = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_abs = 0
    agree0 = 0
    agree1 = 0
    agree_joint = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        abs_cols = [col for col in (reader.fieldnames or []) if col.startswith("abs_diff_readout_")]
        for row in reader:
            total += 1
            if str(row.get("pass", "")).strip() not in {"1", "1.0", "true", "True"}:
                bad += 1
            for col in abs_cols:
                value = safe_float(row.get(col))
                if value is not None:
                    max_abs = max(max_abs, abs(value))
                    sum_abs += abs(value)
                    n_abs += 1
            r0 = safe_float(row.get("ref_readout_0"))
            d0 = safe_float(row.get("dut_readout_0"))
            r1 = safe_float(row.get("ref_readout_1"))
            d1 = safe_float(row.get("dut_readout_1"))
            if None not in (r0, d0, r1, d1):
                a0 = sign_value(float(r0)) == sign_value(float(d0))
                a1 = sign_value(float(r1)) == sign_value(float(d1))
                agree0 += int(a0)
                agree1 += int(a1)
                agree_joint += int(a0 and a1)

    complete = total == expected_rows
    if complete and bad == 0:
        status = "pass"
    elif total > 0:
        status = "partial_or_failed"
    else:
        status = "empty"
    out.update(
        {
            "rows": total,
            "mismatches_or_bad_rows": bad,
            "max_abs_diff": max_abs if n_abs else None,
            "mean_abs_diff": (sum_abs / n_abs) if n_abs else None,
            "sign_agreement_readout_0": (agree0 / total) if total else None,
            "sign_agreement_readout_1": (agree1 / total) if total else None,
            "joint_sign_agreement": (agree_joint / total) if total else None,
            "complete_expected_rows": complete,
            "status": status,
        }
    )
    return out


def parse_csynth_xml(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "csynth_xml": rel(path),
        "csynth_xml_exists": path.exists(),
        "csynth_latency_cycles": None,
        "csynth_interval_cycles": None,
        "csynth_estimated_clock_ns": None,
        "csynth_bram_18k": None,
        "csynth_dsp": None,
        "csynth_ff": None,
        "csynth_lut": None,
    }
    if not path.exists():
        return out
    root = ET.parse(path).getroot()

    def first_text(names: list[str]) -> str | None:
        for name in names:
            elem = root.find(f".//{name}")
            if elem is not None and elem.text is not None:
                return elem.text.strip()
        return None

    out["csynth_latency_cycles"] = safe_int(first_text(["Worst-caseLatency", "Average-caseLatency", "Best-caseLatency"]))
    out["csynth_interval_cycles"] = safe_int(first_text(["Interval-max", "Interval-min"]))
    out["csynth_estimated_clock_ns"] = safe_float(first_text(["EstimatedClockPeriod"]))
    out["csynth_bram_18k"] = safe_int(first_text(["BRAM_18K", "BRAM"]))
    out["csynth_dsp"] = safe_int(first_text(["DSP48E", "DSP"]))
    out["csynth_ff"] = safe_int(first_text(["FF"]))
    out["csynth_lut"] = safe_int(first_text(["LUT"]))
    if out["csynth_latency_cycles"] is not None:
        out["csynth_latency_us_at_100mhz_estimate_only"] = float(out["csynth_latency_cycles"]) * 10.0 / 1000.0
        out["csynth_latency_us_at_90mhz_estimate_only"] = float(out["csynth_latency_cycles"]) * 11.111 / 1000.0
    if out["csynth_interval_cycles"] is not None:
        out["csynth_interval_us_at_100mhz_estimate_only"] = float(out["csynth_interval_cycles"]) * 10.0 / 1000.0
        out["csynth_interval_us_at_90mhz_estimate_only"] = float(out["csynth_interval_cycles"]) * 11.111 / 1000.0
    for key, denom in [("bram_18k", 280), ("dsp", 220), ("ff", 106400), ("lut", 53200)]:
        out[f"csynth_{key}_pct"] = pct(out.get(f"csynth_{key}"), denom)
    return out


def parse_timing_rpt(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "timing_rpt": rel(path),
        "timing_rpt_exists": path.exists(),
        "wns_ns": None,
        "tns_ns": None,
        "setup_failing_endpoints": None,
        "timing_met": None,
    }
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line or "TNS Failing Endpoints" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 8]:
            parts = candidate.split()
            if len(parts) < 3:
                continue
            wns = safe_float(parts[0])
            tns = safe_float(parts[1])
            failing = safe_int(parts[2])
            if wns is not None and tns is not None and failing is not None:
                out["wns_ns"] = wns
                out["tns_ns"] = tns
                out["setup_failing_endpoints"] = failing
                out["timing_met"] = bool(wns >= 0.0 and failing == 0)
                return out
    return out


def parse_util_rpt(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "util_rpt": rel(path),
        "util_rpt_exists": path.exists(),
        "routed_lut": None,
        "routed_ff": None,
        "routed_dsp": None,
        "routed_bram_tiles": None,
        "routed_bram_18k_equiv": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "routed_lut": r"\|\s*Slice LUTs\s*\|\s*([\d,]+)\s*\|",
        "routed_ff": r"\|\s*Slice Registers\s*\|\s*([\d,]+)\s*\|",
        "routed_dsp": r"\|\s*DSPs\s*\|\s*([\d,]+)\s*\|",
        "routed_bram_tiles": r"\|\s*Block RAM Tile\s*\|\s*([\d,]+(?:\.\d+)?)\s*\|",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = safe_float(match.group(1)) if key == "routed_bram_tiles" else safe_int(match.group(1))
    if out["routed_bram_tiles"] is not None:
        out["routed_bram_18k_equiv"] = float(out["routed_bram_tiles"]) * 2.0
    out["routed_lut_pct"] = pct(out["routed_lut"], ZYNQ7020["lut"])
    out["routed_ff_pct"] = pct(out["routed_ff"], ZYNQ7020["ff"])
    out["routed_dsp_pct"] = pct(out["routed_dsp"], ZYNQ7020["dsp"])
    out["routed_bram_18k_pct"] = pct(out["routed_bram_18k_equiv"], ZYNQ7020["bram_18k"])
    return out


def parse_cosim_rpt(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "cosim_rpt": rel(path),
        "cosim_rpt_exists": path.exists(),
        "cosim_status": "missing",
        "cosim_latency_cycles": None,
        "cosim_interval_cycles": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "Verilog" in line and "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 8:
                out["cosim_status"] = cells[1]
                out["cosim_latency_cycles"] = safe_int(cells[4])
                out["cosim_interval_cycles"] = safe_int(cells[7])
                break
    return out


def md_value(value: object, digits: int | None = None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return "missing"
        if digits is not None:
            return f"{value:.{digits}f}"
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    component = ROOT / f"hls_component_{VARIANT}"
    generated = component / "hls_component"
    csynth_xml = generated / "hls" / "syn" / "report" / "tcn_prepad_infer_csynth.xml"
    cosim_rpt = generated / "reports" / "hls_cosim.rpt"
    pnr_base = generated / "hls" / "syn" / "verilog"

    csim = summarize_output_csv(RESULTS / f"tcn_prepad_{VARIANT}_csim_out.csv", EXPECTED_ROWS)
    cosim_csv = summarize_output_csv(RESULTS / f"tcn_prepad_{VARIANT}_cosim_out.csv", EXPECTED_ROWS)
    cosim = parse_cosim_rpt(cosim_rpt)
    csynth = parse_csynth_xml(csynth_xml)

    pnr_rows: list[dict[str, object]] = []
    for label, period in [("100mhz", 10.000), ("90mhz", 11.111)]:
        report_dir = pnr_base / f"direct_tcn_prepad_reports_{VARIANT}_{label}"
        timing = parse_timing_rpt(report_dir / "tcn_prepad_infer_timing_summary_routed.rpt")
        util = parse_util_rpt(report_dir / "tcn_prepad_infer_utilization_routed.rpt")
        pnr_rows.append(
            {
                "variant": VARIANT,
                "clock_label": label,
                "clock_period_ns": period,
                **timing,
                **util,
            }
        )

    summary_row: dict[str, object] = {
        "variant": VARIANT,
        "scope": "07b2_distilled_naive_full_window_recompute",
        "expected_rows": EXPECTED_ROWS,
        "csim_status": csim["status"],
        "csim_rows": csim["rows"],
        "csim_mismatches_or_bad_rows": csim["mismatches_or_bad_rows"],
        "csim_max_abs_diff": csim["max_abs_diff"],
        "csim_mean_abs_diff": csim["mean_abs_diff"],
        "csim_joint_sign_agreement": csim["joint_sign_agreement"],
        "cosim_csv_status": cosim_csv["status"],
        "cosim_csv_rows": cosim_csv["rows"],
        "cosim_csv_bad_rows": cosim_csv["mismatches_or_bad_rows"],
        "cosim_rpt_status": cosim["cosim_status"],
        "cosim_rpt_exists": cosim["cosim_rpt_exists"],
        "cosim_latency_cycles": cosim["cosim_latency_cycles"],
        "cosim_interval_cycles": cosim["cosim_interval_cycles"],
        **csynth,
    }
    write_csv([summary_row], RESULTS / "tcn_hls_closure_summary.csv")
    write_csv(pnr_rows, RESULTS / "tcn_hls_closure_pnr_summary.csv")
    (RESULTS / "tcn_hls_closure_summary.json").write_text(
        json.dumps({"summary": summary_row, "pnr": pnr_rows, "csim": csim, "cosim_csv": cosim_csv}, indent=2),
        encoding="utf-8",
    )

    pnr100 = next(row for row in pnr_rows if row["clock_label"] == "100mhz")
    pnr90 = next(row for row in pnr_rows if row["clock_label"] == "90mhz")
    lines = [
        "status: closure_blocked_by_cosim_timeout_and_pnr_timing_failure",
        "finding: 07b2_distilled_naive_full_window_tcn_csim_passes_1000_rows_but_cosim_did_not_complete_and_pnr_fails_at_100mhz_and_90mhz",
        "# 07B2 TCN HLS Closure Audit",
        "",
        "## Scope Boundary",
        "",
        "- Work was confined to `n2ew_era/07b2_tcn_hls_closure/`; the original 07B directory was read/copied but not modified.",
        "- No model retraining, checkpoint change, threshold change, or NEW5 access was performed.",
        "- The selected closure target is the 07A/07B `distilled` TCN PrePad model because it is the stronger 07A policy row; scratch artifacts are copied into 07B2 but not accepted as closure evidence in this pass.",
        "- Latency definition remains 07B naive full-window recomputation: one new event supplies a 128x16 window, the HLS top prepends 124 zeros, recomputes all five valid-conv TCN blocks, then emits bid/ask readout.",
        "",
        "## Verification Evidence",
        "",
        "| item | result | rows | max abs diff | mean abs diff | joint sign agreement | evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        f"| CSim | {md_value(csim['status'])} | {md_value(csim['rows'])}/{EXPECTED_ROWS} | {md_value(csim['max_abs_diff'], 6)} | {md_value(csim['mean_abs_diff'], 6)} | {md_value(csim['joint_sign_agreement'], 6)} | `{md_value(csim['path'])}` |",
        f"| RTL cosim | timeout/incomplete, no `hls_cosim.rpt` | {md_value(cosim_csv['rows'])}/{EXPECTED_ROWS} partial CSV rows | {md_value(cosim_csv['max_abs_diff'], 6)} | {md_value(cosim_csv['mean_abs_diff'], 6)} | {md_value(cosim_csv['joint_sign_agreement'], 6)} | `{md_value(cosim_csv['path'])}` plus missing `{md_value(cosim['cosim_rpt'])}` |",
        "",
        "The 1000-row cosim command was attempted and stopped after a 30-minute timeout. It reached C TB testing and left a partial CSV, but no HLS cosim report was produced. Therefore no RTL transaction latency/II is accepted from cosim.",
        "",
        "## HLS Csynth Estimate",
        "",
        "| latency cycles | interval cycles | estimated clock ns | latency @100MHz estimate | interval @100MHz estimate | BRAM_18K | DSP | FF | LUT |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {md_value(csynth['csynth_latency_cycles'])} | {md_value(csynth['csynth_interval_cycles'])} | {md_value(csynth['csynth_estimated_clock_ns'], 3)} | {md_value(csynth.get('csynth_latency_us_at_100mhz_estimate_only'), 3)} us | {md_value(csynth.get('csynth_interval_us_at_100mhz_estimate_only'), 3)} us | {md_value(csynth['csynth_bram_18k'])} | {md_value(csynth['csynth_dsp'])} | {md_value(csynth['csynth_ff'])} | {md_value(csynth['csynth_lut'])} |",
        "",
        "These latency/II numbers are csynth estimates only. They are kept for diagnosis but are not used as measured closure latency.",
        "",
        "## Routed P&R",
        "",
        "| target | WNS ns | TNS ns | setup failing endpoints | timing met | routed DSP | routed LUT | routed FF | routed BRAM_18K equiv | evidence |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        f"| 100MHz | {md_value(pnr100['wns_ns'], 3)} | {md_value(pnr100['tns_ns'], 3)} | {md_value(pnr100['setup_failing_endpoints'])} | {md_value(pnr100['timing_met'])} | {md_value(pnr100['routed_dsp'])} ({md_value(pnr100['routed_dsp_pct'], 2)}%) | {md_value(pnr100['routed_lut'])} ({md_value(pnr100['routed_lut_pct'], 2)}%) | {md_value(pnr100['routed_ff'])} ({md_value(pnr100['routed_ff_pct'], 2)}%) | {md_value(pnr100['routed_bram_18k_equiv'], 1)} ({md_value(pnr100['routed_bram_18k_pct'], 2)}%) | `{md_value(pnr100['timing_rpt'])}` |",
        f"| 90MHz | {md_value(pnr90['wns_ns'], 3)} | {md_value(pnr90['tns_ns'], 3)} | {md_value(pnr90['setup_failing_endpoints'])} | {md_value(pnr90['timing_met'])} | {md_value(pnr90['routed_dsp'])} ({md_value(pnr90['routed_dsp_pct'], 2)}%) | {md_value(pnr90['routed_lut'])} ({md_value(pnr90['routed_lut_pct'], 2)}%) | {md_value(pnr90['routed_ff'])} ({md_value(pnr90['routed_ff_pct'], 2)}%) | {md_value(pnr90['routed_bram_18k_equiv'], 1)} ({md_value(pnr90['routed_bram_18k_pct'], 2)}%) | `{md_value(pnr90['timing_rpt'])}` |",
        "",
        "Resource fit is not the blocker: DSP/LUT/FF/BRAM all fit comfortably after routing. The closure blocker is timing: 100MHz fails with negative WNS, and the required 90MHz fallback also fails.",
        "",
        "## PHC-Style Latency Comparison",
        "",
        "| implementation | latency definition | verification level | single-decision latency | steady-state interval / throughput | routed timing | note |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        f"| PHC-lite v8 baseline | single recurrent step | routed P&R + cosim | {PHC_V8_US:.3f} us | 83.170 us interval | met @100MHz | Source: 06F/06S carried-forward PHC baseline. |",
        f"| PHC-lite unroll16 seed2 | single recurrent step | routed P&R + cosim1000/full evidence mix | {PHC_UNROLL16_90MHZ_US:.3f} us | 25.744 us interval | met @90MHz | Source: `n2ew_era/06s_unroll16_verification_closure/AUDIT_unroll16_verification_closure.md`. |",
        f"| TCN PrePad distilled 07B2 | naive full-window recomputation, receptive field 125 | CSim1000 pass; cosim timeout; P&R timing fail | not accepted | not accepted | fail @100MHz and @90MHz | Csynth estimate is {md_value(csynth.get('csynth_latency_us_at_100mhz_estimate_only'), 3)} us @100MHz / {md_value(csynth.get('csynth_latency_us_at_90mhz_estimate_only'), 3)} us @90MHz, but this is not measured closure latency. |",
        "",
        "Important: TCN and PHC rows are not the same latency object. TCN here recomputes a whole window per decision; PHC processes one recurrent step. The table is a verification/closure comparison, not a direct architectural speed ranking.",
        "",
        "## Streaming II=1 Follow-up Boundary",
        "",
        "A separate 07C streaming effort already exists and reaches a resource-fitting csynth-only point (`shared_u1`, 16510 cycles), but it has no cosim or routed timing closure. Because the 07B2 baseline failed both the 1000-row cosim gate and 90MHz P&R gate, this pass did not spend additional time creating a new II=1 streaming variant inside 07B2. Any future streaming result must be reported separately as single-decision latency vs steady-state throughput.",
        "",
        "## Honest Conclusion",
        "",
        "The 07B2 closure task is diagnosed but not closed. The distilled naive full-window TCN C model passes 1000 validation rows, so the copied fixed-point datapath is numerically usable at CSim level. However, 1000-row RTL cosim did not complete and produced no `hls_cosim.rpt`, and routed P&R fails timing at both 100MHz and the requested 90MHz fallback. Therefore the existing 07B TCN latency cannot be promoted to a PHC-style routed/cosim-verified headline. The next engineering target should be the streaming/cached-activation TCN path, not further citation of the naive full-window recompute path.",
        "",
    ]
    (ROOT / "AUDIT_tcn_hls_closure.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {rel(RESULTS / 'tcn_hls_closure_summary.csv')}")
    print(f"wrote {rel(RESULTS / 'tcn_hls_closure_pnr_summary.csv')}")
    print(f"wrote {rel(ROOT / 'AUDIT_tcn_hls_closure.md')}")


if __name__ == "__main__":
    main()
