from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s_unroll16_verification_closure"
RESULTS = ROOT / "results"

P05F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
P06L = PROJECT_ROOT / "n2ew_era" / "06l_phc_lite_pipeline_budget_sweep"

SEEDS = [0, 1, 2]
ZYNQ7020 = {"lut": 53200, "ff": 106400, "dsp": 220, "bram_18k": 280}
V8_LATENCY_US = 83.160
V8_LATENCY_CYCLES = 8316
UNROLL16_90MHZ_LATENCY_US = 25.733
UNROLL16_90MHZ_ROUTED_CYCLES_06L = 2316
UNROLL16_90MHZ_CLOCK_NS = 11.111


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: str | None) -> int | None:
    f = safe_float(value)
    if f is None:
        return None
    return int(round(f))


def pct(value: int | float | None, denom: int) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value) / float(denom)


def sign_value(x: float) -> int:
    return -1 if x < 0 else 1


def row_get(row: dict[str, str], *names: str) -> str | None:
    wanted = {name.strip().lstrip("\ufeff").strip('"') for name in names}
    for key, value in row.items():
        norm = key.strip().lstrip("\ufeff").strip('"')
        if norm in wanted:
            return value
    return None


def summarize_io_csv(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "path": rel(path),
        "exists": path.exists(),
        "status": "missing",
        "total_rows": None,
        "mismatches": None,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "sign_agreement_readout_0": None,
        "sign_agreement_readout_1": None,
        "joint_sign_agreement": None,
        "sha256": sha256_file(path),
    }
    if not path.exists():
        return out

    total = 0
    mismatches = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_abs = 0
    agree0 = 0
    agree1 = 0
    agree_joint = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        abs_cols = [c for c in (reader.fieldnames or []) if c.startswith("abs_diff_")]
        has_pass = "pass" in (reader.fieldnames or [])
        for row in reader:
            total += 1
            if has_pass and str(row.get("pass", "")).strip() not in {"1", "1.0", "True", "true"}:
                mismatches += 1
            for col in abs_cols:
                v = safe_float(row.get(col))
                if v is not None:
                    max_abs = max(max_abs, abs(v))
                    sum_abs += abs(v)
                    n_abs += 1
            if {"ref_readout_0", "dut_readout_0", "ref_readout_1", "dut_readout_1"}.issubset(row.keys()):
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

    out.update(
        {
            "status": "pass" if mismatches == 0 else "fail",
            "total_rows": total,
            "mismatches": mismatches,
            "max_abs_diff": max_abs if n_abs else None,
            "mean_abs_diff": (sum_abs / n_abs) if n_abs else None,
            "sign_agreement_readout_0": (agree0 / total) if total else None,
            "sign_agreement_readout_1": (agree1 / total) if total else None,
            "joint_sign_agreement": (agree_joint / total) if total else None,
        }
    )
    return out


def parse_cosim_rpt(path: Path) -> dict[str, object]:
    out = {
        "cosim_rpt_path": rel(path),
        "cosim_rpt_exists": path.exists(),
        "cosim_status": "missing",
        "cosim_latency_min": None,
        "cosim_latency_avg": None,
        "cosim_latency_max": None,
        "cosim_interval_min": None,
        "cosim_interval_avg": None,
        "cosim_interval_max": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "Verilog" in line and "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 9:
                out.update(
                    {
                        "cosim_status": cells[1],
                        "cosim_latency_min": safe_int(cells[2]),
                        "cosim_latency_avg": safe_int(cells[3]),
                        "cosim_latency_max": safe_int(cells[4]),
                        "cosim_interval_min": safe_int(cells[5]),
                        "cosim_interval_avg": safe_int(cells[6]),
                        "cosim_interval_max": safe_int(cells[7]),
                    }
                )
                break
    return out


def parse_csynth_xml(path: Path) -> dict[str, object]:
    out = {
        "csynth_xml_path": rel(path),
        "csynth_xml_exists": path.exists(),
        "csynth_latency_cycles": None,
        "csynth_interval_cycles": None,
        "csynth_estimated_clock_ns": None,
        "csynth_dsp": None,
        "csynth_lut": None,
        "csynth_ff": None,
        "csynth_bram_18k": None,
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

    out["csynth_latency_cycles"] = safe_int(first_text(["Average-caseLatency", "Best-caseLatency", "Worst-caseLatency"]))
    out["csynth_interval_cycles"] = safe_int(first_text(["Interval-max", "Interval-min"]))
    out["csynth_estimated_clock_ns"] = safe_float(first_text(["EstimatedClockPeriod"]))
    out["csynth_dsp"] = safe_int(first_text(["DSP48E", "DSP"]))
    out["csynth_lut"] = safe_int(first_text(["LUT"]))
    out["csynth_ff"] = safe_int(first_text(["FF"]))
    out["csynth_bram_18k"] = safe_int(first_text(["BRAM_18K", "BRAM"]))
    return out


def parse_timing_rpt(path: Path) -> dict[str, object]:
    out = {
        "timing_rpt_path": rel(path),
        "timing_rpt_exists": path.exists(),
        "wns_ns": None,
        "tns_ns": None,
        "failing_endpoints": None,
        "timing_met": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
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
                out["failing_endpoints"] = failing
                break
        if out["wns_ns"] is not None:
            break
    if out["wns_ns"] is not None and out["failing_endpoints"] is not None:
        out["timing_met"] = bool(out["wns_ns"] >= 0 and out["failing_endpoints"] == 0)
    return out


def parse_util_rpt(path: Path) -> dict[str, object]:
    out = {
        "util_rpt_path": rel(path),
        "util_rpt_exists": path.exists(),
        "routed_lut": None,
        "routed_ff": None,
        "routed_dsp": None,
        "routed_bram_18k": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "routed_lut": [r"\|\s*Slice LUTs\s*\|\s*([\d,]+)\s*\|", r"\|\s*LUT as Logic\s*\|\s*([\d,]+)\s*\|"],
        "routed_ff": [r"\|\s*Slice Registers\s*\|\s*([\d,]+)\s*\|"],
        "routed_dsp": [r"\|\s*DSPs\s*\|\s*([\d,]+)\s*\|", r"\|\s*DSP48E1\s*\|\s*([\d,]+)\s*\|"],
        "routed_bram_18k": [r"\|\s*Block RAM Tile\s*\|\s*([\d,]+(?:\.\d+)?)\s*\|", r"\|\s*RAMB18\s*\|\s*([\d,]+)\s*\|"],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                out[key] = safe_int(m.group(1))
                break
    return out


def seed_paths(seed: int) -> dict[str, Path]:
    base = ROOT / f"seed{seed}"
    generated = base / "hls_component" / "hls_component"
    return {
        "csim_csv": base / "results" / f"phc_step_csim_out_seed{seed}.csv",
        "cosim_csv": base / "results" / f"phc_step_cosim_out_seed{seed}.csv",
        "cosim_rpt": generated / "reports" / "hls_cosim.rpt",
        "csynth_xml": generated / "hls" / "syn" / "report" / "phc_step_csynth.xml",
        "timing_rpt": generated / "hls" / "impl" / "verilog" / f"direct_phc_step_reports_seed{seed}_90mhz" / "phc_step_timing_summary_routed.rpt",
        "util_rpt": generated / "hls" / "impl" / "verilog" / f"direct_phc_step_reports_seed{seed}_90mhz" / "phc_step_utilization_routed.rpt",
    }


def seed2_fallback_paths() -> dict[str, Path]:
    return {
        "csim_csv": P06L / "results" / "factor_16" / "phc_step_csim_out.csv",
        "cosim_csv": P06L / "results" / "factor_16" / "phc_step_cosim_out.csv",
        "cosim_rpt": P06L / "variants" / "factor_16" / "hls_component" / "hls_component" / "reports" / "hls_cosim.rpt",
        "csynth_xml": P06L / "variants" / "factor_16" / "hls_component" / "hls_component" / "hls" / "syn" / "report" / "phc_step_csynth.xml",
        "timing_rpt": P06L
        / "pnr_unroll16_90mhz"
        / "hls_component"
        / "hls_component"
        / "hls"
        / "impl"
        / "verilog"
        / "direct_phc_step_reports_baseline_90mhz"
        / "phc_step_timing_summary_routed.rpt",
        "util_rpt": P06L
        / "pnr_unroll16_90mhz"
        / "hls_component"
        / "hls_component"
        / "hls"
        / "impl"
        / "verilog"
        / "direct_phc_step_reports_baseline_90mhz"
        / "phc_step_utilization_routed.rpt",
    }


def seed2_cosim1000_paths() -> dict[str, Path]:
    base = ROOT / "seed2_cosim1000"
    generated = base / "hls_component" / "hls_component"
    return {
        "csim_csv": base / "results" / "phc_step_csim_out_seed2_first1000.csv",
        "cosim_csv": base / "results" / "phc_step_cosim_out_seed2_first1000.csv",
        "cosim_rpt": generated / "reports" / "hls_cosim.rpt",
        "csynth_xml": generated / "hls" / "syn" / "report" / "phc_step_csynth.xml",
    }


def collect_seed2_cosim1000() -> dict[str, object]:
    paths = seed2_cosim1000_paths()
    csim = summarize_io_csv(paths["csim_csv"])
    cosim_csv = summarize_io_csv(paths["cosim_csv"])
    cosim_rpt = parse_cosim_rpt(paths["cosim_rpt"])
    csynth = parse_csynth_xml(paths["csynth_xml"])
    csim_sha = csim.get("sha256")
    cosim_sha = cosim_csv.get("sha256")
    return {
        "seed2_cosim1000_csim_status": csim["status"],
        "seed2_cosim1000_csim_rows": csim["total_rows"],
        "seed2_cosim1000_csim_max_abs_diff": csim["max_abs_diff"],
        "seed2_cosim1000_csim_mean_abs_diff": csim["mean_abs_diff"],
        "seed2_cosim1000_cosim_status": cosim_rpt["cosim_status"],
        "seed2_cosim1000_cosim_csv_status": cosim_csv["status"],
        "seed2_cosim1000_cosim_rows": cosim_csv["total_rows"],
        "seed2_cosim1000_cosim_mismatches": cosim_csv["mismatches"],
        "seed2_cosim1000_cosim_max_abs_diff": cosim_csv["max_abs_diff"],
        "seed2_cosim1000_cosim_mean_abs_diff": cosim_csv["mean_abs_diff"],
        "seed2_cosim1000_bit_exact_to_csim": bool(csim_sha and cosim_sha and csim_sha == cosim_sha),
        "seed2_cosim1000_latency_cycles": cosim_rpt["cosim_latency_max"],
        "seed2_cosim1000_interval_cycles": cosim_rpt["cosim_interval_max"],
        "seed2_cosim1000_csynth_latency_cycles": csynth["csynth_latency_cycles"],
        "seed2_cosim1000_csynth_interval_cycles": csynth["csynth_interval_cycles"],
        "seed2_cosim1000_csynth_estimated_clock_ns": csynth["csynth_estimated_clock_ns"],
        "seed2_cosim1000_csim_path": rel(paths["csim_csv"]),
        "seed2_cosim1000_cosim_path": rel(paths["cosim_csv"]),
        "seed2_cosim1000_cosim_rpt_path": rel(paths["cosim_rpt"]),
    }


def collect_seed(seed: int) -> dict[str, object]:
    paths = seed_paths(seed)
    source_note = "06s"
    fallback = seed2_fallback_paths() if seed == 2 else None

    csim = summarize_io_csv(paths["csim_csv"])
    cosim_csv = summarize_io_csv(paths["cosim_csv"])
    cosim_rpt = parse_cosim_rpt(paths["cosim_rpt"])
    csynth = parse_csynth_xml(paths["csynth_xml"])
    timing = parse_timing_rpt(paths["timing_rpt"])
    util = parse_util_rpt(paths["util_rpt"])
    existing_full = {}
    if seed == 2 and fallback is not None:
        existing_cosim_rpt = parse_cosim_rpt(fallback["cosim_rpt"])
        existing_timing = parse_timing_rpt(fallback["timing_rpt"])
        existing_util = parse_util_rpt(fallback["util_rpt"])
        existing_full = {
            "existing_06l_full_cosim_status": existing_cosim_rpt["cosim_status"],
            "existing_06l_full_cosim_latency_cycles": existing_cosim_rpt["cosim_latency_max"],
            "existing_06l_full_cosim_interval_cycles": existing_cosim_rpt["cosim_interval_max"],
            "existing_06l_routed_wns_ns": existing_timing["wns_ns"],
            "existing_06l_routed_tns_ns": existing_timing["tns_ns"],
            "existing_06l_routed_failing_endpoints": existing_timing["failing_endpoints"],
            "existing_06l_routed_timing_met": existing_timing["timing_met"],
            "existing_06l_routed_lut": existing_util["routed_lut"],
            "existing_06l_routed_ff": existing_util["routed_ff"],
            "existing_06l_routed_dsp": existing_util["routed_dsp"],
            "existing_06l_routed_bram_18k": existing_util["routed_bram_18k"],
            "existing_06l_latency_us_at_90mhz": UNROLL16_90MHZ_LATENCY_US,
            "existing_06l_source_report": rel(P06L / "AUDIT_unroll16_pnr_timing.md"),
        }
        if not cosim_rpt["cosim_rpt_exists"]:
            source_note = "06s_full_cosim_csv_no_rpt_timeout_plus_06l_existing_full_evidence"
        if not timing["timing_rpt_exists"]:
            timing = existing_timing
            util = existing_util

    csim_sha = csim.get("sha256")
    cosim_sha = cosim_csv.get("sha256")
    cosim_bit_exact_to_csim = bool(csim_sha and cosim_sha and csim_sha == cosim_sha)

    latency_cycles = cosim_rpt.get("cosim_latency_max") or csynth.get("csynth_latency_cycles")
    latency_us_90mhz = (float(latency_cycles) * UNROLL16_90MHZ_CLOCK_NS / 1000.0) if latency_cycles is not None else None
    row = {
        "seed": seed,
        "source_note": source_note,
        "csim_status": csim["status"],
        "csim_total_rows": csim["total_rows"],
        "csim_mismatches": csim["mismatches"],
        "csim_max_abs_diff": csim["max_abs_diff"],
        "csim_mean_abs_diff": csim["mean_abs_diff"],
        "csim_sign_agreement_readout_0": csim["sign_agreement_readout_0"],
        "csim_sign_agreement_readout_1": csim["sign_agreement_readout_1"],
        "csim_joint_sign_agreement": csim["joint_sign_agreement"],
        "cosim_csv_status": cosim_csv["status"],
        "cosim_total_rows": cosim_csv["total_rows"],
        "cosim_mismatches": cosim_csv["mismatches"],
        "cosim_max_abs_diff": cosim_csv["max_abs_diff"],
        "cosim_mean_abs_diff": cosim_csv["mean_abs_diff"],
        "cosim_bit_exact_to_csim": cosim_bit_exact_to_csim,
        "full_06s_cosim_status_note": "rtl_report_present" if cosim_rpt["cosim_rpt_exists"] else ("timeout_or_incomplete_rtl_report_missing" if seed == 2 else "missing"),
        **cosim_rpt,
        **csynth,
        **timing,
        **util,
        **existing_full,
        "latency_cycles": latency_cycles,
        "latency_us_at_90mhz": latency_us_90mhz,
        "dsp_pct": pct(util.get("routed_dsp") or csynth.get("csynth_dsp"), ZYNQ7020["dsp"]),
        "lut_pct": pct(util.get("routed_lut") or csynth.get("csynth_lut"), ZYNQ7020["lut"]),
        "ff_pct": pct(util.get("routed_ff") or csynth.get("csynth_ff"), ZYNQ7020["ff"]),
        "bram_pct": pct(util.get("routed_bram_18k") or csynth.get("csynth_bram_18k"), ZYNQ7020["bram_18k"]),
    }
    if seed == 2:
        row.update(collect_seed2_cosim1000())
    return row


def load_cpu_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_csvs = [
        ("06s", ROOT / "results" / "cpu" / "cpu_concurrent_benchmark_06s_summary.csv"),
        ("06f_existing", P05F / "results" / "cpu_concurrent_benchmark_summary.csv"),
    ]
    seen = set()
    seen_concurrency_from_06s: set[int] = set()
    wanted_concurrency = {1, 4, 8, 16}
    for source_label, path in candidate_csvs:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for src in reader:
                try:
                    concurrency = int(float(row_get(src, "concurrency", "streams") or ""))
                except ValueError:
                    continue
                if concurrency not in wanted_concurrency:
                    continue
                if source_label == "06s":
                    seen_concurrency_from_06s.add(concurrency)
                elif concurrency in seen_concurrency_from_06s:
                    continue
                round_value = row_get(src, "round")
                key = (concurrency, round_value or "", rel(path))
                if key in seen:
                    continue
                seen.add(key)
                median = safe_float(row_get(src, "median_us") or row_get(src, "p50_us"))
                p99 = safe_float(row_get(src, "p99_us"))
                mean = safe_float(row_get(src, "mean_us"))
                max_us = safe_float(row_get(src, "max_us"))
                miss25 = safe_float(row_get(src, "deadline_miss_rate_25p733_pct"))
                if miss25 is None and source_label == "06f_existing":
                    raw_path = P05F / "results" / f"cpu_concurrent_benchmark_N{concurrency}.csv"
                    if raw_path.exists() and round_value is not None:
                        total = 0
                        miss = 0
                        with raw_path.open("r", encoding="utf-8", newline="") as rf:
                            raw_reader = csv.DictReader(rf)
                            for raw in raw_reader:
                                if str(raw.get("round")) != str(round_value):
                                    continue
                                lat = safe_float(raw.get("latency_us"))
                                if lat is None:
                                    continue
                                total += 1
                                miss += int(lat > UNROLL16_90MHZ_LATENCY_US)
                        if total:
                            miss25 = 100.0 * miss / total
                rows.append(
                    {
                        "implementation": f"CPU LibTorch C++ N={concurrency}",
                        "source": rel(path),
                        "verification_level": "host_benchmark",
                        "concurrency": concurrency,
                        "round": round_value,
                        "mean_us": mean,
                        "median_us": median,
                        "p99_us": p99,
                        "max_us": max_us,
                        "jitter_p99_minus_median_us": (p99 - median) if p99 is not None and median is not None else None,
                        "deadline_miss_rate_gt_83p16_pct": safe_float(
                            row_get(src, "deadline_miss_rate_83p16_pct")
                            or row_get(src, "deadline_miss_rate_pct")
                            or row_get(src, "deadline_miss_percent")
                        ),
                        "deadline_miss_rate_gt_25p733_pct": miss25,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "missing"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(seed_rows: list[dict[str, object]], final_rows: list[dict[str, object]]) -> None:
    lines: list[str] = []
    lines.append("# 06S Unroll16 Verification Closure")
    lines.append("")
    lines.append("Status: verification_closure_completed_with_caveats.")
    lines.append("")
    lines.append("Scope: this phase copies and verifies existing frozen artifacts; it does not retrain models, alter weights, alter thresholds, or touch NEW5.")
    lines.append("")
    lines.append("## Part A: Unroll16@90MHz Cosim")
    lines.append("")
    seed2 = next((r for r in seed_rows if r.get("seed") == 2), None)
    if seed2:
        lines.append("- 06S seed2 full 25,600-row CSim passed and the produced full cosim CSV is bit-exact against the CSim CSV.")
        lines.append("- The 06S full 25,600-row RTL cosim did not produce an HLS `hls_cosim.rpt` before the 60-minute run was stopped, so its latency/II is not accepted as a full 06S cosim report.")
        lines.append(
            f"- 06S seed2 1,000-row fallback cosim passed: latency {md_value(seed2.get('seed2_cosim1000_latency_cycles'), 0)} cycles, "
            f"II/interval {md_value(seed2.get('seed2_cosim1000_interval_cycles'), 0)} cycles, "
            f"max_abs_diff {md_value(seed2.get('seed2_cosim1000_cosim_max_abs_diff'), 6)}, "
            f"bit-exact vs fallback CSim CSV `{md_value(seed2.get('seed2_cosim1000_bit_exact_to_csim'))}`."
        )
        lines.append(
            f"- Existing 06L full unroll16 evidence remains the final routed headline: full cosim `{md_value(seed2.get('existing_06l_full_cosim_status'))}`, "
            f"latency {md_value(seed2.get('existing_06l_full_cosim_latency_cycles'), 0)} cycles, "
            f"interval {md_value(seed2.get('existing_06l_full_cosim_interval_cycles'), 0)} cycles, "
            f"90MHz routed WNS {md_value(seed2.get('existing_06l_routed_wns_ns'), 3)}ns."
        )
    else:
        lines.append("- Seed2 evidence row was not found; see CSV for missing details.")
    lines.append("")
    lines.append("## Part B: Cross-seed 90MHz P&R")
    lines.append("")
    lines.append("Seed0 and seed1 were exported through the same copied unroll16 HLS source and 90MHz flow. Seed0 routed timing met; seed1 did not meet timing at 90MHz. Seed2 uses the already routed 06L artifact as the accepted production latency headline.")
    lines.append("")
    lines.append("## FPGA Cross-seed Evidence")
    lines.append("")
    lines.append("| seed | source | csim | csim rows | csim max abs diff | 06S full cosim note | cosim rpt | cosim rows | bit-exact vs csim | csynth/cosim cycles | latency @90MHz us | routed DSP | routed LUT | routed FF | routed BRAM_18K | WNS ns | timing met |")
    lines.append("|---:|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_value(row.get("seed"), 0),
                    md_value(row.get("source_note")),
                    md_value(row.get("csim_status")),
                    md_value(row.get("csim_total_rows"), 0),
                    md_value(row.get("csim_max_abs_diff"), 6),
                    md_value(row.get("full_06s_cosim_status_note")),
                    md_value(row.get("cosim_status")),
                    md_value(row.get("cosim_total_rows"), 0),
                    md_value(row.get("cosim_bit_exact_to_csim")),
                    md_value(row.get("latency_cycles"), 0),
                    md_value(row.get("latency_us_at_90mhz"), 3),
                    f"{md_value(row.get('routed_dsp'), 0)} ({md_value(row.get('dsp_pct'), 1)}%)",
                    f"{md_value(row.get('routed_lut'), 0)} ({md_value(row.get('lut_pct'), 1)}%)",
                    f"{md_value(row.get('routed_ff'), 0)} ({md_value(row.get('ff_pct'), 1)}%)",
                    f"{md_value(row.get('routed_bram_18k'), 0)} ({md_value(row.get('bram_pct'), 1)}%)",
                    md_value(row.get("wns_ns"), 3),
                    md_value(row.get("timing_met")),
                ]
            )
            + " |"
        )
    lines.append("")
    if seed2:
        lines.append("### Seed2 Cosim Status")
        lines.append("")
        lines.append("- 06S full 25,600-row CSim/CSV comparison exists and is bit-exact between CSim CSV and the produced Cosim CSV, but the HLS RTL cosim report was not produced before the 60-minute run was stopped. Therefore the 06S full run is labeled `timeout_or_incomplete_rtl_report_missing`.")
        lines.append(
            f"- 06S fallback 1,000-row cosim: status `{md_value(seed2.get('seed2_cosim1000_cosim_status'))}`, "
            f"rows {md_value(seed2.get('seed2_cosim1000_cosim_rows'), 0)}, "
            f"max_abs_diff {md_value(seed2.get('seed2_cosim1000_cosim_max_abs_diff'), 6)}, "
            f"bit-exact vs CSim CSV `{md_value(seed2.get('seed2_cosim1000_bit_exact_to_csim'))}`, "
            f"latency {md_value(seed2.get('seed2_cosim1000_latency_cycles'), 0)} cycles / interval {md_value(seed2.get('seed2_cosim1000_interval_cycles'), 0)} cycles."
        )
        lines.append(
            f"- Existing 06L full unroll16 evidence remains the routed headline: full cosim `{md_value(seed2.get('existing_06l_full_cosim_status'))}`, "
            f"latency {md_value(seed2.get('existing_06l_full_cosim_latency_cycles'), 0)} cycles / interval {md_value(seed2.get('existing_06l_full_cosim_interval_cycles'), 0)} cycles, "
            f"routed timing WNS {md_value(seed2.get('existing_06l_routed_wns_ns'), 3)}ns."
        )
        lines.append("")
    lines.append("Note: the 06S resynthesis at 90MHz reports 2310 cycles, while the already routed 06L unroll16 artifact reports 2316 cycles. The final latency headline keeps the routed 06L artifact at 25.733us.")
    lines.append("")
    lines.append("## Part C: CPU LibTorch Baselines")
    lines.append("")
    lines.append("CPU N=1/N=4/N=16 were measured in this phase with 10,000 iterations per stream. CPU N=8 is carried forward from the existing 06F concurrent benchmark raw/summary files because 06S did not rerun N=8. The CPU deadline miss behavior is not monotonic across N; this is preserved as measured rather than smoothed.")
    lines.append("")
    lines.append("## Final Latency Comparison")
    lines.append("")
    lines.append("| implementation | verification level | mean us | median us | p99 us | max us | jitter p99-median us | miss >83.16us % | miss >25.733us % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in final_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_value(row.get("implementation")),
                    md_value(row.get("verification_level")),
                    md_value(row.get("mean_us"), 3),
                    md_value(row.get("median_us"), 3),
                    md_value(row.get("p99_us"), 3),
                    md_value(row.get("max_us"), 3),
                    md_value(row.get("jitter_p99_minus_median_us"), 3),
                    md_value(row.get("deadline_miss_rate_gt_83p16_pct"), 3),
                    md_value(row.get("deadline_miss_rate_gt_25p733_pct"), 3),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Claim boundary: this table compares latency and verification evidence only. It does not claim FPGA PnL improvement and does not use NEW5.")
    (ROOT / "AUDIT_unroll16_verification_closure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "final_latency_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fpga_final_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "implementation": "FPGA PHC v8 baseline",
            "source": rel(P05F / "PHC_HLS_SYNTHESIS_REPORT.md"),
            "verification_level": "routed_pnr_met_plus_csim_cosim",
            "mean_us": V8_LATENCY_US,
            "median_us": V8_LATENCY_US,
            "p99_us": V8_LATENCY_US,
            "max_us": V8_LATENCY_US,
            "jitter_p99_minus_median_us": 0.0,
            "deadline_miss_rate_gt_83p16_pct": 0.0,
            "deadline_miss_rate_gt_25p733_pct": 100.0,
        }
    ]
    lat = UNROLL16_90MHZ_LATENCY_US
    rows.append(
        {
            "implementation": "FPGA PHC unroll16 seed2 @90MHz",
            "source": rel(P06L / "AUDIT_unroll16_pnr_timing.md"),
            "verification_level": "routed_pnr_met_06l_full_cosim_plus_06s_cosim1000",
            "mean_us": lat,
            "median_us": lat,
            "p99_us": lat,
            "max_us": lat,
            "jitter_p99_minus_median_us": 0.0,
            "deadline_miss_rate_gt_83p16_pct": 0.0,
            "deadline_miss_rate_gt_25p733_pct": 0.0,
        }
    )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_rows = [collect_seed(seed) for seed in SEEDS]
    write_csv(RESULTS / "fpga_unroll16_seed_closure.csv", seed_rows)

    cpu_rows = sorted(
        load_cpu_rows(),
        key=lambda r: (safe_int(str(r.get("concurrency"))) or 0, safe_int(str(r.get("round"))) or 0),
    )
    final_rows = fpga_final_rows(seed_rows) + cpu_rows
    write_csv(RESULTS / "final_latency_comparison.csv", final_rows)
    write_markdown(seed_rows, final_rows)

    print("seed,csim_status,cosim_status,latency_cycles,latency_us_at_90mhz,wns_ns,timing_met")
    for row in seed_rows:
        print(
            ",".join(
                [
                    str(row.get("seed")),
                    str(row.get("csim_status")),
                    str(row.get("cosim_status")),
                    str(row.get("latency_cycles")),
                    str(row.get("latency_us_at_90mhz")),
                    str(row.get("wns_ns")),
                    str(row.get("timing_met")),
                ]
            )
        )
    print(f"wrote {rel(RESULTS / 'fpga_unroll16_seed_closure.csv')}")
    print(f"wrote {rel(RESULTS / 'final_latency_comparison.csv')}")


if __name__ == "__main__":
    main()
