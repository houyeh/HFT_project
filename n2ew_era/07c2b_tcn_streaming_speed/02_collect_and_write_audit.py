from __future__ import annotations

import csv
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
TMP_ROOT = Path(r"C:\tmp\07c2b_tcn_streaming_speed_shortpath")

ZYNQ = {
    "BRAM_18K": 280,
    "DSP": 220,
    "FF": 106400,
    "LUT": 53200,
}

UNROLL_VARIANTS = [
    {"name": "scratch_shared_u2", "variant": "scratch", "unroll": 2},
    {"name": "scratch_shared_u4", "variant": "scratch", "unroll": 4},
    {"name": "distilled_shared_u2", "variant": "distilled", "unroll": 2},
    {"name": "distilled_shared_u4", "variant": "distilled", "unroll": 4},
]

PIPELINE_VARIANTS = [
    {"name": "scratch_shared_u4", "variant": "scratch", "unroll": 4, "top_pipeline": False},
    {"name": "scratch_shared_u4_pipe", "variant": "scratch", "unroll": 4, "top_pipeline": True},
    {"name": "distilled_shared_u4", "variant": "distilled", "unroll": 4, "top_pipeline": False},
    {"name": "distilled_shared_u4_pipe", "variant": "distilled", "unroll": 4, "top_pipeline": True},
]

LOCALTAP_VARIANTS = [
    {
        "name": "scratch_shared_u4_localtap",
        "variant": "scratch",
        "unroll": 4,
        "status": "completed_diagnostic",
        "note": "local fully-partitioned tap buffer; active only in this wrapper",
    },
    {
        "name": "distilled_shared_u4_localtap",
        "variant": "distilled",
        "unroll": 4,
        "status": "skipped_after_scratch_regression",
        "note": "staged but not run because scratch localtap worsened latency and LUT/FF",
    },
]

PNR_VARIANTS = [
    {"name": "scratch_shared_u4", "variant": "scratch", "clock_mhz": 100.0, "clock_period_ns": 10.0},
    {"name": "distilled_shared_u4", "variant": "distilled", "clock_mhz": 100.0, "clock_period_ns": 10.0},
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def variant_root(name: str) -> Path:
    return TMP_ROOT / name


def component_root(name: str) -> Path:
    return variant_root(name) / f"hls_component_stage2_resource_{name}"


def hls_root(name: str) -> Path:
    return component_root(name) / "hls_component" / "hls"


def syn_report_dir(name: str) -> Path:
    return hls_root(name) / "syn" / "report"


def csim_summary_path(name: str) -> Path:
    return variant_root(name) / "results" / f"stage2_speed_{name}_event1008_tb_summary.csv"


def pnr_report_dir(name: str) -> Path:
    return hls_root(name) / "syn" / "verilog" / f"direct_tcn_streaming_reports_{name}_100mhz"


def xsim_log_path(name: str) -> Path:
    return hls_root(name) / "sim" / "verilog" / "xsim.log"


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel(dst)


def copy_variant_artifacts(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    artifact = ARTIFACTS / name
    report_dst = artifact / "syn_report"
    if syn_report_dir(name).exists():
        report_dst.mkdir(parents=True, exist_ok=True)
        for src in syn_report_dir(name).glob("*csynth.*"):
            copy_if_exists(src, report_dst / src.name)
        out["csynth_xml"] = rel(report_dst / "tcn_streaming_step_csynth.xml")
        out["csynth_rpt"] = rel(report_dst / "tcn_streaming_step_csynth.rpt")
    hls_log = hls_root(name) / "hls.log"
    out["hls_log"] = copy_if_exists(hls_log, artifact / "hls.log")
    csim_log = hls_root(name) / "csim" / "report" / "tcn_streaming_step_csim.log"
    out["csim_log"] = copy_if_exists(csim_log, artifact / "tcn_streaming_step_csim.log")
    csim_csv = csim_summary_path(name)
    out["csim_summary_csv"] = copy_if_exists(csim_csv, RESULTS / csim_csv.name)
    return out


def copy_pnr_artifacts(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    src_dir = pnr_report_dir(name)
    dst_dir = ARTIFACTS / name / "pnr_reports_100mhz"
    if src_dir.exists():
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    out["pnr_timing_report"] = rel(dst_dir / "tcn_streaming_step_timing_summary_routed.rpt")
    out["pnr_util_report"] = rel(dst_dir / "tcn_streaming_step_utilization_routed.rpt")
    return out


def copy_cosim_artifacts(name: str) -> dict[str, str]:
    return {
        "cosim_timeout_log": copy_if_exists(
            xsim_log_path(name),
            ARTIFACTS / name / "cosim_1008_timeout_xsim.log",
        )
    }


def parse_csynth_xml(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()

    def first(tag: str, default: str = "") -> str:
        for elem in root.iter():
            if elem.tag.split("}")[-1] == tag:
                return (elem.text or "").strip()
        return default

    resources: dict[str, int] = {}
    available: dict[str, int] = {}
    in_resources = False
    in_available = False
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Resources":
            in_resources = True
            in_available = False
            continue
        if tag == "AvailableResources":
            in_resources = False
            in_available = True
            continue
        if tag in ZYNQ:
            try:
                value = int((elem.text or "0").strip())
            except ValueError:
                value = 0
            if in_resources:
                resources[tag] = value
            elif in_available:
                available[tag] = value

    latency = int(first("Worst-caseLatency", "0"))
    interval = int(first("Interval-max", "0"))
    row: dict[str, object] = {
        "target_clock_ns": float(first("TargetClockPeriod", "0")),
        "estimated_clock_ns": float(first("EstimatedClockPeriod", "0")),
        "pipeline_type": first("PipelineType", ""),
        "latency_min_cycles": int(first("Best-caseLatency", "0")),
        "latency_max_cycles": latency,
        "latency_max_us_at_100mhz": latency * 0.01,
        "interval_min_cycles": int(first("Interval-min", "0")),
        "interval_max_cycles": interval,
        "interval_max_us_at_100mhz": interval * 0.01,
    }
    for key in ZYNQ:
        row[key] = resources.get(key, 0)
        denom = available.get(key) or ZYNQ[key]
        row[f"{key}_pct"] = 100.0 * resources.get(key, 0) / denom
    row["resource_fit_csynth"] = all(float(row[f"{key}_pct"]) < 100.0 for key in ZYNQ)
    return row


def parse_csynth_instance_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    in_instance = False
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if line.startswith("* Instance:"):
            in_instance = True
            continue
        if in_instance and line.startswith("* DSP:"):
            break
        if not in_instance or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] in {"Instance", ""} or set(cells[0]) == {"-"}:
            continue
        try:
            rows.append(
                {
                    "instance": cells[0],
                    "module": cells[1],
                    "BRAM_18K": int(cells[2]),
                    "DSP": int(cells[3]),
                    "FF": int(cells[4]),
                    "LUT": int(cells[5]),
                    "URAM": int(cells[6]),
                }
            )
        except ValueError:
            continue
    return rows


def parse_csim_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "csim_status": "missing",
            "csim_readout_rows": "",
            "csim_event_calls": "",
            "csim_readout_max_abs_diff": "",
            "csim_readout_rmse": "",
            "csim_sign_agreement_joint": "",
            "csim_valid_alignment_errors": "",
        }
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    readout = next(row for row in rows if row["scope"] == "readout")
    valid_errors = int(float(readout["valid_alignment_errors"]))
    joint = float(readout["sign_agreement_joint"])
    return {
        "csim_status": "pass" if valid_errors == 0 else "fail",
        "csim_readout_rows": int(float(readout["rows"])),
        "csim_event_calls": 1008,
        "csim_readout_max_abs_diff": float(readout["max_abs_diff"]),
        "csim_readout_rmse": float(readout["rmse"]),
        "csim_sign_agreement_joint": joint,
        "csim_valid_alignment_errors": valid_errors,
    }


def parse_timing(path: Path) -> dict[str, object]:
    content = read_text(path)
    match = re.search(
        r"\n\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+)\s+(\d+)\s+"
        r"([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+)\s+(\d+)\s+",
        content,
    )
    if not match:
        raise RuntimeError(f"timing summary row not found: {path}")
    return {
        "wns_ns": float(match.group(1)),
        "tns_ns": float(match.group(2)),
        "setup_failing_endpoints": int(match.group(3)),
        "setup_total_endpoints": int(match.group(4)),
        "whs_ns": float(match.group(5)),
        "ths_ns": float(match.group(6)),
        "hold_failing_endpoints": int(match.group(7)),
        "hold_total_endpoints": int(match.group(8)),
        "timing_met": "All user specified timing constraints are met." in content,
    }


def parse_utilization(path: Path) -> dict[str, object]:
    content = read_text(path)
    patterns = {
        "routed_lut": r"\| Slice LUTs\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_ff": r"\| Slice Registers\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_bram_tiles": r"\| Block RAM Tile\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_dsp": r"\| DSPs\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
    }
    row: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if not match:
            raise RuntimeError(f"utilization {key} not found: {path}")
        row[key] = int(match.group(1))
        row[f"{key}_pct"] = float(match.group(2))
    row["routed_bram_18k_equiv"] = float(row["routed_bram_tiles"]) * 2.0
    row["routed_bram_18k_pct"] = float(row["routed_bram_tiles_pct"])
    row["resource_fit_routed"] = (
        float(row["routed_lut_pct"]) < 100.0
        and float(row["routed_ff_pct"]) < 100.0
        and float(row["routed_dsp_pct"]) < 100.0
        and float(row["routed_bram_18k_pct"]) < 100.0
    )
    return row


def parse_xsim_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "attempted_event_calls": 1008,
            "total_rtl_transactions": "",
            "completed_transactions_at_timeout": 0,
            "final_compare_report": "missing",
            "accepted_as_cosim_pass": False,
        }
    content = read_text(path)
    progress = re.findall(r"RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)", content)
    if not progress:
        return {
            "attempted_event_calls": 1008,
            "total_rtl_transactions": "",
            "completed_transactions_at_timeout": 0,
            "final_compare_report": "missing_timeout",
            "accepted_as_cosim_pass": False,
        }
    done, total = progress[-1]
    rpt = path.parents[1] / "report" / "tcn_streaming_step_cosim.rpt"
    return {
        "attempted_event_calls": 1008,
        "total_rtl_transactions": int(total),
        "completed_transactions_at_timeout": int(done),
        "timeout_budget_seconds": 7200,
        "final_compare_report": "present" if rpt.exists() else "missing_timeout",
        "accepted_as_cosim_pass": False,
    }


def parse_pipeline_conflict(name: str) -> str:
    log = hls_root(name) / "hls.log"
    if not log.exists():
        return ""
    for line in read_text(log).splitlines():
        if "pragma conflict between PIPELINE" in line or "Ignoring PIPELINE" in line:
            return line.strip()
    return ""


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    keys: list[str] = []
    for row in materialized:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(materialized)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def collect_hls_variant(meta: dict[str, object]) -> dict[str, object]:
    name = str(meta["name"])
    artifacts = copy_variant_artifacts(name)
    row: dict[str, object] = dict(meta)
    xml = syn_report_dir(name) / "tcn_streaming_step_csynth.xml"
    if xml.exists():
        row.update(parse_csynth_xml(xml))
    else:
        row["resource_fit_csynth"] = False
    row.update(parse_csim_summary(csim_summary_path(name)))
    row.update(artifacts)
    return row


def collect_unroll_scan(current_hls: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    old_hls = load_csv(RESULTS / "stage2_closure_hls_summary.csv")
    for old in old_hls:
        rows.append(
            {
                "run_source": "07c2_carried_forward",
                "name": f"{old['variant']}_shared_u1",
                "variant": old["variant"],
                "unroll": int(old["unroll"]),
                "top_pipeline": False,
                "local_tap_buffer": False,
                "csim_status": "pass",
                "csim_event_calls": int(old["csim_event_calls"]),
                "csim_readout_rows": int(old["csim_readout_rows"]),
                "csim_valid_alignment_errors": int(float(old["csim_valid_alignment_errors"])),
                "pipeline_type": "no",
                "latency_max_cycles": int(old["latency_max_cycles"]),
                "latency_max_us_at_100mhz": float(old["latency_max_us_at_100mhz"]),
                "interval_max_cycles": int(old["interval_max_cycles"]),
                "interval_max_us_at_100mhz": float(old["interval_max_us_at_100mhz"]),
                "DSP": int(old["DSP"]),
                "DSP_pct": float(old["DSP_pct"]),
                "LUT": int(old["LUT"]),
                "LUT_pct": float(old["LUT_pct"]),
                "FF": int(old["FF"]),
                "FF_pct": float(old["FF_pct"]),
                "BRAM_18K": int(old["BRAM_18K"]),
                "BRAM_18K_pct": float(old["BRAM_18K_pct"]),
                "resource_fit_csynth": old["resource_fit_csynth"],
                "verification_tier": "carried_forward_07c2_csim_csynth_pnr_cosim_timeout",
            }
        )
    for meta in UNROLL_VARIANTS:
        row = dict(current_hls[str(meta["name"])])
        row.update(
            {
                "run_source": "07c2b_current",
                "top_pipeline": False,
                "local_tap_buffer": False,
                "verification_tier": "csim_csynth_only_resource_scan",
            }
        )
        rows.append(row)
    rows.sort(key=lambda r: (str(r["variant"]), int(r["unroll"])))
    return rows


def collect_pipeline_scan(current_hls: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline: dict[str, dict[str, object]] = {}
    for meta in PIPELINE_VARIANTS:
        row = dict(current_hls[str(meta["name"])])
        row["top_pipeline"] = bool(meta["top_pipeline"])
        row["local_tap_buffer"] = False
        row["pipeline_conflict_warning"] = parse_pipeline_conflict(str(meta["name"]))
        row["verification_tier"] = "csim_csynth_only_pipeline_scan"
        key = str(meta["variant"])
        if not bool(meta["top_pipeline"]):
            baseline[key] = row
            row["latency_delta_cycles_vs_nonpipe"] = 0
            row["interval_delta_cycles_vs_nonpipe"] = 0
        else:
            base = baseline[key]
            row["latency_delta_cycles_vs_nonpipe"] = int(row["latency_max_cycles"]) - int(base["latency_max_cycles"])
            row["interval_delta_cycles_vs_nonpipe"] = int(row["interval_max_cycles"]) - int(base["interval_max_cycles"])
        rows.append(row)
    return rows


def collect_localtap(current_hls: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = current_hls["scratch_shared_u4"]
    for meta in LOCALTAP_VARIANTS:
        name = str(meta["name"])
        if name in current_hls:
            row = dict(current_hls[name])
            row["top_pipeline"] = False
            row["local_tap_buffer"] = True
            row["latency_delta_cycles_vs_scratch_u4"] = int(row["latency_max_cycles"]) - int(base["latency_max_cycles"])
            row["interval_delta_cycles_vs_scratch_u4"] = int(row["interval_max_cycles"]) - int(base["interval_max_cycles"])
            row["LUT_delta_vs_scratch_u4"] = int(row["LUT"]) - int(base["LUT"])
            row["FF_delta_vs_scratch_u4"] = int(row["FF"]) - int(base["FF"])
            row["verification_tier"] = "csim_csynth_only_diagnostic"
        else:
            row = dict(meta)
            row.update(
                {
                    "top_pipeline": False,
                    "local_tap_buffer": True,
                    "csim_status": "not_run",
                    "resource_fit_csynth": "",
                    "verification_tier": "not_run_after_scratch_localtap_regression",
                }
            )
        row["status"] = meta["status"]
        row["note"] = meta["note"]
        rows.append(row)
    return rows


def collect_pnr(current_hls: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta in PNR_VARIANTS:
        name = str(meta["name"])
        paths = copy_pnr_artifacts(name)
        row: dict[str, object] = {
            "name": name,
            "variant": meta["variant"],
            "unroll": 4,
            "clock_mhz": meta["clock_mhz"],
            "clock_period_ns": meta["clock_period_ns"],
            "verification_tier": "diagnostic_pnr_met_cosim_timeout",
            "deployable_result": False,
        }
        row.update(parse_timing(pnr_report_dir(name) / "tcn_streaming_step_timing_summary_routed.rpt"))
        row.update(parse_utilization(pnr_report_dir(name) / "tcn_streaming_step_utilization_routed.rpt"))
        row.update(paths)
        row["csynth_latency_cycles"] = current_hls[name]["latency_max_cycles"]
        row["csynth_latency_us_at_100mhz"] = current_hls[name]["latency_max_us_at_100mhz"]
        row["csynth_interval_cycles"] = current_hls[name]["interval_max_cycles"]
        row["csynth_interval_us_at_100mhz"] = current_hls[name]["interval_max_us_at_100mhz"]
        rows.append(row)
    return rows


def collect_cosim() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta in PNR_VARIANTS:
        name = str(meta["name"])
        paths = copy_cosim_artifacts(name)
        row: dict[str, object] = {
            "name": name,
            "variant": meta["variant"],
            "unroll": 4,
            "timeout_budget_seconds": 7200,
        }
        row.update(parse_xsim_progress(xsim_log_path(name)))
        row.update(paths)
        rows.append(row)
    return rows


def fmt_bool(value: object) -> str:
    return "true" if str(value).lower() == "true" or value is True else "false"


def f(value: object, digits: int = 2) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def pct(row: dict[str, object], key: str) -> str:
    return f"{float(row[key]):.2f}%"


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def lookup(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(row for row in rows if row["name"] == name)


def build_comparison_rows(pnr_rows: list[dict[str, object]], cosim_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    phc_rows = load_csv(ROOT.parent / "06u_phc_latency_breakthrough" / "results" / "06u_latency_resource_summary.csv")
    phc_by_variant = {row["variant"]: row for row in phc_rows}
    naive_hls = load_csv(ROOT.parent / "07b2_tcn_hls_closure" / "results" / "tcn_hls_closure_summary.csv")[0]
    naive_pnr = load_csv(ROOT.parent / "07b2_tcn_hls_closure" / "results" / "tcn_hls_closure_pnr_summary.csv")[0]
    old_hls = load_csv(RESULTS / "stage2_closure_hls_summary.csv")
    old_pnr = load_csv(RESULTS / "stage2_closure_pnr_summary.csv")
    old_by_variant = {row["variant"]: row for row in old_hls}
    old_pnr_by_variant = {row["variant"]: row for row in old_pnr}

    rows: list[dict[str, object]] = []
    for variant_key, label in [
        ("baseline_06f_v8_dense16", "PHC-lite 06F v8 baseline"),
        ("v1_unroll16_dataflow_100mhz", "PHC-lite 06U V1 unroll16"),
        ("v3_diagonal16_unroll8_localpart_100mhz", "PHC-lite 06U V3 diagonal u8 localpart"),
    ]:
        row = phc_by_variant[variant_key]
        rows.append(
            {
                "implementation": label,
                "latency_definition": "PHC recurrent single step",
                "verification_level": row["verification_level"],
                "latency_us": row["rtl_latency_us"],
                "interval_us": row["rtl_latency_us"],
                "p_and_r_clock_mhz": row["frequency_mhz"],
                "p_and_r_timing": f"WNS {float(row['wns_ns']):+.3f} ns, timing_met={row['pnr_timing_met']}",
                "post_route_resources": (
                    f"LUT {float(row['post_route_lut_pct']):.2f}%, "
                    f"FF {float(row['post_route_ff_pct']):.2f}%, "
                    f"DSP {float(row['post_route_dsp_pct']):.2f}%, "
                    f"BRAM {float(row['post_route_bram_pct']):.2f}%"
                ),
                "directly_rankable_against_phc": "self",
                "note": "existing PHC-family reference",
            }
        )
    rows.append(
        {
            "implementation": "07B2 naive TCN distilled",
            "latency_definition": "whole receptive-window recomputation",
            "verification_level": "CSim pass + cosim timeout + P&R failed",
            "latency_us": naive_hls["csynth_latency_us_at_100mhz_estimate_only"],
            "interval_us": naive_hls["csynth_interval_us_at_100mhz_estimate_only"],
            "p_and_r_clock_mhz": 100,
            "p_and_r_timing": f"WNS {float(naive_pnr['wns_ns']):+.3f} ns, timing_met={naive_pnr['timing_met']}",
            "post_route_resources": (
                f"LUT {float(naive_pnr['routed_lut_pct']):.2f}%, "
                f"FF {float(naive_pnr['routed_ff_pct']):.2f}%, "
                f"DSP {float(naive_pnr['routed_dsp_pct']):.2f}%, "
                f"BRAM {float(naive_pnr['routed_bram_18k_pct']):.2f}%"
            ),
            "directly_rankable_against_phc": False,
            "note": "naive whole-window TCN is not timing-closed",
        }
    )
    for variant in ["scratch", "distilled"]:
        old = old_by_variant[variant]
        pnr = old_pnr_by_variant[variant]
        rows.append(
            {
                "implementation": f"07C2 streaming TCN {variant} shared_u1",
                "latency_definition": "streaming per-event top call; one readout after warmup/window",
                "verification_level": "csim+csynth+P&R_met; cosim1008_timeout",
                "latency_us": old["latency_max_us_at_100mhz"],
                "interval_us": old["interval_max_us_at_100mhz"],
                "p_and_r_clock_mhz": 100,
                "p_and_r_timing": f"WNS {float(pnr['wns_ns']):+.3f} ns, timing_met={pnr['timing_met']}",
                "post_route_resources": (
                    f"LUT {float(pnr['routed_lut_pct']):.2f}%, "
                    f"FF {float(pnr['routed_ff_pct']):.2f}%, "
                    f"DSP {float(pnr['routed_dsp_pct']):.2f}%, "
                    f"BRAM {float(pnr['routed_bram_18k_pct']):.2f}%"
                ),
                "directly_rankable_against_phc": False,
                "note": "07C2 carried-forward baseline",
            }
        )
    cosim_by_name = {row["name"]: row for row in cosim_rows}
    for row in pnr_rows:
        cosim = cosim_by_name[str(row["name"])]
        rows.append(
            {
                "implementation": f"07C2B streaming TCN {row['variant']} shared_u4",
                "latency_definition": "streaming per-event top call; one readout after warmup/window",
                "verification_level": "csim+csynth+P&R_met; cosim1008_timeout",
                "latency_us": row["csynth_latency_us_at_100mhz"],
                "interval_us": row["csynth_interval_us_at_100mhz"],
                "p_and_r_clock_mhz": 100,
                "p_and_r_timing": f"WNS {float(row['wns_ns']):+.3f} ns, timing_met={row['timing_met']}",
                "post_route_resources": (
                    f"LUT {float(row['routed_lut_pct']):.2f}%, "
                    f"FF {float(row['routed_ff_pct']):.2f}%, "
                    f"DSP {float(row['routed_dsp_pct']):.2f}%, "
                    f"BRAM {float(row['routed_bram_18k_pct']):.2f}%"
                ),
                "directly_rankable_against_phc": False,
                "note": (
                    "diagnostic only; "
                    f"cosim timed out at {cosim['completed_transactions_at_timeout']}/"
                    f"{cosim['total_rtl_transactions']} transactions"
                ),
            }
        )
    return rows


def build_audit(
    unroll_rows: list[dict[str, object]],
    pipeline_rows: list[dict[str, object]],
    localtap_rows: list[dict[str, object]],
    pnr_rows: list[dict[str, object]],
    cosim_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> str:
    old_s = lookup(unroll_rows, "scratch_shared_u1")
    old_d = lookup(unroll_rows, "distilled_shared_u1")
    s4 = lookup(unroll_rows, "scratch_shared_u4")
    d4 = lookup(unroll_rows, "distilled_shared_u4")
    scratch_speedup = float(old_s["latency_max_cycles"]) / float(s4["latency_max_cycles"])
    distilled_speedup = float(old_d["latency_max_cycles"]) / float(d4["latency_max_cycles"])

    resource_table = table(
        [
            "variant",
            "unroll",
            "run source",
            "CSim",
            "align err",
            "lat cycles",
            "lat us",
            "interval cycles",
            "DSP",
            "LUT",
            "FF",
            "BRAM",
            "fit",
        ],
        [
            [
                row["variant"],
                row["unroll"],
                row["run_source"],
                row["csim_status"],
                row["csim_valid_alignment_errors"],
                row["latency_max_cycles"],
                f(row["latency_max_us_at_100mhz"], 2),
                row["interval_max_cycles"],
                pct(row, "DSP_pct"),
                pct(row, "LUT_pct"),
                pct(row, "FF_pct"),
                pct(row, "BRAM_18K_pct"),
                fmt_bool(row["resource_fit_csynth"]),
            ]
            for row in unroll_rows
        ],
    )

    pipeline_table = table(
        [
            "variant",
            "top PIPELINE",
            "PipelineType",
            "lat cycles",
            "interval cycles",
            "delta lat",
            "delta interval",
            "DSP",
            "LUT",
            "FF",
            "fit",
        ],
        [
            [
                row["variant"],
                fmt_bool(row["top_pipeline"]),
                row["pipeline_type"],
                row["latency_max_cycles"],
                row["interval_max_cycles"],
                row["latency_delta_cycles_vs_nonpipe"],
                row["interval_delta_cycles_vs_nonpipe"],
                pct(row, "DSP_pct"),
                pct(row, "LUT_pct"),
                pct(row, "FF_pct"),
                fmt_bool(row["resource_fit_csynth"]),
            ]
            for row in pipeline_rows
        ],
    )

    localtap_table = table(
        [
            "variant",
            "status",
            "CSim",
            "lat cycles",
            "interval cycles",
            "delta lat vs scratch u4",
            "DSP",
            "LUT",
            "FF",
            "fit",
            "note",
        ],
        [
            [
                row["variant"],
                row["status"],
                row.get("csim_status", ""),
                row.get("latency_max_cycles", ""),
                row.get("interval_max_cycles", ""),
                row.get("latency_delta_cycles_vs_scratch_u4", ""),
                pct(row, "DSP_pct") if "DSP_pct" in row else "",
                pct(row, "LUT_pct") if "LUT_pct" in row else "",
                pct(row, "FF_pct") if "FF_pct" in row else "",
                fmt_bool(row["resource_fit_csynth"]) if row.get("resource_fit_csynth") != "" else "",
                row["note"],
            ]
            for row in localtap_rows
        ],
    )

    cosim_table = table(
        [
            "variant",
            "event calls",
            "timeout s",
            "RTL transactions",
            "completed at timeout",
            "final report",
            "accepted pass",
        ],
        [
            [
                row["variant"],
                row["attempted_event_calls"],
                row["timeout_budget_seconds"],
                row["total_rtl_transactions"],
                row["completed_transactions_at_timeout"],
                row["final_compare_report"],
                fmt_bool(row["accepted_as_cosim_pass"]),
            ]
            for row in cosim_rows
        ],
    )

    pnr_table = table(
        [
            "variant",
            "clock",
            "WNS",
            "TNS",
            "failing endpoints",
            "timing met",
            "routed DSP",
            "routed LUT",
            "routed FF",
            "routed BRAM",
            "tier",
        ],
        [
            [
                row["variant"],
                f"{f(row['clock_mhz'], 0)}MHz",
                f"{f(row['wns_ns'], 3)} ns",
                f"{f(row['tns_ns'], 3)} ns",
                row["setup_failing_endpoints"],
                fmt_bool(row["timing_met"]),
                pct(row, "routed_dsp_pct"),
                pct(row, "routed_lut_pct"),
                pct(row, "routed_ff_pct"),
                pct(row, "routed_bram_18k_pct"),
                row["verification_tier"],
            ]
            for row in pnr_rows
        ],
    )

    comparison_table = table(
        [
            "implementation",
            "latency definition",
            "verification",
            "latency us",
            "interval us",
            "P&R timing",
            "post-route resources",
            "note",
        ],
        [
            [
                row["implementation"],
                row["latency_definition"],
                row["verification_level"],
                f(row["latency_us"], 3) if str(row["latency_us"]).replace(".", "", 1).isdigit() else row["latency_us"],
                f(row["interval_us"], 3) if str(row["interval_us"]).replace(".", "", 1).isdigit() else row["interval_us"],
                row["p_and_r_timing"],
                row["post_route_resources"],
                row["note"],
            ]
            for row in comparison_rows
        ],
    )

    instance_rows = []
    for name in ["scratch_shared_u4", "distilled_shared_u4"]:
        for inst in parse_csynth_instance_rows(ARTIFACTS / name / "syn_report" / "tcn_streaming_step_csynth.rpt"):
            if "stage2_resource_stream_block_step" in str(inst["module"]):
                instance_rows.append(
                    [
                        name,
                        str(inst["module"]).replace("p_anonymous_namespace_", ""),
                        inst["BRAM_18K"],
                        inst["DSP"],
                        inst["LUT"],
                        inst["FF"],
                    ]
                )
    hotspot_table = table(
        ["variant", "module", "BRAM_18K", "DSP", "LUT", "FF"],
        instance_rows,
    )

    evidence_table = table(
        ["artifact", "path"],
        [
            ["unroll scan CSV", "results/stage2_speed_unroll_scan.csv"],
            ["pipeline scan CSV", "results/stage2_speed_pipeline_scan.csv"],
            ["local partition diagnostic CSV", "results/stage2_speed_local_partition_diagnostic.csv"],
            ["cosim attempts CSV", "results/stage2_speed_cosim_attempts.csv"],
            ["P&R summary CSV", "results/stage2_speed_pnr_summary.csv"],
            ["comparison CSV", "results/stage2_speed_comparison.csv"],
            ["scratch u4 csynth report", "artifacts/scratch_shared_u4/syn_report/tcn_streaming_step_csynth.rpt"],
            ["distilled u4 csynth report", "artifacts/distilled_shared_u4/syn_report/tcn_streaming_step_csynth.rpt"],
            ["scratch u4 routed timing", "artifacts/scratch_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt"],
            ["distilled u4 routed timing", "artifacts/distilled_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt"],
            ["scratch u4 timeout log", "artifacts/scratch_shared_u4/cosim_1008_timeout_xsim.log"],
            ["distilled u4 timeout log", "artifacts/distilled_shared_u4/cosim_1008_timeout_xsim.log"],
        ],
    )

    return f"""status: mixed_speed_improved_resource_fit_and_pnr_met_but_cosim1008_timeout
finding: shared-history plus unroll=4 is the fastest tested resource-fit point, improving 07C2 shared_u1 from 16,510 cycles to 7,438 cycles, but 1008-event RTL cosim timed out after a two-hour budget. The P&R-met rows are diagnostic only, not deployable closure.

**Scope Boundary**
This audit lives in `n2ew_era/07c2b_tcn_streaming_speed/`. The workdir was created by copying 07C2 outputs as a starting point; `n2ew_era/07c_tcn_streaming_manual_hls/` and `n2ew_era/07c2_tcn_streaming_closure/` were not modified. The only HLS source changes are inside this 07C2B copy. No model weights were changed, no retraining was performed, and NEW5 was not touched.

The latency object here is streaming TCN per-event top-call latency and interval. PHC rows are recurrent single-step latency; 07B naive TCN rows are whole receptive-window recomputation. These objects are not directly rankable as one common latency number.

**Shared-History Unroll Scan**
The shared BRAM history architecture stayed enabled while scanning channel unroll. CSim used 4 full windows, i.e. 1008 streaming event calls, and zero valid-alignment errors was required before csynth.

{resource_table}

Result: the highest tested resource-fit unroll is `TCN_STAGE2_CHANNEL_UNROLL=4` for both scratch and distilled. Against the carried-forward 07C2 shared_u1 point, scratch improved by {scratch_speedup:.2f}x and distilled improved by {distilled_speedup:.2f}x in csynth latency cycles.

**Latency vs Throughput Pipeline**
Top-level `#pragma HLS PIPELINE` was attempted on the selected u4 setting. It did not change latency or interval; the csynth top report still says `PipelineType=no`.

{pipeline_table}

Pipeline warning observed in both pipeline variants: `pragma conflict between PIPELINE ... and PIPELINE off ... Ignoring PIPELINE to honor PIPELINE off`. Therefore the reported steady-state interval remains 7,439 cycles; there is no separate throughput win to claim.

**Local Array Partition Diagnostic**
Because u4 was still much slower than the sub-10us target, one local-tap buffering experiment was run on scratch. It passed CSim, but worsened both latency and LUT/FF, so the distilled localtap run was skipped as diagnostic-only.

{localtap_table}

**Selected U4 Hotspot Snapshot**
The five streaming blocks remain the dominant csynth instance-level resource consumers. Shared BRAM history keeps total resources within budget, but it serializes enough work that u4 still costs 7,438 cycles.

{hotspot_table}

**Cosim Completeness**
The final selected u4 setting was launched for 1008 streaming event calls with a two-hour timeout budget for each variant. Both attempts entered RTL simulation, but neither produced `hls_cosim.rpt` or a final compare result.

{cosim_table}

This is the closure boundary: the u4 P&R data below is valid timing/resource evidence, but must not be cited as deployable `csim+cosim+P&R_met`.

**P&R**
100MHz P&R met timing for both selected u4 variants, so the 90MHz fallback was not run.

{pnr_table}

**Comparison Boundary Table**
Each row states its latency definition. Streaming TCN, naive TCN, and PHC should be read as methodology comparisons, not one direct speed ranking.

{comparison_table}

**Evidence Files**
{evidence_table}

**Honest Conclusion**
07C2B answered the specific speed question: shared-history can be combined with higher unroll, and unroll=4 is the fastest tested resource-fit point. It cuts the 07C2 shared_u1 latency from 165.10 us/event to 74.38 us/event at 100MHz while still meeting Zynq-7020 resource budgets and routed timing. However, this remains far above the <10us objective, the top-level pipeline attempt gave no interval improvement, and the local-tap partition experiment regressed. Most importantly, the required 1000-event RTL cosim did not finish in the two-hour attempts, so the final status is diagnosed but not deployable: resource fit and 100MHz P&R are resolved, full cosim closure remains open.
"""


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    current_metas: list[dict[str, object]] = []
    current_metas.extend(UNROLL_VARIANTS)
    current_metas.extend(meta for meta in PIPELINE_VARIANTS if str(meta["name"]).endswith("_pipe"))
    current_metas.append(LOCALTAP_VARIANTS[0])

    current_hls: dict[str, dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    for meta in current_metas:
        row = collect_hls_variant(meta)
        current_hls[str(row["name"])] = row
        all_rows.append(row)

    unroll_rows = collect_unroll_scan(current_hls)
    pipeline_rows = collect_pipeline_scan(current_hls)
    localtap_rows = collect_localtap(current_hls)
    pnr_rows = collect_pnr(current_hls)
    cosim_rows = collect_cosim()
    comparison_rows = build_comparison_rows(pnr_rows, cosim_rows)

    write_csv(RESULTS / "stage2_speed_all_hls_variants.csv", all_rows)
    write_csv(RESULTS / "stage2_speed_unroll_scan.csv", unroll_rows)
    write_csv(RESULTS / "stage2_speed_pipeline_scan.csv", pipeline_rows)
    write_csv(RESULTS / "stage2_speed_local_partition_diagnostic.csv", localtap_rows)
    write_csv(RESULTS / "stage2_speed_pnr_summary.csv", pnr_rows)
    write_csv(RESULTS / "stage2_speed_cosim_attempts.csv", cosim_rows)
    write_csv(RESULTS / "stage2_speed_comparison.csv", comparison_rows)

    audit = build_audit(unroll_rows, pipeline_rows, localtap_rows, pnr_rows, cosim_rows, comparison_rows)
    (ROOT / "AUDIT_tcn_streaming_speed.md").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
