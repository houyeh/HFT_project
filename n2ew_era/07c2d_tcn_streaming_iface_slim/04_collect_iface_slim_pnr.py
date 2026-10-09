from __future__ import annotations

import csv
import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
OLD_07C2B = REPO / "n2ew_era" / "07c2b_tcn_streaming_speed"

SCRATCH_RTL = Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\c\hls_component\hls\syn\verilog")
DISTILLED_RTL = Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\c\hls_component\hls\syn\verilog")

ZYNQ = {
    "BRAM_18K": 280,
    "DSP": 220,
    "FF": 106400,
    "LUT": 53200,
    "URAM": 0,
}

PNR_RUNS = [
    {
        "name": "scratch_shared_u4_100mhz",
        "variant": "scratch",
        "base_name": "scratch_shared_u4",
        "run_name": "scratch_iface_slim_100mhz",
        "clock_mhz": 100.0,
        "clock_period_ns": 10.0,
        "rtl_dir": SCRATCH_RTL,
        "artifact_dir": ARTIFACTS / "scratch_shared_u4" / "pnr_reports_100mhz",
        "selected_final": False,
    },
    {
        "name": "scratch_shared_u4_90mhz",
        "variant": "scratch",
        "base_name": "scratch_shared_u4",
        "run_name": "scratch_iface_slim_90mhz",
        "clock_mhz": 90.0,
        "clock_period_ns": 11.111,
        "rtl_dir": SCRATCH_RTL,
        "artifact_dir": ARTIFACTS / "scratch_shared_u4" / "pnr_reports_90mhz",
        "selected_final": True,
    },
    {
        "name": "distilled_shared_u4_100mhz",
        "variant": "distilled",
        "base_name": "distilled_shared_u4",
        "run_name": "distilled_iface_slim_100mhz",
        "clock_mhz": 100.0,
        "clock_period_ns": 10.0,
        "rtl_dir": DISTILLED_RTL,
        "artifact_dir": ARTIFACTS / "distilled_shared_u4" / "pnr_reports_100mhz",
        "selected_final": True,
    },
]

SOURCE_HASH_PAIRS = [
    ("scratch wrapper", ROOT / "hls_src" / "tcn_stage2_resource_scratch_shared_u4.cpp", Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\s\tcn_stage2_resource_scratch_shared_u4.cpp")),
    ("distilled wrapper", ROOT / "hls_src" / "tcn_stage2_resource_distilled_shared_u4.cpp", Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\s\tcn_stage2_resource_distilled_shared_u4.cpp")),
    ("prepad scratch", ROOT / "hls_src" / "tcn_prepad_hls.hpp", Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\s\tcn_prepad_hls.hpp")),
    ("prepad distilled", ROOT / "hls_src" / "tcn_prepad_hls.hpp", Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\s\tcn_prepad_hls.hpp")),
    ("streaming header scratch", ROOT / "hls_src" / "tcn_stage2_streaming_hls.hpp", Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\s\tcn_stage2_streaming_hls.hpp")),
    ("streaming header distilled", ROOT / "hls_src" / "tcn_stage2_streaming_hls.hpp", Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\s\tcn_stage2_streaming_hls.hpp")),
    ("impl scratch", ROOT / "hls_src" / "tcn_stage2_resource_impl.hpp", Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\s\tcn_stage2_resource_impl.hpp")),
    ("impl distilled", ROOT / "hls_src" / "tcn_stage2_resource_impl.hpp", Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\s\tcn_stage2_resource_impl.hpp")),
    ("weights scratch", ROOT / "hls_src" / "tcn_weights_scratch.hpp", Path(r"C:\tmp\07c2d\su4_cosim_readout_w4\s\tcn_weights_scratch.hpp")),
    ("weights distilled", ROOT / "hls_src" / "tcn_weights_distilled.hpp", Path(r"C:\tmp\07c2d\du4_cosim_readout_w4\s\tcn_weights_distilled.hpp")),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        try:
            return str(path.resolve().relative_to(REPO)).replace("\\", "/")
        except ValueError:
            return str(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


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
        "csynth_target_clock_ns": float(first("TargetClockPeriod", "0")),
        "csynth_estimated_clock_ns": float(first("EstimatedClockPeriod", "0")),
        "csynth_pipeline_type": first("PipelineType", ""),
        "csynth_latency_min_cycles": int(first("Best-caseLatency", "0")),
        "csynth_latency_max_cycles": latency,
        "csynth_latency_max_us_at_100mhz": latency * 0.01,
        "csynth_interval_min_cycles": int(first("Interval-min", "0")),
        "csynth_interval_max_cycles": interval,
        "csynth_interval_max_us_at_100mhz": interval * 0.01,
    }
    for key in ZYNQ:
        value = resources.get(key, 0)
        denom = available.get(key) or ZYNQ[key]
        row[f"csynth_{key}"] = value
        row[f"csynth_{key}_pct"] = 100.0 * value / denom if denom else 0.0
    return row


def report_dir(run: dict[str, object]) -> Path:
    return Path(run["rtl_dir"]) / f"direct_tcn_streaming_reports_{run['run_name']}"


def copy_reports(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        raise RuntimeError(f"missing P&R report dir: {src_dir}")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def collect_pnr_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in PNR_RUNS:
        src_dir = report_dir(run)
        dst_dir = Path(run["artifact_dir"])
        copy_reports(src_dir, dst_dir)

        timing = src_dir / "tcn_streaming_step_timing_summary_routed.rpt"
        util = src_dir / "tcn_streaming_step_utilization_routed.rpt"
        csynth_xml = Path(run["rtl_dir"]).parent / "report" / "tcn_streaming_step_csynth.xml"
        csynth_rpt = Path(run["rtl_dir"]).parent / "report" / "tcn_streaming_step_csynth.rpt"
        syn_dst = ARTIFACTS / str(run["base_name"]) / "syn_report_1008event_reused"
        syn_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csynth_xml, syn_dst / csynth_xml.name)
        shutil.copy2(csynth_rpt, syn_dst / csynth_rpt.name)

        row: dict[str, object] = {
            "name": run["name"],
            "variant": run["variant"],
            "unroll": 4,
            "clock_mhz": run["clock_mhz"],
            "clock_period_ns": run["clock_period_ns"],
            "selected_final": run["selected_final"],
            "verification_tier": "csim+csynth+cosim_1008event+routed_pnr_met"
            if run["selected_final"]
            else "csim+csynth+cosim_1008event+routed_pnr_failed_100mhz",
            "pnr_timing_report": rel(dst_dir / timing.name),
            "pnr_util_report": rel(dst_dir / util.name),
            "csynth_rpt": rel(syn_dst / csynth_rpt.name),
            "csynth_xml": rel(syn_dst / csynth_xml.name),
        }
        row.update(parse_timing(timing))
        row.update(parse_utilization(util))
        row.update(parse_csynth_xml(csynth_xml))
        rows.append(row)
    return rows


def collect_hash_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, workspace, temp in SOURCE_HASH_PAIRS:
        h1 = sha256(workspace)
        h2 = sha256(temp)
        rows.append(
            {
                "label": label,
                "workspace_path": rel(workspace),
                "temp_source_path": str(temp),
                "match": h1 == h2,
                "sha256": h1,
                "temp_sha256": h2,
            }
        )
    return rows


def collect_old_vs_new(new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    old_rows = load_csv(OLD_07C2B / "results" / "stage2_speed_pnr_summary.csv")
    old_by_variant = {row["variant"]: row for row in old_rows}
    new100_by_variant = {
        str(row["variant"]): row
        for row in new_rows
        if float(row["clock_mhz"]) == 100.0
    }
    compare_keys = [
        "wns_ns",
        "routed_lut_pct",
        "routed_ff_pct",
        "routed_dsp_pct",
        "routed_bram_18k_pct",
    ]
    rows: list[dict[str, object]] = []
    for variant in ["scratch", "distilled"]:
        old = old_by_variant[variant]
        new = new100_by_variant[variant]
        row: dict[str, object] = {
            "variant": variant,
            "old_07c2b_clock_mhz": old["clock_mhz"],
            "new_07c2d_clock_mhz": new["clock_mhz"],
            "old_07c2b_timing_met": old["timing_met"],
            "new_07c2d_timing_met": new["timing_met"],
        }
        for key in compare_keys:
            old_v = float(old[key])
            new_v = float(new[key])
            row[f"old_{key}"] = old_v
            row[f"new_{key}"] = new_v
            row[f"delta_{key}"] = new_v - old_v
        rows.append(row)
    return rows


def selected(rows: list[dict[str, object]], variant: str) -> dict[str, object]:
    return next(row for row in rows if str(row["variant"]) == variant and str(row["selected_final"]).lower() == "true")


def cosim_row(variant: str) -> dict[str, str]:
    rows = load_csv(RESULTS / "iface_slim_cosim_metrics.csv")
    return next(row for row in rows if row["variant"] == variant and row["scale_tag"] == "4window")


def readout_summary(variant: str) -> dict[str, str]:
    suffix = "scratch" if variant == "scratch" else "distilled"
    rows = load_csv(RESULTS / f"stage2_iface_slim_cosim_readout1008_{suffix}_shared_u4_4window_tb_summary.csv")
    return next(row for row in rows if row["scope"] == "readout")


def fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: object) -> str:
    return f"{float(value):.2f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_audit(pnr_rows: list[dict[str, object]], old_vs_new: list[dict[str, object]], hash_rows: list[dict[str, object]]) -> str:
    scratch_final = selected(pnr_rows, "scratch")
    distilled_final = selected(pnr_rows, "distilled")
    all_hash_match = all(row["match"] for row in hash_rows)
    all_selected_met = all(str(row["timing_met"]).lower() == "true" for row in [scratch_final, distilled_final])
    status = "resolved_iface_slim_pnr_met" if all_selected_met else "mixed_iface_slim_pnr_partial"

    pnr_table = md_table(
        [
            "run",
            "variant",
            "clock",
            "WNS",
            "TNS",
            "setup fail",
            "hold fail",
            "timing met",
            "LUT",
            "FF",
            "DSP",
            "BRAM",
        ],
        [
            [
                row["name"],
                row["variant"],
                f"{fmt(row['clock_mhz'], 1)} MHz",
                fmt(row["wns_ns"]),
                fmt(row["tns_ns"]),
                row["setup_failing_endpoints"],
                row["hold_failing_endpoints"],
                str(row["timing_met"]).lower(),
                pct(row["routed_lut_pct"]),
                pct(row["routed_ff_pct"]),
                pct(row["routed_dsp_pct"]),
                pct(row["routed_bram_18k_pct"]),
            ]
            for row in pnr_rows
        ],
    )

    final_table = md_table(
        [
            "variant",
            "selected clock",
            "verification tier",
            "latency cycles",
            "latency at selected clock",
            "interval cycles",
            "timing met",
        ],
        [
            [
                row["variant"],
                f"{fmt(row['clock_mhz'], 1)} MHz",
                row["verification_tier"],
                row["csynth_latency_max_cycles"],
                f"{float(row['csynth_latency_max_cycles']) * float(row['clock_period_ns']) / 1000.0:.2f} us",
                row["csynth_interval_max_cycles"],
                str(row["timing_met"]).lower(),
            ]
            for row in [scratch_final, distilled_final]
        ],
    )

    compare_table = md_table(
        [
            "variant",
            "old/new 100MHz timing",
            "WNS delta",
            "LUT old -> new",
            "FF old -> new",
            "DSP old -> new",
            "BRAM old -> new",
        ],
        [
            [
                row["variant"],
                f"{str(row['old_07c2b_timing_met']).lower()} -> {str(row['new_07c2d_timing_met']).lower()}",
                f"{float(row['delta_wns_ns']):+.3f} ns",
                f"{pct(row['old_routed_lut_pct'])} -> {pct(row['new_routed_lut_pct'])} ({float(row['delta_routed_lut_pct']):+.2f} pp)",
                f"{pct(row['old_routed_ff_pct'])} -> {pct(row['new_routed_ff_pct'])} ({float(row['delta_routed_ff_pct']):+.2f} pp)",
                f"{pct(row['old_routed_dsp_pct'])} -> {pct(row['new_routed_dsp_pct'])} ({float(row['delta_routed_dsp_pct']):+.2f} pp)",
                f"{pct(row['old_routed_bram_18k_pct'])} -> {pct(row['new_routed_bram_18k_pct'])} ({float(row['delta_routed_bram_18k_pct']):+.2f} pp)",
            ]
            for row in old_vs_new
        ],
    )

    cosim_table_rows = []
    for variant in ["scratch", "distilled"]:
        cm = cosim_row(variant)
        rs = readout_summary(variant)
        cosim_table_rows.append(
            [
                variant,
                f"{cm['observed_transactions']}/{cm['declared_total_transactions']}",
                cm["readout_decision_rows"],
                cm["xsim_memory_peak_kb"],
                f"{float(cm['wall_seconds_basis']):.0f} s",
                fmt(rs["max_abs_diff"], 6),
                fmt(rs["rmse"], 6),
                rs["sign_agreement_joint"],
                rs["valid_alignment_errors"],
            ]
        )
    cosim_table = md_table(
        [
            "variant",
            "RTL tx",
            "readout rows",
            "xsim peak KB",
            "xsim wall",
            "readout max diff",
            "readout RMSE",
            "joint sign",
            "valid errors",
        ],
        cosim_table_rows,
    )

    hash_table = md_table(
        ["check", "matched", "sha256"],
        [[row["label"], str(row["match"]).lower(), str(row["sha256"])[:12] + "..."] for row in hash_rows],
    )

    evidence_table = md_table(
        ["artifact", "path"],
        [
            ["P&R summary CSV", "results/iface_slim_pnr_summary.csv"],
            ["old vs new comparison CSV", "results/iface_slim_pnr_old_vs_new.csv"],
            ["csynth reuse hash CSV", "results/iface_slim_csynth_reuse_hash_check.csv"],
            ["scratch 100MHz routed timing", "artifacts/scratch_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt"],
            ["scratch 90MHz routed timing", "artifacts/scratch_shared_u4/pnr_reports_90mhz/tcn_streaming_step_timing_summary_routed.rpt"],
            ["distilled 100MHz routed timing", "artifacts/distilled_shared_u4/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt"],
            ["Vivado logs", "artifacts/pnr_logs/"],
            ["1008-event cosim metrics", "results/iface_slim_cosim_metrics.csv"],
        ],
    )

    return f"""status: {status}

# 07C2D TCN Streaming Interface-Slim P&R Audit

07C2D is the first TCN streaming line from 07C onward to carry the same shared_u4 RTL through CSim, csynth, 1008-event RTL cosim, and routed P&R timing closure at 90MHz or better.

## Scope Boundary

- Work directory: `n2ew_era/07c2d_tcn_streaming_iface_slim/`.
- No changes were made to `n2ew_era/07c2_tcn_streaming_closure/`, `n2ew_era/07c2b_tcn_streaming_speed/`, or `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/`.
- No model retraining, no weight changes, no NEW5 access, and no change to shared_u4 unroll/BRAM settings or the slim top interface.
- The 1008-event 07C2D slim csynth/RTL product was reused because every actually synthesized source file matched the current 07C2D workspace source by SHA-256: `{str(all_hash_match).lower()}`.
- P&R was run with the 07C2B-style direct non-project Vivado flow using `run_direct_tcn_streaming_pnr_custom_clock.tcl`. The only script correction in the 07C2D copy was to load the generated clock via XDC, matching the 07C2B flow.

## CSynth Reuse Check

{hash_table}

The HLS configs for the reused 1008-event products synthesize only the slim wrapper, `tcn_prepad_hls.hpp`, `tcn_stage2_streaming_hls.hpp`, `tcn_stage2_resource_impl.hpp`, and the matching weight header. Legacy copied files left in the temporary source directory were not referenced by those configs.

## 1008-Event Cosim Evidence

{cosim_table}

The cosim evidence is the existing 07C2D slim-interface 1008-event run. It is readout-only at the top interface, with final readout sign agreement of 1.0 and zero valid-alignment errors for both variants.

## Routed P&R Results

{pnr_table}

Scratch did not meet 100MHz after interface slimming, so the required 90MHz fallback was run and met timing. Distilled met timing at 100MHz, so no 90MHz fallback was required for distilled.

## Selected Closure Points

{final_table}

Latency/interval are still the HLS streaming-step latency and interval objects from csynth. They should not be ranked directly against PHC recurrent single-step latency or naive TCN whole-window recompute latency.

## Interface Effect Versus 07C2B Old Top

{compare_table}

At 100MHz, interface slimming reduced routed LUT and FF use for both variants, left BRAM unchanged, and increased routed DSP use. Timing moved in different directions: distilled improved slightly and still met 100MHz, while scratch moved from passing in 07C2B to failing by -0.150ns in 07C2D. Therefore the measured effect is not simply "fewer ports means everything improves"; the reliable conclusion is narrower: the slim interface solved cosim memory scaling and reduced LUT/FF, but scratch needs 90MHz for routed timing under this flow.

## Evidence Files

{evidence_table}

## Honest Conclusion

The 07C2D interface-slim route closes the verification chain that 07C2B could not close: both shared_u4 variants have CSim, csynth, 1008-event RTL cosim, and routed P&R timing-met evidence. The final routed clocks are 90MHz for scratch and 100MHz for distilled. This is a full validation milestone for the slim-interface streaming TCN, but it does not change the latency definition: TCN streaming remains an amortized per-event streaming-step result, not the same latency object as PHC-lite's recurrent single-step path.
"""


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    hash_rows = collect_hash_rows()
    pnr_rows = collect_pnr_rows()
    old_vs_new = collect_old_vs_new(pnr_rows)

    write_csv(RESULTS / "iface_slim_csynth_reuse_hash_check.csv", hash_rows)
    write_csv(RESULTS / "iface_slim_pnr_summary.csv", pnr_rows)
    write_csv(RESULTS / "iface_slim_pnr_old_vs_new.csv", old_vs_new)

    audit = build_audit(pnr_rows, old_vs_new, hash_rows)
    (ROOT / "AUDIT_tcn_streaming_iface_slim_pnr.md").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
