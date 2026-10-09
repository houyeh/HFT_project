from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

ZYNQ = {
    "BRAM_18K": 280,
    "DSP": 220,
    "FF": 106400,
    "LUT": 53200,
}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_csynth_xml(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()

    def first(tag: str) -> str:
        for elem in root.iter():
            if elem.tag.split("}")[-1] == tag:
                return (elem.text or "").strip()
        return ""

    resources = {}
    available = {}
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

    row: dict[str, object] = {
        "target_clock_ns": float(first("TargetClockPeriod")),
        "estimated_clock_ns": float(first("EstimatedClockPeriod")),
        "latency_min_cycles": int(first("Best-caseLatency")),
        "latency_max_cycles": int(first("Worst-caseLatency")),
        "latency_max_us_at_100mhz": int(first("Worst-caseLatency")) * 0.01,
        "interval_min_cycles": int(first("Interval-min")),
        "interval_max_cycles": int(first("Interval-max")),
        "interval_max_us_at_100mhz": int(first("Interval-max")) * 0.01,
    }
    for key in ZYNQ:
        row[key] = resources.get(key, 0)
        denom = available.get(key) or ZYNQ[key]
        row[f"{key}_pct"] = 100.0 * resources.get(key, 0) / denom
    return row


def parse_csynth_instance_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    in_instance = False
    for raw in text(path).splitlines():
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


def parse_timing(path: Path) -> dict[str, object]:
    content = text(path)
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
    patterns = {
        "routed_lut": r"\| Slice LUTs\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_ff": r"\| Slice Registers\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_bram_tiles": r"\| Block RAM Tile\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
        "routed_dsp": r"\| DSPs\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)\s+\|",
    }
    content = text(path)
    row: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if not match:
            raise RuntimeError(f"utilization {key} not found: {path}")
        row[key] = int(match.group(1))
        row[f"{key}_pct"] = float(match.group(2))
    row["routed_bram_18k_equiv"] = float(row["routed_bram_tiles"]) * 2.0
    row["routed_bram_18k_pct"] = float(row["routed_bram_tiles_pct"])
    return row


def parse_csim_summary(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    readout = next(row for row in rows if row["scope"] == "readout")
    block_max = max(float(row["max_abs_diff"]) for row in rows if row["scope"] == "block")
    block_rmse = max(float(row["rmse"]) for row in rows if row["scope"] == "block")
    return {
        "csim_readout_rows": int(float(readout["rows"])),
        "csim_readout_compared_values": int(float(readout["compared_values"])),
        "csim_readout_max_abs_diff": float(readout["max_abs_diff"]),
        "csim_readout_rmse": float(readout["rmse"]),
        "csim_sign_agreement_joint": float(readout["sign_agreement_joint"]),
        "csim_valid_alignment_errors": int(float(readout["valid_alignment_errors"])),
        "csim_block_max_abs_diff": block_max,
        "csim_block_rmse_max": block_rmse,
    }


def parse_xsim_progress(path: Path) -> dict[str, object]:
    content = text(path)
    progress = re.findall(r"RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)", content)
    if not progress:
        return {
            "cosim_attempted_event_calls": 1008,
            "cosim_total_transactions": "",
            "cosim_completed_transactions_at_timeout": 0,
            "cosim_final_report": "missing",
        }
    last_done, total = progress[-1]
    return {
        "cosim_attempted_event_calls": 1008,
        "cosim_total_transactions": int(total),
        "cosim_completed_transactions_at_timeout": int(last_done),
        "cosim_final_report": "missing_timeout",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_reference_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sweep_path = RESULTS / "stage2_resource_sweep_summary_07c_carried_forward.csv"
    with sweep_path.open(newline="", encoding="utf-8") as fh:
        sweep_rows = list(csv.DictReader(fh))
    phc_path = ROOT.parent / "06u_phc_latency_breakthrough" / "results" / "06u_latency_resource_summary.csv"
    with phc_path.open(newline="", encoding="utf-8") as fh:
        phc_rows = list(csv.DictReader(fh))
    return sweep_rows, phc_rows


def collect_variant(variant: str) -> dict[str, object]:
    base = ARTIFACTS / f"shortpath_{variant}_shared_u1"
    csynth_xml = base / "syn_report" / "tcn_streaming_step_csynth.xml"
    csynth_rpt = base / "syn_report" / "tcn_streaming_step_csynth.rpt"
    pnr = base / "pnr_reports_100mhz"
    csim_csv = RESULTS / f"stage2_closure_{variant}_shared_u1_event1008_tb_summary.csv"
    row: dict[str, object] = {
        "variant": variant,
        "suffix": "shared_u1",
        "unroll": 1,
        "shared_history_storage": True,
        "clock_mhz": 100,
        "clock_period_ns": 10.0,
        "verification_tier": "csim_csynth_pnr_met_cosim1008_timeout",
        "deployable_result": False,
        "deployable_blocker": "1008-event RTL cosim timed out before final compare report",
    }
    row.update(parse_csim_summary(csim_csv))
    row.update(parse_csynth_xml(csynth_xml))
    row.update(parse_timing(pnr / "tcn_streaming_step_timing_summary_routed.rpt"))
    row.update(parse_utilization(pnr / "tcn_streaming_step_utilization_routed.rpt"))
    row.update(parse_xsim_progress(base / "cosim_1008_timeout_xsim.log"))
    row["resource_fit_csynth"] = all(float(row[f"{key}_pct"]) < 100.0 for key in ZYNQ)
    row["resource_fit_routed"] = (
        float(row["routed_lut_pct"]) < 100.0
        and float(row["routed_ff_pct"]) < 100.0
        and float(row["routed_dsp_pct"]) < 100.0
        and float(row["routed_bram_18k_pct"]) < 100.0
    )
    row["csynth_report"] = str(csynth_rpt.relative_to(ROOT))
    row["pnr_timing_report"] = str((pnr / "tcn_streaming_step_timing_summary_routed.rpt").relative_to(ROOT))
    row["pnr_util_report"] = str((pnr / "tcn_streaming_step_utilization_routed.rpt").relative_to(ROOT))
    row["cosim_timeout_log"] = str((base / "cosim_1008_timeout_xsim.log").relative_to(ROOT))
    return row


def make_audit(rows: list[dict[str, object]], sweep_rows: list[dict[str, object]], phc_rows: list[dict[str, object]]) -> str:
    s = next(row for row in rows if row["variant"] == "scratch")
    d = next(row for row in rows if row["variant"] == "distilled")
    naive = {
        "latency_us": "13,703.85",
        "verification": "CSim pass + cosim timeout + P&R failed",
        "timing": "100MHz WNS -1.112 ns; 90MHz WNS -0.358 ns",
        "resources": "LUT 24.85%, FF 11.94%, DSP 21.82%, BRAM18K equiv 22.50%",
    }
    phc_lookup = {row["variant"]: row for row in phc_rows}
    phc_v8 = phc_lookup["baseline_06f_v8_dense16"]
    phc_v1 = phc_lookup["v1_unroll16_dataflow_100mhz"]
    phc_v3 = phc_lookup["v3_diagonal16_unroll8_localpart_100mhz"]

    def pct(row: dict[str, object], key: str) -> str:
        return f"{float(row[key]):.2f}%"

    def f(row: dict[str, object], key: str, ndigits: int = 3) -> str:
        return f"{float(row[key]):.{ndigits}f}"

    sweep_fit = [
        row
        for row in sweep_rows
        if row["suffix"] in {"u2", "u1", "shared_u1"}
    ]
    sweep_lines = []
    for row in sweep_fit:
        sweep_lines.append(
            "| {variant} | {suffix} | {csim} | {lat:.2f} | {ii} | {dsp:.2f}% | {lut:.2f}% | {ff:.2f}% | {bram:.2f}% | {fit} |".format(
                variant=row["variant"],
                suffix=row["suffix"],
                csim=row["hlsrun_csim_state"],
                lat=float(row["latency_us_100mhz"]),
                ii=row["ii_max_cycles"],
                dsp=float(row["DSP_pct"]),
                lut=float(row["LUT_pct"]),
                ff=float(row["FF_pct"]),
                bram=float(row["BRAM_18K_pct"]),
                fit=row["resource_fit_zynq7020"],
            )
        )

    instance_lines = []
    for variant in ("scratch", "distilled"):
        inst_rows = parse_csynth_instance_rows(
            ARTIFACTS / f"shortpath_{variant}_shared_u1" / "syn_report" / "tcn_streaming_step_csynth.rpt"
        )
        for inst in inst_rows:
            if "stage2_resource_stream_block_step" in str(inst["module"]):
                instance_lines.append(
                    "| {variant} | {module} | {BRAM_18K} | {DSP} | {LUT} | {FF} |".format(
                        variant=variant,
                        module=str(inst["module"]).replace("p_anonymous_namespace_", ""),
                        BRAM_18K=inst["BRAM_18K"],
                        DSP=inst["DSP"],
                        LUT=inst["LUT"],
                        FF=inst["FF"],
                    )
                )

    return f"""status: mixed_resource_fit_and_pnr_met_but_1008_event_cosim_timeout
finding: shared_u1 BRAM-backed history storage fixes the 07C streaming TCN resource overflow and routes at 100MHz, but full 1008-event RTL cosim did not complete within the one-hour budget, so the closure is not deployable yet.

**Scope**
This audit lives in `n2ew_era/07c2_tcn_streaming_closure/`. It copied 07C Stage 2 HLS source/results into a new workspace and did not modify `n2ew_era/07c_tcn_streaming_manual_hls/` or any other prior phase. No model weights were changed, no retraining was performed, and NEW5 was not touched.

The latency object here is streaming TCN per-event top-call latency/interval. It is not the same object as PHC recurrent single-step latency or 07B naive whole-window recomputation latency.

**Environment Fix**
The original 07C distilled csynth failure (`Spawn failed: The operation completed successfully`) did not reproduce after rebuilding in a clean 07C2 shortpath workdir under `C:\\tmp`. Distilled `shared_u1` produced a complete csynth top report, so the likely cause was stale/long-path/tool workdir state rather than invalid distilled source or weights.

**Resource Sweep**
These rows are the carried-forward 07C sweep plus the 07C2-verified selected setting. The resource-fit point is `shared_u1`: `TCN_STAGE2_CHANNEL_UNROLL=1` with BRAM-backed shared history storage.

| variant | setting | CSim state | csynth latency us @100MHz | II max cycles | DSP | LUT | FF | BRAM_18K | fit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(sweep_lines)}

**Selected Variant CSim**
07C2 reran CSim on the first 4 full windows. The testbench internally calls `tcn_streaming_step` over 252 events per window, so this is 1008 streaming event calls.

| variant | readout rows | event calls | readout max abs diff | readout RMSE | joint sign agreement | alignment errors |
|---|---:|---:|---:|---:|---:|---:|
| scratch shared_u1 | {s["csim_readout_rows"]} | 1008 | {f(s, "csim_readout_max_abs_diff", 6)} | {f(s, "csim_readout_rmse", 6)} | {f(s, "csim_sign_agreement_joint", 3)} | {s["csim_valid_alignment_errors"]} |
| distilled shared_u1 | {d["csim_readout_rows"]} | 1008 | {f(d, "csim_readout_max_abs_diff", 6)} | {f(d, "csim_readout_rmse", 6)} | {f(d, "csim_sign_agreement_joint", 3)} | {d["csim_valid_alignment_errors"]} |

**Csynth And P&R**

| variant | csynth latency cycles | csynth interval cycles | csynth DSP/LUT/FF/BRAM | routed timing @100MHz | routed resources | verification tier |
|---|---:|---:|---|---|---|---|
| scratch shared_u1 | {s["latency_max_cycles"]} | {s["interval_max_cycles"]} | DSP {pct(s, "DSP_pct")}, LUT {pct(s, "LUT_pct")}, FF {pct(s, "FF_pct")}, BRAM {pct(s, "BRAM_18K_pct")} | WNS {f(s, "wns_ns")} ns, TNS {f(s, "tns_ns")} ns, failing endpoints {s["setup_failing_endpoints"]}, timing_met={s["timing_met"]} | LUT {pct(s, "routed_lut_pct")}, FF {pct(s, "routed_ff_pct")}, DSP {pct(s, "routed_dsp_pct")}, BRAM {pct(s, "routed_bram_18k_pct")} | {s["verification_tier"]} |
| distilled shared_u1 | {d["latency_max_cycles"]} | {d["interval_max_cycles"]} | DSP {pct(d, "DSP_pct")}, LUT {pct(d, "LUT_pct")}, FF {pct(d, "FF_pct")}, BRAM {pct(d, "BRAM_18K_pct")} | WNS {f(d, "wns_ns")} ns, TNS {f(d, "tns_ns")} ns, failing endpoints {d["setup_failing_endpoints"]}, timing_met={d["timing_met"]} | LUT {pct(d, "routed_lut_pct")}, FF {pct(d, "routed_ff_pct")}, DSP {pct(d, "routed_dsp_pct")}, BRAM {pct(d, "routed_bram_18k_pct")} | {d["verification_tier"]} |

100MHz P&R met timing for both selected variants, so the 90MHz fallback was not run.

**Speed Cost**
The resource-fit mechanism is effective but expensive. The original over-budget scratch `u4` point was 558 cycles / 5.58 us at csynth, while `shared_u1` is 16,510 cycles / 165.10 us. This is still much faster than 07B naive whole-window recomputation, but it is slower than the routed PHC 06F v8 baseline and does not meet the sub-10us goal. The selected rows therefore close resource and timing, not the latency objective.

**Function Hotspot**
The selected `shared_u1` design still spends most csynth resources inside the five TCN stream blocks, but binding the history buffers to BRAM removes the LUT/FF explosion seen in `u1` and `u2`.

| variant | module | BRAM_18K | DSP | LUT | FF |
|---|---|---:|---:|---:|---:|
{chr(10).join(instance_lines)}

**Cosim Boundary**
The requested 1000-event cosim was attempted on 1008 streaming event calls. Both variants got past RTL elaboration in the shortpath flow, unlike the first long-path attempt, but neither completed final compare within one hour:

| variant | attempted event calls | total RTL transactions | completed transactions at timeout | final compare report |
|---|---:|---:|---:|---|
| scratch shared_u1 | {s["cosim_attempted_event_calls"]} | {s["cosim_total_transactions"]} | {s["cosim_completed_transactions_at_timeout"]} | {s["cosim_final_report"]} |
| distilled shared_u1 | {d["cosim_attempted_event_calls"]} | {d["cosim_total_transactions"]} | {d["cosim_completed_transactions_at_timeout"]} | {d["cosim_final_report"]} |

This means the P&R-met rows are timing/resource evidence, not deployable hardware closure. They must not be listed as `csim+cosim+P&R_met`, and no RTL mismatch count can be claimed from the timed-out 1008-event attempts.

**Comparison Table**

| implementation | latency definition | verification level | latency | steady-state interval / throughput | routed timing | routed resources | note |
|---|---|---|---:|---:|---|---|---|
| PHC-lite 06F v8 baseline | PHC recurrent single step | {phc_v8["verification_level"]} | {float(phc_v8["rtl_latency_us"]):.3f} us | {float(phc_v8["rtl_latency_us"]):.3f} us interval | WNS {float(phc_v8["wns_ns"]):+.3f} ns @100MHz | LUT {float(phc_v8["post_route_lut_pct"]):.2f}%, FF {float(phc_v8["post_route_ff_pct"]):.2f}%, DSP {float(phc_v8["post_route_dsp_pct"]):.2f}%, BRAM {float(phc_v8["post_route_bram_pct"]):.2f}% | Existing routed PHC baseline. |
| PHC-lite 06U V1 unroll16 | PHC recurrent single step | {phc_v1["verification_level"]} | {float(phc_v1["rtl_latency_us"]):.3f} us | {float(phc_v1["rtl_latency_us"]):.3f} us interval | WNS {float(phc_v1["wns_ns"]):+.3f} ns @100MHz | LUT {float(phc_v1["post_route_lut_pct"]):.2f}%, FF {float(phc_v1["post_route_ff_pct"]):.2f}%, DSP {float(phc_v1["post_route_dsp_pct"]):.2f}%, BRAM {float(phc_v1["post_route_bram_pct"]):.2f}% | Dense PHC speedup point with narrow WNS. |
| PHC-lite 06U V3 diagonal u8 localpart | PHC recurrent single step | {phc_v3["verification_level"]} | {float(phc_v3["rtl_latency_us"]):.3f} us | {float(phc_v3["rtl_latency_us"]):.3f} us interval | WNS {float(phc_v3["wns_ns"]):+.3f} ns @100MHz | LUT {float(phc_v3["post_route_lut_pct"]):.2f}%, FF {float(phc_v3["post_route_ff_pct"]):.2f}%, DSP {float(phc_v3["post_route_dsp_pct"]):.2f}%, BRAM {float(phc_v3["post_route_bram_pct"]):.2f}% | Fastest fully verified PHC-family point. |
| 07B2 naive TCN distilled | whole receptive-window recomputation | {naive["verification"]} | {naive["latency_us"]} us | same as whole-window recomputation | {naive["timing"]} | {naive["resources"]} | Not timing-closed. |
| 07C2 streaming TCN scratch shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met, but 1008-event cosim timeout | {float(s["latency_max_us_at_100mhz"]):.3f} us csynth / timing object | {float(s["interval_max_us_at_100mhz"]):.3f} us interval | WNS {float(s["wns_ns"]):+.3f} ns @100MHz | LUT {float(s["routed_lut_pct"]):.2f}%, FF {float(s["routed_ff_pct"]):.2f}%, DSP {float(s["routed_dsp_pct"]):.2f}%, BRAM {float(s["routed_bram_18k_pct"]):.2f}% | Diagnostic only until cosim completes. |
| 07C2 streaming TCN distilled shared_u1 | streaming per-event top call; one readout after warmup/window | csim+csynth+P&R_met, but 1008-event cosim timeout | {float(d["latency_max_us_at_100mhz"]):.3f} us csynth / timing object | {float(d["interval_max_us_at_100mhz"]):.3f} us interval | WNS {float(d["wns_ns"]):+.3f} ns @100MHz | LUT {float(d["routed_lut_pct"]):.2f}%, FF {float(d["routed_ff_pct"]):.2f}%, DSP {float(d["routed_dsp_pct"]):.2f}%, BRAM {float(d["routed_bram_18k_pct"]):.2f}% | Diagnostic only until cosim completes. |

**Evidence Files**
Primary generated CSVs:
- `results/stage2_closure_hls_summary.csv`
- `results/stage2_closure_pnr_summary.csv`
- `results/stage2_closure_cosim_attempts.csv`
- `results/stage2_closure_comparison.csv`

Primary report artifacts:
- `artifacts/shortpath_scratch_shared_u1/syn_report/tcn_streaming_step_csynth.rpt`
- `artifacts/shortpath_scratch_shared_u1/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt`
- `artifacts/shortpath_scratch_shared_u1/cosim_1008_timeout_xsim.log`
- `artifacts/shortpath_distilled_shared_u1/syn_report/tcn_streaming_step_csynth.rpt`
- `artifacts/shortpath_distilled_shared_u1/pnr_reports_100mhz/tcn_streaming_step_timing_summary_routed.rpt`
- `artifacts/shortpath_distilled_shared_u1/cosim_1008_timeout_xsim.log`

**Honest Conclusion**
The resource bottleneck in 07C Stage 2 was not DSP after unroll reduction; it was LUT/FF pressure from fully materialized history storage. The `shared_u1` BRAM-backed history setting solves that resource problem and the routed RTL meets 100MHz for both scratch and distilled variants. The cost is severe: the resource-fit point regresses from the 5.58 us over-budget `u4` point to 165.10 us/event, so it is not the desired low-latency streaming solution. In addition, the required 1000-event RTL cosim did not finish within the one-hour attempts, so this task is only partially closed: resource-fit and P&R timing are resolved, but low-latency and full cosim closure remain open. The current valid claim is `csim + csynth + routed P&R timing-met, diagnostic only because cosim1008 timed out`.
"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = [collect_variant("scratch"), collect_variant("distilled")]
    sweep_rows, phc_rows = load_reference_rows()

    hls_rows = []
    pnr_rows = []
    cosim_rows = []
    comparison_rows = []
    for row in rows:
        hls_rows.append(
            {
                "variant": row["variant"],
                "suffix": row["suffix"],
                "unroll": row["unroll"],
                "shared_history_storage": row["shared_history_storage"],
                "latency_max_cycles": row["latency_max_cycles"],
                "latency_max_us_at_100mhz": row["latency_max_us_at_100mhz"],
                "interval_max_cycles": row["interval_max_cycles"],
                "interval_max_us_at_100mhz": row["interval_max_us_at_100mhz"],
                "BRAM_18K": row["BRAM_18K"],
                "BRAM_18K_pct": row["BRAM_18K_pct"],
                "DSP": row["DSP"],
                "DSP_pct": row["DSP_pct"],
                "FF": row["FF"],
                "FF_pct": row["FF_pct"],
                "LUT": row["LUT"],
                "LUT_pct": row["LUT_pct"],
                "resource_fit_csynth": row["resource_fit_csynth"],
                "csim_readout_rows": row["csim_readout_rows"],
                "csim_event_calls": 1008,
                "csim_readout_max_abs_diff": row["csim_readout_max_abs_diff"],
                "csim_readout_rmse": row["csim_readout_rmse"],
                "csim_sign_agreement_joint": row["csim_sign_agreement_joint"],
                "csim_valid_alignment_errors": row["csim_valid_alignment_errors"],
                "csynth_report": row["csynth_report"],
            }
        )
        pnr_rows.append(
            {
                "variant": row["variant"],
                "clock_mhz": row["clock_mhz"],
                "clock_period_ns": row["clock_period_ns"],
                "wns_ns": row["wns_ns"],
                "tns_ns": row["tns_ns"],
                "setup_failing_endpoints": row["setup_failing_endpoints"],
                "timing_met": row["timing_met"],
                "routed_lut": row["routed_lut"],
                "routed_lut_pct": row["routed_lut_pct"],
                "routed_ff": row["routed_ff"],
                "routed_ff_pct": row["routed_ff_pct"],
                "routed_dsp": row["routed_dsp"],
                "routed_dsp_pct": row["routed_dsp_pct"],
                "routed_bram_tiles": row["routed_bram_tiles"],
                "routed_bram_18k_equiv": row["routed_bram_18k_equiv"],
                "routed_bram_18k_pct": row["routed_bram_18k_pct"],
                "resource_fit_routed": row["resource_fit_routed"],
                "pnr_timing_report": row["pnr_timing_report"],
                "pnr_util_report": row["pnr_util_report"],
            }
        )
        cosim_rows.append(
            {
                "variant": row["variant"],
                "attempted_event_calls": row["cosim_attempted_event_calls"],
                "total_rtl_transactions": row["cosim_total_transactions"],
                "completed_transactions_at_timeout": row["cosim_completed_transactions_at_timeout"],
                "timeout_budget_seconds": 3600,
                "final_compare_report": row["cosim_final_report"],
                "accepted_as_cosim_pass": False,
                "cosim_timeout_log": row["cosim_timeout_log"],
            }
        )
        comparison_rows.append(
            {
                "implementation": f"07C2 streaming TCN {row['variant']} shared_u1",
                "latency_definition": "streaming per-event top call; one readout after warmup/window",
                "verification_level": "csim+csynth+P&R_met; cosim1008_timeout",
                "latency_us": row["latency_max_us_at_100mhz"],
                "interval_us": row["interval_max_us_at_100mhz"],
                "timing_met": row["timing_met"],
                "wns_ns": row["wns_ns"],
                "routed_lut_pct": row["routed_lut_pct"],
                "routed_ff_pct": row["routed_ff_pct"],
                "routed_dsp_pct": row["routed_dsp_pct"],
                "routed_bram_pct": row["routed_bram_18k_pct"],
                "directly_rankable_against_phc": False,
            }
        )

    write_csv(RESULTS / "stage2_closure_hls_summary.csv", hls_rows)
    write_csv(RESULTS / "stage2_closure_pnr_summary.csv", pnr_rows)
    write_csv(RESULTS / "stage2_closure_cosim_attempts.csv", cosim_rows)
    write_csv(RESULTS / "stage2_closure_comparison.csv", comparison_rows)

    audit = make_audit(rows, sweep_rows, phc_rows)
    (ROOT / "AUDIT_tcn_streaming_closure.md").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
