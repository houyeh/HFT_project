from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"


PROGRESS_RE = re.compile(
    r"RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)\s*\[[^\]]+\]\s*@\s*\"(\d+)\""
)
ELAPSED_RE = re.compile(r"run:\s+Time\s+\(s\):.*?elapsed\s*=\s*([0-9:]+)")
CPU_RE = re.compile(r"Simulation CPU Usage:\s*([0-9]+)\s*ms")
MEM_RE = re.compile(r"Simulation Memory Usage:\s*([0-9]+)\s*KB\s*\(Peak:\s*([0-9]+)\s*KB\)")
PROTOCOL_RE = re.compile(r"\|\s*[^|]+\|\s*(?:in|out|inout)\|\s*\d+\|\s*(ap_ctrl_hs|ap_none|ap_vld|ap_ovld|ap_memory)\|")


def hms_to_seconds(text: str) -> float:
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_log(path: Path) -> dict[str, object]:
    progress: list[tuple[int, int, int]] = []
    elapsed_seconds = math.nan
    cpu_ms = math.nan
    memory_kb = math.nan
    memory_peak_kb = math.nan
    pass_detected = False
    fail_detected = False
    oom_detected = False
    simulation_not_responding = False
    output_feedback_mentions = 0
    svr_active_mentions = 0
    uvm_agent_mentions = 0
    has_wdb = False
    has_simmode_gui = False
    explicit_log_wave = False
    commented_log_wave = False

    scan_paths = [path]
    if path.exists() and path.name == "hls_run_cosim.log":
        component_root = path.parent.parent
        xsim_dir = component_root / "hls" / "sim" / "verilog" / "xsim.dir"
        scan_paths.extend(sorted(xsim_dir.glob("*/xsimkernel.log")))

    if not path.exists():
        return {"log_exists": False}

    for scan_path in scan_paths:
        if not scan_path.exists():
            continue
        with scan_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                m = PROGRESS_RE.search(line)
                if m:
                    progress.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                    continue
                m = ELAPSED_RE.search(line)
                if m:
                    elapsed_seconds = hms_to_seconds(m.group(1))
                m = CPU_RE.search(line)
                if m:
                    cpu_ms = float(m.group(1))
                m = MEM_RE.search(line)
                if m:
                    memory_kb = float(m.group(1))
                    memory_peak_kb = float(m.group(2))
                stripped = line.strip()
                if stripped.startswith("#log_wave"):
                    commented_log_wave = True
                elif "log_wave" in stripped:
                    explicit_log_wave = True
                if ".wdb" in line or " -wdb " in line:
                    has_wdb = True
                if "-simmode gui" in line:
                    has_simmode_gui = True
                if "*** C/RTL co-simulation finished: PASS" in line:
                    pass_detected = True
                if "*** C/RTL co-simulation finished: FAIL" in line:
                    fail_detected = True
                if "Out of memory" in line:
                    oom_detected = True
                if "Simulation engine not responding" in line:
                    simulation_not_responding = True
                if "output feeding back to input case" in line:
                    output_feedback_mentions += 1
                if "SVR_ACTIVE" in line:
                    svr_active_mentions += 1
                if "uvm_agent" in line:
                    uvm_agent_mentions += 1

    deltas = []
    for prev, cur in zip(progress, progress[1:]):
        prev_tx, _, prev_ps = prev
        tx, _, sim_ps = cur
        if tx <= prev_tx:
            continue
        deltas.append((sim_ps - prev_ps) / 10000.0)

    observed = max((item[0] for item in progress), default=0)
    declared = progress[-1][1] if progress else ""
    last_ps = progress[-1][2] if progress else ""
    outcome = "timeout_or_incomplete"
    if pass_detected:
        outcome = "pass"
    elif oom_detected:
        outcome = "oom_fail"
    elif fail_detected:
        outcome = "fail"

    return {
        "log_exists": True,
        "detected_outcome": outcome,
        "observed_transactions": observed,
        "declared_total_transactions": declared,
        "last_sim_time_ps": last_ps,
        "delta_cycles_p50": percentile(deltas, 0.50) if deltas else "",
        "delta_cycles_p90": percentile(deltas, 0.90) if deltas else "",
        "delta_cycles_max": max(deltas) if deltas else "",
        "wall_seconds_basis": elapsed_seconds if not math.isnan(elapsed_seconds) else "",
        "wall_seconds_per_observed_transaction": (
            elapsed_seconds / observed if observed and not math.isnan(elapsed_seconds) else ""
        ),
        "xsim_cpu_ms": cpu_ms if not math.isnan(cpu_ms) else "",
        "xsim_memory_kb": memory_kb if not math.isnan(memory_kb) else "",
        "xsim_memory_peak_kb": memory_peak_kb if not math.isnan(memory_peak_kb) else "",
        "pass_detected": pass_detected,
        "fail_detected": fail_detected,
        "oom_detected": oom_detected,
        "simulation_not_responding": simulation_not_responding,
        "has_wdb": has_wdb,
        "has_simmode_gui": has_simmode_gui,
        "explicit_log_wave": explicit_log_wave,
        "commented_log_wave": commented_log_wave,
        "output_feedback_mentions": output_feedback_mentions,
        "svr_active_mentions": svr_active_mentions,
        "uvm_agent_mentions": uvm_agent_mentions,
    }


def parse_interface_counts(csynth: Path) -> dict[str, object]:
    counts = {
        "interface_port_rows": "",
        "ap_ctrl_hs_ports": "",
        "ap_memory_ports": "",
        "ap_none_ports": "",
        "ap_vld_ports": "",
        "ap_ovld_ports": "",
    }
    if not csynth.exists():
        return counts
    protocol_counts = {
        "ap_ctrl_hs": 0,
        "ap_memory": 0,
        "ap_none": 0,
        "ap_vld": 0,
        "ap_ovld": 0,
    }
    total = 0
    with csynth.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = PROTOCOL_RE.search(line)
            if not m:
                continue
            total += 1
            protocol_counts[m.group(1)] += 1
    counts["interface_port_rows"] = total
    counts["ap_ctrl_hs_ports"] = protocol_counts["ap_ctrl_hs"]
    counts["ap_memory_ports"] = protocol_counts["ap_memory"]
    counts["ap_none_ports"] = protocol_counts["ap_none"]
    counts["ap_vld_ports"] = protocol_counts["ap_vld"]
    counts["ap_ovld_ports"] = protocol_counts["ap_ovld"]
    return counts


def parse_autowrap(component: Path) -> dict[str, object]:
    candidates = list((component / "hls_component" / "hls" / "sim" / "autowrap").glob("**/apatb_tcn_streaming_step.cpp"))
    if not candidates:
        return {
            "autowrap_exists": False,
            "autowrap_cpp_bytes": "",
            "tv_marker_occurrences": "",
            "tvin_marker_occurrences": "",
            "tvout_marker_occurrences": "",
        }
    path = candidates[0]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "autowrap_exists": True,
        "autowrap_cpp_bytes": path.stat().st_size,
        "tv_marker_occurrences": text.count("AUTOTB_TVOUT_PC_") + text.count("AUTOTB_TVIN_"),
        "tvin_marker_occurrences": text.count("AUTOTB_TVIN_"),
        "tvout_marker_occurrences": text.count("AUTOTB_TVOUT_"),
    }


def load_manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest in sorted(ARTIFACTS.glob("iface_slim_manifest_cosim_readout_*.json")):
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        for row in loaded:
            row["manifest"] = str(manifest)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_rows: list[dict[str, object]] = []
    for row in load_manifest_rows():
        component = Path(str(row["component"]))
        csynth = component / "hls_component" / "hls" / "syn" / "report" / "tcn_streaming_step_csynth.rpt"
        hls_log = component / "hls_component" / "logs" / "hls_run_cosim.log"
        merged = dict(row)
        merged.update(parse_interface_counts(csynth))
        merged.update(parse_autowrap(component))
        merged.update(parse_log(hls_log))
        out_rows.append(merged)

    out = RESULTS / "iface_slim_cosim_metrics.csv"
    write_csv(out, out_rows)
    for row in out_rows:
        print(
            f"{row['name']} {row['scale_tag']}: outcome={row.get('detected_outcome')} "
            f"tx={row.get('observed_transactions')}/{row.get('declared_total_transactions')} "
            f"mem_peak_kb={row.get('xsim_memory_peak_kb')} ports={row.get('interface_port_rows')}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
