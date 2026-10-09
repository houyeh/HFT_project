from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


PROGRESS_RE = re.compile(
    r"RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)\s*\[[^\]]+\]\s*@\s*\"(\d+)\""
)
ELAPSED_RE = re.compile(
    r"run:\s+Time\s+\(s\):.*?elapsed\s*=\s*([0-9:]+)"
)
CPU_RE = re.compile(r"Simulation CPU Usage:\s*([0-9]+)\s*ms")
MEM_RE = re.compile(r"Simulation Memory Usage:\s*([0-9]+)\s*KB\s*\(Peak:\s*([0-9]+)\s*KB\)")
XSIMK_RE = re.compile(r"Running:\s*(.+?xsimk\.exe.+)")


KNOWN_LOGS = [
    {
        "label": "phc_06u_v3_1000_full",
        "architecture": "PHC 06U V3",
        "status": "pass",
        "timeout_seconds": "",
        "log": Path(
            "C:/tmp/06u_v3_diag16_u8lp_100_20260730_0730/"
            "hls_component/hls_component/logs/hls_run_cosim.log"
        ),
    },
    {
        "label": "tcn_07c2b_scratch_shared_u4_1008_timeout",
        "architecture": "TCN streaming shared_u4 scratch",
        "status": "timeout",
        "timeout_seconds": "7200",
        "log": Path(
            "C:/tmp/07c2b_tcn_streaming_speed_shortpath/scratch_shared_u4/"
            "hls_component_stage2_resource_scratch_shared_u4/hls_component/logs/hls_run_cosim.log"
        ),
    },
    {
        "label": "tcn_07c2b_distilled_shared_u4_1008_timeout",
        "architecture": "TCN streaming shared_u4 distilled",
        "status": "timeout",
        "timeout_seconds": "7200",
        "log": Path(
            "C:/tmp/07c2b_tcn_streaming_speed_shortpath/distilled_shared_u4/"
            "hls_component_stage2_resource_distilled_shared_u4/hls_component/logs/hls_run_cosim.log"
        ),
    },
]


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


def parse_log(meta: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = Path(meta["log"])
    scan_paths = [path]
    if path.name == "hls_run_cosim.log":
        component_root = path.parent.parent
        xsim_dir = component_root / "hls" / "sim" / "verilog" / "xsim.dir"
        scan_paths.extend(sorted(xsim_dir.glob("*/xsimkernel.log")))
    progress: list[tuple[int, int, int]] = []
    elapsed_seconds = math.nan
    cpu_ms = math.nan
    memory_kb = math.nan
    memory_peak_kb = math.nan
    xsimk_line = ""
    explicit_log_wave = False
    commented_log_wave = False
    has_autoloadwcfg = False
    has_wdb = False
    has_simmode_gui = False
    ram_2p_mentions = 0
    bind_storage_mentions = 0
    pass_detected = False
    fail_detected = False
    oom_detected = False
    simulation_not_responding = False
    output_feedback_mentions = 0
    svr_active_mentions = 0
    uvm_agent_mentions = 0

    if not path.exists():
        row = {
            **meta,
            "log": str(path),
            "exists": False,
            "observed_transactions": 0,
            "declared_total_transactions": "",
            "last_sim_time_ps": "",
        }
        return row, []

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
                m = XSIMK_RE.search(line)
                if m:
                    xsimk_line = m.group(1).strip()
                stripped = line.strip()
                if stripped.startswith("#log_wave"):
                    commented_log_wave = True
                elif "log_wave" in stripped:
                    explicit_log_wave = True
                if "-autoloadwcfg" in line:
                    has_autoloadwcfg = True
                if ".wdb" in line or " -wdb " in line:
                    has_wdb = True
                if "-simmode gui" in line:
                    has_simmode_gui = True
                if "ram_2p" in line:
                    ram_2p_mentions += 1
                if "BIND_STORAGE" in line:
                    bind_storage_mentions += 1
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

    deltas: list[dict[str, object]] = []
    for prev, cur in zip(progress, progress[1:]):
        tx, total, sim_ps = cur
        prev_tx, _, prev_sim_ps = prev
        if tx <= prev_tx:
            continue
        delta_ps = sim_ps - prev_sim_ps
        deltas.append(
            {
                "label": meta["label"],
                "architecture": meta["architecture"],
                "transaction": tx,
                "declared_total_transactions": total,
                "sim_time_ps": sim_ps,
                "delta_ps": delta_ps,
                "delta_cycles_at_100mhz": delta_ps / 10000.0,
                "delta_sim_us": delta_ps / 1_000_000.0,
            }
        )

    delta_cycles = [float(row["delta_cycles_at_100mhz"]) for row in deltas]
    delta_us = [float(row["delta_sim_us"]) for row in deltas]
    observed_transactions = max((item[0] for item in progress), default=0)
    declared_total = progress[-1][1] if progress else ""
    last_sim_time_ps = progress[-1][2] if progress else ""

    timeout_seconds = meta.get("timeout_seconds") or ""
    wall_seconds_basis = elapsed_seconds
    wall_basis = "xsim_elapsed"
    if math.isnan(wall_seconds_basis) and timeout_seconds != "":
        wall_seconds_basis = float(timeout_seconds)
        wall_basis = "external_timeout_budget"

    row = {
        **meta,
        "log": str(path),
        "exists": True,
        "detected_outcome": (
            "pass"
            if pass_detected
            else "oom_fail"
            if oom_detected
            else "fail"
            if fail_detected
            else "timeout_or_incomplete"
        ),
        "pass_detected": pass_detected,
        "fail_detected": fail_detected,
        "oom_detected": oom_detected,
        "simulation_not_responding": simulation_not_responding,
        "observed_transactions": observed_transactions,
        "declared_total_transactions": declared_total,
        "last_sim_time_ps": last_sim_time_ps,
        "last_sim_time_ms": (float(last_sim_time_ps) / 1_000_000_000.0) if last_sim_time_ps != "" else "",
        "delta_count": len(deltas),
        "delta_cycles_min": min(delta_cycles) if delta_cycles else "",
        "delta_cycles_p50": percentile(delta_cycles, 0.50) if delta_cycles else "",
        "delta_cycles_p90": percentile(delta_cycles, 0.90) if delta_cycles else "",
        "delta_cycles_p99": percentile(delta_cycles, 0.99) if delta_cycles else "",
        "delta_cycles_max": max(delta_cycles) if delta_cycles else "",
        "delta_sim_us_min": min(delta_us) if delta_us else "",
        "delta_sim_us_p50": percentile(delta_us, 0.50) if delta_us else "",
        "delta_sim_us_p90": percentile(delta_us, 0.90) if delta_us else "",
        "delta_sim_us_p99": percentile(delta_us, 0.99) if delta_us else "",
        "delta_sim_us_max": max(delta_us) if delta_us else "",
        "wall_seconds_basis": wall_seconds_basis if not math.isnan(wall_seconds_basis) else "",
        "wall_seconds_basis_type": wall_basis if not math.isnan(wall_seconds_basis) else "",
        "wall_seconds_per_observed_transaction": (
            wall_seconds_basis / observed_transactions
            if observed_transactions and not math.isnan(wall_seconds_basis)
            else ""
        ),
        "xsim_cpu_ms": cpu_ms if not math.isnan(cpu_ms) else "",
        "xsim_memory_kb": memory_kb if not math.isnan(memory_kb) else "",
        "xsim_memory_peak_kb": memory_peak_kb if not math.isnan(memory_peak_kb) else "",
        "xsimk_line": xsimk_line,
        "explicit_log_wave": explicit_log_wave,
        "commented_log_wave": commented_log_wave,
        "has_autoloadwcfg": has_autoloadwcfg,
        "has_wdb": has_wdb,
        "has_simmode_gui": has_simmode_gui,
        "ram_2p_mentions": ram_2p_mentions,
        "bind_storage_mentions": bind_storage_mentions,
        "output_feedback_mentions": output_feedback_mentions,
        "svr_active_mentions": svr_active_mentions,
        "uvm_agent_mentions": uvm_agent_mentions,
    }
    return row, deltas


def load_reduced_manifest(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, object]] = []
    for row in rows:
        name = str(row["name"])
        comp = Path(row["component"])
        suffix = f"{row['windows']}window"
        if row.get("event_limit") is not None:
            suffix = f"event{row['event_limit']}"
        out.append(
            {
                "label": f"tcn_07c2c_{name}_{suffix}",
                "architecture": f"TCN streaming {name} reduced {suffix}",
                "status": "reduced_cosim_attempt",
                "timeout_seconds": "",
                "log": comp / "hls_component" / "logs" / "hls_run_cosim.log",
            }
        )
    return out


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
    )
    args = parser.parse_args()

    manifest_paths = args.manifest
    if manifest_paths is None:
        manifest_paths = [
            ROOT / "artifacts" / "reduced_cosim_manifest_1window.json",
            ROOT / "artifacts" / "reduced_cosim_manifest_event174.json",
        ]

    metas = list(KNOWN_LOGS)
    for manifest in manifest_paths:
        metas.extend(load_reduced_manifest(manifest))
    summary_rows: list[dict[str, object]] = []
    all_deltas: list[dict[str, object]] = []
    for meta in metas:
        summary, deltas = parse_log(meta)
        summary_rows.append(summary)
        all_deltas.extend(deltas)

    write_csv(RESULTS / "xsim_transaction_time_stats.csv", summary_rows)
    write_csv(RESULTS / "xsim_transaction_deltas.csv", all_deltas)
    print(f"wrote {RESULTS / 'xsim_transaction_time_stats.csv'}")
    print(f"wrote {RESULTS / 'xsim_transaction_deltas.csv'}")


if __name__ == "__main__":
    main()
