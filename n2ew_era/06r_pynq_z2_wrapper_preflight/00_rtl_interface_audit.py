from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "n2ew_era" / "06r_pynq_z2_wrapper_preflight" / "results"

DESIGNS = {
    "06f_v8_frozen_100mhz": ROOT
    / "n2ew_era"
    / "06f_phc_lite_manual_hls"
    / "hls_component"
    / "hls_component"
    / "hls"
    / "impl"
    / "verilog"
    / "phc_step.v",
    "06l_unroll16_90mhz": ROOT
    / "n2ew_era"
    / "06l_phc_lite_pipeline_budget_sweep"
    / "pnr_unroll16_90mhz"
    / "hls_component"
    / "hls_component"
    / "hls"
    / "impl"
    / "verilog"
    / "phc_step.v",
}


PORT_RE = re.compile(r"^(input|output)\s+(?:\[(\d+):(\d+)\]\s+)?(\w+);")
CORE_RE = re.compile(r'HLS_SYN_CLOCK=([^,}]+).*?HLS_SYN_LAT=([^,}]+).*?HLS_SYN_MEM=([^,}]+).*?HLS_SYN_DSP=([^,}]+).*?HLS_SYN_FF=([^,}]+).*?HLS_SYN_LUT=([^,}]+)')


def parse_ports(path: Path) -> list[dict[str, str]]:
    ports: list[dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        match = PORT_RE.match(line.strip())
        if not match:
            continue
        direction, msb, lsb, name = match.groups()
        if msb is None:
            width = 1
            range_text = ""
        else:
            width = abs(int(msb) - int(lsb)) + 1
            range_text = f"[{msb}:{lsb}]"
        ports.append(
            {
                "name": name,
                "direction": direction,
                "width_bits": str(width),
                "range": range_text,
                "line": str(line_no),
            }
        )
    return ports


def parse_core_info(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = CORE_RE.search(text)
    if not match:
        return {}
    keys = ["hls_syn_clock_ns", "hls_syn_latency_cycles", "hls_syn_mem", "hls_syn_dsp", "hls_syn_ff", "hls_syn_lut"]
    return dict(zip(keys, match.groups()))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = {name: parse_ports(path) for name, path in DESIGNS.items()}
    core_rows = []
    for name, path in DESIGNS.items():
        rows = [{"design": name, "rtl_path": str(path.relative_to(ROOT)), **row} for row in parsed[name]]
        write_csv(OUT_DIR / f"{name}_port_list.csv", rows, ["design", "rtl_path", "name", "direction", "width_bits", "range", "line"])
        info = parse_core_info(path)
        core_rows.append({"design": name, "rtl_path": str(path.relative_to(ROOT)), **info})

    write_csv(
        OUT_DIR / "rtl_core_generation_info.csv",
        core_rows,
        ["design", "rtl_path", "hls_syn_clock_ns", "hls_syn_latency_cycles", "hls_syn_mem", "hls_syn_dsp", "hls_syn_ff", "hls_syn_lut"],
    )

    names = sorted(set(row["name"] for rows in parsed.values() for row in rows))
    rows = []
    by_design = {design: {row["name"]: row for row in design_rows} for design, design_rows in parsed.items()}
    for name in names:
        left = by_design["06f_v8_frozen_100mhz"].get(name)
        right = by_design["06l_unroll16_90mhz"].get(name)
        if left and right:
            status = "same" if (left["direction"], left["width_bits"]) == (right["direction"], right["width_bits"]) else "changed"
        elif left:
            status = "only_06f"
        else:
            status = "only_06l"
        rows.append(
            {
                "port": name,
                "status": status,
                "dir_06f": left["direction"] if left else "",
                "bits_06f": left["width_bits"] if left else "",
                "dir_06l": right["direction"] if right else "",
                "bits_06l": right["width_bits"] if right else "",
            }
        )
    write_csv(OUT_DIR / "rtl_port_diff.csv", rows, ["port", "status", "dir_06f", "bits_06f", "dir_06l", "bits_06l"])

    diff_count = sum(1 for row in rows if row["status"] != "same")
    print(f"ports_06f={len(parsed['06f_v8_frozen_100mhz'])}")
    print(f"ports_06l={len(parsed['06l_unroll16_90mhz'])}")
    print(f"diff_count={diff_count}")
    for row in rows:
        if row["status"] != "same":
            print(row)


if __name__ == "__main__":
    main()
