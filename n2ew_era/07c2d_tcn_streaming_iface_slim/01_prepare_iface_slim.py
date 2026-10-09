from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_RESULTS = ROOT.parent / "07c2b_tcn_streaming_speed" / "results"
HLS_SRC = ROOT / "hls_src"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SHORT_ROOT_BASE = Path("C:/tmp/07c2d")


VARIANTS = {
    "scratch_shared_u4": ("scratch", "tcn_stage2_resource_scratch_shared_u4.cpp", "su4"),
    "distilled_shared_u4": ("distilled", "tcn_stage2_resource_distilled_shared_u4.cpp", "du4"),
}

TB_MODES = {
    "csim_debug": "tcn_stage2_iface_slim_csim_testbench.cpp",
    "cosim_readout": "tcn_stage2_iface_slim_cosim_testbench.cpp",
}


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def subset_vectors(variant: str, windows: int) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = SOURCE_RESULTS / f"stage2_streaming_{variant}_test_vectors_first4_windows.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    dest = RESULTS / f"stage2_streaming_{variant}_test_vectors_first{windows}_window.csv"
    with source.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        with dest.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, lineterminator="\n")
            header = next(reader)
            writer.writerow(header)
            for idx, row in enumerate(reader):
                if idx >= windows:
                    break
                writer.writerow(row)
    return dest


def prepare_component(
    name: str,
    windows: int,
    event_limit: int | None,
    tb_mode: str,
    clean: bool,
) -> dict[str, object]:
    variant, wrapper, short_name = VARIANTS[name]
    tb_name = TB_MODES[tb_mode]
    vector_src = subset_vectors(variant, windows)

    event_count_per_window = 252 if event_limit is None else event_limit
    scale_tag = f"{windows}window" if event_limit is None else f"event{event_limit}"
    short_scale = f"w{windows}" if event_limit is None else f"e{event_limit}"
    short_root = SHORT_ROOT_BASE / f"{short_name}_{tb_mode}_{short_scale}"
    comp = short_root / "c"
    short_src = short_root / "s"
    short_results = short_root / "r"

    if clean and short_root.exists():
        shutil.rmtree(short_root)
    short_src.mkdir(parents=True, exist_ok=True)
    short_results.mkdir(parents=True, exist_ok=True)
    comp.mkdir(parents=True, exist_ok=True)

    for path in HLS_SRC.glob("*"):
        if path.is_file():
            shutil.copy2(path, short_src / path.name)

    vector_dst = short_results / vector_src.name
    shutil.copy2(vector_src, vector_dst)
    summary_events = windows * event_count_per_window
    summary_dst = short_results / f"stage2_iface_slim_{tb_mode}{summary_events}_{name}_{scale_tag}_tb_summary.csv"
    argv_tail = "" if event_limit is None else f" {event_limit}"

    cfg_text = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=tcn_streaming_step
syn.file={posix(short_src / wrapper)}
syn.file={posix(short_src / "tcn_prepad_hls.hpp")}
syn.file={posix(short_src / "tcn_stage2_streaming_hls.hpp")}
syn.file={posix(short_src / "tcn_stage2_resource_impl.hpp")}
syn.file={posix(short_src / f"tcn_weights_{variant}.hpp")}
tb.file={posix(short_src / tb_name)}
clock=10ns
csim.argv={posix(vector_dst)} {posix(summary_dst)}{argv_tail}
cosim.argv={posix(vector_dst)} {posix(summary_dst)}{argv_tail}
"""
    cfg_path = comp / "hls_config.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    tcl_text = "\n".join(
        [
            "open_project -reset hls_component",
            "set_top tcn_streaming_step",
            f"add_files ../s/{wrapper}",
            "add_files ../s/tcn_prepad_hls.hpp",
            "add_files ../s/tcn_stage2_streaming_hls.hpp",
            "add_files ../s/tcn_stage2_resource_impl.hpp",
            f"add_files ../s/tcn_weights_{variant}.hpp",
            f"add_files -tb ../s/{tb_name}",
            "open_solution -flow_target vivado hls",
            "set_part xc7z020-clg400-1",
            "create_clock -period 10 -name default",
            "config_compile -pipeline_loops 0",
            "csynth_design",
            "exit",
            "",
        ]
    )
    tcl_path = comp / "run_csynth_only.tcl"
    tcl_path.write_text(tcl_text, encoding="utf-8")

    return {
        "name": name,
        "variant": variant,
        "wrapper": wrapper,
        "testbench": tb_name,
        "tb_mode": tb_mode,
        "windows": windows,
        "event_limit": event_limit or 252,
        "scale_tag": scale_tag,
        "component": str(comp),
        "hls_config": str(cfg_path),
        "csynth_tcl": str(tcl_path),
        "subset_vectors_workspace": str(vector_src),
        "subset_vectors_shortpath": str(vector_dst),
        "tb_summary_shortpath": str(summary_dst),
        "tb_summary_workspace": str(RESULTS / summary_dst.name),
        "streaming_top_event_vectors": windows * event_count_per_window,
        "expected_rtl_transactions": windows * (event_count_per_window + 1),
        "readout_decision_rows": windows if event_count_per_window >= 252 else 0,
        "new5_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS) + ["all"], default="all")
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--tb-mode", choices=sorted(TB_MODES), required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.windows < 1 or args.windows > 4:
        raise ValueError("--windows must be in [1, 4] for the existing first4 vector files")
    if args.event_limit is not None and not (1 <= args.event_limit <= 252):
        raise ValueError("--event-limit must be in [1, 252]")

    names = sorted(VARIANTS) if args.variant == "all" else [args.variant]
    rows = [
        prepare_component(name, args.windows, args.event_limit, args.tb_mode, args.clean)
        for name in names
    ]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    scale_tag = f"{args.windows}window" if args.event_limit is None else f"event{args.event_limit}"
    out = ARTIFACTS / f"iface_slim_manifest_{args.tb_mode}_{scale_tag}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        print(f"{row['name']}: prepared {row['component']}")
        print(f"  tb_mode: {row['tb_mode']}")
        print(f"  expected RTL transactions: {row['expected_rtl_transactions']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
