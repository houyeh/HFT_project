from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SHORT_ROOT_BASE = Path("C:/tmp/07c2_tcn_streaming_closure_shortpath")
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def wrapper_name(variant: str) -> str:
    return f"tcn_stage2_resource_{variant}_shared_u1.cpp"


def prepare_variant(variant: str, clean: bool) -> dict[str, object]:
    if variant not in {"scratch", "distilled"}:
        raise ValueError(f"unknown variant: {variant}")

    short_root = SHORT_ROOT_BASE / variant
    comp = short_root / f"hls_component_stage2_resource_{variant}_shared_u1"
    short_src = short_root / "hls_src"
    short_results = short_root / "results"

    if clean and short_root.exists():
        shutil.rmtree(short_root)
    short_src.mkdir(parents=True, exist_ok=True)
    short_results.mkdir(parents=True, exist_ok=True)
    comp.mkdir(parents=True, exist_ok=True)

    for path in HLS_SRC.glob("*"):
        if path.is_file():
            shutil.copy2(path, short_src / path.name)

    vector_src = RESULTS / f"stage2_streaming_{variant}_test_vectors_first4_windows.csv"
    if not vector_src.exists():
        raise FileNotFoundError(vector_src)
    vector_dst = short_results / vector_src.name
    shutil.copy2(vector_src, vector_dst)

    summary_dst = short_results / f"stage2_closure_{variant}_shared_u1_event1008_tb_summary.csv"

    cfg_text = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=tcn_streaming_step
syn.file={posix(short_src / wrapper_name(variant))}
syn.file={posix(short_src / "tcn_prepad_hls.hpp")}
syn.file={posix(short_src / "tcn_stage2_streaming_hls.hpp")}
syn.file={posix(short_src / "tcn_stage2_resource_impl.hpp")}
syn.file={posix(short_src / f"tcn_weights_{variant}.hpp")}
tb.file={posix(short_src / "tcn_stage2_resource_testbench.cpp")}
clock=10ns
csim.argv={posix(vector_dst)} {posix(summary_dst)}
cosim.argv={posix(vector_dst)} {posix(summary_dst)}
"""
    cfg_path = comp / "hls_config.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    tcl_text = "\n".join(
        [
            "open_project -reset hls_component",
            "set_top tcn_streaming_step",
            f"add_files ../hls_src/{wrapper_name(variant)}",
            "add_files ../hls_src/tcn_prepad_hls.hpp",
            "add_files ../hls_src/tcn_stage2_streaming_hls.hpp",
            "add_files ../hls_src/tcn_stage2_resource_impl.hpp",
            f"add_files ../hls_src/tcn_weights_{variant}.hpp",
            "add_files -tb ../hls_src/tcn_stage2_resource_testbench.cpp",
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
        "variant": variant,
        "short_root": str(short_root),
        "component": str(comp),
        "hls_config": str(cfg_path),
        "csynth_tcl": str(tcl_path),
        "subset_vectors": str(vector_dst),
        "tb_summary": str(summary_dst),
        "streaming_top_event_vectors": 1008,
        "readout_decision_rows": 4,
        "new5_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["scratch", "distilled", "both"], default="both")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    variants = ["scratch", "distilled"] if args.variant == "both" else [args.variant]
    rows = [prepare_variant(variant, args.clean) for variant in variants]
    out = RESULTS / "stage2_shortpath_manifest.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    merged = {row["variant"]: row for row in old if isinstance(row, dict) and "variant" in row}
    for row in rows:
        merged[row["variant"]] = row
        print(f"{row['variant']}: prepared {row['component']}")
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
