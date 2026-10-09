from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"

VARIANTS = ("scratch", "distilled")
SETTINGS = (
    {"mode": "unroll_only", "suffix": "u2", "unroll": 2, "shared_history": False},
    {"mode": "unroll_only", "suffix": "u1", "unroll": 1, "shared_history": False},
    {"mode": "shared_history", "suffix": "shared_u1", "unroll": 1, "shared_history": True},
)


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def component_dir(variant: str, suffix: str) -> Path:
    return ROOT / f"hls_component_stage2_resource_{variant}_{suffix}"


def wrapper_cpp(variant: str, suffix: str) -> Path:
    return HLS_SRC / f"tcn_stage2_resource_{variant}_{suffix}.cpp"


def vectors_path(variant: str) -> Path:
    return RESULTS / f"stage2_streaming_{variant}_test_vectors.csv"


def summary_path(variant: str, suffix: str) -> Path:
    return RESULTS / f"stage2_resource_{variant}_{suffix}_csim_summary.csv"


def write_config(variant: str, setting: dict[str, Any], clean: bool) -> dict[str, Any]:
    suffix = str(setting["suffix"])
    unroll = int(setting["unroll"])
    comp = component_dir(variant, suffix)
    if clean and comp.exists():
        shutil.rmtree(comp)
    comp.mkdir(parents=True, exist_ok=True)

    vector_csv = vectors_path(variant)
    if not vector_csv.exists():
        raise FileNotFoundError(f"missing Stage 2 vector CSV: {vector_csv}")

    weight = HLS_SRC / f"tcn_weights_{variant}.hpp"
    if not weight.exists():
        raise FileNotFoundError(f"missing weight header: {weight}")

    text = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=tcn_streaming_step
syn.file={posix(wrapper_cpp(variant, suffix))}
syn.file={posix(HLS_SRC / "tcn_prepad_hls.hpp")}
syn.file={posix(HLS_SRC / "tcn_stage2_streaming_hls.hpp")}
syn.file={posix(HLS_SRC / "tcn_stage2_resource_impl.hpp")}
syn.file={posix(weight)}
tb.file={posix(HLS_SRC / "tcn_stage2_resource_testbench.cpp")}
clock=10ns
csim.argv={posix(vector_csv)} {posix(summary_path(variant, suffix))}
"""
    cfg_path = comp / "hls_config.cfg"
    cfg_path.write_text(text, encoding="utf-8")

    tcl_text = "\n".join(
        [
            "open_project -reset hls_component",
            "set_top tcn_streaming_step",
            f"add_files ../hls_src/{wrapper_cpp(variant, suffix).name}",
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
        "mode": setting["mode"],
        "suffix": suffix,
        "unroll": unroll,
        "shared_history_storage": bool(setting["shared_history"]),
        "component": str(comp),
        "hls_config": str(cfg_path),
        "csynth_tcl": str(tcl_path),
        "wrapper_cpp": str(wrapper_cpp(variant, suffix)),
        "impl_header": str(HLS_SRC / "tcn_stage2_resource_impl.hpp"),
        "testbench": str(HLS_SRC / "tcn_stage2_resource_testbench.cpp"),
        "vectors": str(vector_csv),
        "csim_summary": str(summary_path(variant, suffix)),
        "new5_used": False,
        "resource_only_change": (
            "outer/inner accumulation PIPELINE off; inner TCN_STAGE2_CHANNEL_UNROLL set by wrapper"
            + ("; history buffers BIND_STORAGE ram_2p bram and history write/reset loops not unrolled" if setting["shared_history"] else "")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove and recreate only resource-sweep component dirs.")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in VARIANTS:
        for setting in SETTINGS:
            item = write_config(variant, setting, clean=args.clean)
            manifest.append(item)
            print(f"{variant} {item['suffix']}: wrote {item['hls_config']}")

    manifest_path = RESULTS / "stage2_resource_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
