from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s_unroll16_verification_closure"
SOURCE_SEED2 = ROOT / "seed2"
SUBSET = ROOT / "seed2_cosim1000"
N_ROWS = 1000
CLOCK_PERIOD_NS = "11.111"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def win_abs(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def copy_tree_src() -> None:
    src = SOURCE_SEED2 / "hls_src"
    dst = SUBSET / "hls_src"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_subset_vectors() -> Path:
    src = SOURCE_SEED2 / "results" / "phc_step_test_vectors_seed2.csv"
    dst = SUBSET / "results" / "phc_step_test_vectors_seed2_first1000.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as f_in, dst.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out, lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        count = 0
        for row in reader:
            if count >= N_ROWS:
                break
            writer.writerow(row)
            count += 1
    if count != N_ROWS:
        raise RuntimeError(f"expected {N_ROWS} rows, wrote {count}")
    return dst


def write_hls_files(vectors: Path) -> None:
    hls_component = SUBSET / "hls_component"
    hls_component.mkdir(parents=True, exist_ok=True)
    csim_out = SUBSET / "results" / "phc_step_csim_out_seed2_first1000.csv"
    cosim_out = SUBSET / "results" / "phc_step_cosim_out_seed2_first1000.csv"
    config = "\n".join(
        [
            "part=xc7z020clg400-1",
            "",
            "[hls]",
            "flow_target=vivado",
            "package.output.format=ip_catalog",
            "package.output.syn=false",
            "syn.top=phc_step",
            "syn.file=../hls_src/phc_step_hls.cpp",
            "syn.file=../hls_src/phc_step_hls.hpp",
            "syn.file=../hls_src/phc_weights.hpp",
            "tb.file=../hls_src/phc_step_testbench.cpp",
            f"clock={CLOCK_PERIOD_NS}ns",
            f"cosim.argv={win_abs(vectors)} {win_abs(cosim_out)}",
            f"csim.argv={win_abs(vectors)} {win_abs(csim_out)}",
            "",
        ]
    )
    (hls_component / "hls_config.cfg").write_text(config, encoding="utf-8")
    tcl = "\n".join(
        [
            "open_project hls_component",
            "set_top phc_step",
            "add_files ../hls_src/phc_step_hls.cpp",
            "add_files ../hls_src/phc_step_hls.hpp",
            "add_files ../hls_src/phc_weights.hpp",
            "add_files -tb ../hls_src/phc_step_testbench.cpp",
            "open_solution -flow_target vivado hls",
            "set_part xc7z020-clg400-1",
            f"create_clock -period {CLOCK_PERIOD_NS} -name default",
            "csynth_design",
            "exit",
            "",
        ]
    )
    (hls_component / "run_csynth_only_90mhz.tcl").write_text(tcl, encoding="utf-8")


def main() -> None:
    copy_tree_src()
    vectors = write_subset_vectors()
    write_hls_files(vectors)
    manifest = {
        "phase": "06s",
        "purpose": "seed2 unroll16 1000-row cosim fallback after full 25,600-row cosim timeout",
        "source_seed_dir": rel(SOURCE_SEED2),
        "subset_dir": rel(SUBSET),
        "rows": N_ROWS,
        "clock_period_ns": CLOCK_PERIOD_NS,
        "vectors": rel(vectors),
        "new5_used": False,
    }
    (SUBSET / "results" / "cosim1000_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
