from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s_unroll16_verification_closure"
SOURCE_UNROLL16 = PROJECT_ROOT / "n2ew_era" / "06l_phc_lite_pipeline_budget_sweep" / "variants" / "factor_16" / "hls_src"
SOURCE_PNR_TCL = (
    PROJECT_ROOT
    / "n2ew_era"
    / "06l_phc_lite_pipeline_budget_sweep"
    / "pnr_unroll16_90mhz"
    / "hls_component"
    / "hls_component"
    / "hls"
    / "impl"
    / "verilog"
    / "run_direct_phc_step_pnr_custom_clock.tcl"
)
SOURCE_SEED2_VECTORS = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results" / "phc_step_test_vectors.csv"
SOURCE_SEED2_WEIGHTS = SOURCE_UNROLL16 / "phc_weights.hpp"
SOURCE_CROSS_SEED = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "cross_seed"

SEEDS = [0, 1, 2]
CLOCK_PERIOD_NS = "11.111"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def win_abs(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_root(seed: int) -> Path:
    return ROOT / f"seed{seed}"


def seed_paths(seed: int) -> dict[str, Path]:
    base = seed_root(seed)
    paths = {
        "root": base,
        "hls_src": base / "hls_src",
        "hls_component": base / "hls_component",
        "results": base / "results",
        "models": base / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def source_vectors(seed: int) -> Path:
    if seed == 2:
        return SOURCE_SEED2_VECTORS
    return SOURCE_CROSS_SEED / f"seed{seed}" / "results" / f"phc_step_test_vectors_seed{seed}.csv"


def source_weights(seed: int) -> Path:
    if seed == 2:
        return SOURCE_SEED2_WEIGHTS
    return SOURCE_CROSS_SEED / f"seed{seed}" / "hls_src" / "phc_weights.hpp"


def copy_checked(src: Path, dst: Path) -> dict[str, object]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "src": rel(src),
        "dst": rel(dst),
        "src_size_bytes": src.stat().st_size,
        "dst_size_bytes": dst.stat().st_size,
        "src_sha256": sha256_file(src),
        "dst_sha256": sha256_file(dst),
        "sha256_match": sha256_file(src) == sha256_file(dst),
    }


def write_hls_config(seed: int, paths: dict[str, Path]) -> Path:
    vectors = paths["results"] / f"phc_step_test_vectors_seed{seed}.csv"
    csim_out = paths["results"] / f"phc_step_csim_out_seed{seed}.csv"
    cosim_out = paths["results"] / f"phc_step_cosim_out_seed{seed}.csv"
    text = "\n".join(
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
    out = paths["hls_component"] / "hls_config.cfg"
    out.write_text(text, encoding="utf-8")
    return out


def write_csynth_tcl(paths: dict[str, Path]) -> Path:
    text = "\n".join(
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
    out = paths["hls_component"] / "run_csynth_only_90mhz.tcl"
    out.write_text(text, encoding="utf-8")
    return out


def prepare_seed(seed: int) -> dict[str, object]:
    paths = seed_paths(seed)
    copied = []
    for name in ["phc_step_hls.cpp", "phc_step_hls.hpp", "phc_step_testbench.cpp"]:
        copied.append(copy_checked(SOURCE_UNROLL16 / name, paths["hls_src"] / name))
    copied.append(copy_checked(source_weights(seed), paths["hls_src"] / "phc_weights.hpp"))
    copied.append(copy_checked(source_vectors(seed), paths["results"] / f"phc_step_test_vectors_seed{seed}.csv"))
    copied.append(copy_checked(SOURCE_PNR_TCL, paths["hls_component"] / "run_direct_phc_step_pnr_custom_clock.tcl"))
    config = write_hls_config(seed, paths)
    csynth_tcl = write_csynth_tcl(paths)
    manifest = {
        "phase": "06s",
        "seed": seed,
        "clock_period_ns": CLOCK_PERIOD_NS,
        "clock_frequency_mhz": 1000.0 / float(CLOCK_PERIOD_NS),
        "source_unroll16_hls_src": rel(SOURCE_UNROLL16),
        "source_weights": rel(source_weights(seed)),
        "source_vectors": rel(source_vectors(seed)),
        "hls_config": rel(config),
        "csynth_tcl": rel(csynth_tcl),
        "copied_files": copied,
        "new5_used": False,
        "source_modified": False,
    }
    (paths["results"] / f"prepare_manifest_seed{seed}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    manifests = [prepare_seed(seed) for seed in SEEDS]
    (ROOT / "results" / "prepare_manifest_all_seeds.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    for manifest in manifests:
        print(f"prepared seed{manifest['seed']}: {manifest['hls_config']}")


if __name__ == "__main__":
    main()
