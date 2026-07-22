from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06l_phc_lite_pipeline_budget_sweep"
SOURCE_06F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
BASE_HLS_SRC = ROOT / "hls_src"
BASE_CONFIG = ROOT / "hls_component" / "hls_config.cfg"
VARIANTS = ROOT / "variants"
RESULTS = ROOT / "results"
FACTORS = [1, 2, 4, 8, 16]

LINEAR_FUNCTIONS = {
    "linear_16_16_fixed",
    "linear_16_16_acc",
    "linear_16_16_fixed_no_bias",
    "linear_16_2_acc",
}


def ensure_base_copy() -> None:
    BASE_HLS_SRC.mkdir(parents=True, exist_ok=True)
    (ROOT / "hls_component").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for src in (SOURCE_06F / "hls_src").glob("*"):
        if src.is_file():
            shutil.copy2(src, BASE_HLS_SRC / src.name)
    base_cfg = (SOURCE_06F / "hls_component" / "hls_config.cfg").read_text(encoding="utf-8")
    base_cfg = base_cfg.replace(
        (SOURCE_06F / "results" / "phc_step_cosim_out.csv").as_posix(),
        (RESULTS / "root_baseline" / "phc_step_cosim_out.csv").as_posix(),
    )
    base_cfg = base_cfg.replace(
        (SOURCE_06F / "results" / "phc_step_csim_out.csv").as_posix(),
        (RESULTS / "root_baseline" / "phc_step_csim_out.csv").as_posix(),
    )
    BASE_CONFIG.write_text(base_cfg, encoding="utf-8")


def transform_cpp(base_cpp: str, factor: int) -> str:
    lines = base_cpp.splitlines()
    out: list[str] = []
    in_linear = False
    brace_depth = 0
    seen_open_brace = False
    pending_inner_for = False
    pending_function_name = ""

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(f"void {name}(") for name in LINEAR_FUNCTIONS):
            in_linear = True
            pending_function_name = stripped.split("(")[0].replace("void ", "")
            brace_depth = 0
            seen_open_brace = False

        out.append(line)

        if in_linear:
            if "{" in line:
                seen_open_brace = True
            brace_depth += line.count("{") - line.count("}")
            if stripped == "#pragma HLS INLINE":
                if factor > 1:
                    out.append(f"#pragma HLS ARRAY_PARTITION variable=w cyclic factor={factor} dim=2")
                else:
                    out.append("// factor=1: no ARRAY_PARTITION; this intentionally matches the v8 baseline semantics.")
            if pending_inner_for and stripped == "#pragma HLS PIPELINE off":
                if factor > 1:
                    out.append(f"#pragma HLS UNROLL factor={factor}")
                else:
                    out.append("// factor=1: no UNROLL; this intentionally matches the v8 baseline semantics.")
                pending_inner_for = False
            elif stripped == "for (int i = 0; i < 16; ++i) {":
                pending_inner_for = True
            if seen_open_brace and brace_depth <= 0 and pending_function_name:
                in_linear = False
                pending_function_name = ""
                seen_open_brace = False
                pending_inner_for = False

    header = [
        "// Phase 06L generated variant.",
        f"// Inner i-loop unroll factor: {factor}.",
        "// Outer j-loops and all non-linear/elementwise/layernorm loops retain explicit PIPELINE off pragmas.",
        "// Do not edit this generated file by hand; edit 00_generate_factor_variants.py instead.",
        "",
    ]
    return "\n".join(header + out) + "\n"


def config_text(factor: int) -> str:
    result_dir = RESULTS / f"factor_{factor}"
    result_dir.mkdir(parents=True, exist_ok=True)
    test_vectors = SOURCE_06F / "results" / "phc_step_test_vectors.csv"
    csim_out = result_dir / "phc_step_csim_out.csv"
    cosim_out = result_dir / "phc_step_cosim_out.csv"
    return "\n".join(
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
            "clock=10ns",
            f"cosim.argv={test_vectors.as_posix()} {cosim_out.as_posix()}",
            f"csim.argv={test_vectors.as_posix()} {csim_out.as_posix()}",
            "",
        ]
    )


def generate_variant(factor: int) -> None:
    variant = VARIANTS / f"factor_{factor}"
    hls_src = variant / "hls_src"
    hls_component = variant / "hls_component"
    hls_src.mkdir(parents=True, exist_ok=True)
    hls_component.mkdir(parents=True, exist_ok=True)

    for name in ["phc_step_hls.hpp", "phc_weights.hpp", "phc_step_testbench.cpp"]:
        shutil.copy2(BASE_HLS_SRC / name, hls_src / name)

    base_cpp = (BASE_HLS_SRC / "phc_step_hls.cpp").read_text(encoding="utf-8")
    (hls_src / "phc_step_hls.cpp").write_text(transform_cpp(base_cpp, factor), encoding="utf-8")
    (hls_component / "hls_config.cfg").write_text(config_text(factor), encoding="utf-8")


def write_instructions() -> None:
    lines = [
        "# PHC-lite Pipeline Budget Sweep Instructions",
        "",
        "This phase prepares synthesis variants only. It does not modify `n2ew_era/06f_phc_lite_manual_hls/`.",
        "",
        "For each factor, run from the corresponding variant component directory:",
        "",
        "```powershell",
        "cd C:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\n2ew_era\\06l_phc_lite_pipeline_budget_sweep\\variants\\factor_<N>\\hls_component",
        "& 'C:\\AMDDesignTools\\2025.2\\Vitis\\bin\\vitis-run.bat' --mode hls --csim --config hls_config.cfg --work_dir hls_component",
        "& 'C:\\AMDDesignTools\\2025.2\\Vitis\\bin\\vitis-run.bat' --mode hls --syn  --config hls_config.cfg --work_dir hls_component",
        "```",
        "",
        "Only if DSP < 220, LUT < 53200, FF < 106400, and BRAM_18K < 280 should you run:",
        "",
        "```powershell",
        "& 'C:\\AMDDesignTools\\2025.2\\Vitis\\bin\\vitis-run.bat' --mode hls --cosim --config hls_config.cfg --work_dir hls_component",
        "```",
        "",
        "After running the desired factors, collect the table with:",
        "",
        "```powershell",
        "python n2ew_era\\06l_phc_lite_pipeline_budget_sweep\\01_collect_sweep_results.py",
        "```",
        "",
        "Factor 1 intentionally preserves the v8 semantics: no inner unroll and no weight partition. Factors 2/4/8/16 add `#pragma HLS UNROLL factor=<N>` to the inner `i` loops of the four linear helper functions and `#pragma HLS ARRAY_PARTITION variable=w cyclic factor=<N> dim=2` to the corresponding formal weight argument.",
        "",
        "Do not raise `PHC_ABS_TOL`, do not edit the 06F baseline, and do not use these synthesis numbers as PnL claims.",
    ]
    (ROOT / "RUN_SWEEP_INSTRUCTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_base_copy()
    for factor in FACTORS:
        generate_variant(factor)
    write_instructions()
    print(f"Generated variants under {VARIANTS}")
    for factor in FACTORS:
        print(VARIANTS / f"factor_{factor}" / "hls_component" / "hls_config.cfg")


if __name__ == "__main__":
    main()
