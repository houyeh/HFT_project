from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HLS_SRC = ROOT / "hls_src"
RESULTS = ROOT / "results"

VARIANTS = {
    "scratch": {
        "component": ROOT / "hls_component_scratch",
        "cpp": HLS_SRC / "tcn_prepad_hls_scratch.cpp",
        "vectors": RESULTS / "tcn_prepad_scratch_test_vectors.csv",
        "csim_out": RESULTS / "tcn_prepad_scratch_csim_out.csv",
        "cosim_out": RESULTS / "tcn_prepad_scratch_cosim_out.csv",
    },
    "distilled": {
        "component": ROOT / "hls_component_distilled",
        "cpp": HLS_SRC / "tcn_prepad_hls_distilled.cpp",
        "vectors": RESULTS / "tcn_prepad_distilled_test_vectors.csv",
        "csim_out": RESULTS / "tcn_prepad_distilled_csim_out.csv",
        "cosim_out": RESULTS / "tcn_prepad_distilled_cosim_out.csv",
    },
}


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def write_config(variant: str, cfg: dict[str, Path]) -> None:
    cfg["component"].mkdir(parents=True, exist_ok=True)
    text = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=tcn_prepad_infer
syn.file={posix(cfg["cpp"])}
syn.file={posix(HLS_SRC / "tcn_prepad_hls.hpp")}
syn.file={posix(HLS_SRC / "tcn_prepad_hls_impl.hpp")}
syn.file={posix(HLS_SRC / f"tcn_weights_{variant}.hpp")}
tb.file={posix(HLS_SRC / "tcn_prepad_testbench.cpp")}
clock=10ns
csim.argv={posix(cfg["vectors"])} {posix(cfg["csim_out"])}
cosim.argv={posix(cfg["vectors"])} {posix(cfg["cosim_out"])}
"""
    (cfg["component"] / "hls_config.cfg").write_text(text, encoding="utf-8")


def main() -> None:
    for variant, cfg in VARIANTS.items():
        write_config(variant, cfg)
        print(f"{variant}: wrote {cfg['component'] / 'hls_config.cfg'}")


if __name__ == "__main__":
    main()

