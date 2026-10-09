from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07c_tcn_streaming_manual_hls"
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"
PHASE07B = PROJECT_ROOT / "n2ew_era" / "07b_tcn_prepad_manual_hls"
STAGE0_SCRIPT = ROOT / "01_stage0_streaming_python_validation.py"

VARIANTS: dict[str, dict[str, Path]] = {
    "scratch": {
        "component": ROOT / "hls_component_stage2_scratch",
        "cpp": HLS_SRC / "tcn_stage2_streaming_scratch.cpp",
        "weight_src": PHASE07B / "hls_src" / "tcn_weights_scratch.hpp",
        "weight_dst": HLS_SRC / "tcn_weights_scratch.hpp",
        "vectors": RESULTS / "stage2_streaming_scratch_test_vectors.csv",
        "block_csim_out": RESULTS / "stage2_streaming_scratch_block_csim_out.csv",
        "readout_csim_out": RESULTS / "stage2_streaming_scratch_readout_csim_out.csv",
    },
    "distilled": {
        "component": ROOT / "hls_component_stage2_distilled",
        "cpp": HLS_SRC / "tcn_stage2_streaming_distilled.cpp",
        "weight_src": PHASE07B / "hls_src" / "tcn_weights_distilled.hpp",
        "weight_dst": HLS_SRC / "tcn_weights_distilled.hpp",
        "vectors": RESULTS / "stage2_streaming_distilled_test_vectors.csv",
        "block_csim_out": RESULTS / "stage2_streaming_distilled_block_csim_out.csv",
        "readout_csim_out": RESULTS / "stage2_streaming_distilled_readout_csim_out.csv",
    },
}

BLOCK_OUTPUTS = {
    0: 248,
    1: 240,
    2: 224,
    3: 192,
    4: 128,
}


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def load_stage0_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage0_streaming_reference", STAGE0_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {STAGE0_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_readonly_07b_headers() -> None:
    HLS_SRC.mkdir(parents=True, exist_ok=True)
    header_dst = HLS_SRC / "tcn_prepad_hls.hpp"
    if not header_dst.exists():
        shutil.copy2(PHASE07B / "hls_src" / "tcn_prepad_hls.hpp", header_dst)
    for cfg in VARIANTS.values():
        if not cfg["weight_dst"].exists():
            shutil.copy2(cfg["weight_src"], cfg["weight_dst"])


def write_vectors(stage0: Any, variant: str, cfg: dict[str, Path]) -> dict[str, Any]:
    model, ckpt = stage0.load_model(stage0.VARIANTS[variant]["checkpoint"])
    x_np, ref_readout_csv = stage0.load_test_vectors(stage0.VARIANTS[variant]["test_vectors"])
    stream = stage0.streaming_trace(model, x_np, float(ckpt["target_scale"]))

    header = ["row_id", "sample_idx"]
    header += [f"x_{t}_{f}" for t in range(128) for f in range(16)]
    for block_idx, length in BLOCK_OUTPUTS.items():
        header += [f"ref_b{block_idx}_{t}_{c}" for t in range(length) for c in range(16)]
    header += [f"ref_readout_{i}" for i in range(2)]

    with cfg["vectors"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(x_np.shape[0]):
            row: list[Any] = [i, i]
            row.extend(f"{float(v):.9g}" for v in x_np[i].reshape(-1))
            for block_idx in range(5):
                key = f"block{block_idx}_out"
                expected_shape = (x_np.shape[0], BLOCK_OUTPUTS[block_idx], 16)
                if stream[key].shape != expected_shape:
                    raise ValueError(f"{variant}:{key} shape {stream[key].shape}, expected {expected_shape}")
                row.extend(f"{float(v):.9g}" for v in stream[key][i].reshape(-1))
            row.extend(f"{float(v):.9g}" for v in ref_readout_csv[i].reshape(-1))
            writer.writerow(row)

    return {
        "variant": variant,
        "source_stage0_script": str(STAGE0_SCRIPT),
        "source_checkpoint": str(stage0.VARIANTS[variant]["checkpoint"]),
        "source_07b_test_vectors": str(stage0.VARIANTS[variant]["test_vectors"]),
        "stage2_test_vectors": str(cfg["vectors"]),
        "block_csim_output": str(cfg["block_csim_out"]),
        "readout_csim_output": str(cfg["readout_csim_out"]),
        "sample_count": int(x_np.shape[0]),
        "input_events_per_sample": 252,
        "block_output_lengths": BLOCK_OUTPUTS,
        "readout_dim": 2,
        "channel_unroll": 4,
        "new5_used": False,
    }


def write_config(variant: str, cfg: dict[str, Path]) -> None:
    cfg["component"].mkdir(parents=True, exist_ok=True)
    text = f"""part=xc7z020clg400-1

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=tcn_streaming_step
syn.file={posix(cfg["cpp"])}
syn.file={posix(HLS_SRC / "tcn_prepad_hls.hpp")}
syn.file={posix(HLS_SRC / "tcn_stage2_streaming_hls.hpp")}
syn.file={posix(HLS_SRC / "tcn_stage2_streaming_impl.hpp")}
syn.file={posix(cfg["weight_dst"])}
tb.file={posix(HLS_SRC / "tcn_stage2_streaming_testbench.cpp")}
clock=10ns
csim.argv={posix(cfg["vectors"])} {posix(cfg["block_csim_out"])} {posix(cfg["readout_csim_out"])}
"""
    (cfg["component"] / "hls_config.cfg").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    HLS_SRC.mkdir(parents=True, exist_ok=True)
    ensure_readonly_07b_headers()
    stage0 = load_stage0_module()
    manifest = []
    for variant, cfg in VARIANTS.items():
        manifest.append(write_vectors(stage0, variant, cfg))
        write_config(variant, cfg)
        print(f"{variant}: wrote {cfg['vectors']}")
        print(f"{variant}: wrote {cfg['component'] / 'hls_config.cfg'}")
    with (RESULTS / "stage2_streaming_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {RESULTS / 'stage2_streaming_manifest.json'}")


if __name__ == "__main__":
    main()
