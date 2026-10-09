from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
BRAM = ROOT / "bram"
TOTAL_BITS = 24
FRAC_BITS = 12
SIGNED_MIN = -(1 << (TOTAL_BITS - 1))
SIGNED_MAX = (1 << (TOTAL_BITS - 1)) - 1
WORD_HEX_DIGITS = 6
FEATURE_DIM = 16
READOUT_DIM = 2
HIDDEN_DIM = 16


def q24_12_round_int(values: np.ndarray) -> np.ndarray:
    scaled = np.rint(values.astype(np.float64) * (1 << FRAC_BITS))
    return np.clip(scaled, SIGNED_MIN, SIGNED_MAX).astype(np.int64)


def q24_12_trunc_int(values: np.ndarray) -> np.ndarray:
    # HLS phc_fixed_t is ap_fixed<24,12> with default AP_TRN quantization.
    # In the fixed-point two's-complement representation AP_TRN is a bit-drop:
    # negative values floor by one LSB rather than truncating toward zero.
    scaled = np.floor(values.astype(np.float64) * (1 << FRAC_BITS))
    return np.clip(scaled, SIGNED_MIN, SIGNED_MAX).astype(np.int64)


def int24_to_hex(value: int) -> str:
    if value < 0:
        value = (1 << TOTAL_BITS) + value
    return f"{int(value) & ((1 << TOTAL_BITS) - 1):0{WORD_HEX_DIGITS}X}"


def pack_words(values: np.ndarray, *, rounding: str = "round") -> str:
    ints = q24_12_trunc_int(values) if rounding == "trunc" else q24_12_round_int(values)
    return "".join(int24_to_hex(int(v)) for v in ints)


def write_mem_and_coe(stem: str, rows: list[str]) -> None:
    (BRAM / f"{stem}.mem").write_text("\n".join(rows) + "\n", encoding="ascii")
    (BRAM / f"{stem}.coe").write_text(
        "memory_initialization_radix=16;\n"
        "memory_initialization_vector=\n"
        + ",\n".join(rows)
        + ";\n",
        encoding="ascii",
    )


def main() -> None:
    source_features = RESULTS / "demo_replay_input_features.csv"
    csim_out = Path(os.environ.get("PHC_CSIM_OUT", RESULTS / "demo_replay_hls_csim_out.csv"))
    if not source_features.exists():
        raise FileNotFoundError(source_features)
    if not csim_out.exists():
        raise FileNotFoundError(
            f"Missing HLS CSim output. Run vitis-run --mode hls --csim in hls_replay_component first: {csim_out}"
        )
    features = pd.read_csv(source_features)
    csim = pd.read_csv(csim_out)
    if len(features) != len(csim):
        raise RuntimeError(f"Row count mismatch: features={len(features)} csim={len(csim)}")
    if "pass" in csim.columns and not csim["pass"].astype(int).eq(1).all():
        bad = int((~csim["pass"].astype(int).eq(1)).sum())
        raise RuntimeError(f"CSim output contains failed tolerance rows: {bad}")
    x_cols = [f"x_t_{i}" for i in range(FEATURE_DIM)]
    readout_cols = [f"dut_readout_{i}" for i in range(READOUT_DIM)]
    state_cols = [f"dut_state_next_{i}" for i in range(HIDDEN_DIM)]
    missing = [c for c in [*x_cols, *readout_cols, *state_cols] if c not in [*features.columns, *csim.columns]]
    if missing:
        raise RuntimeError(f"Missing columns for BRAM update: {missing}")
    packed_x_readout: list[str] = []
    packed_state: list[str] = []
    for i in range(len(features)):
        x = features.loc[i, x_cols].to_numpy(dtype=np.float64)
        readout = csim.loc[i, readout_cols].to_numpy(dtype=np.float64)
        state = csim.loc[i, state_cols].to_numpy(dtype=np.float64)
        x_int = q24_12_trunc_int(x)
        readout_int = q24_12_round_int(readout)
        packed_x_readout.append("".join(int24_to_hex(int(v)) for v in np.concatenate([x_int, readout_int])))
        packed_state.append(pack_words(state))
    write_mem_and_coe("demo_replay_x_readout_q24_12", packed_x_readout)
    write_mem_and_coe("demo_replay_expected_state_q24_12", packed_state)
    manifest = {
        "status": "updated_from_hls_csim",
        "csim_output": str(csim_out.relative_to(PROJECT_ROOT)),
        "row_count": int(len(csim)),
        "max_abs_diff_vs_python_ref": float(csim["max_abs_diff"].max()) if "max_abs_diff" in csim.columns else None,
        "source": "HLS CSim DUT output; input feature packing uses AP_TRN bit-drop semantics (floor for negative values) to match ap_fixed casts",
    }
    (RESULTS / "bram_expected_from_hls_csim_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
