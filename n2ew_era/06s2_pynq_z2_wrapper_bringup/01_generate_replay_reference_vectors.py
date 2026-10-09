from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
BRAM = ROOT / "bram"
PHASE06E = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml"

if str(PHASE06E) not in sys.path:
    sys.path.insert(0, str(PHASE06E))

import common_06e as p06e  # noqa: E402


TOTAL_BITS = 24
FRAC_BITS = 12
SIGNED_MIN = -(1 << (TOTAL_BITS - 1))
SIGNED_MAX = (1 << (TOTAL_BITS - 1)) - 1
WORD_HEX_DIGITS = 6
FEATURE_DIM = 16
READOUT_DIM = 2
HIDDEN_DIM = 16


def ensure_dirs() -> None:
    for path in [RESULTS, BRAM]:
        path.mkdir(parents=True, exist_ok=True)


def q24_12_round_int(values: np.ndarray) -> np.ndarray:
    scaled = np.rint(values.astype(np.float64) * (1 << FRAC_BITS))
    clipped = np.clip(scaled, SIGNED_MIN, SIGNED_MAX)
    return clipped.astype(np.int64)


def q24_12_trunc_int(values: np.ndarray) -> np.ndarray:
    # HLS ap_fixed<24,12> AP_TRN drops fractional bits in the two's-complement
    # fixed-point representation. For negative values this matches floor(),
    # not toward-zero truncation.
    scaled = np.floor(values.astype(np.float64) * (1 << FRAC_BITS))
    clipped = np.clip(scaled, SIGNED_MIN, SIGNED_MAX)
    return clipped.astype(np.int64)


def int24_to_hex(value: int) -> str:
    if value < 0:
        value = (1 << TOTAL_BITS) + value
    return f"{int(value) & ((1 << TOTAL_BITS) - 1):0{WORD_HEX_DIGITS}X}"


def pack_words(values: np.ndarray, *, rounding: str = "round") -> str:
    ints = q24_12_trunc_int(values) if rounding == "trunc" else q24_12_round_int(values)
    return "".join(int24_to_hex(int(v)) for v in ints)


def write_mem_and_coe(stem: str, rows: list[str], width_bits: int) -> None:
    mem_path = BRAM / f"{stem}.mem"
    coe_path = BRAM / f"{stem}.coe"
    mem_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    coe_path.write_text(
        "memory_initialization_radix=16;\n"
        "memory_initialization_vector=\n"
        + ",\n".join(rows)
        + ";\n",
        encoding="ascii",
    )
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    input_path = RESULTS / "demo_replay_input_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Run 00_build_demo_replay_staleness.py first: {input_path}")
    features = pd.read_csv(input_path)
    x_cols = [f"x_t_{i}" for i in range(FEATURE_DIM)]
    missing = [c for c in x_cols if c not in features.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")
    x_np = features[x_cols].to_numpy(dtype=np.float32)
    step_model, payload = p06e.load_step_model()
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step model payload is marked new5_used.")
    step_model.eval()
    state = torch.zeros(1, HIDDEN_DIM, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    packed_x_readout: list[str] = []
    packed_expected_state: list[str] = []
    with torch.no_grad():
        for i in range(len(x_np)):
            x_t = torch.from_numpy(x_np[i : i + 1]).float()
            state_prev = state.clone()
            x_cat = torch.cat([x_t, state_prev], dim=-1)
            out_step = step_model(x_cat)
            readout, state = step_model.split_output(out_step)
            x_arr = x_t.numpy().reshape(-1)
            state_prev_arr = state_prev.numpy().reshape(-1)
            readout_arr = readout.numpy().reshape(-1)
            state_next_arr = state.numpy().reshape(-1)
            row: dict[str, Any] = {
                "row_id": int(i),
                "source_replay_event_id": int(features["replay_event_id"].iloc[i]),
                "event_index": int(features["event_index"].iloc[i]),
                "timestamp_us": float(features["timestamp_us"].iloc[i]),
            }
            for j, value in enumerate(x_arr):
                row[f"x_t_{j}"] = float(value)
            for j, value in enumerate(state_prev_arr):
                row[f"state_prev_{j}"] = float(value)
            for j, value in enumerate(readout_arr):
                row[f"ref_readout_{j}"] = float(value)
            for j, value in enumerate(state_next_arr):
                row[f"ref_state_next_{j}"] = float(value)
            rows.append(row)
            packed_x = q24_12_trunc_int(x_arr)
            packed_readout = q24_12_round_int(readout_arr)
            packed_x_readout.append("".join(int24_to_hex(int(v)) for v in np.concatenate([packed_x, packed_readout])))
            packed_expected_state.append(pack_words(state_next_arr))
    out_df = pd.DataFrame(rows)
    vector_csv = RESULTS / "demo_replay_step_reference_vectors.csv"
    out_df.to_csv(vector_csv, index=False)
    hls_cols = ["row_id"]
    hls_df = pd.DataFrame(
        {
            "row_id": out_df["row_id"].astype(int),
            "sample_idx": 0,
            "timestep": out_df["row_id"].astype(int),
        }
    )
    for name in [f"x_t_{i}" for i in range(FEATURE_DIM)]:
        hls_df[name] = out_df[name]
    for name in [f"state_prev_{i}" for i in range(HIDDEN_DIM)]:
        hls_df[name] = out_df[name]
    for name in [f"ref_readout_{i}" for i in range(READOUT_DIM)]:
        hls_df[name] = out_df[name]
    for name in [f"ref_state_next_{i}" for i in range(HIDDEN_DIM)]:
        hls_df[name] = out_df[name]
    hls_vector_csv = RESULTS / "demo_replay_hls_csim_vectors.csv"
    hls_df.to_csv(hls_vector_csv, index=False)
    write_mem_and_coe("demo_replay_x_readout_q24_12", packed_x_readout, (FEATURE_DIM + READOUT_DIM) * TOTAL_BITS)
    write_mem_and_coe("demo_replay_expected_state_q24_12", packed_expected_state, HIDDEN_DIM * TOTAL_BITS)
    x_int = q24_12_trunc_int(x_np)
    readout_int = q24_12_round_int(out_df[[f"ref_readout_{i}" for i in range(READOUT_DIM)]].to_numpy(dtype=np.float32))
    state_int = q24_12_round_int(out_df[[f"ref_state_next_{i}" for i in range(HIDDEN_DIM)]].to_numpy(dtype=np.float32))
    manifest = {
        "status": "generated",
        "source_feature_csv": str(input_path.relative_to(PROJECT_ROOT)),
        "reference_vector_csv": str(vector_csv.relative_to(PROJECT_ROOT)),
        "hls_csim_vector_csv": str(hls_vector_csv.relative_to(PROJECT_ROOT)),
        "step_model_checkpoint": str(p06e.STEP_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(out_df)),
        "feature_dim": FEATURE_DIM,
        "readout_dim": READOUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "quantization": "Q24.12 ap_fixed<24,12>; input casts use AP_TRN bit-drop semantics (floor for negative values)",
        "x_quantized_min": int(x_int.min()),
        "x_quantized_max": int(x_int.max()),
        "readout_quantized_min": int(readout_int.min()),
        "readout_quantized_max": int(readout_int.max()),
        "state_next_quantized_min": int(state_int.min()),
        "state_next_quantized_max": int(state_int.max()),
        "packed_x_readout_width_bits": int((FEATURE_DIM + READOUT_DIM) * TOTAL_BITS),
        "packed_state_width_bits": int(HIDDEN_DIM * TOTAL_BITS),
        "x_readout_mem": str((BRAM / "demo_replay_x_readout_q24_12.mem").relative_to(PROJECT_ROOT)),
        "x_readout_coe": str((BRAM / "demo_replay_x_readout_q24_12.coe").relative_to(PROJECT_ROOT)),
        "expected_state_mem": str((BRAM / "demo_replay_expected_state_q24_12.mem").relative_to(PROJECT_ROOT)),
        "expected_state_coe": str((BRAM / "demo_replay_expected_state_q24_12.coe").relative_to(PROJECT_ROOT)),
        "new5_used": False,
    }
    write_json(RESULTS / "replay_vector_manifest.json", manifest)
    (BRAM / "BRAM_VECTOR_PACKING.md").write_text(
        "\n".join(
            [
                "# BRAM Vector Packing",
                "",
                "status: `replay_vectors_generated`",
                "",
                "- Quantization: signed Q24.12, matching `ap_fixed<24,12>`.",
                "- Input feature words use AP_TRN bit-drop semantics to match HLS `(phc_fixed_t)double` casts; negative values floor by one LSB rather than truncating toward zero.",
                "- Expected readout/state words are packed from already-quantized fixed-point DUT values.",
                "- Hex word width: 24 bits = 6 hex digits, two's complement for negative values.",
                "- `demo_replay_x_readout_q24_12.mem/.coe` line width: 432 bits.",
                "- Word order per line: `x_t_0 ... x_t_15 ref_readout_0 ref_readout_1`.",
                "- `demo_replay_expected_state_q24_12.mem/.coe` line width: 384 bits.",
                "- Word order per line: `ref_state_next_0 ... ref_state_next_15`.",
                "- The board wrapper should reset recurrent state to zero at replay start and compare readout after each `ap_done`.",
                "- Expected state is provided for XSim debug or optional hardware checking; the primary demo comparator uses readout only to save BRAM.",
                "",
                "Capacity note: 4096 rows × 432 bits = 1,769,472 bits, approximately 96 BRAM_18K before IP overhead. Full 25,600 rows would require about 614 BRAM_18K for this packed input+readout memory alone, so the demo uses the selected burst subsample.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
