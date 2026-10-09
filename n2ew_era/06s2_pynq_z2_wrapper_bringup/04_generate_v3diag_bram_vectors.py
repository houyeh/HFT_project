from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

TOTAL_BITS = 24
FRAC_BITS = 12
SIGNED_MIN = -(1 << (TOTAL_BITS - 1))
SIGNED_MAX = (1 << (TOTAL_BITS - 1)) - 1
WORD_HEX_DIGITS = 6
FEATURE_DIM = 16
READOUT_DIM = 2
HIDDEN_DIM = 16
EXPECTED_SESSION = "NEW4_20260112_TXFA6"
EXPECTED_ROWS = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q24_12_trunc_int(value: float) -> int:
    scaled = math.floor(float(value) * (1 << FRAC_BITS))
    return min(max(scaled, SIGNED_MIN), SIGNED_MAX)


def q24_12_round_int(value: float) -> int:
    # CSim outputs are exact Q24.12 multiples, so tie behavior is immaterial.
    scaled = round(float(value) * (1 << FRAC_BITS))
    return min(max(scaled, SIGNED_MIN), SIGNED_MAX)


def int24_to_hex(value: int) -> str:
    if value < 0:
        value = (1 << TOTAL_BITS) + value
    return f"{value & ((1 << TOTAL_BITS) - 1):0{WORD_HEX_DIGITS}X}"


def write_mem_and_coe(output_dir: Path, stem: str, rows: list[str]) -> tuple[Path, Path]:
    mem_path = output_dir / f"{stem}.mem"
    coe_path = output_dir / f"{stem}.coe"
    mem_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    coe_path.write_text(
        "memory_initialization_radix=16;\n"
        "memory_initialization_vector=\n"
        + ",\n".join(rows)
        + ";\n",
        encoding="ascii",
    )
    return mem_path, coe_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def differing_lines(left: Path, right: Path) -> tuple[int, int]:
    left_lines = left.read_text(encoding="ascii").splitlines()
    right_lines = right.read_text(encoding="ascii").splitlines()
    total = max(len(left_lines), len(right_lines))
    count = sum(
        1
        for i in range(total)
        if (left_lines[i] if i < len(left_lines) else None)
        != (right_lines[i] if i < len(right_lines) else None)
    )
    return count, total


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Pack V3 diagonal sequential CSim replay vectors.")
    parser.add_argument(
        "--features",
        type=Path,
        default=root / "results" / "demo_replay_input_features.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "results" / "demo_replay_metadata.json",
    )
    parser.add_argument(
        "--csim-output",
        type=Path,
        default=root / "results_v3_diag" / "demo_replay_v3diag_hls_csim_sequential_out.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "bram_v3_diag")
    parser.add_argument(
        "--old-readout-mem",
        type=Path,
        default=root / "bram" / "demo_replay_x_readout_q24_12.mem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.features, args.metadata, args.csim_output, args.old_readout_mem):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    selected = metadata.get("selected_window", {})
    if selected.get("session_id") != EXPECTED_SESSION:
        raise RuntimeError(f"unexpected replay session: {selected.get('session_id')}")

    features = read_csv(args.features)
    csim = read_csv(args.csim_output)
    if len(features) != EXPECTED_ROWS or len(csim) != EXPECTED_ROWS:
        raise RuntimeError(f"expected 4096 rows: features={len(features)} csim={len(csim)}")
    sessions = {row["session_id"] for row in features}
    if sessions != {EXPECTED_SESSION}:
        raise RuntimeError(f"unexpected feature sessions: {sorted(sessions)}")

    x_cols = [f"x_t_{i}" for i in range(FEATURE_DIM)]
    readout_cols = [f"dut_readout_{i}" for i in range(READOUT_DIM)]
    state_cols = [f"dut_state_next_{i}" for i in range(HIDDEN_DIM)]
    packed_x_readout: list[str] = []
    packed_state: list[str] = []
    for index, (feature_row, csim_row) in enumerate(zip(features, csim)):
        if feature_row["replay_event_id"] != csim_row["replay_event_id"]:
            raise RuntimeError(f"row alignment mismatch at index {index}")
        x_ints = [q24_12_trunc_int(float(feature_row[col])) for col in x_cols]
        readout_ints = [q24_12_round_int(float(csim_row[col])) for col in readout_cols]
        state_ints = [q24_12_round_int(float(csim_row[col])) for col in state_cols]
        packed_x_readout.append("".join(int24_to_hex(v) for v in x_ints + readout_ints))
        packed_state.append("".join(int24_to_hex(v) for v in state_ints))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    readout_mem, readout_coe = write_mem_and_coe(
        args.output_dir, "demo_replay_x_readout_q24_12", packed_x_readout
    )
    state_mem, state_coe = write_mem_and_coe(
        args.output_dir, "demo_replay_expected_state_q24_12", packed_state
    )
    diff_count, comparison_rows = differing_lines(readout_mem, args.old_readout_mem)
    if comparison_rows != EXPECTED_ROWS:
        raise RuntimeError(f"old/new readout comparison does not cover 4096 rows: {comparison_rows}")
    if diff_count == 0:
        raise RuntimeError("V3 and legacy readout memories are identical; wrong model source suspected")

    packing_doc = args.output_dir / "BRAM_VECTOR_PACKING.md"
    packing_doc.write_text(
        "# V3 Diagonal BRAM Vector Packing\n\n"
        "status: `generated_from_v3_diagonal_sequential_hls_csim`\n\n"
        f"- Replay session: `{EXPECTED_SESSION}`.\n"
        f"- Replay rows: {EXPECTED_ROWS}.\n"
        "- Quantization: signed Q24.12 (`ap_fixed<24,12>`).\n"
        "- Inputs use AP_TRN bit-drop semantics (`floor(value * 4096)`).\n"
        "- Expected readout/state are the sequential V3 HLS CSim DUT outputs.\n"
        "- State starts at zero and each DUT `state_next` is fed into the next row.\n"
        "- `demo_replay_x_readout_q24_12`: 16 inputs followed by 2 expected readouts.\n"
        "- `demo_replay_expected_state_q24_12`: 16 expected next-state words.\n"
        f"- Lines different from the legacy 05C4 readout memory: {diff_count} / {EXPECTED_ROWS}.\n",
        encoding="utf-8",
    )

    outputs = [readout_mem, readout_coe, state_mem, state_coe, packing_doc]
    manifest = {
        "status": "stage_c_v3diag_vectors_generated",
        "session_id": EXPECTED_SESSION,
        "source_group": selected.get("source_group"),
        "split": selected.get("split"),
        "peak_count_1s": selected.get("peak_count_1s"),
        "row_count": EXPECTED_ROWS,
        "feedback_mode": "hls_v3diag_dut_state_feedback",
        "features_path": str(args.features),
        "features_sha256": sha256(args.features),
        "csim_output_path": str(args.csim_output),
        "csim_output_sha256": sha256(args.csim_output),
        "legacy_readout_mem_path": str(args.old_readout_mem),
        "legacy_readout_mem_sha256": sha256(args.old_readout_mem),
        "new_old_readout_differing_lines": diff_count,
        "comparison_rows": comparison_rows,
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = args.output_dir / "v3diag_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
