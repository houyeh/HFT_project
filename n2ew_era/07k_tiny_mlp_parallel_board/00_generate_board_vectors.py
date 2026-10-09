from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "n2ew_era"
    / "07j_tiny_mlp_parallel_hls"
    / "results"
    / "tiny_mlp_q28_test_vectors.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bram" / "tiny_mlp_board_vectors.mem"


def twos_complement(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def pack_row(row: dict[str, str]) -> int:
    word = 0
    shift = 0
    for index in range(16):
        word |= twos_complement(int(row[f"x{index}_q"]), 28) << shift
        shift += 28
    for index in range(3):
        word |= twos_complement(int(row[f"logit{index}_q"]), 40) << shift
        shift += 40
    word |= twos_complement(int(row["action"]), 2) << shift
    shift += 2
    assert shift == 570
    return word


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=4096)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if len(rows) == args.count:
                break
    if len(rows) != args.count:
        raise RuntimeError(f"expected {args.count} rows, found {len(rows)}")

    hex_digits = (570 + 3) // 4
    with args.output.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            handle.write(f"{pack_row(row):0{hex_digits}x}\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "rows": len(rows),
        "packed_width_bits": 570,
        "hex_digits_per_row": hex_digits,
        "layout_lsb_to_msb": [
            "x0_q..x15_q: 16 x signed 28-bit",
            "logit0_q..logit2_q: 3 x signed 40-bit",
            "action: unsigned 2-bit",
        ],
        "sha256": digest,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] rows={len(rows)} width=570 sha256={digest}")
    print(f"[OK] output={args.output}")


if __name__ == "__main__":
    main()
