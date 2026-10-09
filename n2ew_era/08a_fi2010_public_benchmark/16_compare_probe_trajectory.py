from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = [
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
]


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPU probe epochs with the CPU fair512 reference.")
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu = read_rows(args.gpu)
    cpu = read_rows(args.cpu)
    if sorted(gpu) != [1, 2, 3] or sorted(cpu) != [1, 2, 3]:
        raise RuntimeError(f"Expected exactly epochs 1-3; gpu={sorted(gpu)}, cpu={sorted(cpu)}")

    rows: list[dict[str, object]] = []
    for epoch in [1, 2, 3]:
        for metric in METRICS:
            cpu_value = float(cpu[epoch][metric])
            gpu_value = float(gpu[epoch][metric])
            abs_delta = abs(gpu_value - cpu_value)
            rows.append(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "cpu_value": cpu_value,
                    "gpu_value": gpu_value,
                    "signed_delta_gpu_minus_cpu": gpu_value - cpu_value,
                    "abs_delta": abs_delta,
                    "relative_abs_delta": abs_delta / max(abs(cpu_value), 1e-15),
                    "gpu_epoch_wall_seconds": gpu[epoch].get("epoch_wall_seconds", ""),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
