from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from fi2010_phc_lite import PHCLiteFI2010Classifier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "v3_speed_benchmark.csv"


def bench(batch_size: int, threads: int) -> dict[str, object]:
    torch.set_num_threads(threads)
    model = PHCLiteFI2010Classifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, 100, 40)
    y = torch.randint(0, 3, (batch_size,))

    t0 = time.time()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    dt = time.time() - t0
    row = {
        "threads": threads,
        "batch_size": batch_size,
        "seconds": dt,
        "samples_per_second": batch_size / dt,
    }
    print(row, flush=True)
    return row


def main() -> None:
    rows = []
    for threads in [4, 8]:
        for batch_size in [128, 256, 512, 1024]:
            rows.append(bench(batch_size, threads))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print({"wrote": str(OUT.relative_to(ROOT))}, flush=True)


if __name__ == "__main__":
    main()
