from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def detect_symbol_from_raw_dir(data_dir: Path, preferred: str = "TXFA6") -> str:
    counts: dict[str, int] = {}
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir))):
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 34:
                    continue
                try:
                    bid = float(row[21])
                    ask = float(row[22])
                except ValueError:
                    continue
                if bid <= 0 or ask <= 0:
                    continue
                symbol = row[3].strip()
                counts[symbol] = counts.get(symbol, 0) + 1
    if preferred in counts:
        return preferred
    if not counts:
        raise RuntimeError(f"No valid symbol found under {data_dir}")
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def infer_tick_size_simple(lob_df: pd.DataFrame) -> float:
    prices = pd.concat(
        [
            lob_df["bid_px_1"].astype(float),
            lob_df["ask_px_1"].astype(float),
        ],
        ignore_index=True,
    )
    diffs = np.diff(np.unique(prices[np.isfinite(prices)].to_numpy(dtype=float)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    candidates = diffs[diffs >= 0.1]
    if len(candidates) == 0:
        candidates = diffs
    return float(np.round(np.min(candidates), 6))
