from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "spread_ticks",
    "queue_imbalance_calc",
    "deep_imbalance_calc",
    "top_depth",
    "depth_ratio",
    "volatility_past_50",
    "abs_return_past_50",
    "event_rate_past_50",
    "queue_imbalance_change_past",
    "normalized_spread_calc",
]


def audit_split(path: Path, split: str) -> dict:
    df = pd.read_parquet(path)
    y = df["negative_ev_moderate"].astype(int)
    severe = df.get("severe_negative_ev_moderate", (df["quote_ev_moderate"].astype(float) <= -5000).astype(int)).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return {
        "split": split,
        "rows": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "negative_ev_count": pos,
        "non_negative_ev_count": neg,
        "negative_ev_rate": float(y.mean()) if len(y) else 0.0,
        "severe_negative_ev_count": int(severe.sum()),
        "severe_negative_ev_rate": float(severe.mean()) if len(severe) else 0.0,
        "class_imbalance_non_negative_to_negative": float(neg / pos) if pos else np.inf,
        "quote_ev_sum": float(df["quote_ev_moderate"].astype(float).sum()),
        "quote_ev_mean": float(df["quote_ev_moderate"].astype(float).mean()) if len(df) else 0.0,
        "sessions": int(df["session_id"].nunique()) if "session_id" in df else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Directory containing train.parquet and val.parquet")
    args = parser.parse_args()

    root = Path(args.data_dir)
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    rows = [audit_split(root / "train.parquet", "train"), audit_split(root / "val.parquet", "val")]
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "training_pipeline_audit.csv", index=False)
    print(out.to_string(index=False))
    print(f"Wrote {out_dir / 'training_pipeline_audit.csv'}")


if __name__ == "__main__":
    main()

