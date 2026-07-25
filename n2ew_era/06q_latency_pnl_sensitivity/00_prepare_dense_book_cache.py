from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06q_latency_pnl_sensitivity"
RESULTS = ROOT / "results"
PHASE04B_ACTIONS = (
    PROJECT_ROOT
    / "n2ew_era"
    / "04b_txf_only_final_development_rerun"
    / "data"
    / "txf_only_action_universe.parquet"
)

SPLITS = ["val", "new3", "new4"]
DAY_US = 86_400_000_000


def unwrap_session_timestamp_us(ts: np.ndarray) -> np.ndarray:
    out = np.rint(np.asarray(ts, dtype=np.float64)).astype(np.int64)
    if out.size == 0:
        return out
    first = int(out[0])
    if first >= 12 * 60 * 60 * 1_000_000:
        shifted = out < (first - 6 * 60 * 60 * 1_000_000)
        out = out + shifted.astype(np.int64) * DAY_US
    return out


def build_split(split: str) -> dict[str, Any]:
    if "NEW5" in split.upper():
        raise RuntimeError(f"Forbidden NEW5-like split requested: {split}")
    split_dir = PHASE04B_ACTIONS / f"split={split}"
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split}: {split_dir}")
    columns = [
        "split",
        "source_group",
        "session_id",
        "event_index",
        "timestamp_us",
        "action_slot",
        "top_depth",
        "bid_sz_1",
        "ask_sz_1",
    ]
    session_labels: list[str] = []
    source_group_labels: list[str] = []
    session_code_chunks: list[np.ndarray] = []
    event_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    top_chunks: list[np.ndarray] = []
    bid_chunks: list[np.ndarray] = []
    ask_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for path in files:
        if "NEW5" in str(path).upper():
            raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if frame["source_group"].astype(str).str.upper().eq("NEW5").any():
            raise RuntimeError(f"Forbidden NEW5 source_group encountered in {path}")
        frame = frame[frame["action_slot"].astype(int).eq(0)].copy()
        frame = frame.sort_values("event_index", kind="mergesort").drop_duplicates("event_index", keep="first")
        if frame.empty:
            continue
        session_id = str(frame["session_id"].iloc[0])
        source_group = str(frame["source_group"].iloc[0])
        session_code = len(session_labels)
        session_labels.append(session_id)
        source_group_labels.append(source_group)
        event_index = pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        ts_raw = pd.to_numeric(frame["timestamp_us"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        ts = unwrap_session_timestamp_us(ts_raw)
        order = np.lexsort((event_index, ts))
        sorted_ts = ts[order].astype(np.int64)
        sorted_event = event_index[order].astype(np.int64)
        diffs = np.diff(sorted_ts).astype(np.float64)
        nonzero = diffs[diffs > 0]
        rows.append(
            {
                "split": split,
                "session_id": session_id,
                "source_group": source_group,
                "dense_event_count": int(len(frame)),
                "interval_count": int(len(diffs)),
                "same_timestamp_rate": float((diffs == 0).mean()) if len(diffs) else 0.0,
                "min_nonzero_dt_us": float(nonzero.min()) if len(nonzero) else None,
            }
        )
        session_code_chunks.append(np.full(len(order), session_code, dtype=np.int16))
        event_chunks.append(sorted_event)
        timestamp_chunks.append(sorted_ts)
        top_chunks.append(
            pd.to_numeric(frame["top_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[order]
        )
        bid_chunks.append(
            pd.to_numeric(frame["bid_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[order]
        )
        ask_chunks.append(
            pd.to_numeric(frame["ask_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[order]
        )

    out_path = RESULTS / f"dense_book_cache_{split}.npz"
    np.savez(
        out_path,
        session_code=np.concatenate(session_code_chunks).astype(np.int16),
        session_labels=np.asarray(session_labels, dtype="<U96"),
        source_group_labels=np.asarray(source_group_labels, dtype="<U16"),
        event_index=np.concatenate(event_chunks).astype(np.int64),
        timestamp_unwrapped_us=np.concatenate(timestamp_chunks).astype(np.int64),
        top_depth=np.concatenate(top_chunks).astype(np.float32),
        bid_depth=np.concatenate(bid_chunks).astype(np.float32),
        ask_depth=np.concatenate(ask_chunks).astype(np.float32),
    )
    return {
        "split": split,
        "cache_path": str(out_path),
        "cache_size_bytes": int(out_path.stat().st_size),
        "session_count": int(len(session_labels)),
        "dense_event_count": int(sum(len(x) for x in event_chunks)),
        "sessions": rows,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = [build_split(split) for split in SPLITS]
    manifest_path = RESULTS / "dense_book_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
