from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06t_event_shift_staleness_cost"
RESULTS = ROOT / "results"
ACTION_UNIVERSE = (
    PROJECT_ROOT
    / "n2ew_era"
    / "04b_txf_only_final_development_rerun"
    / "data"
    / "txf_only_action_universe.parquet"
)

SPLITS = ["val", "new3", "new4"]
ACTION_BID = 0
ACTION_ASK = 1


def assert_allowed_path(path: Path) -> None:
    if "NEW5" in str(path).upper():
        raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")


def aligned_slot_frames(raw: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bid = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].sort_values("event_index", kind="mergesort").reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(ACTION_ASK)].sort_values("event_index", kind="mergesort").reset_index(drop=True)
    if len(bid) != len(ask):
        raise RuntimeError(f"Bid/ask row mismatch in {path}: {len(bid)} vs {len(ask)}")
    for col in ["session_id", "event_index", "best_bid", "best_ask", "mid_price", "future_mid_H20"]:
        if not bid[col].reset_index(drop=True).equals(ask[col].reset_index(drop=True)):
            raise RuntimeError(f"Bid/ask alignment mismatch col={col} path={path}")
    return bid, ask


def prepare_split(split: str) -> dict[str, Any]:
    if split not in SPLITS or "NEW5" in split.upper():
        raise RuntimeError(f"Forbidden split requested: {split}")
    out_path = RESULTS / f"dense_book_shift_cache_{split}.npz"
    columns = [
        "split",
        "source_group",
        "session_id",
        "event_index",
        "action_slot",
        "best_bid",
        "best_ask",
        "mid_price",
        "future_mid_H20",
        "excess_pnl_H20",
    ]
    session_labels: list[str] = []
    source_group_labels: list[str] = []
    arrays: dict[str, list[np.ndarray]] = {
        "session_code": [],
        "event_index": [],
        "best_bid": [],
        "best_ask": [],
        "mid_price": [],
        "future_mid_h20": [],
        "bid_excess_pnl_h20": [],
        "ask_excess_pnl_h20": [],
    }
    for code, path in enumerate(sorted((ACTION_UNIVERSE / f"split={split}").glob("*.parquet"))):
        assert_allowed_path(path)
        raw = pd.read_parquet(path, columns=columns)
        if raw["source_group"].astype(str).str.upper().eq("NEW5").any():
            raise RuntimeError(f"Forbidden NEW5 source group encountered in {path}")
        bid, ask = aligned_slot_frames(raw, path)
        session_labels.append(str(bid["session_id"].iloc[0]))
        source_group_labels.append(str(bid["source_group"].iloc[0]))
        n = len(bid)
        arrays["session_code"].append(np.full(n, code, dtype=np.int16))
        arrays["event_index"].append(pd.to_numeric(bid["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64))
        arrays["best_bid"].append(pd.to_numeric(bid["best_bid"], errors="coerce").to_numpy(dtype=np.float32))
        arrays["best_ask"].append(pd.to_numeric(bid["best_ask"], errors="coerce").to_numpy(dtype=np.float32))
        arrays["mid_price"].append(pd.to_numeric(bid["mid_price"], errors="coerce").to_numpy(dtype=np.float32))
        arrays["future_mid_h20"].append(pd.to_numeric(bid["future_mid_H20"], errors="coerce").to_numpy(dtype=np.float32))
        arrays["bid_excess_pnl_h20"].append(pd.to_numeric(bid["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        arrays["ask_excess_pnl_h20"].append(pd.to_numeric(ask["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        print(f"cached split={split} session={session_labels[-1]} events={n}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {key: np.concatenate(value) for key, value in arrays.items()}
    payload["session_labels"] = np.asarray(session_labels)
    payload["source_group_labels"] = np.asarray(source_group_labels)
    np.savez_compressed(out_path, **payload)
    return {
        "split": split,
        "cache_path": str(out_path.relative_to(ROOT)),
        "session_count": len(session_labels),
        "event_count": int(len(payload["event_index"])),
        "source_groups": ",".join(sorted(set(source_group_labels))),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [prepare_split(split) for split in SPLITS]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS / "dense_book_shift_cache_manifest.csv", index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
