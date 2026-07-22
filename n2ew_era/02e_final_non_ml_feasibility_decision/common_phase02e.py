from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "02e_final_non_ml_feasibility_decision"
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

PHASE01_ROOT = PROJECT_ROOT / "n2ew_era" / "01_full_l2_action_universe_builder"
PHASE01_DATA = PHASE01_ROOT / "data"
PHASE01_RESULTS = PHASE01_ROOT / "results"
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
PHASE02D_DIR = PROJECT_ROOT / "n2ew_era" / "02d_final_strict_baseline_audit_package"
NEW3_RAW_DIR = PROJECT_ROOT / "data" / "NEW3"

if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))
if str(PHASE01_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE01_ROOT))

from common_phase01 import (  # noqa: E402
    ACTION_SIDES,
    BOUNDARY_FIRST_TICKS,
    BOUNDARY_LAST_TICKS,
    HORIZONS,
    MAKER_FEE,
    POINT_VALUE,
    SOURCE_COLUMNS,
)
from common_phase02 import ACTION_SLOT_ASK, ACTION_SLOT_BID, ACTION_SLOT_NO_QUOTE  # noqa: E402
from common_phase02 import RANDOM_SEED, policy_catalog, policy_decision  # noqa: E402


SELECTED_POLICY_ID = "side_by_queue_imbalance_abs_0p1"
OLD_BID_POLICY_ID = "old_bid_only_spread_ge_1"
NO_QUOTE_POLICY_ID = "no_quote"
SELECTED_HORIZON = 20
EVAL_SPLITS = ["val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
FEE_MULTIPLIERS = [1.0, 2.0, 5.0]
POSITIVE_FILL_KEEP_RATES = [1.0, 0.75, 0.5, 0.25, 0.1]
ADVERSE_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
DELAY_EVENTS = [0, 1, 2, 5, 10]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def phase01_builder_module() -> Any:
    path = PHASE01_ROOT / "01_build_full_l2_action_universe.py"
    spec = importlib.util.spec_from_file_location("phase01_builder_for_02e", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Phase 01 builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_by_id(policy_id: str) -> dict[str, Any]:
    lookup = {p["policy_id"]: p for p in policy_catalog()}
    if policy_id not in lookup:
        raise KeyError(f"Policy not found: {policy_id}")
    return lookup[policy_id]


def policy_set() -> dict[str, dict[str, Any]]:
    return {
        NO_QUOTE_POLICY_ID: policy_by_id(NO_QUOTE_POLICY_ID),
        OLD_BID_POLICY_ID: policy_by_id(OLD_BID_POLICY_ID),
        SELECTED_POLICY_ID: policy_by_id(SELECTED_POLICY_ID),
    }


def split_dataset_dir(split: str) -> Path:
    if split == "new3":
        return DATA_DIR / "full_l2_action_universe_new3.parquet"
    return PHASE01_DATA / f"full_l2_action_universe_{split}.parquet"


def part_paths(split: str) -> list[Path]:
    path = split_dataset_dir(split)
    return sorted(path.glob("*.parquet"), key=lambda p: p.name)


def hash_int64(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def event_frame_from_part(path: Path, horizons: Iterable[int] = (SELECTED_HORIZON,)) -> pd.DataFrame:
    label_cols: list[str] = []
    for h in horizons:
        label_cols.extend([f"future_mid_H{h}", f"pnl_H{h}", f"markout_ticks_H{h}"])
    cols = [
        "split",
        "session_id",
        "symbol",
        "event_index",
        "timestamp",
        "action_slot",
        "best_bid",
        "best_ask",
        "mid_price",
        "tick_size",
        "spread_ticks",
        "queue_imbalance",
        "deep_imbalance",
        "top_depth",
        "volatility_past_50",
        *label_cols,
    ]
    raw = pd.read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int) == ACTION_SLOT_BID].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int) == ACTION_SLOT_ASK].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int) == ACTION_SLOT_NO_QUOTE].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet mismatch in {path}")
    out = bid[
        [
            "split",
            "session_id",
            "symbol",
            "event_index",
            "timestamp",
            "best_bid",
            "best_ask",
            "mid_price",
            "tick_size",
            "spread_ticks",
            "queue_imbalance",
            "deep_imbalance",
            "top_depth",
            "volatility_past_50",
        ]
    ].copy()
    for h in horizons:
        out[f"future_mid_H{h}"] = pd.to_numeric(bid[f"future_mid_H{h}"], errors="coerce").to_numpy(dtype=float)
        out[f"bid_pnl_H{h}"] = pd.to_numeric(bid[f"pnl_H{h}"], errors="coerce").to_numpy(dtype=float)
        out[f"ask_pnl_H{h}"] = pd.to_numeric(ask[f"pnl_H{h}"], errors="coerce").to_numpy(dtype=float)
        out[f"bid_markout_ticks_H{h}"] = pd.to_numeric(bid[f"markout_ticks_H{h}"], errors="coerce").to_numpy(dtype=float)
        out[f"ask_markout_ticks_H{h}"] = pd.to_numeric(ask[f"markout_ticks_H{h}"], errors="coerce").to_numpy(dtype=float)
    return out


def iter_event_frames(split: str, horizons: Iterable[int] = (SELECTED_HORIZON,)) -> Iterable[pd.DataFrame]:
    for path in part_paths(split):
        yield event_frame_from_part(path, horizons=horizons)


def delayed_decision(decision: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return decision.copy()
    out = np.zeros_like(decision)
    if delay < len(decision):
        out[delay:] = decision[:-delay]
    return out


def pnl_from_decision(frame: pd.DataFrame, decision: np.ndarray, horizon: int = SELECTED_HORIZON, fee_multiplier: float = 1.0) -> np.ndarray:
    bid_base = frame[f"bid_pnl_H{horizon}"].to_numpy(dtype=float) - MAKER_FEE * (float(fee_multiplier) - 1.0)
    ask_base = frame[f"ask_pnl_H{horizon}"].to_numpy(dtype=float) - MAKER_FEE * (float(fee_multiplier) - 1.0)
    out = np.zeros(len(frame), dtype=float)
    bid_mask = decision == (ACTION_SLOT_BID + 1)
    ask_mask = decision == (ACTION_SLOT_ASK + 1)
    out[bid_mask] = np.nan_to_num(bid_base[bid_mask], nan=0.0)
    out[ask_mask] = np.nan_to_num(ask_base[ask_mask], nan=0.0)
    return out


def stress_pnl(base_pnl: np.ndarray, positive_fill_keep_rate: float, adverse_multiplier: float) -> np.ndarray:
    return np.where(base_pnl > 0.0, base_pnl * float(positive_fill_keep_rate), base_pnl * float(adverse_multiplier))


def max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    equity = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    return float((np.maximum.accumulate(equity) - equity).max())


def summarize_pnl(policy: dict[str, Any], split: str, horizon: int = SELECTED_HORIZON) -> dict[str, Any]:
    total = 0.0
    selected = 0
    wins = 0
    losses = 0
    state_count = 0
    dd_equity: list[np.ndarray] = []
    bid_count = 0
    ask_count = 0
    for frame in iter_event_frames(split, horizons=[horizon]):
        decision = policy_decision(policy, frame)
        pnl = pnl_from_decision(frame, decision, horizon=horizon)
        chosen = decision != 0
        selected_values = pnl[chosen]
        total += float(pnl.sum())
        selected += int(chosen.sum())
        bid_count += int((decision == 1).sum())
        ask_count += int((decision == 2).sum())
        wins += int((selected_values > 0).sum())
        losses += int((selected_values < 0).sum())
        state_count += int(len(frame))
        dd_equity.append(pnl)
    all_pnl = np.concatenate(dd_equity) if dd_equity else np.array([], dtype=float)
    denom = max(selected, 1)
    return {
        "split": split,
        "policy_id": policy["policy_id"],
        "horizon": int(horizon),
        "state_count": int(state_count),
        "selected_action_count": int(selected),
        "selection_rate": float(selected / state_count) if state_count else 0.0,
        "bid_count": int(bid_count),
        "ask_count": int(ask_count),
        "total_pnl": float(total),
        "win_rate": float(wins / denom) if selected else 0.0,
        "loss_rate": float(losses / denom) if selected else 0.0,
        "max_drawdown": max_drawdown_from_pnl(all_pnl),
    }


def spread_bucket_values(spread: pd.Series | np.ndarray) -> np.ndarray:
    return pd.cut(
        pd.to_numeric(pd.Series(spread), errors="coerce"),
        bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
        labels=["<=1", "(1,2]", "(2,3]", ">3"],
    ).astype(str).to_numpy(dtype=object)


def matched_random_trial_pnls(split: str, base_policy: dict[str, Any], horizon: int, trials: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 9020 + len(split))
    group_values: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    required: dict[tuple[str, str, str], int] = defaultdict(int)
    model_pnl = 0.0
    selected_count = 0
    for frame in iter_event_frames(split, horizons=[horizon]):
        decision = policy_decision(base_policy, frame)
        session_id = str(frame["session_id"].iloc[0])
        buckets = spread_bucket_values(frame["spread_ticks"])
        for side, code, pnl_col in [
            ("bid", ACTION_SLOT_BID + 1, f"bid_pnl_H{horizon}"),
            ("ask", ACTION_SLOT_ASK + 1, f"ask_pnl_H{horizon}"),
        ]:
            values = np.nan_to_num(frame[pnl_col].to_numpy(dtype=float), nan=0.0)
            selected_mask = decision == code
            model_pnl += float(values[selected_mask].sum())
            selected_count += int(selected_mask.sum())
            for bucket in np.unique(buckets):
                mask = buckets == bucket
                key = (side, session_id, str(bucket))
                group_values[key].append(values[mask])
                required[key] += int((selected_mask & mask).sum())
    flat_values = {key: np.concatenate(chunks) for key, chunks in group_values.items()}
    trial_pnls = np.zeros(trials, dtype=float)
    for trial in range(trials):
        total = 0.0
        for key, values in flat_values.items():
            k = int(required.get(key, 0))
            if k <= 0 or len(values) == 0:
                continue
            if k >= len(values):
                total += float(values.sum())
            else:
                idx = rng.choice(len(values), size=k, replace=False)
                total += float(values[idx].sum())
        trial_pnls[trial] = total
    return pd.DataFrame(
        {
            "split": split,
            "policy_id": base_policy["policy_id"],
            "horizon": int(horizon),
            "trial": np.arange(trials, dtype=int),
            "random_pnl": trial_pnls,
            "model_pnl": float(model_pnl),
            "selected_action_count": int(selected_count),
        }
    )


def write_action_part(action_frame: pd.DataFrame, session_id: str) -> dict[str, Any]:
    full_dir = DATA_DIR / "full_l2_action_universe_new3.parquet"
    full_dir.mkdir(parents=True, exist_ok=True)
    path = full_dir / f"part-{session_id}.parquet"
    table = pa.Table.from_pandas(action_frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", use_dictionary=True, row_group_size=250_000)
    return {
        "split": "new3",
        "session_id": str(session_id),
        "source_rows": int(len(action_frame) // len(ACTION_SIDES)),
        "action_rows": int(len(action_frame)),
        "part_path": str(path),
    }


def build_new3_action_universe() -> pd.DataFrame:
    builder = phase01_builder_module()
    target = DATA_DIR / "full_l2_action_universe_new3.parquet"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if not NEW3_RAW_DIR.exists():
        raise FileNotFoundError(f"NEW3 raw directory missing: {NEW3_RAW_DIR}")

    symbol = builder.detect_symbol_from_raw_dir(NEW3_RAW_DIR, preferred="TXFA6")
    lob_frames, _ = builder.load_new_tw_data(NEW3_RAW_DIR, [symbol])
    if symbol not in lob_frames or lob_frames[symbol].empty:
        raise RuntimeError(f"No L2 frame loaded for symbol {symbol} from {NEW3_RAW_DIR}")
    lob = lob_frames[symbol].sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    tick_size = float(builder.infer_tick_size_simple(lob))

    depth_cols: list[str] = []
    for level in range(1, 6):
        depth_cols.extend([f"bid_px_{level}", f"bid_vol_{level}", f"ask_px_{level}", f"ask_vol_{level}"])

    report_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    for session_id, group in lob.groupby("session_id", sort=True):
        g = group.sort_values("timestamp").reset_index(drop=True).copy()
        raw_rows = int(len(g))
        spread = pd.to_numeric(g["ask_px_1"], errors="coerce") - pd.to_numeric(g["bid_px_1"], errors="coerce")
        idx = np.arange(raw_rows)
        rule_masks = {
            "invalid_bid_ask": builder.finite_positive(g, ["bid_px_1", "ask_px_1"]),
            "nonpositive_spread": np.isfinite(spread.to_numpy(dtype=float)) & (spread.to_numpy(dtype=float) > 0.0),
            "missing_depth": builder.finite_positive(g, depth_cols),
            "session_boundary": (idx >= int(BOUNDARY_FIRST_TICKS)) & (idx < max(raw_rows - int(BOUNDARY_LAST_TICKS), 0)),
            "feature_na": np.ones(raw_rows, dtype=bool),
        }
        drop_counts, eligible_mask = builder.sequential_filter_counts(rule_masks)
        local_row_id = np.flatnonzero(eligible_mask).astype(np.int64)
        out = g.loc[eligible_mask, ["timestamp", *depth_cols]].copy()
        out.insert(0, "dataset", "NEW3")
        out.insert(1, "symbol", symbol)
        out.insert(2, "session_id", str(session_id))
        out.insert(3, "split", "new3")
        out.insert(4, "local_row_id", local_row_id)
        out["tick_size"] = tick_size
        out["mid_price"] = (out["bid_px_1"].astype(float) + out["ask_px_1"].astype(float)) / 2.0
        out["spread_ticks"] = (out["ask_px_1"].astype(float) - out["bid_px_1"].astype(float)) / max(tick_size, 1e-9)
        out["quote_eligible"] = True
        for col in ["queue_imbalance", "deep_obi", "queue_imbalance_change", "normalized_spread", "trade_intensity_short", "cancel_pressure_short"]:
            out[col] = np.nan
        action_frame = builder.make_action_frame(out[SOURCE_COLUMNS])
        part_rows.append(write_action_part(action_frame, str(session_id)))
        report_rows.append(
            {
                "split": "new3",
                "dataset": "NEW3",
                "symbol": symbol,
                "session_id": str(session_id),
                "source_path": str(NEW3_RAW_DIR),
                "raw_rows": raw_rows,
                "valid_book_rows": int(raw_rows - drop_counts["dropped_invalid_bid_ask"] - drop_counts["dropped_nonpositive_spread"] - drop_counts["dropped_missing_depth"]),
                "quote_eligible_rows": int(eligible_mask.sum()),
                **drop_counts,
                "tick_size": tick_size,
                "timestamp_hash": hash_int64(g["timestamp"].to_numpy(dtype=np.int64)),
                "eligible_timestamp_hash": hash_int64(out["timestamp"].to_numpy(dtype=np.int64)),
            }
        )
        print(f"built NEW3 session={session_id}: source_rows={len(out):,} action_rows={len(action_frame):,}")

    report = pd.DataFrame(report_rows)
    parts = pd.DataFrame(part_rows)
    report.to_csv(RESULTS / "new3_source_filter_counts.csv", index=False)
    parts.to_csv(RESULTS / "new3_build_parts_manifest.csv", index=False)
    return report


@dataclass
class InventoryMetrics:
    pnl: float
    fill_count: int
    forced_no_quote_count: int
    max_abs_inventory: int
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float

