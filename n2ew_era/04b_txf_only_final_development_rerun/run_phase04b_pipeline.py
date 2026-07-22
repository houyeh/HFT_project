from __future__ import annotations

import json
import math
import re
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"

REGISTRY = PROJECT_ROOT / "n2ew_era" / "04a1_dataset_registry_patch" / "results" / "dataset_registry_expanded.csv"

TARGET_PRODUCT_RE = re.compile(r"^TXF[A-Z0-9]*$")
HORIZON = 20
POINT_VALUE = 200.0
MAKER_FEE = 15.0
BOUNDARY_FIRST_TICKS = 100
BOUNDARY_LAST_TICKS = 100
RANDOM_TRIALS = 1000
RANDOM_SEED = 20260602
TRAIN_SAMPLE_ROWS = 400_000
VAL_SAMPLE_ROWS = 200_000
CONFUSION_SAMPLE_ROWS = 150_000

ACTION_BID = 0
ACTION_ASK = 1
ACTION_NO_QUOTE = 2
DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2

POLICY_IDS = ["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"]
OOS_SPLITS = ["new3", "new4"]
ALL_SPLITS = ["train", "val", *OOS_SPLITS]
ML_FORBIDDEN_FEATURES = {"best_bid", "best_ask", "mid_price", "quote_price", "product_id"}

RAW_COLUMN_MAP = {
    "1th_buy_price": "bid_px_1",
    "1th_buy_qty": "bid_sz_1",
    "2th_buy_price": "bid_px_2",
    "2th_buy_qty": "bid_sz_2",
    "3th_buy_price": "bid_px_3",
    "3th_buy_qty": "bid_sz_3",
    "4th_buy_price": "bid_px_4",
    "4th_buy_qty": "bid_sz_4",
    "5th_buy_price": "bid_px_5",
    "5th_buy_qty": "bid_sz_5",
    "1th_sell_price": "ask_px_1",
    "1th_sell_qty": "ask_sz_1",
    "2th_sell_price": "ask_px_2",
    "2th_sell_qty": "ask_sz_2",
    "3th_sell_price": "ask_px_3",
    "3th_sell_qty": "ask_sz_3",
    "4th_sell_price": "ask_px_4",
    "4th_sell_qty": "ask_sz_4",
    "5th_sell_price": "ask_px_5",
    "5th_sell_qty": "ask_sz_5",
}

ML_FEATURES = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "abs_return_past_50",
    "imbalance_change_past",
    "spread_change_past",
    "hawkes_excite",
    "action_is_bid",
    "action_is_ask",
    "side_sign",
    "side_queue_imbalance",
    "side_deep_imbalance",
    "side_imbalance_change_past",
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS, MODELS, PREDICTIONS, action_dataset_dir()]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def action_dataset_dir() -> Path:
    return DATA / "txf_only_action_universe.parquet"


def split_dir(split: str) -> Path:
    return action_dataset_dir() / f"split={split}"


def part_paths(split: str) -> list[Path]:
    return sorted(split_dir(split).glob("*.parquet"))


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def parse_info_time(value: Any) -> tuple[float | None, bool]:
    parts = str(value).strip().split(":")
    if len(parts) != 5:
        return None, False
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2])
        millisecond = int(parts[3])
        fifth = int(parts[4])
    except ValueError:
        return None, False
    if fifth != 0:
        return None, False
    us = float(((hour * 60 + minute) * 60 + second) * 1_000_000 + millisecond * 1_000)
    return us, True


def session_split(source_group: str) -> str | None:
    if source_group == "NEW2":
        return "train"
    if source_group == "NEW1":
        return "val"
    if source_group == "NEW3":
        return "new3"
    if source_group == "NEW4":
        return "new4"
    return None


def registry_for_04b() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    registry = registry[registry["source_group"].isin(["NEW1", "NEW2", "NEW3", "NEW4"])].copy()
    registry["split"] = registry["source_group"].map(session_split)
    if registry["source_group"].eq("NEW5").any():
        raise RuntimeError("NEW5 must not be loaded in Phase 04B.")
    registry = registry.sort_values(["source_group", "session_id", "file_name"]).reset_index(drop=True)
    return registry


def is_txf(product_id: Any) -> bool:
    return bool(TARGET_PRODUCT_RE.match(str(product_id).strip()))


def load_session_txf_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for file_order, row in enumerate(rows.itertuples(index=False)):
        path = Path(row.physical_path)
        raw = pd.read_csv(path)
        raw = clean_columns(raw)
        raw["raw_row_in_file"] = np.arange(len(raw), dtype=np.int64)
        raw["file_order"] = int(file_order)
        parsed = raw["info_time"].map(parse_info_time)
        raw["timestamp_us"] = [item[0] for item in parsed]
        raw["valid_info_time"] = [item[1] for item in parsed]
        malformed = raw[~raw["valid_info_time"].astype(bool)]
        for bad in malformed.head(50).itertuples(index=False):
            logs.append(
                {
                    "source_group": row.source_group,
                    "split": row.split,
                    "calendar_session_id": row.session_id,
                    "file_name": row.file_name,
                    "physical_path": row.physical_path,
                    "global_row_index": int(getattr(bad, "raw_row_in_file")),
                    "product_id": str(getattr(bad, "product_id", "")),
                    "info_time": str(getattr(bad, "info_time", "")),
                    "loader_action": "drop_before_feature_or_label_construction",
                }
            )
        raw = raw[raw["valid_info_time"].astype(bool)].copy()
        products = raw["product_id"].fillna("").astype(str).str.strip()
        raw = raw[products.map(is_txf)].copy()
        if raw.empty:
            continue
        raw["product_id"] = raw["product_id"].astype(str).str.strip()
        keep_cols = ["product_id", "info_time", "timestamp_us", "file_order", "raw_row_in_file", *RAW_COLUMN_MAP.keys()]
        frame = raw[keep_cols].rename(columns=RAW_COLUMN_MAP)
        frame.insert(0, "source_group", row.source_group)
        frame.insert(1, "split", row.split)
        frame.insert(2, "calendar_session_id", str(row.session_id))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), logs
    return pd.concat(frames, ignore_index=True), logs


def safe_ratio(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> np.ndarray:
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.divide(n, d, out=np.zeros_like(n, dtype=float), where=np.isfinite(d) & (d != 0.0))
    out[~np.isfinite(out)] = 0.0
    return out


def add_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values(["file_order", "raw_row_in_file"]).reset_index(drop=True).copy()
    for col in [*RAW_COLUMN_MAP.values(), "timestamp_us"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g["best_bid"] = g["bid_px_1"].astype(float)
    g["best_ask"] = g["ask_px_1"].astype(float)
    g["mid_price"] = (g["best_bid"] + g["best_ask"]) / 2.0
    g["tick_size"] = 1.0
    g["spread_ticks"] = g["best_ask"] - g["best_bid"]

    bid_top = g["bid_sz_1"].astype(float)
    ask_top = g["ask_sz_1"].astype(float)
    bid_depth = sum(g[f"bid_sz_{i}"].astype(float) for i in range(1, 6))
    ask_depth = sum(g[f"ask_sz_{i}"].astype(float) for i in range(1, 6))
    g["queue_imbalance"] = safe_ratio(bid_top - ask_top, bid_top + ask_top)
    g["deep_imbalance"] = safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)
    g["top_depth"] = (bid_top + ask_top).fillna(0.0)
    g["depth_ratio"] = safe_ratio(bid_depth, ask_depth)
    g["normalized_spread"] = g["spread_ticks"].astype(float)

    timestamp = g["timestamp_us"].astype(float)
    dt_us = timestamp.diff().to_numpy(dtype=float)
    dt_us[dt_us < 0] += 24.0 * 60.0 * 60.0 * 1_000_000.0
    dt_us[~np.isfinite(dt_us)] = 0.0
    g["inter_arrival_ms"] = np.clip(dt_us / 1_000.0, 0.0, None)
    event_rate = np.divide(1_000_000.0, dt_us, out=np.zeros_like(dt_us), where=dt_us > 0.0)
    event_rate = np.clip(event_rate, 0.0, 1_000_000.0)
    event_rate_s = pd.Series(event_rate)
    g["event_rate_past_50"] = event_rate_s.rolling(50, min_periods=3).mean().fillna(0.0).to_numpy()
    mid_ret = g["mid_price"].astype(float).diff()
    g["volatility_past_50"] = mid_ret.rolling(50, min_periods=3).std().fillna(0.0)
    g["abs_return_past_50"] = mid_ret.abs().rolling(50, min_periods=3).mean().fillna(0.0)
    g["imbalance_change_past"] = g["queue_imbalance"].astype(float).diff().fillna(0.0)
    g["spread_change_past"] = g["spread_ticks"].astype(float).diff().fillna(0.0)
    g["hawkes_excite"] = event_rate_s.ewm(span=50, adjust=False).mean().fillna(0.0).to_numpy()
    return g


def valid_book_mask(frame: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for level in range(1, 6):
        for col in [f"bid_px_{level}", f"ask_px_{level}", f"bid_sz_{level}", f"ask_sz_{level}"]:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            mask &= np.isfinite(values) & (values > 0.0)
    spread = pd.to_numeric(frame["spread_ticks"], errors="coerce").to_numpy(dtype=float)
    mask &= np.isfinite(spread) & (spread > 0.0)
    return mask


def build_action_frame(source: pd.DataFrame) -> pd.DataFrame:
    g = add_features(source)
    valid_book = valid_book_mask(g)
    g = g[valid_book].reset_index(drop=True)
    n0 = len(g)
    if n0 == 0:
        return pd.DataFrame()
    keep = np.ones(n0, dtype=bool)
    idx = np.arange(n0)
    keep &= idx >= BOUNDARY_FIRST_TICKS
    keep &= idx < max(n0 - BOUNDARY_LAST_TICKS, 0)
    g = g[keep].reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    n = len(g)
    product = str(g["product_id"].iloc[0])
    source_group = str(g["source_group"].iloc[0])
    split = str(g["split"].iloc[0])
    calendar_session = str(g["calendar_session_id"].iloc[0])
    session_id = f"{source_group}_{calendar_session}_{product}"
    event_index = np.arange(n, dtype=np.int64)
    action_slot = np.tile(np.array([ACTION_BID, ACTION_ASK, ACTION_NO_QUOTE], dtype=np.int8), n)
    action_side = np.tile(np.array(["bid", "ask", "no_quote"], dtype=object), n)

    best_bid = g["best_bid"].to_numpy(dtype=float)
    best_ask = g["best_ask"].to_numpy(dtype=float)
    mid = g["mid_price"].to_numpy(dtype=float)
    future_mid = pd.Series(mid).shift(-HORIZON).to_numpy(dtype=float)
    half_spread_bid = (mid - best_bid) * POINT_VALUE
    half_spread_ask = (best_ask - mid) * POINT_VALUE
    raw_bid = (future_mid - best_bid) * POINT_VALUE - MAKER_FEE
    raw_ask = (best_ask - future_mid) * POINT_VALUE - MAKER_FEE
    excess_bid = (future_mid - mid) * POINT_VALUE - MAKER_FEE
    excess_ask = (mid - future_mid) * POINT_VALUE - MAKER_FEE

    quote_price = np.empty(n * 3, dtype=float)
    quote_price[0::3] = best_bid
    quote_price[1::3] = best_ask
    quote_price[2::3] = np.nan
    raw_pnl = np.zeros(n * 3, dtype=float)
    raw_pnl[0::3] = raw_bid
    raw_pnl[1::3] = raw_ask
    raw_pnl[2::3] = 0.0
    excess = np.zeros(n * 3, dtype=float)
    excess[0::3] = excess_bid
    excess[1::3] = excess_ask
    excess[2::3] = 0.0
    half_spread = np.zeros(n * 3, dtype=float)
    half_spread[0::3] = half_spread_bid
    half_spread[1::3] = half_spread_ask
    half_spread[2::3] = 0.0

    out: dict[str, Any] = {
        "split": np.repeat(split, n * 3),
        "source_group": np.repeat(source_group, n * 3),
        "calendar_session_id": np.repeat(calendar_session, n * 3),
        "session_id": np.repeat(session_id, n * 3),
        "symbol": np.repeat(product, n * 3),
        "product_id": np.repeat(product, n * 3),
        "event_index": np.repeat(event_index, 3),
        "timestamp_us": np.repeat(g["timestamp_us"].to_numpy(dtype=float), 3),
        "action_side": action_side,
        "action_slot": action_slot,
        "quote_price": quote_price,
        "best_bid": np.repeat(best_bid, 3),
        "best_ask": np.repeat(best_ask, 3),
        "mid_price": np.repeat(mid, 3),
        "tick_size": np.repeat(1.0, n * 3),
        "spread_ticks": np.repeat(g["spread_ticks"].to_numpy(dtype=float), 3),
        "future_mid_H20": np.repeat(future_mid, 3),
        "raw_pnl_H20": raw_pnl,
        "half_spread_component_H20": half_spread,
        "excess_pnl_H20": excess,
        "positive_excess_pnl_H20": np.where(np.isfinite(excess), excess > 0.0, False),
        "negative_excess_pnl_H20": np.where(np.isfinite(excess), excess < 0.0, False),
    }
    for level in range(1, 6):
        for prefix in ["bid", "ask"]:
            for suffix in ["px", "sz"]:
                col = f"{prefix}_{suffix}_{level}"
                out[col] = np.repeat(g[col].to_numpy(dtype=float), 3)
    for col in [
        "queue_imbalance",
        "deep_imbalance",
        "top_depth",
        "depth_ratio",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "normalized_spread",
    ]:
        out[col] = np.repeat(g[col].to_numpy(dtype=float), 3)
    return pd.DataFrame(out)


def write_action_part(frame: pd.DataFrame, split: str, session_id: str) -> Path:
    target = split_dir(split)
    target.mkdir(parents=True, exist_ok=True)
    safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
    path = target / f"part-{safe_session}.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd", use_dictionary=True, row_group_size=250_000)
    return path


def build_txf_action_universe() -> pd.DataFrame:
    manifest_path = RESULTS / "txf_action_universe_manifest.csv"
    if manifest_path.exists() and any(action_dataset_dir().glob("split=*/*.parquet")):
        print("TXF-only action universe already exists; reusing generated parquet parts.")
        return pd.read_csv(manifest_path)

    if action_dataset_dir().exists():
        shutil.rmtree(action_dataset_dir())
    action_dataset_dir().mkdir(parents=True, exist_ok=True)

    registry = registry_for_04b()
    malformed_logs: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for (source_group, calendar_session), rows in registry.groupby(["source_group", "session_id"], sort=True):
        split = str(rows["split"].iloc[0])
        raw_txf, logs = load_session_txf_rows(rows)
        malformed_logs.extend(logs)
        raw_txf_rows = int(len(raw_txf))
        if raw_txf.empty:
            filter_rows.append(
                {
                    "source_group": source_group,
                    "split": split,
                    "calendar_session_id": calendar_session,
                    "raw_txf_rows": 0,
                    "valid_book_rows": 0,
                    "boundary_kept_rows": 0,
                    "action_rows": 0,
                }
            )
            continue
        for product, group in raw_txf.groupby("product_id", sort=True):
            featured = add_features(group)
            valid_book_rows = int(valid_book_mask(featured).sum())
            action = build_action_frame(group)
            boundary_kept = int(len(action) // 3)
            session_id = f"{source_group}_{calendar_session}_{product}"
            action_rows = int(len(action))
            filter_rows.append(
                {
                    "source_group": source_group,
                    "split": split,
                    "calendar_session_id": calendar_session,
                    "product_id": product,
                    "raw_txf_rows": int(len(group)),
                    "valid_book_rows": valid_book_rows,
                    "boundary_kept_rows": boundary_kept,
                    "action_rows": action_rows,
                    "boundary_first_ticks": BOUNDARY_FIRST_TICKS,
                    "boundary_last_ticks": BOUNDARY_LAST_TICKS,
                }
            )
            if action.empty:
                continue
            path = write_action_part(action, split=split, session_id=session_id)
            manifest_rows.append(
                {
                    "split": split,
                    "source_group": source_group,
                    "calendar_session_id": calendar_session,
                    "session_id": session_id,
                    "product_id": product,
                    "source_rows": boundary_kept,
                    "action_rows": action_rows,
                    "part_path": str(path),
                }
            )
            print(f"built {split} {session_id}: source_rows={boundary_kept:,} action_rows={action_rows:,}")

    manifest = pd.DataFrame(manifest_rows)
    filter_report = pd.DataFrame(filter_rows)
    malformed = pd.DataFrame(malformed_logs)
    manifest.to_csv(RESULTS / "txf_action_universe_manifest.csv", index=False)
    filter_report.to_csv(RESULTS / "txf_action_universe_filter_report.csv", index=False)
    malformed.to_csv(DATA / "malformed_info_time_drop_log.csv", index=False)
    split_counts = (
        manifest.groupby("split", as_index=False)
        .agg(
            sessions=("session_id", "nunique"),
            source_rows=("source_rows", "sum"),
            action_rows=("action_rows", "sum"),
        )
        .sort_values("split")
    )
    split_counts.to_csv(RESULTS / "txf_action_counts_by_split.csv", index=False)
    write_markdown(
        ROOT / "TXF_ACTION_UNIVERSE_REPORT.md",
        [
            "# TXF-only Action Universe Report",
            "",
            "Phase 04B rebuilds the action universe from raw CSVs using the frozen Phase 04A.2 TXF product filter `^TXF[A-Z0-9]*$` and the frozen malformed `info_time` drop-and-log loader contract.",
            "",
            "Dataset role mapping: `NEW2 -> train`, `NEW1 -> val`, `NEW3 -> new3 diagnostic OOS`, `NEW4 -> new4 diagnostic OOS`. `NEW5` is not loaded.",
            "",
            "Each eligible L2 state emits exactly three candidate actions: `bid`, `ask`, and `no_quote`. Labels are H20 future-mid markout labels; `excess_pnl_H20` removes the mechanical half-spread component and remains the main target.",
            "",
            "## Counts by Split",
            "",
            split_counts.to_markdown(index=False),
            "",
            "## Loader Notes",
            "",
            f"Malformed `info_time` rows dropped and logged: `{len(malformed)}`.",
            f"Boundary rule: drop first `{BOUNDARY_FIRST_TICKS}` and last `{BOUNDARY_LAST_TICKS}` events within each session/product stream.",
            "",
            "Output dataset: `data/txf_only_action_universe.parquet/`.",
        ],
    )
    return manifest


def read_event_frame(path: Path) -> pd.DataFrame:
    cols = [
        "split",
        "session_id",
        "calendar_session_id",
        "symbol",
        "event_index",
        "timestamp_us",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    raw = pd.read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(ACTION_ASK)].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int).eq(ACTION_NO_QUOTE)].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet mismatch in {path}")
    if not bid["event_index"].reset_index(drop=True).equals(ask["event_index"].reset_index(drop=True)):
        raise RuntimeError(f"Bid/ask event alignment mismatch in {path}")
    out = bid.drop(columns=["action_slot", "excess_pnl_H20"]).copy()
    out["bid_excess_pnl_H20"] = bid["excess_pnl_H20"].to_numpy(dtype=float)
    out["ask_excess_pnl_H20"] = ask["excess_pnl_H20"].to_numpy(dtype=float)
    return out


def policy_decision(policy_id: str, frame: pd.DataFrame) -> np.ndarray:
    decision = np.zeros(len(frame), dtype=np.int8)
    if policy_id == "no_quote":
        return decision
    if policy_id == "old_bid_only_spread_ge_1":
        spread = pd.to_numeric(frame["spread_ticks"], errors="coerce").to_numpy(dtype=float)
        decision[np.isfinite(spread) & (spread >= 1.0)] = DECISION_BID
        return decision
    if policy_id == "side_by_queue_imbalance_abs_0p1":
        q = pd.to_numeric(frame["queue_imbalance"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(q)
        decision[finite & (q > 0.1)] = DECISION_BID
        decision[finite & (q < -0.1)] = DECISION_ASK
        return decision
    raise ValueError(f"Unknown policy_id: {policy_id}")


def decision_pnl(frame: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frame), dtype=float)
    bid = frame["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = frame["ask_excess_pnl_H20"].to_numpy(dtype=float)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    out[~np.isfinite(out)] = 0.0
    return out


@dataclass
class Metric:
    state_count: int = 0
    selected_count: int = 0
    bid_count: int = 0
    ask_count: int = 0
    pnl_sum: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown: float = 0.0
    equity: float = 0.0
    peak: float = 0.0


def update_metric(metric: Metric, decision: np.ndarray, pnl: np.ndarray) -> None:
    selected = decision != DECISION_NO_QUOTE
    metric.state_count += int(len(decision))
    metric.selected_count += int(selected.sum())
    metric.bid_count += int((decision == DECISION_BID).sum())
    metric.ask_count += int((decision == DECISION_ASK).sum())
    selected_pnl = pnl[selected]
    selected_pnl = selected_pnl[np.isfinite(selected_pnl)]
    metric.pnl_sum += float(selected_pnl.sum())
    metric.win_count += int((selected_pnl > 0.0).sum())
    metric.loss_count += int((selected_pnl < 0.0).sum())
    if len(pnl):
        equity = metric.equity + np.cumsum(pnl)
        peak = np.maximum.accumulate(np.maximum(equity, metric.peak))
        dd = peak - equity
        metric.max_drawdown = max(metric.max_drawdown, float(dd.max()) if len(dd) else 0.0)
        metric.peak = max(metric.peak, float(peak[-1]) if len(peak) else metric.peak)
        metric.equity = float(equity[-1])


def finalize_metric(split: str, policy_id: str, metric: Metric) -> dict[str, Any]:
    selected = max(metric.selected_count, 1)
    return {
        "split": split,
        "policy_id": policy_id,
        "horizon": HORIZON,
        "state_count": metric.state_count,
        "selected_action_count": metric.selected_count,
        "selection_rate": metric.selected_count / metric.state_count if metric.state_count else 0.0,
        "bid_count": metric.bid_count,
        "ask_count": metric.ask_count,
        "total_excess_pnl_H20": metric.pnl_sum,
        "avg_excess_pnl_per_selected": metric.pnl_sum / selected if metric.selected_count else 0.0,
        "win_rate": metric.win_count / selected if metric.selected_count else 0.0,
        "loss_rate": metric.loss_count / selected if metric.selected_count else 0.0,
        "max_drawdown": metric.max_drawdown,
    }


def label_decomposition_audit() -> pd.DataFrame:
    rows = []
    for split in ALL_SPLITS:
        max_abs_error = 0.0
        checked = 0
        for path in part_paths(split):
            raw = pd.read_parquet(path, columns=["action_side", "raw_pnl_H20", "half_spread_component_H20", "excess_pnl_H20"])
            values = (
                pd.to_numeric(raw["raw_pnl_H20"], errors="coerce")
                - pd.to_numeric(raw["half_spread_component_H20"], errors="coerce")
                - pd.to_numeric(raw["excess_pnl_H20"], errors="coerce")
            )
            finite = values.replace([np.inf, -np.inf], np.nan).dropna().abs()
            if len(finite):
                max_abs_error = max(max_abs_error, float(finite.max()))
                checked += int(len(finite))
        rows.append(
            {
                "split": split,
                "horizon": HORIZON,
                "rows_checked": checked,
                "max_abs_reconstruction_error": max_abs_error,
                "formula": "raw_pnl_H20 = half_spread_component_H20 + excess_pnl_H20",
                "excess_pnl_H20_confirmed_main_target": max_abs_error <= 1e-9,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "txf_label_decomposition_audit.csv", index=False)
    return out


def evaluate_non_ml() -> pd.DataFrame:
    rows = []
    for split in ALL_SPLITS:
        metrics = {policy_id: Metric() for policy_id in POLICY_IDS}
        for path in part_paths(split):
            frame = read_event_frame(path)
            for policy_id in POLICY_IDS:
                decision = policy_decision(policy_id, frame)
                update_metric(metrics[policy_id], decision, decision_pnl(frame, decision))
        for policy_id, metric in metrics.items():
            rows.append(finalize_metric(split, policy_id, metric))
        print(f"non-ML evaluated split={split}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "txf_non_ml_baseline_eval.csv", index=False)
    write_markdown(
        ROOT / "TXF_NON_ML_BASELINE_REPORT.md",
        [
            "# TXF-only Non-ML Baseline Report",
            "",
            "Policies evaluated on TXF-only `excess_pnl_H20`. `NEW5` was not loaded or evaluated.",
            "",
            out.to_markdown(index=False),
            "",
            "Interpretation: this is proxy short-horizon excess markout, not true exchange replay or real-world profitability.",
        ],
    )
    return out


def matched_random_trials(policy_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    summary_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for split in ["val", *OOS_SPLITS]:
        event_frames = [read_event_frame(path) for path in part_paths(split)]
        for policy_id in policy_ids:
            required: dict[tuple[str, str], int] = defaultdict(int)
            candidates: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
            model_pnl = 0.0
            for frame in event_frames:
                session = str(frame["session_id"].iloc[0])
                decision = policy_decision(policy_id, frame)
                pnl = decision_pnl(frame, decision)
                model_pnl += float(pnl.sum())
                bid_values = frame["bid_excess_pnl_H20"].to_numpy(dtype=float)
                ask_values = frame["ask_excess_pnl_H20"].to_numpy(dtype=float)
                candidates[(session, "bid")].append(np.nan_to_num(bid_values, nan=0.0))
                candidates[(session, "ask")].append(np.nan_to_num(ask_values, nan=0.0))
                required[(session, "bid")] += int((decision == DECISION_BID).sum())
                required[(session, "ask")] += int((decision == DECISION_ASK).sum())
            candidate_arrays = {key: np.concatenate(vals) for key, vals in candidates.items()}
            trial_values = np.zeros(RANDOM_TRIALS, dtype=float)
            for trial in range(RANDOM_TRIALS):
                total = 0.0
                for key, k in required.items():
                    if k <= 0:
                        continue
                    values = candidate_arrays[key]
                    if k >= len(values):
                        total += float(values.sum())
                    else:
                        idx = rng.choice(len(values), size=k, replace=False)
                        total += float(values[idx].sum())
                trial_values[trial] = total
                trial_rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "trial": trial,
                        "random_total_excess_pnl_H20": total,
                    }
                )
            p_value = float((trial_values >= model_pnl).mean())
            summary_rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "model_total_excess_pnl_H20": model_pnl,
                    "random_trials": RANDOM_TRIALS,
                    "random_mean": float(trial_values.mean()),
                    "random_std": float(trial_values.std(ddof=0)),
                    "random_95p": float(np.quantile(trial_values, 0.95)),
                    "random_99p": float(np.quantile(trial_values, 0.99)),
                    "empirical_p_value_fraction_random_ge_model": p_value,
                    "beats_random_95p": bool(model_pnl > np.quantile(trial_values, 0.95)),
                    "matching": "exact_count_side_session",
                    "test_new3_new4_used_for_selection": False,
                }
            )
            print(f"matched random split={split} policy={policy_id}")
    trials = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(summary_rows)
    trials.to_csv(RESULTS / "txf_matched_random_trials.csv", index=False)
    summary.to_csv(RESULTS / "txf_matched_random_summary.csv", index=False)
    return trials, summary


@dataclass
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def compute_fill_norms() -> FillNorms:
    values = {"top_depth": [], "bid_sz_1": [], "ask_sz_1": []}
    for path in part_paths("train"):
        frame = read_event_frame(path)
        for col in values:
            s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s):
                values[col].append(s.to_numpy(dtype=float))
    med = {}
    for col, chunks in values.items():
        arr = np.concatenate(chunks) if chunks else np.array([1.0])
        positive = arr[arr > 0.0]
        med[col] = float(np.nanmedian(positive)) if len(positive) else 1.0
    return FillNorms(med["top_depth"], med["bid_sz_1"], med["ask_sz_1"])


def stable_uniform(frame: pd.DataFrame, decision: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str).to_numpy(),
            "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "decision": np.asarray(decision, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(abs(hash(salt)) % (2**64))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def fill_probabilities(frame: pd.DataFrame, decision: np.ndarray, norms: FillNorms) -> np.ndarray:
    out = np.zeros(len(frame), dtype=float)
    selected = decision != DECISION_NO_QUOTE
    if not selected.any():
        return out
    top_depth = pd.to_numeric(frame["top_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bid_depth = pd.to_numeric(frame["bid_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ask_depth = pd.to_numeric(frame["ask_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
    side_depth_norm = np.zeros(len(frame), dtype=float)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    side_depth_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
    side_depth_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
    side = 1.0 / (1.0 + np.maximum(side_depth_norm, 0.0))
    out[selected] = np.clip(0.5 * depth[selected] * side[selected], 0.01, 1.0)
    return out


@njit(cache=True)
def simulate_scheduled(decision: np.ndarray, fills: np.ndarray, base_pnl: np.ndarray, horizon: int, inventory_cap: int) -> tuple[float, int, int, int, float, int, int]:
    n = len(decision)
    close_side_sum = np.zeros(n, dtype=np.int64)
    close_pnl_sum = np.zeros(n, dtype=np.float64)
    inventory = 0
    max_abs_inventory = 0
    forced_no_quote = 0
    fill_count = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for t in range(n):
        if close_side_sum[t] != 0 or close_pnl_sum[t] != 0.0:
            inventory -= close_side_sum[t]
            equity += close_pnl_sum[t]
            total_pnl += close_pnl_sum[t]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_drawdown:
                max_drawdown = dd
        if not fills[t]:
            continue
        side_sign = 1 if decision[t] == DECISION_BID else -1
        next_inventory = inventory + side_sign
        if abs(next_inventory) > inventory_cap:
            forced_no_quote += 1
            continue
        trade_pnl = base_pnl[t]
        inventory = next_inventory
        if abs(inventory) > max_abs_inventory:
            max_abs_inventory = abs(inventory)
        close_idx = t + horizon
        if close_idx >= n:
            close_idx = n - 1
        close_side_sum[close_idx] += side_sign
        close_pnl_sum[close_idx] += trade_pnl
        fill_count += 1
        if trade_pnl > 0.0:
            win_count += 1
        elif trade_pnl < 0.0:
            loss_count += 1
    return total_pnl, fill_count, forced_no_quote, max_abs_inventory, max_drawdown, win_count, loss_count


def causal_fill_sequential(policy_ids: list[str]) -> pd.DataFrame:
    norms = compute_fill_norms()
    rows = []
    for split in ["val", *OOS_SPLITS]:
        for policy_id in policy_ids:
            accum = {
                "pnl": 0.0,
                "fill_count": 0,
                "forced": 0,
                "max_inv": 0,
                "max_dd": 0.0,
                "win": 0,
                "loss": 0,
            }
            for path in part_paths(split):
                frame = read_event_frame(path)
                decision = policy_decision(policy_id, frame)
                probs = fill_probabilities(frame, decision, norms)
                fills = (decision != DECISION_NO_QUOTE) & (stable_uniform(frame, decision, "04b_conservative_l2_stress") < probs)
                pnl = decision_pnl(frame, decision)
                result = simulate_scheduled(decision.astype(np.int8), fills.astype(np.bool_), pnl.astype(np.float64), HORIZON, 1)
                accum["pnl"] += float(result[0])
                accum["fill_count"] += int(result[1])
                accum["forced"] += int(result[2])
                accum["max_inv"] = max(accum["max_inv"], int(result[3]))
                accum["max_dd"] = max(accum["max_dd"], float(result[4]))
                accum["win"] += int(result[5])
                accum["loss"] += int(result[6])
            fills_n = max(accum["fill_count"], 1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "horizon": HORIZON,
                    "fill_model": "conservative_l2_stress",
                    "inventory_cap": 1,
                    "sequential_excess_pnl": accum["pnl"],
                    "fill_count": accum["fill_count"],
                    "forced_no_quote_count": accum["forced"],
                    "max_abs_inventory": accum["max_inv"],
                    "max_drawdown": accum["max_dd"],
                    "win_rate": accum["win"] / fills_n if accum["fill_count"] else 0.0,
                    "loss_rate": accum["loss"] / fills_n if accum["fill_count"] else 0.0,
                    "avg_trade_excess_pnl": accum["pnl"] / fills_n if accum["fill_count"] else 0.0,
                    "fill_uses_future_pnl": False,
                }
            )
            print(f"sequential split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "txf_causal_fill_sequential_eval.csv", index=False)
    return out


def action_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    slot = out["action_slot"].astype(np.int8).to_numpy()
    out["action_is_bid"] = (slot == ACTION_BID).astype(np.float32)
    out["action_is_ask"] = (slot == ACTION_ASK).astype(np.float32)
    side_sign = np.zeros(len(out), dtype=np.float32)
    side_sign[slot == ACTION_BID] = 1.0
    side_sign[slot == ACTION_ASK] = -1.0
    out["side_sign"] = side_sign
    out["side_queue_imbalance"] = side_sign * pd.to_numeric(out["queue_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_deep_imbalance"] = side_sign * pd.to_numeric(out["deep_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_imbalance_change_past"] = side_sign * pd.to_numeric(out["imbalance_change_past"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    for col in ML_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["target_excess_pnl_H20"] = pd.to_numeric(out["excess_pnl_H20"], errors="coerce").fillna(0.0).astype(np.float32)
    out["target_positive_excess_H20"] = out["target_excess_pnl_H20"] > 0.0
    return out


def sample_action_rows(split: str, max_rows: int, seed: int) -> pd.DataFrame:
    paths = part_paths(split)
    if not paths:
        return pd.DataFrame()
    per_part = max(1, math.ceil(max_rows / len(paths)))
    rng = np.random.default_rng(seed)
    frames = []
    cols = [
        "split",
        "session_id",
        "event_index",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    for path in paths:
        raw = pd.read_parquet(path, columns=cols)
        if len(raw) > per_part:
            raw = raw.sample(n=per_part, random_state=int(rng.integers(0, 2**31 - 1)))
        frames.append(raw)
    out = pd.concat(frames, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return action_feature_frame(out.reset_index(drop=True))


def safe_spearman(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 3:
        return float("nan")
    return float(pd.Series(score).corr(pd.Series(target), method="spearman"))


def build_models() -> dict[str, Any]:
    return {
        "ridge_l2": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", C=0.5, solver="saga", max_iter=200, n_jobs=-1, random_state=RANDOM_SEED),
        ),
        "hgb_regressor_shallow": HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
        ),
    }


def model_scores(model: Any, model_id: str, matrix: pd.DataFrame) -> np.ndarray:
    x = matrix[ML_FEATURES].to_numpy(dtype=np.float32)
    if model_id.startswith("logistic"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    return model.predict(x).astype(np.float32)


def train_simple_ml() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    warnings.filterwarnings("ignore", category=UserWarning)
    if any(col in ML_FEATURES for col in ML_FORBIDDEN_FEATURES):
        raise RuntimeError("Forbidden ML feature included in Phase 04B feature schema.")
    train = sample_action_rows("train", TRAIN_SAMPLE_ROWS, RANDOM_SEED)
    val = sample_action_rows("val", VAL_SAMPLE_ROWS, RANDOM_SEED + 1)
    x_train = train[ML_FEATURES].to_numpy(dtype=np.float32)
    y_train_reg = train["target_excess_pnl_H20"].to_numpy(dtype=np.float32)
    y_train_cls = train["target_positive_excess_H20"].astype(int).to_numpy()
    x_val = val[ML_FEATURES].to_numpy(dtype=np.float32)
    y_val_reg = val["target_excess_pnl_H20"].to_numpy(dtype=np.float32)
    y_val_cls = val["target_positive_excess_H20"].astype(int).to_numpy()

    metric_rows = []
    convergence_rows = []
    for model_id, model in build_models().items():
        is_classifier = model_id.startswith("logistic")
        y_train = y_train_cls if is_classifier else y_train_reg
        model.fit(x_train, y_train)
        joblib.dump(model, MODELS / f"{model_id}.joblib")
        score = model_scores(model, model_id, val)
        rmse = float(np.sqrt(mean_squared_error(y_val_reg, score))) if not is_classifier else float("nan")
        mae = float(mean_absolute_error(y_val_reg, score)) if not is_classifier else float("nan")
        pr_auc = float(average_precision_score(y_val_cls, score)) if len(np.unique(y_val_cls)) > 1 else float("nan")
        roc_auc = float(roc_auc_score(y_val_cls, score)) if len(np.unique(y_val_cls)) > 1 else float("nan")
        metric_rows.append(
            {
                "model_id": model_id,
                "train_sample_rows": len(train),
                "val_sample_rows": len(val),
                "feature_count": len(ML_FEATURES),
                "val_rmse_vs_excess": rmse,
                "val_mae_vs_excess": mae,
                "val_pr_auc_positive_excess": pr_auc,
                "val_roc_auc_positive_excess": roc_auc,
                "val_spearman_score_vs_excess": safe_spearman(score, y_val_reg),
                "test_new3_new4_used_for_training_or_selection": False,
            }
        )
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        raw_n_iter = getattr(estimator, "n_iter_", "")
        raw_max_iter = getattr(estimator, "max_iter", "")
        n_iter_value = ""
        if raw_n_iter is not None and raw_n_iter != "":
            try:
                n_iter_value = int(np.max(raw_n_iter))
            except TypeError:
                n_iter_value = str(raw_n_iter)
        hit_max_iter = False
        if raw_max_iter is not None and raw_max_iter != "" and n_iter_value != "":
            try:
                hit_max_iter = int(n_iter_value) >= int(raw_max_iter)
            except (TypeError, ValueError):
                hit_max_iter = False
        convergence_rows.append(
            {
                "model_id": model_id,
                "n_iter": n_iter_value,
                "max_iter": "" if raw_max_iter is None else raw_max_iter,
                "hit_max_iter": hit_max_iter,
                "convergence_note": "diagnostic only; model/threshold selection uses validation only",
            }
        )
        print(f"trained {model_id}")
    metrics = pd.DataFrame(metric_rows)
    convergence = pd.DataFrame(convergence_rows)
    metrics.to_csv(RESULTS / "txf_simple_ml_training_metrics.csv", index=False)
    convergence.to_csv(RESULTS / "txf_ml_convergence_log.csv", index=False)
    return metrics, convergence, {"train": train, "val": val}


def action_matrix_from_part(path: Path) -> pd.DataFrame:
    cols = [
        "split",
        "session_id",
        "event_index",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    return action_feature_frame(pd.read_parquet(path, columns=cols))


def ml_policy_decision_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    bid = scores[0::3]
    ask = scores[1::3]
    n = len(bid)
    decision = np.zeros(n, dtype=np.int8)
    choose_bid = (bid >= ask) & (bid > threshold)
    choose_ask = (ask > bid) & (ask > threshold)
    decision[choose_bid] = DECISION_BID
    decision[choose_ask] = DECISION_ASK
    return decision


def evaluate_ml_thresholds() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    val_threshold_rows = []
    eval_rows = []
    selected_policy: dict[str, Any] | None = None
    best_val = -np.inf
    for model_path in sorted(MODELS.glob("*.joblib")):
        model_id = model_path.stem
        model = joblib.load(model_path)
        val_sample = sample_action_rows("val", 100_000, RANDOM_SEED + 9)
        sample_scores = model_scores(model, model_id, val_sample)
        finite_scores = sample_scores[np.isfinite(sample_scores)]
        if model_id.startswith("logistic"):
            thresholds = sorted(set([0.5, *np.quantile(finite_scores, [0.5, 0.75, 0.90, 0.95]).tolist()]))
        else:
            thresholds = sorted(set([-50.0, -15.0, 0.0, 15.0, *np.quantile(finite_scores, [0.5, 0.75, 0.90, 0.95]).tolist()]))
        for threshold in thresholds:
            metric = Metric()
            for path in part_paths("val"):
                matrix = action_matrix_from_part(path)
                scores = model_scores(model, model_id, matrix)
                decision = ml_policy_decision_from_scores(scores, float(threshold))
                event = read_event_frame(path)
                update_metric(metric, decision, decision_pnl(event, decision))
            row = finalize_metric("val", f"ml_{model_id}", metric)
            row["model_id"] = model_id
            row["threshold"] = float(threshold)
            row["selection_split"] = "val"
            row["test_new3_new4_used_for_selection"] = False
            val_threshold_rows.append(row)
            if row["total_excess_pnl_H20"] > best_val and row["selected_action_count"] > 0:
                best_val = float(row["total_excess_pnl_H20"])
                selected_policy = {
                    "policy_id": f"ml_{model_id}",
                    "model_id": model_id,
                    "threshold": float(threshold),
                    "selection_split": "val",
                    "validation_total_excess_pnl_H20": float(row["total_excess_pnl_H20"]),
                    "validation_selected_action_count": int(row["selected_action_count"]),
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                }
    if selected_policy is None:
        selected_policy = {
            "policy_id": "no_ml_policy_selected",
            "model_id": "",
            "threshold": 0.0,
            "selection_split": "val",
            "validation_total_excess_pnl_H20": 0.0,
            "validation_selected_action_count": 0,
            "test_new3_new4_used_for_selection": False,
            "new5_used": False,
        }
    write_json(RESULTS / "txf_selected_ml_policy.json", selected_policy)
    val_grid = pd.DataFrame(val_threshold_rows)
    val_grid.to_csv(RESULTS / "txf_ml_validation_threshold_grid.csv", index=False)

    if selected_policy["model_id"]:
        model = joblib.load(MODELS / f"{selected_policy['model_id']}.joblib")
        for split in ["val", *OOS_SPLITS]:
            metric = Metric()
            for path in part_paths(split):
                matrix = action_matrix_from_part(path)
                scores = model_scores(model, selected_policy["model_id"], matrix)
                decision = ml_policy_decision_from_scores(scores, float(selected_policy["threshold"]))
                event = read_event_frame(path)
                update_metric(metric, decision, decision_pnl(event, decision))
            row = finalize_metric(split, selected_policy["policy_id"], metric)
            row["model_id"] = selected_policy["model_id"]
            row["threshold"] = selected_policy["threshold"]
            row["selected_on_val_only"] = True
            eval_rows.append(row)
    ml_eval = pd.DataFrame(eval_rows)
    ml_eval.to_csv(RESULTS / "txf_ml_selected_policy_eval.csv", index=False)
    return val_grid, selected_policy, ml_eval


def ml_diagnostics(selected_policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not selected_policy.get("model_id"):
        empty = pd.DataFrame()
        empty.to_csv(RESULTS / "txf_ml_confusion_matrices.csv", index=False)
        empty.to_csv(RESULTS / "txf_ml_feature_importance.csv", index=False)
        return empty, empty
    model_id = selected_policy["model_id"]
    model = joblib.load(MODELS / f"{model_id}.joblib")
    threshold = float(selected_policy["threshold"])
    confusion_rows = []
    for split in ["val", *OOS_SPLITS]:
        sample = sample_action_rows(split, CONFUSION_SAMPLE_ROWS, RANDOM_SEED + len(split))
        scores = model_scores(model, model_id, sample)
        pred = scores > threshold
        y = sample["target_positive_excess_H20"].astype(bool).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
        confusion_rows.append(
            {
                "split": split,
                "model_id": model_id,
                "threshold_selected_on_val": threshold,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "test_new3_new4_used_for_selection": False,
            }
        )
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(RESULTS / "txf_ml_confusion_matrices.csv", index=False)

    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).reshape(-1)
        importance = pd.DataFrame({"feature": ML_FEATURES, "importance": np.abs(coef), "signed_value": coef})
    else:
        val = sample_action_rows("val", 50_000, RANDOM_SEED + 22)
        base_score = model_scores(model, model_id, val)
        base = safe_spearman(base_score, val["target_excess_pnl_H20"].to_numpy(dtype=float))
        rows = []
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for feature in ML_FEATURES:
            shuffled = val.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            score = model_scores(model, model_id, shuffled)
            rows.append({"feature": feature, "importance": base - safe_spearman(score, shuffled["target_excess_pnl_H20"].to_numpy(dtype=float)), "signed_value": np.nan})
        importance = pd.DataFrame(rows)
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(RESULTS / "txf_ml_feature_importance.csv", index=False)
    return confusion, importance


def write_ml_reports(metrics: pd.DataFrame, convergence: pd.DataFrame, val_grid: pd.DataFrame, selected_policy: dict[str, Any], ml_eval: pd.DataFrame, confusion: pd.DataFrame, importance: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "TXF_SIMPLE_ML_REPORT.md",
        [
            "# TXF-only Simple ML Report",
            "",
            "Simple ML was trained only on TXF-only training data (`NEW2 -> train`) and selected only on TXF-only validation data (`NEW1 -> val`). `NEW3/NEW4` were frozen diagnostic OOS. `NEW5` was not loaded.",
            "",
            "Forbidden features excluded: `best_bid`, `best_ask`, `mid_price`, `quote_price`, `product_id`, and all future labels except as targets.",
            "",
            "## Training Metrics",
            "",
            metrics.to_markdown(index=False),
            "",
            "## Validation Threshold Selection",
            "",
            f"Selected ML policy: `{selected_policy}`",
            "",
            val_grid.sort_values("total_excess_pnl_H20", ascending=False).head(15).to_markdown(index=False) if not val_grid.empty else "No validation grid rows.",
            "",
            "## Frozen Diagnostic OOS Evaluation",
            "",
            ml_eval.to_markdown(index=False) if not ml_eval.empty else "No ML policy selected.",
        ],
    )
    write_markdown(
        ROOT / "TXF_ML_TRAINING_DIAGNOSTICS_REPORT.md",
        [
            "# TXF-only ML Training Diagnostics Report",
            "",
            "Diagnostics are for simple Ridge/Logistic/shallow HGB only. No TCN/PHC/deep sequence training was run.",
            "",
            "## Convergence Log",
            "",
            convergence.to_markdown(index=False),
            "",
            "## Confusion Matrices",
            "",
            confusion.to_markdown(index=False) if not confusion.empty else "No selected ML policy.",
            "",
            "## Feature Importance",
            "",
            importance.head(25).to_markdown(index=False) if not importance.empty else "No feature importance available.",
        ],
    )


def final_decision(non_ml: pd.DataFrame, random_summary: pd.DataFrame, sequential: pd.DataFrame, ml_eval: pd.DataFrame) -> None:
    candidate_rows = []
    for policy_id in POLICY_IDS:
        if policy_id == "no_quote":
            continue
        pnl_ok = bool(
            non_ml[
                non_ml["split"].isin(["val", *OOS_SPLITS]) & non_ml["policy_id"].eq(policy_id)
            ]["total_excess_pnl_H20"].gt(0.0).all()
        )
        random_ok = bool(
            random_summary[
                random_summary["split"].isin(["val", *OOS_SPLITS]) & random_summary["policy_id"].eq(policy_id)
            ]["beats_random_95p"].astype(bool).all()
        )
        seq_ok = bool(
            sequential[
                sequential["split"].isin(["val", *OOS_SPLITS]) & sequential["policy_id"].eq(policy_id)
            ]["sequential_excess_pnl"].gt(0.0).all()
        )
        candidate_rows.append(
            {
                "candidate_policy_id": policy_id,
                "strategy_family": "non_ml",
                "positive_excess_pnl_val_new3_new4": pnl_ok,
                "beats_matched_random_95p_val_new3_new4": random_ok,
                "positive_causal_fill_sequential_val_new3_new4": seq_ok,
                "eligible_for_new5_holdout": bool(pnl_ok and random_ok and seq_ok),
            }
        )
    if not ml_eval.empty:
        ml_policy_id = str(ml_eval["policy_id"].iloc[0])
        pnl_ok = bool(ml_eval[ml_eval["split"].isin(["val", *OOS_SPLITS])]["total_excess_pnl_H20"].gt(0.0).all())
        # Matched random for ML is not required as a non-ML rerun gate in this phase unless generated separately.
        queue = non_ml[non_ml["policy_id"].eq("side_by_queue_imbalance_abs_0p1")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "queue_pnl"})
        ml = ml_eval[["split", "total_excess_pnl_H20"]].merge(queue, on="split", how="left")
        beats_queue = bool((ml[ml["split"].isin(["val", *OOS_SPLITS])]["total_excess_pnl_H20"] > ml[ml["split"].isin(["val", *OOS_SPLITS])]["queue_pnl"]).all())
        candidate_rows.append(
            {
                "candidate_policy_id": ml_policy_id,
                "strategy_family": "simple_ml",
                "positive_excess_pnl_val_new3_new4": pnl_ok,
                "beats_matched_random_95p_val_new3_new4": False,
                "positive_causal_fill_sequential_val_new3_new4": False,
                "beats_queue_rule_val_new3_new4": beats_queue,
                "eligible_for_new5_holdout": False,
                "eligibility_note": "diagnostic_only_until_matched_random_and_causal_fill_hurdles_are_run",
            }
        )
    scorecard = pd.DataFrame(candidate_rows)
    scorecard.to_csv(RESULTS / "txf_final_candidate_scorecard.csv", index=False)
    eligible = scorecard[scorecard["eligible_for_new5_holdout"].astype(bool)]
    if eligible.empty:
        decision = "no_strategy_ready_for_new5"
        selected = ""
    else:
        selected = str(eligible.iloc[0]["candidate_policy_id"])
        decision = "one_candidate_ready_for_new5_holdout_metadata_locked"
    write_markdown(
        ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT.md",
        [
            "# TXF-only Final Development Decision Report",
            "",
            f"Final decision: `{decision}`.",
            f"Selected candidate for upcoming NEW5 holdout: `{selected}`." if selected else "Selected candidate for upcoming NEW5 holdout: none.",
            "",
            "This decision is based only on TXF-only development validation plus NEW3/NEW4 diagnostic OOS. NEW5 was not loaded or evaluated.",
            "",
            "A strategy must remain positive under TXF-only excess markout, beat matched random 95p where applicable, and survive the causal-fill sequential proxy before being proposed for NEW5.",
            "",
            "## Candidate Scorecard",
            "",
            scorecard.to_markdown(index=False),
            "",
            "## Non-ML Summary",
            "",
            non_ml.to_markdown(index=False),
            "",
            "## Matched Random Summary",
            "",
            random_summary.to_markdown(index=False),
            "",
            "When `empirical_p_value_fraction_random_ge_model` is `0` in the CSV, it means that in 1000 exact matched random trials no random trial achieved PnL greater than or equal to the model/policy PnL; the advisor-safe wording is empirical p-value `< 0.001`, not exactly zero.",
            "",
            "## Causal-Fill Sequential Summary",
            "",
            sequential.to_markdown(index=False),
            "",
            "## Simple ML Selected Policy Summary",
            "",
            ml_eval.to_markdown(index=False) if not ml_eval.empty else "No ML selected policy evaluation available.",
            "",
            "Allowed claim: TXF-only development rerun completed under frozen product and loader contracts.",
            "Forbidden claims: real-world profitability, true exchange replay, FPGA PnL improvement, or any NEW5 result.",
        ],
    )


def main() -> None:
    ensure_dirs()
    manifest = build_txf_action_universe()
    if manifest.empty:
        raise RuntimeError("TXF-only action universe is empty.")
    label_path = RESULTS / "txf_label_decomposition_audit.csv"
    label_decomposition = pd.read_csv(label_path) if label_path.exists() else label_decomposition_audit()
    non_ml_path = RESULTS / "txf_non_ml_baseline_eval.csv"
    non_ml = pd.read_csv(non_ml_path) if non_ml_path.exists() else evaluate_non_ml()
    random_summary_path = RESULTS / "txf_matched_random_summary.csv"
    if random_summary_path.exists() and (RESULTS / "txf_matched_random_trials.csv").exists():
        random_summary = pd.read_csv(random_summary_path)
    else:
        _, random_summary = matched_random_trials(["old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"])
    sequential_path = RESULTS / "txf_causal_fill_sequential_eval.csv"
    sequential = pd.read_csv(sequential_path) if sequential_path.exists() else causal_fill_sequential(POLICY_IDS)
    metrics, convergence, _ = train_simple_ml()
    val_grid, selected_policy, ml_eval = evaluate_ml_thresholds()
    confusion, importance = ml_diagnostics(selected_policy)
    write_ml_reports(metrics, convergence, val_grid, selected_policy, ml_eval, confusion, importance)
    final_decision(non_ml, random_summary, sequential, ml_eval)
    print("Phase 04B TXF-only final development rerun complete.")
    print(ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT.md")
    print("Label decomposition:")
    print(label_decomposition.to_string(index=False))


if __name__ == "__main__":
    main()
