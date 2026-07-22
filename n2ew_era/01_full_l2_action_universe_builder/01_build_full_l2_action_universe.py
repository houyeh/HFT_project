from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common_phase01 import (
    ACTION_SIDES,
    BOUNDARY_FIRST_TICKS,
    BOUNDARY_LAST_TICKS,
    DATA_DIR,
    FULL_DATASET_DIR,
    HORIZONS,
    MAKER_FEE,
    NEW4_SPLIT,
    POINT_VALUE,
    RESULTS_DIR,
    SEVERE_NEGATIVE_PNL,
    SOURCE_COLUMNS,
    SOURCE_FILTER_REPORT,
    SOURCE_NEW4_DIR,
    SOURCE_QUOTE_ELIGIBLE,
    ensure_dirs,
    split_dataset_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
for path in [PROJECT_ROOT, V26_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from new_era018_helpers import detect_symbol_from_raw_dir, infer_tick_size_simple  # noqa: E402


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def finite_positive(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values) & (values > 0.0)
    return mask


def sequential_filter_counts(rule_masks: dict[str, np.ndarray]) -> tuple[dict[str, int], np.ndarray]:
    if not rule_masks:
        return {}, np.array([], dtype=bool)
    n = len(next(iter(rule_masks.values())))
    remaining = np.ones(n, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in rule_masks.items():
        keep = np.asarray(mask, dtype=bool)
        dropped = remaining & ~keep
        counts[f"dropped_{name}"] = int(dropped.sum())
        remaining &= keep
    return counts, remaining


def historical_filter_counts_by_split() -> pd.DataFrame:
    report = pd.read_csv(SOURCE_FILTER_REPORT)
    split_report = report[report["scope"].astype(str) == "split"].copy()
    rows: list[dict[str, Any]] = []
    for row in split_report.itertuples(index=False):
        raw_rows = int(row.raw_rows)
        dropped_invalid = int(row.dropped_invalid_bid_ask)
        dropped_nonpositive = int(row.dropped_nonpositive_spread)
        dropped_missing_depth = int(row.dropped_missing_depth)
        valid_book = raw_rows - dropped_invalid - dropped_nonpositive - dropped_missing_depth
        rows.append(
            {
                "split": str(row.split),
                "dataset": str(row.dataset),
                "symbol": str(row.symbol),
                "source_path": str(SOURCE_QUOTE_ELIGIBLE),
                "raw_rows": raw_rows,
                "valid_book_rows": int(valid_book),
                "quote_eligible_rows": int(row.eligible_rows),
                "dropped_invalid_bid_ask": dropped_invalid,
                "dropped_nonpositive_spread": dropped_nonpositive,
                "dropped_missing_depth": dropped_missing_depth,
                "dropped_session_boundary": int(row.dropped_session_boundary),
                "dropped_feature_na": int(row.dropped_feature_na),
                "boundary_first_ticks": int(row.boundary_first_ticks),
                "boundary_last_ticks": int(row.boundary_last_ticks),
                "tick_size": float(row.tick_size),
                "new4_loaded": False,
                "new4_compatibility_status": "",
            }
        )
    return pd.DataFrame(rows)


def build_new4_quote_eligible() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "source_new4_dir": str(SOURCE_NEW4_DIR),
        "new4_loaded": False,
        "new4_compatibility_status": "missing",
    }
    if not SOURCE_NEW4_DIR.exists():
        return pd.DataFrame(columns=SOURCE_COLUMNS), pd.DataFrame(), metadata

    symbol = detect_symbol_from_raw_dir(SOURCE_NEW4_DIR, preferred="TXFA6")
    lob_frames, _ = load_new_tw_data(SOURCE_NEW4_DIR, [symbol])
    if symbol not in lob_frames or lob_frames[symbol].empty:
        metadata.update({"symbol": symbol, "new4_compatibility_status": "no_lob_frame"})
        return pd.DataFrame(columns=SOURCE_COLUMNS), pd.DataFrame(), metadata

    lob = lob_frames[symbol].sort_values(["session_id", "timestamp"]).reset_index(drop=True).copy()
    tick_size = float(infer_tick_size_simple(lob))
    metadata.update(
        {
            "new4_loaded": True,
            "new4_compatibility_status": "loaded_raw_l2_5_levels",
            "symbol": symbol,
            "tick_size": tick_size,
            "raw_rows": int(len(lob)),
            "session_count": int(lob["session_id"].nunique()),
        }
    )

    output_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    depth_cols: list[str] = []
    for level in range(1, 6):
        depth_cols.extend([f"bid_px_{level}", f"bid_vol_{level}", f"ask_px_{level}", f"ask_vol_{level}"])

    for session_id, group in lob.groupby("session_id", sort=True):
        g = group.sort_values("timestamp").reset_index(drop=True).copy()
        raw_rows = int(len(g))
        spread = pd.to_numeric(g["ask_px_1"], errors="coerce") - pd.to_numeric(g["bid_px_1"], errors="coerce")
        idx = np.arange(raw_rows)
        rule_masks = {
            "invalid_bid_ask": finite_positive(g, ["bid_px_1", "ask_px_1"]),
            "nonpositive_spread": np.isfinite(spread.to_numpy(dtype=float)) & (spread.to_numpy(dtype=float) > 0.0),
            "missing_depth": finite_positive(g, depth_cols),
            "session_boundary": (idx >= int(BOUNDARY_FIRST_TICKS)) & (idx < max(raw_rows - int(BOUNDARY_LAST_TICKS), 0)),
            "feature_na": np.ones(raw_rows, dtype=bool),
        }
        drop_counts, eligible_mask = sequential_filter_counts(rule_masks)
        local_row_id = np.flatnonzero(eligible_mask).astype(np.int64)

        out = g.loc[eligible_mask, ["timestamp", *depth_cols]].copy()
        out.insert(0, "dataset", "NEW4")
        out.insert(1, "symbol", symbol)
        out.insert(2, "session_id", str(session_id))
        out.insert(3, "split", NEW4_SPLIT)
        out.insert(4, "local_row_id", local_row_id)
        out["tick_size"] = tick_size
        out["mid_price"] = (out["bid_px_1"].astype(float) + out["ask_px_1"].astype(float)) / 2.0
        out["spread_ticks"] = (out["ask_px_1"].astype(float) - out["bid_px_1"].astype(float)) / max(tick_size, 1e-9)
        out["quote_eligible"] = True
        for col in ["queue_imbalance", "deep_obi", "queue_imbalance_change", "normalized_spread", "trade_intensity_short", "cancel_pressure_short"]:
            out[col] = np.nan
        output_frames.append(out[SOURCE_COLUMNS])

        report_rows.append(
            {
                "split": NEW4_SPLIT,
                "dataset": "NEW4",
                "symbol": symbol,
                "source_path": str(SOURCE_NEW4_DIR),
                "session_id": str(session_id),
                "raw_rows": raw_rows,
                "valid_book_rows": int(raw_rows - drop_counts["dropped_invalid_bid_ask"] - drop_counts["dropped_nonpositive_spread"] - drop_counts["dropped_missing_depth"]),
                "quote_eligible_rows": int(eligible_mask.sum()),
                **drop_counts,
                "boundary_first_ticks": int(BOUNDARY_FIRST_TICKS),
                "boundary_last_ticks": int(BOUNDARY_LAST_TICKS),
                "tick_size": tick_size,
                "new4_loaded": True,
                "new4_compatibility_status": "loaded_raw_l2_5_levels",
            }
        )

    if not output_frames:
        metadata["new4_compatibility_status"] = "no_eligible_sessions"
        return pd.DataFrame(columns=SOURCE_COLUMNS), pd.DataFrame(), metadata

    frame = pd.concat(output_frames, ignore_index=True)
    session_report = pd.DataFrame(report_rows)
    split_row = {
        "split": NEW4_SPLIT,
        "dataset": "NEW4",
        "symbol": symbol,
        "source_path": str(SOURCE_NEW4_DIR),
        "session_id": "__ALL__",
        "raw_rows": int(session_report["raw_rows"].sum()),
        "valid_book_rows": int(session_report["valid_book_rows"].sum()),
        "quote_eligible_rows": int(session_report["quote_eligible_rows"].sum()),
        "dropped_invalid_bid_ask": int(session_report["dropped_invalid_bid_ask"].sum()),
        "dropped_nonpositive_spread": int(session_report["dropped_nonpositive_spread"].sum()),
        "dropped_missing_depth": int(session_report["dropped_missing_depth"].sum()),
        "dropped_session_boundary": int(session_report["dropped_session_boundary"].sum()),
        "dropped_feature_na": int(session_report["dropped_feature_na"].sum()),
        "boundary_first_ticks": int(BOUNDARY_FIRST_TICKS),
        "boundary_last_ticks": int(BOUNDARY_LAST_TICKS),
        "tick_size": tick_size,
        "new4_loaded": True,
        "new4_compatibility_status": "loaded_raw_l2_5_levels",
    }
    metadata["quote_eligible_rows"] = int(len(frame))
    return frame, pd.DataFrame([split_row]), metadata


def add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True).copy()
    out = out.rename(columns={f"bid_vol_{i}": f"bid_sz_{i}" for i in range(1, 6)} | {f"ask_vol_{i}": f"ask_sz_{i}" for i in range(1, 6)})
    out["best_bid"] = out["bid_px_1"].astype(float)
    out["best_ask"] = out["ask_px_1"].astype(float)

    bid_top = out["bid_sz_1"].astype(float)
    ask_top = out["ask_sz_1"].astype(float)
    bid_depth = sum(out[f"bid_sz_{i}"].astype(float) for i in range(1, 6))
    ask_depth = sum(out[f"ask_sz_{i}"].astype(float) for i in range(1, 6))
    out["queue_imbalance"] = safe_ratio(bid_top - ask_top, bid_top + ask_top)
    out["deep_imbalance"] = safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)
    out["top_depth"] = (bid_top + ask_top).fillna(0.0)
    out["depth_ratio"] = safe_ratio(bid_depth, ask_depth)
    out["normalized_spread"] = out["spread_ticks"].astype(float)

    parts: list[pd.DataFrame] = []
    for _, group in out.groupby("session_id", sort=False):
        g = group.sort_values("local_row_id").copy()
        timestamp = g["timestamp"].astype(float)
        dt_ns = timestamp.diff()
        g["inter_arrival_ms"] = (dt_ns / 1_000_000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        event_rate = (1_000_000_000.0 / dt_ns.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1_000_000.0)
        g["event_rate_past_50"] = event_rate.rolling(50, min_periods=3).mean().fillna(0.0)
        mid_ret = g["mid_price"].astype(float).diff()
        g["volatility_past_50"] = mid_ret.rolling(50, min_periods=3).std().fillna(0.0)
        g["abs_return_past_50"] = mid_ret.abs().rolling(50, min_periods=3).mean().fillna(0.0)
        g["imbalance_change_past"] = g["queue_imbalance"].astype(float).diff().fillna(0.0)
        g["spread_change_past"] = g["spread_ticks"].astype(float).diff().fillna(0.0)
        g["hawkes_excite"] = event_rate.ewm(span=50, adjust=False).mean().fillna(0.0)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out


def repeat_series(frame: pd.DataFrame, col: str, dtype: str | None = None) -> np.ndarray:
    arr = frame[col].to_numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return np.repeat(arr, len(ACTION_SIDES))


def make_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_causal_features(frame)
    n = len(frame)
    side = np.tile(np.asarray(ACTION_SIDES, dtype=object), n)
    action_slot = np.tile(np.arange(len(ACTION_SIDES), dtype=np.int8), n)
    best_bid = frame["best_bid"].to_numpy(dtype=float)
    best_ask = frame["best_ask"].to_numpy(dtype=float)
    quote_price = np.empty(n * len(ACTION_SIDES), dtype=float)
    quote_price[0::3] = best_bid
    quote_price[1::3] = best_ask
    quote_price[2::3] = np.nan

    out: dict[str, Any] = {
        "split": repeat_series(frame, "split"),
        "session_id": repeat_series(frame, "session_id"),
        "symbol": repeat_series(frame, "symbol"),
        "event_index": repeat_series(frame, "local_row_id", "int64"),
        "timestamp": repeat_series(frame, "timestamp", "int64"),
        "action_side": side,
        "action_slot": action_slot,
        "quote_price": quote_price,
        "best_bid": np.repeat(best_bid, 3),
        "best_ask": np.repeat(best_ask, 3),
        "mid_price": repeat_series(frame, "mid_price", "float64"),
        "tick_size": repeat_series(frame, "tick_size", "float64"),
        "spread_ticks": repeat_series(frame, "spread_ticks", "float64"),
        "quote_eligible": repeat_series(frame, "quote_eligible").astype(bool),
    }

    for level in range(1, 6):
        for prefix in ["bid", "ask"]:
            out[f"{prefix}_px_{level}"] = repeat_series(frame, f"{prefix}_px_{level}", "float64")
            out[f"{prefix}_sz_{level}"] = repeat_series(frame, f"{prefix}_sz_{level}", "float64")

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
        out[col] = repeat_series(frame, col, "float64")

    mid = frame["mid_price"].astype(float)
    tick = frame["tick_size"].astype(float).replace(0, np.nan)
    for horizon in HORIZONS:
        future_mid = mid.shift(-horizon).to_numpy(dtype=float)
        future_mid_rep = np.repeat(future_mid, 3)
        out[f"future_mid_H{horizon}"] = future_mid_rep

        markout = np.empty(n * 3, dtype=float)
        markout[0::3] = (future_mid - best_bid) / tick.to_numpy(dtype=float)
        markout[1::3] = (best_ask - future_mid) / tick.to_numpy(dtype=float)
        markout[2::3] = 0.0
        out[f"markout_ticks_H{horizon}"] = markout

        pnl = np.empty(n * 3, dtype=float)
        pnl[0::3] = (future_mid - best_bid) * POINT_VALUE - MAKER_FEE
        pnl[1::3] = (best_ask - future_mid) * POINT_VALUE - MAKER_FEE
        pnl[2::3] = 0.0
        out[f"pnl_H{horizon}"] = pnl
        out[f"positive_pnl_H{horizon}"] = np.where(np.isfinite(pnl), pnl > 0.0, False)
        out[f"negative_pnl_H{horizon}"] = np.where(np.isfinite(pnl), pnl < 0.0, False)
        out[f"severe_negative_pnl_H{horizon}"] = np.where(np.isfinite(pnl), pnl <= SEVERE_NEGATIVE_PNL, False)

    return pd.DataFrame(out)


def write_part(action_frame: pd.DataFrame, split: str, session_id: str) -> dict[str, Any]:
    table = pa.Table.from_pandas(action_frame, preserve_index=False)
    full_dir = FULL_DATASET_DIR / f"split={split}" / f"session_id={session_id}"
    split_dir = split_dataset_dir(split)
    full_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    part_name = f"part-{session_id}.parquet"
    full_path = full_dir / part_name
    split_path = split_dir / part_name
    pq.write_table(table, full_path, compression="zstd", use_dictionary=True, row_group_size=250_000)
    if split_path.exists():
        split_path.unlink()
    try:
        os.link(full_path, split_path)
        split_link_type = "hardlink"
    except OSError:
        shutil.copyfile(full_path, split_path)
        split_link_type = "copy"
    return {
        "split": split,
        "session_id": session_id,
        "source_rows": int(len(action_frame) // 3),
        "action_rows": int(len(action_frame)),
        "bid_actions": int((action_frame["action_side"] == "bid").sum()),
        "ask_actions": int((action_frame["action_side"] == "ask").sum()),
        "no_quote_actions": int((action_frame["action_side"] == "no_quote").sum()),
        "full_part_path": str(full_path),
        "split_part_path": str(split_path),
        "split_link_type": split_link_type,
    }


def main() -> None:
    ensure_dirs()
    historical = pd.read_parquet(SOURCE_QUOTE_ELIGIBLE, columns=SOURCE_COLUMNS)
    new4, new4_filter_counts, new4_metadata = build_new4_quote_eligible()
    frames = [historical]
    if not new4.empty:
        frames.append(new4)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["split", "session_id", "local_row_id"]).reset_index(drop=True)
    filter_counts = pd.concat([historical_filter_counts_by_split(), new4_filter_counts], ignore_index=True)
    filter_counts.to_csv(RESULTS_DIR / "source_filter_counts_by_split.csv", index=False)

    # Clean old generated parquet parts from this phase only.
    generated_splits = sorted(set(frame["split"].astype(str).unique().tolist()) | {"train", "val", "test", NEW4_SPLIT})
    for target in [FULL_DATASET_DIR, *[split_dataset_dir(s) for s in generated_splits]]:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    part_rows: list[dict[str, Any]] = []
    for (split, session_id), group in frame.groupby(["split", "session_id"], sort=True):
        action_frame = make_action_frame(group.reset_index(drop=True))
        part_rows.append(write_part(action_frame, str(split), str(session_id)))
        print(f"built split={split} session={session_id}: source_rows={len(group):,} action_rows={len(action_frame):,}")

    manifest = pd.DataFrame(part_rows)
    manifest.to_csv(RESULTS_DIR / "build_parts_manifest.csv", index=False)
    split_counts = (
        manifest.groupby("split", as_index=False)
        .agg(
            quote_eligible_rows=("source_rows", "sum"),
            action_rows=("action_rows", "sum"),
            bid_actions=("bid_actions", "sum"),
            ask_actions=("ask_actions", "sum"),
            no_quote_actions=("no_quote_actions", "sum"),
            session_count=("session_id", "nunique"),
        )
        .sort_values("split")
    )
    split_counts.to_csv(RESULTS_DIR / "build_action_counts_by_split.csv", index=False)
    config = {
        "source_quote_eligible": str(SOURCE_QUOTE_ELIGIBLE),
        "source_new4_dir": str(SOURCE_NEW4_DIR),
        "new4_metadata": new4_metadata,
        "full_dataset_dir": str(FULL_DATASET_DIR),
        "split_dataset_pattern": str(DATA_DIR / "full_l2_action_universe_{split}.parquet"),
        "horizons": HORIZONS,
        "point_value": POINT_VALUE,
        "maker_fee": MAKER_FEE,
        "pnl_available": True,
        "severe_negative_pnl_threshold": SEVERE_NEGATIVE_PNL,
        "action_generation_rule": "For every quote-eligible row, emit bid, ask, and no_quote. No future fill proxy is used for action filtering.",
        "label_formula": {
            "bid": "pnl_H = (future_mid_H - best_bid) * point_value - maker_fee",
            "ask": "pnl_H = (best_ask - future_mid_H) * point_value - maker_fee",
            "no_quote": "pnl_H = 0",
        },
        "boolean_labels": {
            "positive_pnl_H": "pnl_H > 0",
            "negative_pnl_H": "pnl_H < 0",
            "severe_negative_pnl_H": f"pnl_H <= {SEVERE_NEGATIVE_PNL}",
        },
    }
    (DATA_DIR / "build_config.json").write_text(json.dumps(json_sanitize(config), indent=2), encoding="utf-8")

    print("Total action rows by split:")
    print(split_counts.to_string(index=False))
    print(f"Total action rows: {int(split_counts['action_rows'].sum()):,}")


if __name__ == "__main__":
    main()
