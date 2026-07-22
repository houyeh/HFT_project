from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy"
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PHASE01_DATA = PROJECT_ROOT / "n2ew_era" / "01_full_l2_action_universe_builder" / "data"
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
PHASE02E_DIR = PROJECT_ROOT / "n2ew_era" / "02e_final_non_ml_feasibility_decision"
PHASE02E_DATA = PHASE02E_DIR / "data"

if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))

from common_phase02 import policy_catalog, policy_decision  # noqa: E402


HORIZONS = [1, 5, 10, 20, 50, 100]
POINT_VALUE = 200.0
MAKER_FEE = 15.0
ACTION_SLOT_BID = 0
ACTION_SLOT_ASK = 1
ACTION_SLOT_NO_QUOTE = 2
SELECTED_POLICY_ID = "side_by_queue_imbalance_abs_0p1"
OLD_BID_POLICY_ID = "old_bid_only_spread_ge_1"
NO_QUOTE_POLICY_ID = "no_quote"
EVAL_SPLITS = ["val", "test", "new4", "new3"]
ALL_SPLITS = ["train", "val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
INVENTORY_CAPS = [1, 2, 5]
FEE_MULTIPLIERS = [1.0, 2.0, 5.0]
FILL_MODELS = ["depth_haircut", "queue_side_haircut", "conservative_l2_stress"]
RANDOM_95_Z = NormalDist().inv_cdf(0.95)

BASE_COLUMNS = [
    "split",
    "session_id",
    "symbol",
    "event_index",
    "timestamp",
    "action_side",
    "action_slot",
    "quote_price",
    "best_bid",
    "best_ask",
    "mid_price",
    "tick_size",
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "volatility_past_50",
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA_DIR, RESULTS, FIGURES]:
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def source_part_paths(split: str) -> list[Path]:
    if split == "new3":
        return sorted((PHASE02E_DATA / "full_l2_action_universe_new3.parquet").glob("*.parquet"))
    return sorted((PHASE01_DATA / f"full_l2_action_universe_{split}.parquet").glob("*.parquet"))


def excess_dataset_dir() -> Path:
    return DATA_DIR / "full_action_universe_with_excess_markout.parquet"


def excess_part_paths(split: str) -> list[Path]:
    return sorted((excess_dataset_dir() / f"split={split}").glob("*.parquet"))


def policy_by_id(policy_id: str) -> dict[str, Any]:
    lookup = {p["policy_id"]: p for p in policy_catalog()}
    if policy_id not in lookup:
        raise KeyError(policy_id)
    return lookup[policy_id]


def policy_set() -> dict[str, dict[str, Any]]:
    return {
        NO_QUOTE_POLICY_ID: policy_by_id(NO_QUOTE_POLICY_ID),
        OLD_BID_POLICY_ID: policy_by_id(OLD_BID_POLICY_ID),
        SELECTED_POLICY_ID: policy_by_id(SELECTED_POLICY_ID),
    }


def add_excess_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    action_slot = out["action_slot"].astype(int).to_numpy()
    mid = out["mid_price"].astype(float).to_numpy()
    for horizon in HORIZONS:
        future = out[f"future_mid_H{horizon}"].astype(float).to_numpy()
        bid_excess = (future - mid) * POINT_VALUE - MAKER_FEE
        ask_excess = (mid - future) * POINT_VALUE - MAKER_FEE
        excess = np.zeros(len(out), dtype=float)
        excess[action_slot == ACTION_SLOT_BID] = bid_excess[action_slot == ACTION_SLOT_BID]
        excess[action_slot == ACTION_SLOT_ASK] = ask_excess[action_slot == ACTION_SLOT_ASK]
        excess[action_slot == ACTION_SLOT_NO_QUOTE] = 0.0
        out[f"excess_bid_pnl_H{horizon}"] = bid_excess
        out[f"excess_ask_pnl_H{horizon}"] = ask_excess
        out[f"excess_pnl_H{horizon}"] = excess
        out[f"positive_excess_pnl_H{horizon}"] = np.where(np.isfinite(excess), excess > 0.0, False)
        out[f"negative_excess_pnl_H{horizon}"] = np.where(np.isfinite(excess), excess < 0.0, False)
    return out


def write_excess_part(frame: pd.DataFrame, split: str, session_id: str) -> Path:
    target_dir = excess_dataset_dir() / f"split={split}"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"part-{session_id}.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd", use_dictionary=True, row_group_size=250_000)
    return path


def reset_excess_dataset() -> None:
    target = excess_dataset_dir()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _source_path_for_event_frame(path: Path, split: str, session_id: str) -> Path | None:
    expected = f"part-{session_id}.parquet"
    for source_path in source_part_paths(split):
        if source_path.name == expected:
            return source_path
    return None


def _attach_event_rate_from_source(out: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "event_rate_past_50" in out.columns:
        return out
    if out.empty:
        out["event_rate_past_50"] = np.array([], dtype=float)
        return out
    split = str(out["split"].iloc[0])
    session_id = str(out["session_id"].iloc[0])
    source_path = _source_path_for_event_frame(path, split, session_id)
    if source_path is None:
        raise RuntimeError(f"Cannot find source action-universe part for split={split} session={session_id}")
    source_cols = ["session_id", "event_index", "action_slot", "event_rate_past_50"]
    missing = set(source_cols) - _parquet_columns(source_path)
    if missing:
        raise RuntimeError(f"Source {source_path} is missing required latency columns: {sorted(missing)}")
    source = pd.read_parquet(source_path, columns=source_cols)
    source = source[source["action_slot"].astype(int) == ACTION_SLOT_BID][["session_id", "event_index", "event_rate_past_50"]].reset_index(drop=True)
    if len(source) != len(out):
        raise RuntimeError(f"Event-rate source length mismatch for {path}: source={len(source)} event_frame={len(out)}")
    same_session = source["session_id"].astype(str).to_numpy() == out["session_id"].astype(str).to_numpy()
    same_event = pd.to_numeric(source["event_index"], errors="coerce").to_numpy() == pd.to_numeric(out["event_index"], errors="coerce").to_numpy()
    if not bool(np.all(same_session & same_event)):
        raise RuntimeError(f"Event-rate source key mismatch for {path}")
    out["event_rate_past_50"] = pd.to_numeric(source["event_rate_past_50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return out


def read_excess_event_frame(path: Path, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
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
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "volatility_past_50",
    ]
    available = _parquet_columns(path)
    if "event_rate_past_50" in available:
        cols.append("event_rate_past_50")
    for horizon in horizons:
        cols.extend([f"excess_bid_pnl_H{horizon}", f"excess_ask_pnl_H{horizon}", f"excess_pnl_H{horizon}"])
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
            "bid_sz_1",
            "ask_sz_1",
            "top_depth",
            "queue_imbalance",
            "deep_imbalance",
            *([] if "event_rate_past_50" not in raw.columns else ["event_rate_past_50"]),
            "volatility_past_50",
        ]
    ].copy()
    out = _attach_event_rate_from_source(out, path)
    for horizon in horizons:
        out[f"bid_excess_pnl_H{horizon}"] = bid[f"excess_pnl_H{horizon}"].to_numpy(dtype=float)
        out[f"ask_excess_pnl_H{horizon}"] = ask[f"excess_pnl_H{horizon}"].to_numpy(dtype=float)
    return out


def iter_excess_event_frames(split: str, horizons: Iterable[int] = HORIZONS) -> Iterable[pd.DataFrame]:
    for path in excess_part_paths(split):
        yield read_excess_event_frame(path, horizons=horizons)


def decision_pnl(frame: pd.DataFrame, decision: np.ndarray, horizon: int, fee_multiplier: float = 1.0) -> np.ndarray:
    bid = frame[f"bid_excess_pnl_H{horizon}"].to_numpy(dtype=float) - MAKER_FEE * (float(fee_multiplier) - 1.0)
    ask = frame[f"ask_excess_pnl_H{horizon}"].to_numpy(dtype=float) - MAKER_FEE * (float(fee_multiplier) - 1.0)
    out = np.zeros(len(frame), dtype=float)
    bid_mask = decision == (ACTION_SLOT_BID + 1)
    ask_mask = decision == (ACTION_SLOT_ASK + 1)
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    out[~np.isfinite(out)] = 0.0
    return out


def delayed_decision(decision: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return decision.copy()
    out = np.zeros_like(decision)
    if delay < len(decision):
        out[delay:] = decision[:-delay]
    return out


def spread_bucket_values(spread: pd.Series | np.ndarray) -> np.ndarray:
    return pd.cut(
        pd.to_numeric(pd.Series(spread), errors="coerce"),
        bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
        labels=["<=1", "(1,2]", "(2,3]", ">3"],
    ).astype(str).to_numpy(dtype=object)


def finite_population_random_hurdle(values: np.ndarray, required_count: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    k = int(required_count)
    if k <= 0 or n <= 0:
        return 0.0, 0.0, 0.0
    if k >= n:
        total = float(values.sum())
        return total, total, total
    mean = float(values.mean())
    var = float(values.var(ddof=0))
    expected = k * mean
    variance = k * (1.0 - k / n) * var * (n / max(n - 1, 1))
    std = math.sqrt(max(variance, 0.0))
    return expected, expected + RANDOM_95_Z * std, std


def matched_random_hurdle(split: str, policy: dict[str, Any], horizon: int) -> dict[str, Any]:
    grouped_values: dict[tuple[str, str, str], list[np.ndarray]] = {}
    required: dict[tuple[str, str, str], int] = {}
    model_pnl = 0.0
    selected_total = 0
    for frame in iter_excess_event_frames(split, horizons=[horizon]):
        decision = policy_decision(policy, frame)
        session_id = str(frame["session_id"].iloc[0])
        buckets = spread_bucket_values(frame["spread_ticks"])
        for side, code, pnl_col in [
            ("bid", ACTION_SLOT_BID + 1, f"bid_excess_pnl_H{horizon}"),
            ("ask", ACTION_SLOT_ASK + 1, f"ask_excess_pnl_H{horizon}"),
        ]:
            values = np.nan_to_num(frame[pnl_col].to_numpy(dtype=float), nan=0.0)
            selected = decision == code
            model_pnl += float(values[selected].sum())
            selected_total += int(selected.sum())
            for bucket in np.unique(buckets):
                mask = buckets == bucket
                key = (side, session_id, str(bucket))
                grouped_values.setdefault(key, []).append(values[mask])
                required[key] = required.get(key, 0) + int((selected & mask).sum())
    random_mean = 0.0
    random_var = 0.0
    for key, chunks in grouped_values.items():
        mean, _, std = finite_population_random_hurdle(np.concatenate(chunks), required.get(key, 0))
        random_mean += mean
        random_var += std * std
    random_std = math.sqrt(max(random_var, 0.0))
    random_95p = random_mean + RANDOM_95_Z * random_std
    return {
        "split": split,
        "base_policy_id": policy["policy_id"],
        "horizon": int(horizon),
        "model_excess_pnl": float(model_pnl),
        "selected_action_count": int(selected_total),
        "matched_random_mean": float(random_mean),
        "matched_random_95p": float(random_95p),
        "matched_random_std_normal_approx": float(random_std),
        "beats_matched_random_95p": bool(model_pnl > random_95p),
        "random_method": "finite_population_normal_approx_side_session_spread_bucket",
    }


def stable_uniform(session_id: pd.Series, event_index: pd.Series, side_code: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": session_id.astype(str).to_numpy(),
            "event_index": pd.to_numeric(event_index, errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "side_code": np.asarray(side_code, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little"))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


@dataclass
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def compute_train_fill_norms() -> FillNorms:
    values = {"top_depth": [], "bid_sz_1": [], "ask_sz_1": []}
    for frame in iter_excess_event_frames("train", horizons=[20]):
        for col in values:
            s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s):
                values[col].append(s.to_numpy(dtype=float))
    med = {}
    for col, chunks in values.items():
        arr = np.concatenate(chunks) if chunks else np.array([1.0])
        med[col] = float(np.nanmedian(arr[arr > 0])) if np.any(arr > 0) else 1.0
    return FillNorms(top_depth_median=med["top_depth"], bid_sz_1_median=med["bid_sz_1"], ask_sz_1_median=med["ask_sz_1"])


def _fill_probabilities_from_depth_arrays(
    decision: np.ndarray,
    model: str,
    norms: FillNorms,
    top_depth: np.ndarray,
    bid_depth: np.ndarray,
    ask_depth: np.ndarray,
) -> np.ndarray:
    decision = np.asarray(decision, dtype=np.int8)
    top_depth = np.asarray(top_depth, dtype=float)
    bid_depth = np.asarray(bid_depth, dtype=float)
    ask_depth = np.asarray(ask_depth, dtype=float)
    out = np.zeros(len(decision), dtype=float)
    selected = decision != 0
    if not selected.any():
        return out
    if model == "depth_haircut":
        norm = np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0)
        out[selected] = 1.0 / (1.0 + norm[selected])
    elif model == "queue_side_haircut":
        side_depth_norm = np.zeros(len(decision), dtype=float)
        bid_mask = decision == (ACTION_SLOT_BID + 1)
        ask_mask = decision == (ACTION_SLOT_ASK + 1)
        side_depth_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
        side_depth_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
        out[selected] = 1.0 / (1.0 + np.maximum(side_depth_norm[selected], 0.0))
    elif model == "conservative_l2_stress":
        depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
        side_depth_norm = np.zeros(len(decision), dtype=float)
        bid_mask = decision == (ACTION_SLOT_BID + 1)
        ask_mask = decision == (ACTION_SLOT_ASK + 1)
        side_depth_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
        side_depth_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
        side = 1.0 / (1.0 + np.maximum(side_depth_norm, 0.0))
        out[selected] = np.clip(0.5 * depth[selected] * side[selected], 0.01, 1.0)
    else:
        raise ValueError(f"Unknown fill proxy model: {model}")
    return np.clip(out, 0.0, 1.0)


def fill_probabilities(frame: pd.DataFrame, decision: np.ndarray, model: str, norms: FillNorms) -> np.ndarray:
    top_depth = pd.to_numeric(frame["top_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bid_depth = pd.to_numeric(frame["bid_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ask_depth = pd.to_numeric(frame["ask_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return _fill_probabilities_from_depth_arrays(decision, model, norms, top_depth, bid_depth, ask_depth)


def deterministic_fill_mask(frame: pd.DataFrame, decision: np.ndarray, model: str, norms: FillNorms) -> tuple[np.ndarray, np.ndarray]:
    probs = fill_probabilities(frame, decision, model, norms)
    uniforms = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=f"02f|{model}")
    return (decision != 0) & (uniforms < probs), probs


def _latency_array_us(n: int, latency_us: float | np.ndarray) -> np.ndarray:
    if np.isscalar(latency_us):
        latency = np.full(n, float(latency_us), dtype=float)
    else:
        latency = np.asarray(latency_us, dtype=float)
        if len(latency) != n:
            raise ValueError(f"latency_us length mismatch: got {len(latency)} expected {n}")
    latency = np.nan_to_num(latency, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(latency, 0.0)


def latency_events_to_skip(frame: pd.DataFrame, latency_us: float | np.ndarray) -> np.ndarray:
    # Legacy event-rate approximation. This is known to be too coarse for
    # microsecond-scale latency in this dataset because inter-event gaps are
    # usually milliseconds; keep it only for high-event-density diagnostics and
    # unit checks of event_rate_past_50 units.
    if "event_rate_past_50" not in frame.columns:
        raise KeyError("event_rate_past_50 is required for event-rate latency approximation.")
    n = len(frame)
    event_rate = pd.to_numeric(frame["event_rate_past_50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    event_rate = np.nan_to_num(event_rate, nan=0.0, posinf=0.0, neginf=0.0)
    event_rate = np.maximum(event_rate, 0.0)
    latency = _latency_array_us(n, latency_us)
    # event_rate_past_50 is generated as events/second in Phase 01, so us -> s uses / 1e6.
    skips = np.floor(latency * event_rate / 1_000_000.0 + 0.5).astype(np.int64)
    return np.maximum(skips, 0)


def latency_events_to_skip_by_timestamp(frame: pd.DataFrame, latency_us: float | np.ndarray) -> np.ndarray:
    if "timestamp" not in frame.columns:
        raise KeyError("timestamp is required for timestamp-based latency lookup.")
    n = len(frame)
    latency = _latency_array_us(n, latency_us)
    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(timestamp)):
        raise ValueError("timestamp contains non-finite values; cannot perform latency lookup.")
    session = frame["session_id"].astype(str).to_numpy()
    source_idx = np.full(n, -1, dtype=np.int64)
    for session_id in pd.unique(pd.Series(session)):
        idx = np.flatnonzero(session == session_id)
        if len(idx) == 0:
            continue
        ts = timestamp[idx]
        target_ts = ts + latency[idx] * 1_000.0
        local_target = np.searchsorted(ts, target_ts, side="right") - 1
        # Defensive clamp: latency is non-negative, so the last observed book at
        # target time should never precede the decision event. If timestamps are
        # duplicated or numeric precision is awkward, keep the decision event as
        # the minimum valid state. This also encodes the common case where no new
        # book update occurs during the latency window: source_idx == row_idx.
        local_target = np.maximum(local_target, np.arange(len(idx), dtype=np.int64))
        valid = local_target < len(idx)
        source_idx[idx[valid]] = idx[local_target[valid]]
    return source_idx


def latency_aware_fill_probabilities(
    frame: pd.DataFrame,
    decision: np.ndarray,
    model: str,
    norms: FillNorms,
    latency_us: float | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(frame)
    decision = np.asarray(decision, dtype=np.int8)
    if len(decision) != n:
        raise ValueError(f"decision length mismatch: got {len(decision)} expected {n}")
    selected = decision != 0
    source_idx = latency_events_to_skip_by_timestamp(frame, latency_us)
    in_bounds = source_idx >= 0
    session = frame["session_id"].astype(str).to_numpy()
    same_session = np.zeros(n, dtype=bool)
    valid_idx = in_bounds.nonzero()[0]
    if len(valid_idx):
        same_session[valid_idx] = session[valid_idx] == session[source_idx[valid_idx]]
    valid_shift = selected & in_bounds & same_session
    overflow = selected & ~valid_shift

    top_depth = np.zeros(n, dtype=float)
    bid_depth = np.zeros(n, dtype=float)
    ask_depth = np.zeros(n, dtype=float)
    raw_top = pd.to_numeric(frame["top_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_bid = pd.to_numeric(frame["bid_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_ask = pd.to_numeric(frame["ask_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    top_depth[valid_shift] = raw_top[source_idx[valid_shift]]
    bid_depth[valid_shift] = raw_bid[source_idx[valid_shift]]
    ask_depth[valid_shift] = raw_ask[source_idx[valid_shift]]

    probs = _fill_probabilities_from_depth_arrays(decision, model, norms, top_depth, bid_depth, ask_depth)
    probs[overflow] = 0.0
    selected_count = int(selected.sum())
    overflow_count = int(overflow.sum())
    effective_skips = np.zeros(n, dtype=np.int64)
    effective_skips[valid_shift] = source_idx[valid_shift] - np.arange(n, dtype=np.int64)[valid_shift]
    valid_selected_count = int(valid_shift.sum())
    meta = {
        "latency_model": "book_state_shift_by_timestamp_searchsorted",
        "selected_count": selected_count,
        "valid_shift_selected_count": valid_selected_count,
        "latency_no_fill_count": overflow_count,
        "latency_no_fill_rate": float(overflow_count / selected_count) if selected_count else 0.0,
        "mean_events_to_skip_selected": float(effective_skips[valid_shift].mean()) if valid_selected_count else 0.0,
        "p50_events_to_skip_selected": float(np.quantile(effective_skips[valid_shift], 0.50)) if valid_selected_count else 0.0,
        "p99_events_to_skip_selected": float(np.quantile(effective_skips[valid_shift], 0.99)) if valid_selected_count else 0.0,
        "max_events_to_skip_selected": int(effective_skips[valid_shift].max()) if valid_selected_count else 0,
    }
    return np.clip(probs, 0.0, 1.0), meta


def deterministic_latency_aware_fill_mask(
    frame: pd.DataFrame,
    decision: np.ndarray,
    model: str,
    norms: FillNorms,
    latency_us: float | np.ndarray,
    salt_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probs, meta = latency_aware_fill_probabilities(frame, decision, model, norms, latency_us)
    salt = f"02f|latency|{model}"
    if salt_suffix:
        salt = f"{salt}|{salt_suffix}"
    uniforms = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=salt)
    fills = (np.asarray(decision, dtype=np.int8) != 0) & (uniforms < probs)
    return fills, probs, meta
