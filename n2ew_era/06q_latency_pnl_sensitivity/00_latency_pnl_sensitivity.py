from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06q_latency_pnl_sensitivity"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE06P = PROJECT_ROOT / "n2ew_era" / "06p_event_interarrival_latency_budget"
PHASE04B_ACTIONS = (
    PROJECT_ROOT
    / "n2ew_era"
    / "04b_txf_only_final_development_rerun"
    / "data"
    / "txf_only_action_universe.parquet"
)

if str(PHASE05C4) not in sys.path:
    sys.path.insert(0, str(PHASE05C4))

import common_05c4 as p05c4  # noqa: E402

p05a = p05c4.p05a


SPLITS = ["val", "new3", "new4"]
POLICIES = [
    "side_by_queue_imbalance_abs_0p1",
    "ml_hgb_regressor_shallow_convergence_selected",
    "phc_stp_v2_cosine_lr",
]
POLICY_DISPLAY = {
    "side_by_queue_imbalance_abs_0p1": "Queue rule",
    "ml_hgb_regressor_shallow_convergence_selected": "HGB",
    "phc_stp_v2_cosine_lr": "PHC STP v2 seed2",
}
FILL_MODEL = "conservative_l2_stress"
INVENTORY_CAP = 1
HORIZON = 20
DAY_US = 86_400_000_000
LATENCY_REFS_US = [23.16, 83.16]
MATERIAL_DEGRADATION_PCT = 0.01

SCRIPT_REL = "n2ew_era/06q_latency_pnl_sensitivity/00_latency_pnl_sensitivity.py"
SUMMARY_CSV = RESULTS / "latency_pnl_sensitivity.csv"
SHIFT_CSV = RESULTS / "latency_shift_summary.csv"
RESOLUTION_CSV = RESULTS / "timestamp_resolution_sensitivity_scope.csv"
BASELINE_CHECK_CSV = RESULTS / "baseline_l0_consistency_check.csv"
REPORT_PATH = ROOT / "AUDIT_latency_pnl_sensitivity.md"
FIGURE_PATH = FIGURES / "latency_pnl_curve.png"


@dataclass(frozen=True)
class SessionLookup:
    positions_sorted: np.ndarray
    timestamps_sorted_us: np.ndarray
    local_index_by_global: dict[int, int]


@dataclass(frozen=True)
class DenseBookSession:
    session_id: str
    source_group: str
    positions_sorted: np.ndarray
    event_index_sorted: np.ndarray
    timestamps_sorted_us: np.ndarray
    top_depth_sorted: np.ndarray
    bid_depth_sorted: np.ndarray
    ask_depth_sorted: np.ndarray
    local_index_by_event_index: dict[int, int]


@dataclass(frozen=True)
class DenseBookSplit:
    sessions: dict[str, DenseBookSession]
    resolution_rows: list[dict[str, Any]]
    event_count: int


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unwrap_session_timestamp_us(ts: np.ndarray) -> np.ndarray:
    """Unwrap time-of-day microseconds for night sessions crossing midnight."""
    out = np.rint(np.asarray(ts, dtype=np.float64)).astype(np.int64)
    if out.size == 0:
        return out
    first = int(out[0])
    if first >= 12 * 60 * 60 * 1_000_000:
        shifted = out < (first - 6 * 60 * 60 * 1_000_000)
        out = out + shifted.astype(np.int64) * DAY_US
    return out


def build_session_lookups(payload: dict[str, Any]) -> tuple[dict[int, SessionLookup], np.ndarray]:
    session_code = payload["session_code"].numpy().astype(np.int64)
    event_index = payload["event_index"].numpy().astype(np.int64)
    timestamp_raw = payload["timestamp_us"].numpy().astype(np.int64)
    unwrapped_global = np.empty(len(event_index), dtype=np.int64)
    lookups: dict[int, SessionLookup] = {}
    for code in np.unique(session_code):
        pos = np.flatnonzero(session_code == code)
        ts_unwrapped = unwrap_session_timestamp_us(timestamp_raw[pos])
        unwrapped_global[pos] = ts_unwrapped
        order = np.lexsort((event_index[pos], ts_unwrapped))
        sorted_pos = pos[order]
        sorted_ts = ts_unwrapped[order]
        lookups[int(code)] = SessionLookup(
            positions_sorted=sorted_pos.astype(np.int64),
            timestamps_sorted_us=sorted_ts.astype(np.int64),
            local_index_by_global={int(global_pos): int(local_i) for local_i, global_pos in enumerate(sorted_pos)},
        )
    return lookups, unwrapped_global


def load_dense_book_split(split: str) -> DenseBookSplit:
    """Load dense TXF L2 book states for timestamp latency lookup.

    The PHC policy and causal-fill scheduler operate on sequence-window terminal
    states, but order-book latency should be measured against the dense L2 event
    stream. This avoids mistaking the 32-event PHC stride for market latency.
    """
    if "NEW5" in split.upper():
        raise RuntimeError(f"Forbidden NEW5-like split requested: {split}")
    cache_path = RESULTS / f"dense_book_cache_{split}.npz"
    if cache_path.exists():
        return load_dense_book_split_cache(split, cache_path)
    split_dir = PHASE04B_ACTIONS / f"split={split}"
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No dense 04B action-universe parquet files found for split={split}: {split_dir}")
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
    sessions: dict[str, DenseBookSession] = {}
    resolution_rows: list[dict[str, Any]] = []
    event_count = 0
    for path in files:
        if "NEW5" in str(path).upper():
            raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if frame["source_group"].astype(str).str.upper().eq("NEW5").any():
            raise RuntimeError(f"Forbidden NEW5 source group encountered in {path}")
        frame = frame[frame["action_slot"].astype(int).eq(0)].copy()
        if frame.empty:
            continue
        frame = frame.sort_values("event_index", kind="mergesort").drop_duplicates("event_index", keep="first")
        session_id = str(frame["session_id"].iloc[0])
        source_group = str(frame["source_group"].iloc[0])
        event_index = pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        ts_raw = pd.to_numeric(frame["timestamp_us"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        ts_unwrapped = unwrap_session_timestamp_us(ts_raw)
        order = np.lexsort((event_index, ts_unwrapped))
        sorted_ts = ts_unwrapped[order].astype(np.int64)
        sorted_event_index = event_index[order].astype(np.int64)
        diffs = np.diff(sorted_ts).astype(np.float64)
        nonzero = diffs[diffs > 0]
        resolution_rows.append(
            {
                "split": split,
                "session_id": session_id,
                "source_group": source_group,
                "dense_event_count": int(len(frame)),
                "interval_count": int(len(diffs)),
                "same_timestamp_count": int((diffs == 0).sum()),
                "same_timestamp_rate": float((diffs == 0).mean()) if len(diffs) else 0.0,
                "min_nonzero_dt_us": float(nonzero.min()) if len(nonzero) else np.nan,
                "p50_dt_us": float(np.quantile(diffs, 0.50)) if len(diffs) else np.nan,
                "p99_dt_us": float(np.quantile(diffs, 0.99)) if len(diffs) else np.nan,
            }
        )
        sessions[session_id] = DenseBookSession(
            session_id=session_id,
            source_group=source_group,
            positions_sorted=np.arange(len(frame), dtype=np.int64)[order],
            event_index_sorted=sorted_event_index,
            timestamps_sorted_us=sorted_ts,
            top_depth_sorted=pd.to_numeric(frame["top_depth"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)[order],
            bid_depth_sorted=pd.to_numeric(frame["bid_sz_1"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)[order],
            ask_depth_sorted=pd.to_numeric(frame["ask_sz_1"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)[order],
            local_index_by_event_index={int(evt): int(local_i) for local_i, evt in enumerate(sorted_event_index)},
        )
        event_count += int(len(frame))
    return DenseBookSplit(sessions=sessions, resolution_rows=resolution_rows, event_count=event_count)


def load_dense_book_split_cache(split: str, cache_path: Path) -> DenseBookSplit:
    payload = np.load(cache_path, allow_pickle=False)
    session_code = payload["session_code"].astype(np.int64)
    session_labels = payload["session_labels"].astype(str).tolist()
    source_group_labels = payload["source_group_labels"].astype(str).tolist()
    event_index = payload["event_index"].astype(np.int64)
    timestamps = payload["timestamp_unwrapped_us"].astype(np.int64)
    top_depth = payload["top_depth"].astype(np.float64)
    bid_depth = payload["bid_depth"].astype(np.float64)
    ask_depth = payload["ask_depth"].astype(np.float64)
    sessions: dict[str, DenseBookSession] = {}
    resolution_rows: list[dict[str, Any]] = []
    event_count = 0
    for code, session_id in enumerate(session_labels):
        pos = np.flatnonzero(session_code == code)
        if len(pos) == 0:
            continue
        source_group = source_group_labels[code] if code < len(source_group_labels) else ""
        sorted_ts = timestamps[pos]
        sorted_event_index = event_index[pos]
        diffs = np.diff(sorted_ts).astype(np.float64)
        nonzero = diffs[diffs > 0]
        resolution_rows.append(
            {
                "split": split,
                "session_id": session_id,
                "source_group": source_group,
                "dense_event_count": int(len(pos)),
                "interval_count": int(len(diffs)),
                "same_timestamp_count": int((diffs == 0).sum()),
                "same_timestamp_rate": float((diffs == 0).mean()) if len(diffs) else 0.0,
                "min_nonzero_dt_us": float(nonzero.min()) if len(nonzero) else np.nan,
                "p50_dt_us": float(np.quantile(diffs, 0.50)) if len(diffs) else np.nan,
                "p99_dt_us": float(np.quantile(diffs, 0.99)) if len(diffs) else np.nan,
            }
        )
        sessions[str(session_id)] = DenseBookSession(
            session_id=str(session_id),
            source_group=str(source_group),
            positions_sorted=pos.astype(np.int64),
            event_index_sorted=sorted_event_index.astype(np.int64),
            timestamps_sorted_us=sorted_ts.astype(np.int64),
            top_depth_sorted=top_depth[pos].astype(np.float64),
            bid_depth_sorted=bid_depth[pos].astype(np.float64),
            ask_depth_sorted=ask_depth[pos].astype(np.float64),
            local_index_by_event_index={int(evt): int(local_i) for local_i, evt in enumerate(sorted_event_index)},
        )
        event_count += int(len(pos))
    return DenseBookSplit(sessions=sessions, resolution_rows=resolution_rows, event_count=event_count)


def find_effective_dense_book(
    payload: dict[str, Any],
    dense: DenseBookSplit,
    latency_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return dense book depths for the first dense event at/after t + latency.

    Mechanism: a decision is made at event t. The quote is considered active for
    fill testing from the first dense L2 event in the same session whose
    timestamp is greater than or equal to timestamp(t) + latency_us. The source
    event is clamped to the decision event for latency=0 and duplicate timestamp
    edge cases.
    """
    session_code = payload["session_code"].numpy().astype(np.int64)
    event_index = payload["event_index"].numpy().astype(np.int64)
    labels = payload["session_labels"]
    source_local = np.full(len(session_code), -1, dtype=np.int64)
    row_shift = np.zeros(len(session_code), dtype=np.int64)
    event_index_shift = np.zeros(len(session_code), dtype=np.int64)
    timestamp_shift_us = np.zeros(len(session_code), dtype=np.float64)
    top_depth = np.zeros(len(session_code), dtype=np.float64)
    bid_depth = np.zeros(len(session_code), dtype=np.float64)
    ask_depth = np.zeros(len(session_code), dtype=np.float64)
    latency = max(0.0, float(latency_us))

    for code in np.unique(session_code):
        rows = np.flatnonzero(session_code == code)
        session_id = str(labels[int(code)])
        if session_id not in dense.sessions:
            raise KeyError(f"Payload session_id not found in dense 04B book split: {session_id}")
        book = dense.sessions[session_id]
        events = event_index[rows]
        missing = [int(evt) for evt in events if int(evt) not in book.local_index_by_event_index]
        if missing:
            raise KeyError(
                f"Payload event_index not found in dense book session={session_id}; first missing={missing[:5]}"
            )
        current_local = np.asarray([book.local_index_by_event_index[int(evt)] for evt in events], dtype=np.int64)
        current_ts = book.timestamps_sorted_us[current_local].astype(np.float64)
        if latency == 0.0:
            target_local = current_local.copy()
        else:
            target_ts = current_ts + latency
            target_local = np.searchsorted(book.timestamps_sorted_us, target_ts, side="left").astype(np.int64)
            target_local = np.maximum(target_local, current_local)
        valid = target_local < len(book.timestamps_sorted_us)
        valid_rows = rows[valid]
        valid_target = target_local[valid]
        valid_current = current_local[valid]
        source_local[valid_rows] = valid_target
        row_shift[valid_rows] = (valid_target - valid_current).astype(np.int64)
        event_index_shift[valid_rows] = (book.event_index_sorted[valid_target] - events[valid]).astype(np.int64)
        timestamp_shift_us[valid_rows] = (
            book.timestamps_sorted_us[valid_target].astype(np.float64) - current_ts[valid]
        )
        top_depth[valid_rows] = book.top_depth_sorted[valid_target]
        bid_depth[valid_rows] = book.bid_depth_sorted[valid_target]
        ask_depth[valid_rows] = book.ask_depth_sorted[valid_target]

    valid = source_local >= 0
    return top_depth, bid_depth, ask_depth, valid, {
        "valid_count": int(valid.sum()),
        "overflow_count": int((~valid).sum()),
        "row_shift": row_shift,
        "event_index_shift": event_index_shift,
        "timestamp_shift_us": timestamp_shift_us,
    }


def latency_shifted_fill_probabilities(
    payload: dict[str, Any],
    decision: np.ndarray,
    norms: p05a.FillNorms,
    latency_us: float,
    dense: DenseBookSplit,
) -> tuple[np.ndarray, dict[str, Any]]:
    decision = np.asarray(decision, dtype=np.int8)
    selected = decision != p05a.DECISION_NO_QUOTE
    top_depth, bid_depth, ask_depth, source_valid, meta = find_effective_dense_book(payload, dense, latency_us)
    valid_shift = selected & source_valid
    overflow = selected & ~source_valid

    out = np.zeros(len(decision), dtype=np.float64)
    if valid_shift.any():
        depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
        side_depth_norm = np.zeros(len(decision), dtype=np.float64)
        bid = decision == p05a.DECISION_BID
        ask = decision == p05a.DECISION_ASK
        side_depth_norm[bid] = bid_depth[bid] / max(norms.bid_sz_1_median, 1.0)
        side_depth_norm[ask] = ask_depth[ask] / max(norms.ask_sz_1_median, 1.0)
        side = 1.0 / (1.0 + np.maximum(side_depth_norm, 0.0))
        out[valid_shift] = np.clip(0.5 * depth[valid_shift] * side[valid_shift], 0.01, 1.0)
    out[overflow] = 0.0

    shift_mask = valid_shift
    row_shift = np.asarray(meta["row_shift"], dtype=np.int64)
    event_shift = np.asarray(meta["event_index_shift"], dtype=np.int64)
    ts_shift = np.asarray(meta["timestamp_shift_us"], dtype=np.float64)
    selected_count = int(selected.sum())
    valid_selected_count = int(valid_shift.sum())
    meta_out = {
        "latency_us": float(latency_us),
        "selected_action_count": selected_count,
        "latency_valid_fill_probability_count": valid_selected_count,
        "latency_no_fill_count": int(overflow.sum()),
        "latency_no_fill_rate": float(overflow.sum() / selected_count) if selected_count else 0.0,
        "row_shift_mean_selected": float(row_shift[shift_mask].mean()) if valid_selected_count else 0.0,
        "row_shift_p50_selected": float(np.quantile(row_shift[shift_mask], 0.50)) if valid_selected_count else 0.0,
        "row_shift_p99_selected": float(np.quantile(row_shift[shift_mask], 0.99)) if valid_selected_count else 0.0,
        "row_shift_max_selected": int(row_shift[shift_mask].max()) if valid_selected_count else 0,
        "event_index_shift_mean_selected": float(event_shift[shift_mask].mean()) if valid_selected_count else 0.0,
        "event_index_shift_p50_selected": float(np.quantile(event_shift[shift_mask], 0.50)) if valid_selected_count else 0.0,
        "event_index_shift_p99_selected": float(np.quantile(event_shift[shift_mask], 0.99)) if valid_selected_count else 0.0,
        "event_index_shift_max_selected": int(event_shift[shift_mask].max()) if valid_selected_count else 0,
        "timestamp_shift_us_mean_selected": float(ts_shift[shift_mask].mean()) if valid_selected_count else 0.0,
        "timestamp_shift_us_p50_selected": float(np.quantile(ts_shift[shift_mask], 0.50)) if valid_selected_count else 0.0,
        "timestamp_shift_us_p99_selected": float(np.quantile(ts_shift[shift_mask], 0.99)) if valid_selected_count else 0.0,
        "timestamp_shift_us_max_selected": float(ts_shift[shift_mask].max()) if valid_selected_count else 0.0,
        "nonzero_book_shift_rate_selected": float((row_shift[shift_mask] > 0).mean()) if valid_selected_count else 0.0,
    }
    return np.clip(out, 0.0, 1.0), meta_out


def timestamp_resolution_for_payload(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    session_code = payload["session_code"].numpy().astype(np.int64)
    event_index = payload["event_index"].numpy().astype(np.int64)
    timestamps = payload["timestamp_us"].numpy().astype(np.int64)
    rows: list[dict[str, Any]] = []
    for code, label in enumerate(payload["session_labels"]):
        pos = np.flatnonzero(session_code == code)
        ts = unwrap_session_timestamp_us(timestamps[pos])
        order = np.lexsort((event_index[pos], ts))
        diffs = np.diff(ts[order]).astype(np.float64)
        nonzero = diffs[diffs > 0]
        rows.append(
            {
                "split": split,
                "session_id": label,
                "window_count": int(len(pos)),
                "interval_count": int(len(diffs)),
                "same_timestamp_count": int((diffs == 0).sum()),
                "same_timestamp_rate": float((diffs == 0).mean()) if len(diffs) else 0.0,
                "min_nonzero_dt_us": float(nonzero.min()) if len(nonzero) else np.nan,
                "p50_dt_us": float(np.quantile(diffs, 0.50)) if len(diffs) else np.nan,
                "p99_dt_us": float(np.quantile(diffs, 0.99)) if len(diffs) else np.nan,
            }
        )
    return rows


def latency_grid_from_resolution(min_nonzero_dt_us: float) -> list[float]:
    if not np.isfinite(min_nonzero_dt_us) or min_nonzero_dt_us <= 0:
        min_nonzero_dt_us = 1_000.0
    requested = [0.0, float(min_nonzero_dt_us), 1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0, 500_000.0]
    out: list[float] = []
    for val in requested:
        if val == 0.0 or val >= min_nonzero_dt_us:
            rounded = float(val)
            if not any(math.isclose(rounded, old, rel_tol=0.0, abs_tol=1e-9) for old in out):
                out.append(rounded)
    return out


def latency_label(latency_us: float) -> str:
    if latency_us == 0:
        return "0"
    if latency_us < 1_000.0:
        return f"{latency_us:g}us"
    if latency_us < 1_000_000.0:
        return f"{latency_us / 1_000.0:g}ms"
    return f"{latency_us / 1_000_000.0:g}s"


def policy_decisions(
    payload: dict[str, Any],
    stp_model: Any,
    stp_ckpt: dict[str, Any],
    stp_threshold: float,
) -> dict[str, np.ndarray]:
    return {
        "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
        "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
        "phc_stp_v2_cosine_lr": p05c4.stp_decision_for_payload(payload, stp_model, stp_ckpt, stp_threshold)[0],
    }


def salt_for_policy(policy_id: str) -> str:
    if policy_id == "phc_stp_v2_cosine_lr":
        return p05c4.FILL_RANDOM_SALT
    return "05a_conservative_l2_stress"


def existing_l0_baselines() -> pd.DataFrame:
    base05a = pd.read_csv(p05a.RESULTS / "phc_lite_causal_fill_sequential_eval.csv")
    base05a = base05a[
        base05a["policy_id"].isin(
            [
                "side_by_queue_imbalance_abs_0p1",
                "ml_hgb_regressor_shallow_convergence_selected",
            ]
        )
    ][["split", "policy_id", "sequential_excess_pnl", "fill_count"]]
    base05c4 = pd.read_csv(p05c4.RESULTS / "phc_stp_v2_causal_fill_eval.csv")
    base05c4 = base05c4[base05c4["policy_id"].eq("phc_stp_v2_cosine_lr")][
        ["split", "policy_id", "sequential_excess_pnl", "fill_count"]
    ]
    return pd.concat([base05a, base05c4], ignore_index=True).rename(
        columns={
            "sequential_excess_pnl": "existing_l0_sequential_excess_pnl",
            "fill_count": "existing_l0_fill_count",
        }
    )


def add_l0_deltas(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["latency_us"].eq(0.0)][
        ["split", "policy_id", "sequential_excess_pnl", "fill_count", "avg_trade_excess_pnl"]
    ].rename(
        columns={
            "sequential_excess_pnl": "l0_sequential_excess_pnl",
            "fill_count": "l0_fill_count",
            "avg_trade_excess_pnl": "l0_avg_trade_excess_pnl",
        }
    )
    out = df.merge(baseline, on=["split", "policy_id"], how="left")
    out["delta_pnl_vs_0us"] = out["sequential_excess_pnl"] - out["l0_sequential_excess_pnl"]
    out["delta_pct_pnl_vs_0us"] = out["delta_pnl_vs_0us"] / out["l0_sequential_excess_pnl"].replace(0, np.nan)
    out["delta_fill_count_vs_0us"] = out["fill_count"] - out["l0_fill_count"]
    out["delta_avg_trade_excess_vs_0us"] = out["avg_trade_excess_pnl"] - out["l0_avg_trade_excess_pnl"]
    return out


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    stp_model, stp_ckpt, selected = p05c4.selected_model()
    if selected.get("new5_used_for_selection"):
        raise RuntimeError("Refusing selected PHC policy marked new5_used_for_selection=True")
    stp_threshold = float(selected["threshold"])
    stp_seed = int(selected["seed"])
    if stp_seed != 2:
        raise RuntimeError(f"Expected frozen PHC seed=2; got seed={stp_seed}")
    norms = p05a.compute_fill_norms_from_train_sequences()

    payloads = {split: p05a.load_sequences(split) for split in SPLITS}
    for split, payload in payloads.items():
        if payload.get("new5_used"):
            raise RuntimeError(f"Forbidden payload marked new5_used=True for split={split}")
        labels = [str(x).upper() for x in payload.get("session_labels", [])]
        if any("NEW5" in label for label in labels) or "NEW5" in str(split).upper():
            raise RuntimeError(f"Forbidden NEW5 session/split encountered in split={split}")

    dense_books: dict[str, DenseBookSplit] = {}
    resolution_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        dense_books[split] = load_dense_book_split(split)
        resolution_rows.extend(dense_books[split].resolution_rows)
    resolution_df = pd.DataFrame(resolution_rows)
    nonzero_dt = resolution_df["min_nonzero_dt_us"].dropna().to_numpy(dtype=float)
    min_nonzero_dt_us = float(np.nanmin(nonzero_dt)) if len(nonzero_dt) else 1_000.0
    latencies = latency_grid_from_resolution(min_nonzero_dt_us)

    rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    for split, payload in payloads.items():
        dense = dense_books[split]
        decisions = policy_decisions(payload, stp_model, stp_ckpt, stp_threshold)
        for policy_id in POLICIES:
            decision = decisions[policy_id]
            selected_mask = decision != p05a.DECISION_NO_QUOTE
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            uniforms = p05a.stable_uniform(payload, decision, salt_for_policy(policy_id))
            zero_probs: np.ndarray | None = None
            for latency_us in latencies:
                probs, meta = latency_shifted_fill_probabilities(payload, decision, norms, latency_us, dense)
                if latency_us == 0.0:
                    zero_probs = probs.copy()
                fills = selected_mask & (uniforms < probs)
                total, fill_count, forced, max_abs_inventory, max_dd, win, loss, _equity = p05a.simulate_scheduled_by_event_index(
                    payload, decision, fills, pnl, inventory_cap=INVENTORY_CAP
                )
                avg_trade = float(total / fill_count) if fill_count else 0.0
                prob_diff = np.abs(probs[selected_mask] - zero_probs[selected_mask]) if zero_probs is not None else np.array([])
                row = {
                    "split": split,
                    "policy_id": policy_id,
                    "policy_display": POLICY_DISPLAY[policy_id],
                    "horizon": HORIZON,
                    "fill_model": FILL_MODEL,
                    "inventory_cap": INVENTORY_CAP,
                    "latency_us": float(latency_us),
                    "latency_label": latency_label(latency_us),
                    "timestamp_resolution_min_nonzero_us": min_nonzero_dt_us,
                    "below_microsecond_reference_resolution": bool(0.0 < latency_us < min_nonzero_dt_us),
                    "state_count": int(len(decision)),
                    "selected_action_count": int(selected_mask.sum()),
                    "bid_count": int((decision == p05a.DECISION_BID).sum()),
                    "ask_count": int((decision == p05a.DECISION_ASK).sum()),
                    "no_quote_count": int((decision == p05a.DECISION_NO_QUOTE).sum()),
                    "sequential_excess_pnl": float(total),
                    "fill_count": int(fill_count),
                    "fill_rate_selected": float(fill_count / selected_mask.sum()) if selected_mask.sum() else 0.0,
                    "forced_no_quote_count": int(forced),
                    "max_abs_inventory": int(max_abs_inventory),
                    "max_drawdown": float(max_dd),
                    "win_count": int(win),
                    "loss_count": int(loss),
                    "win_rate": float(win / fill_count) if fill_count else 0.0,
                    "loss_rate": float(loss / fill_count) if fill_count else 0.0,
                    "avg_trade_excess_pnl": avg_trade,
                    "probability_diff_vs_0us_max_selected": float(prob_diff.max()) if len(prob_diff) else 0.0,
                    "probability_diff_vs_0us_mean_selected": float(prob_diff.mean()) if len(prob_diff) else 0.0,
                }
                row.update(meta)
                rows.append(row)
                shift_rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "latency_us": float(latency_us),
                        **meta,
                    }
                )

    out = add_l0_deltas(pd.DataFrame(rows))
    resolution_df.to_csv(RESOLUTION_CSV, index=False)
    out.to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(shift_rows).to_csv(SHIFT_CSV, index=False)
    baseline = existing_l0_baselines()
    check = out[out["latency_us"].eq(0.0)][["split", "policy_id", "sequential_excess_pnl", "fill_count"]].merge(
        baseline, on=["split", "policy_id"], how="left"
    )
    check["delta_existing_l0_pnl"] = check["sequential_excess_pnl"] - check["existing_l0_sequential_excess_pnl"]
    check["delta_existing_l0_fill_count"] = check["fill_count"] - check["existing_l0_fill_count"]
    check.to_csv(BASELINE_CHECK_CSV, index=False)
    return out, resolution_df, check


def breakpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, policy_id), group in df.sort_values("latency_us").groupby(["split", "policy_id"]):
        base = group[group["latency_us"].eq(0.0)].iloc[0]
        threshold = abs(float(base["sequential_excess_pnl"])) * MATERIAL_DEGRADATION_PCT
        candidates = group[(group["latency_us"] > 0.0) & (group["delta_pnl_vs_0us"] <= -threshold)]
        if len(candidates):
            bp = candidates.iloc[0]
            status = "material_degradation_found"
            latency = float(bp["latency_us"])
        else:
            status = "no_material_degradation_in_scanned_range"
            latency = math.nan
        rows.append(
            {
                "split": split,
                "policy_id": policy_id,
                "baseline_pnl": float(base["sequential_excess_pnl"]),
                "material_drop_threshold_pnl": float(threshold),
                "breakpoint_status": status,
                "first_material_degradation_latency_us": latency,
            }
        )
    return pd.DataFrame(rows)


def plot_curve(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    positive_nonzero = sorted(x for x in plot_df["latency_us"].unique() if x > 0)
    zero_plot_x = (positive_nonzero[0] / 10.0) if positive_nonzero else 100.0
    plot_df["plot_latency_us"] = plot_df["latency_us"].where(plot_df["latency_us"] > 0, zero_plot_x)
    split_order = ["val", "new3", "new4"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    colors = {
        "side_by_queue_imbalance_abs_0p1": "#1f77b4",
        "ml_hgb_regressor_shallow_convergence_selected": "#ff7f0e",
        "phc_stp_v2_cosine_lr": "#2ca02c",
    }
    for ax, split in zip(axes, split_order):
        sub = plot_df[plot_df["split"].eq(split)]
        for policy_id in POLICIES:
            g = sub[sub["policy_id"].eq(policy_id)].sort_values("latency_us")
            ax.plot(
                g["plot_latency_us"],
                g["sequential_excess_pnl"],
                marker="o",
                linewidth=1.8,
                color=colors[policy_id],
                label=POLICY_DISPLAY[policy_id],
            )
        for ref in LATENCY_REFS_US:
            ax.axvline(ref, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_xscale("log")
        ticks = [zero_plot_x] + positive_nonzero
        labels = ["0"] + [latency_label(x) for x in positive_nonzero]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(split.upper())
        ax.set_xlabel("Injected latency L")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Causal-fill sequential PnL")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Latency PnL Sensitivity on Frozen TXF Sequence Policies")
    fig.text(
        0.5,
        0.01,
        "Vertical references: 23.16us and 83.16us. They are below the verified 1ms timestamp resolution and are visual markers only.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIGURE_PATH, dpi=170)
    plt.close(fig)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3%}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(work.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, resolution_df: pd.DataFrame, baseline_check: pd.DataFrame) -> None:
    bp = breakpoint_table(df)
    bp.to_csv(RESULTS / "latency_pnl_breakpoints.csv", index=False)
    min_tick = float(df["timestamp_resolution_min_nonzero_us"].dropna().iloc[0])
    max_latency = float(df["latency_us"].max())
    material_found = bp["breakpoint_status"].eq("material_degradation_found").any()
    if material_found and min_tick > max(LATENCY_REFS_US):
        status = "latency_breakpoint_found_in_ms_grid_sub_ms_refs_resolution_limited"
    elif min_tick > max(LATENCY_REFS_US):
        status = "resolution_limited_for_sub_ms_latency_references"
    elif material_found:
        status = "latency_breakpoint_found_in_scanned_range"
    else:
        status = "flat_over_scanned_range"

    compact_cols = [
        "split",
        "policy_id",
        "latency_label",
        "sequential_excess_pnl",
        "delta_pnl_vs_0us",
        "delta_pct_pnl_vs_0us",
        "fill_count",
        "delta_fill_count_vs_0us",
        "avg_trade_excess_pnl",
        "nonzero_book_shift_rate_selected",
        "latency_no_fill_rate",
    ]
    compact = df[compact_cols].copy()
    compact["delta_pct_pnl_vs_0us"] = compact["delta_pct_pnl_vs_0us"].map(format_pct)
    compact["nonzero_book_shift_rate_selected"] = compact["nonzero_book_shift_rate_selected"].map(format_pct)
    compact["latency_no_fill_rate"] = compact["latency_no_fill_rate"].map(format_pct)

    l0_ok = baseline_check.copy()
    l0_ok["pnl_matches_existing"] = l0_ok["delta_existing_l0_pnl"].abs() < 1e-9
    l0_ok["fill_count_matches_existing"] = l0_ok["delta_existing_l0_fill_count"].eq(0)

    lines = [
        "# AUDIT: Latency PnL Sensitivity",
        "",
        f"Status: `{status}`.",
        "",
        "## Delay Injection Mechanism",
        "",
        "A frozen policy observes the original PHC-window terminal state at event `t` and emits the same decision it emitted in the no-latency evaluation. No model, feature, threshold, seed, or checkpoint is retrained or reselected.",
        "",
        "For an injected deterministic latency `L`, the quote is considered eligible for fill testing from the first dense TXF L2 event in the same session whose `timestamp_us >= timestamp_us(t) + L`. The book-state features used by the conservative L2 stress fill-probability model (`top_depth`, `bid_sz_1`, `ask_sz_1`) are taken from that effective dense event. If the lookup runs past the end of the session, the decision is treated as no-fill for latency reasons.",
        "",
        "The H20 excess outcome and inventory scheduling remain the existing `simulate_scheduled_by_event_index()` proxy on the PHC-window decision-state universe. In other words, latency shifts the fill-probability book state using the dense L2 stream, but it does not create a new stale-order exchange replay. This preserves the prior causal-fill simulator semantics and should be read as a latency sensitivity proxy, not true exchange matching.",
        "",
        "## Timestamp Resolution Statement",
        "",
        f"The verified minimum non-zero timestamp step in this dense L2 book-lookup scope is `{min_tick:.0f}us` (`{min_tick / 1000.0:.3g}ms`). Phase 06P independently verified the underlying NEW1-NEW4 TXF event stream also has a 1ms minimum non-zero timestamp difference. Therefore `23.16us` and `83.16us` are below data resolution and are shown only as visual references; this dataset cannot honestly measure separate PnL effects at those two sub-millisecond latencies.",
        "",
        f"The scanned latency grid is: {', '.join(latency_label(x) for x in sorted(df['latency_us'].unique()))}.",
        "",
        "NEW5 is not loaded or touched. Splits are limited to val, NEW3, and NEW4.",
        "",
        "## Relation To The 2026-06-26 Null Result",
        "",
        "The prior val-only check found CPU empirical vs FPGA fixed latency changed selected-row fill probabilities for only `0.09111724%` of decisions under the then-current proxy. This audit extends that null result into a deterministic latency curve. If the curve stays flat through the measurable range, the correct conclusion is a boundary result: under this timestamp resolution and market-density representation, sub-millisecond CPU/FPGA latency differences are not economically distinguishable by the proxy.",
        "",
        "## Baseline Consistency Check",
        "",
        "The `L=0` rows were compared against the frozen causal-fill CSVs from Phase 05A and 05C4. Exact agreement means the new script did not change the no-latency simulator path.",
        "",
        markdown_table(l0_ok),
        "",
        "## PnL Sensitivity Table",
        "",
        markdown_table(compact),
        "",
        "## Breakpoint Rule",
        "",
        f"For this audit, a material degradation breakpoint is the first scanned `L > 0` where causal-fill PnL is at least `{MATERIAL_DEGRADATION_PCT:.1%}` below the same strategy/split's `L=0` PnL. This is a diagnostic threshold for readability, not a tuned trading rule.",
        "",
        markdown_table(bp),
        "",
        "## Conclusion",
        "",
    ]
    if material_found:
        first = bp[bp["breakpoint_status"].eq("material_degradation_found")].sort_values(
            "first_material_degradation_latency_us"
        ).iloc[0]
        lines.append(
            f"A material degradation point appears in the scanned range: first observed at `{latency_label(float(first['first_material_degradation_latency_us']))}` for `{first['policy_id']}` on `{first['split']}` under the 1% diagnostic rule."
        )
        phc_bp = bp[bp["policy_id"].eq("phc_stp_v2_cosine_lr")]
        hgb_bp = bp[bp["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")]
        if phc_bp["breakpoint_status"].eq("no_material_degradation_in_scanned_range").all():
            lines.append("For the frozen PHC seed2 policy, no material PnL degradation appears through `500ms` on val/NEW3/NEW4.")
        if hgb_bp["breakpoint_status"].eq("no_material_degradation_in_scanned_range").all():
            lines.append("For the frozen HGB policy, no material PnL degradation appears through `500ms` on val/NEW3/NEW4.")
        lines.append(
            "Several delayed rows show higher proxy PnL because shifted dense-book fill probabilities increase fill counts and sometimes per-fill average PnL. This is a proxy artifact/diagnostic result, not evidence that real trading latency is beneficial."
        )
    else:
        lines.append(
            f"No strategy/split crossed the 1% material degradation threshold through `{latency_label(max_latency)}`. Within the measurable grid, the curve is effectively flat by this rule."
        )
    if min_tick > max(LATENCY_REFS_US):
        lines.append(
            "However, the project still cannot claim a measured economic distinction between `23.16us` and `83.16us`, because both are below the verified 1ms timestamp resolution."
        )
    lines.extend(
        [
            "",
            "Safe claim: this is a latency-to-fill proxy on PHC-window sequence states. It does not prove real-world profitability, does not validate exchange queue replay, and does not imply FPGA latency changes PnL unless the measured curve supports that at the available timestamp resolution.",
            "",
            "## Outputs",
            "",
            f"- Calculation script: `{SCRIPT_REL}`",
            "- Main CSV: `results/latency_pnl_sensitivity.csv`",
            "- Shift CSV: `results/latency_shift_summary.csv`",
            "- Breakpoint CSV: `results/latency_pnl_breakpoints.csv`",
            "- Resolution CSV: `results/timestamp_resolution_sensitivity_scope.csv`",
            "- Baseline check CSV: `results/baseline_l0_consistency_check.csv`",
            "- Figure: `figures/latency_pnl_curve.png`",
        ]
    )
    write_markdown(REPORT_PATH, lines)


def main() -> None:
    df, resolution_df, baseline_check = evaluate()
    plot_curve(df)
    write_report(df, resolution_df, baseline_check)
    print(json.dumps(
        {
            "summary_csv": str(SUMMARY_CSV),
            "report": str(REPORT_PATH),
            "figure": str(FIGURE_PATH),
            "rows": int(len(df)),
            "latencies_us": sorted(df["latency_us"].unique().tolist()),
        },
        indent=2,
    ))
    print(df[["split", "policy_id", "latency_label", "sequential_excess_pnl", "fill_count", "delta_pct_pnl_vs_0us"]].to_string(index=False))


if __name__ == "__main__":
    main()
