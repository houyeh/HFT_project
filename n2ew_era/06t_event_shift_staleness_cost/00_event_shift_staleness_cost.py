from __future__ import annotations

import json
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
ROOT = PROJECT_ROOT / "n2ew_era" / "06t_event_shift_staleness_cost"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE06R = PROJECT_ROOT / "n2ew_era" / "06r_lobster_us_latency_budget"

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
K_VALUES = [0, 1, 2, 3, 5]
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260728
POINT_VALUE = 200.0
HORIZON = 20

BASELINE_CSV = RESULTS / "baseline_consistency_check.csv"
BY_K_CSV = RESULTS / "staleness_cost_by_k.csv"
SYNTHESIS_CSV = RESULTS / "synthesis_estimate.csv"
REPORT = ROOT / "AUDIT_event_shift_staleness_cost.md"
FIGURE = FIGURES / "staleness_decay_curve.png"


@dataclass(frozen=True)
class DenseSession:
    session_id: str
    source_group: str
    event_index: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    mid_price: np.ndarray
    future_mid_h20: np.ndarray
    bid_excess_pnl_h20: np.ndarray
    ask_excess_pnl_h20: np.ndarray
    top_change_positions: np.ndarray
    local_by_event_index: dict[int, int]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def assert_allowed_split(split: str) -> None:
    if split not in SPLITS or "NEW5" in split.upper():
        raise RuntimeError(f"Forbidden or unknown split requested: {split}")


def assert_no_new5_path(path: Path) -> None:
    if "NEW5" in str(path).upper():
        raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)


def read_reference_baselines() -> dict[tuple[str, str], dict[str, Any]]:
    ref: dict[tuple[str, str], dict[str, Any]] = {}
    compare_path = PHASE05C4 / "results" / "phc_stp_v2_vs_v1_hgb_queue.csv"
    if not compare_path.exists():
        raise FileNotFoundError(compare_path)
    compare = pd.read_csv(compare_path)
    for split in SPLITS:
        for policy_id in POLICIES:
            row = compare[(compare["split"].eq(split)) & (compare["policy_id"].eq(policy_id))]
            if row.empty:
                raise RuntimeError(f"Missing frozen baseline row split={split} policy_id={policy_id}")
            r = row.iloc[0]
            ref[(split, policy_id)] = {
                "baseline_source": str(compare_path.relative_to(PROJECT_ROOT)),
                "state_count": int(r["state_count"]),
                "selected_action_count": int(r["selected_action_count"]),
                "bid_count": int(r["bid_count"]),
                "ask_count": int(r["ask_count"]),
                "session_count": int(r["session_count"]),
                "total_excess_pnl_H20": float(r["total_excess_pnl_H20"]),
            }
    return ref


def selected_model_for_phc() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model, ckpt, selected = p05c4.selected_model()
    for key in ["new3_used_for_selection", "new4_used_for_selection", "new5_used_for_selection"]:
        if bool(selected.get(key, False)):
            raise RuntimeError(f"Refusing PHC selected policy marked {key}=True")
    if int(selected["seed"]) != 2:
        raise RuntimeError(f"Expected PHC seed=2, got seed={selected['seed']}")
    return model, ckpt, selected


def decisions_for_split(
    split: str,
    payload: dict[str, Any],
    stp_model: Any,
    stp_ckpt: dict[str, Any],
    stp_selected: dict[str, Any],
) -> dict[str, np.ndarray]:
    assert_allowed_split(split)
    if payload.get("new5_used"):
        raise RuntimeError(f"Refusing payload marked new5_used=True for split={split}")
    threshold = float(stp_selected["threshold"])
    return {
        "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
        "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
        "phc_stp_v2_cosine_lr": p05c4.stp_decision_for_payload(payload, stp_model, stp_ckpt, threshold)[0],
    }


def aligned_slot_frames(raw: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bid = raw[raw["action_slot"].astype(int).eq(p05a.ACTION_BID)].sort_values("event_index", kind="mergesort").reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(p05a.ACTION_ASK)].sort_values("event_index", kind="mergesort").reset_index(drop=True)
    if len(bid) != len(ask):
        raise RuntimeError(f"Bid/ask row mismatch in {path}: {len(bid)} vs {len(ask)}")
    for col in ["session_id", "event_index", "best_bid", "best_ask", "mid_price", "future_mid_H20"]:
        if not bid[col].reset_index(drop=True).equals(ask[col].reset_index(drop=True)):
            raise RuntimeError(f"Bid/ask alignment mismatch col={col} path={path}")
    return bid, ask


def load_dense_sessions_cache(split: str, cache_path: Path) -> dict[str, DenseSession]:
    payload = np.load(cache_path, allow_pickle=False)
    session_code = payload["session_code"].astype(np.int64)
    session_labels = payload["session_labels"].astype(str).tolist()
    source_group_labels = payload["source_group_labels"].astype(str).tolist()
    sessions: dict[str, DenseSession] = {}
    for code, session_id in enumerate(session_labels):
        pos = np.flatnonzero(session_code == code)
        if len(pos) == 0:
            continue
        best_bid = payload["best_bid"][pos].astype(np.float64)
        best_ask = payload["best_ask"][pos].astype(np.float64)
        event_index = payload["event_index"][pos].astype(np.int64)
        top_change = np.zeros(len(pos), dtype=bool)
        if len(pos) > 1:
            top_change[1:] = (best_bid[1:] != best_bid[:-1]) | (best_ask[1:] != best_ask[:-1])
        source_group = source_group_labels[code] if code < len(source_group_labels) else ""
        if str(source_group).upper() == "NEW5":
            raise RuntimeError(f"Forbidden NEW5 source group encountered in dense cache split={split}")
        sessions[str(session_id)] = DenseSession(
            session_id=str(session_id),
            source_group=str(source_group),
            event_index=event_index,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=payload["mid_price"][pos].astype(np.float64),
            future_mid_h20=payload["future_mid_h20"][pos].astype(np.float64),
            bid_excess_pnl_h20=payload["bid_excess_pnl_h20"][pos].astype(np.float64),
            ask_excess_pnl_h20=payload["ask_excess_pnl_h20"][pos].astype(np.float64),
            top_change_positions=np.flatnonzero(top_change).astype(np.int64),
            local_by_event_index={int(evt): int(i) for i, evt in enumerate(event_index)},
        )
    return sessions


def load_dense_sessions(split: str) -> dict[str, DenseSession]:
    assert_allowed_split(split)
    cache_path = RESULTS / f"dense_book_shift_cache_{split}.npz"
    if cache_path.exists():
        return load_dense_sessions_cache(split, cache_path)
    sessions: dict[str, DenseSession] = {}
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
    for path in p05a.part_paths(split):
        assert_no_new5_path(path)
        raw = pd.read_parquet(path, columns=columns)
        if raw["source_group"].astype(str).str.upper().eq("NEW5").any():
            raise RuntimeError(f"Forbidden NEW5 source group encountered in {path}")
        bid, ask = aligned_slot_frames(raw, path)
        session_id = str(bid["session_id"].iloc[0])
        if session_id in sessions:
            raise RuntimeError(f"Duplicate dense session encountered: {session_id}")
        event_index = pd.to_numeric(bid["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        best_bid = pd.to_numeric(bid["best_bid"], errors="coerce").to_numpy(dtype=np.float64)
        best_ask = pd.to_numeric(bid["best_ask"], errors="coerce").to_numpy(dtype=np.float64)
        mid_price = pd.to_numeric(bid["mid_price"], errors="coerce").to_numpy(dtype=np.float64)
        future_mid = pd.to_numeric(bid["future_mid_H20"], errors="coerce").to_numpy(dtype=np.float64)
        bid_label = pd.to_numeric(bid["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        ask_label = pd.to_numeric(ask["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        top_change = np.zeros(len(event_index), dtype=bool)
        if len(event_index) > 1:
            top_change[1:] = (best_bid[1:] != best_bid[:-1]) | (best_ask[1:] != best_ask[:-1])
        sessions[session_id] = DenseSession(
            session_id=session_id,
            source_group=str(bid["source_group"].iloc[0]),
            event_index=event_index,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            future_mid_h20=future_mid,
            bid_excess_pnl_h20=bid_label,
            ask_excess_pnl_h20=ask_label,
            top_change_positions=np.flatnonzero(top_change).astype(np.int64),
            local_by_event_index={int(evt): int(i) for i, evt in enumerate(event_index)},
        )
    return sessions


def values_for_side(dense: DenseSession, local: np.ndarray, side: np.ndarray) -> np.ndarray:
    out = np.empty(len(local), dtype=np.float64)
    bid = side == p05a.DECISION_BID
    ask = side == p05a.DECISION_ASK
    out[bid] = dense.bid_excess_pnl_h20[local[bid]]
    out[ask] = dense.ask_excess_pnl_h20[local[ask]]
    return out


def original_quote_for_side(dense: DenseSession, local: np.ndarray, side: np.ndarray) -> np.ndarray:
    out = np.empty(len(local), dtype=np.float64)
    bid = side == p05a.DECISION_BID
    ask = side == p05a.DECISION_ASK
    out[bid] = dense.best_bid[local[bid]]
    out[ask] = dense.best_ask[local[ask]]
    return out


def raw_quote_markout(orig_quote: np.ndarray, future_mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    out = np.empty(len(orig_quote), dtype=np.float64)
    bid = side == p05a.DECISION_BID
    ask = side == p05a.DECISION_ASK
    out[bid] = (future_mid[bid] - orig_quote[bid]) * POINT_VALUE
    out[ask] = (orig_quote[ask] - future_mid[ask]) * POINT_VALUE
    return out


def target_positions_after_k_book_changes(current_local: np.ndarray, changes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return current_local.copy(), np.ones(len(current_local), dtype=bool)
    insertion = np.searchsorted(changes, current_local, side="right")
    target_order = insertion + k - 1
    valid = target_order < len(changes)
    target = np.full(len(current_local), -1, dtype=np.int64)
    target[valid] = changes[target_order[valid]]
    return target, valid


def bootstrap_mean_ci(delta: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    if np.all(delta == delta[0]):
        value = float(delta[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    batch = max(1, min(100, 20_000_000 // max(n, 1)))
    filled = 0
    while filled < n_boot:
        b = min(batch, n_boot - filled)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        means[filled : filled + b] = delta[idx].mean(axis=1)
        filled += b
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_policy_split(
    split: str,
    policy_id: str,
    payload: dict[str, Any],
    decision: np.ndarray,
    dense_sessions: dict[str, DenseSession],
) -> list[dict[str, Any]]:
    session_code = payload["session_code"].numpy().astype(np.int64)
    event_index = payload["event_index"].numpy().astype(np.int64)
    rows = []
    selected_total = int((decision != p05a.DECISION_NO_QUOTE).sum())
    for k in K_VALUES:
        base_values: list[np.ndarray] = []
        shifted_values: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        v2_deltas: list[np.ndarray] = []
        missing_event_count = 0
        tail_overflow_count = 0
        valid_count = 0
        for code, session_id in enumerate(payload["session_labels"]):
            pos = np.flatnonzero(session_code == code)
            sess_decision = decision[pos].astype(np.int8)
            selected = sess_decision != p05a.DECISION_NO_QUOTE
            if not selected.any():
                continue
            selected_events = event_index[pos][selected]
            selected_side = sess_decision[selected]
            dense = dense_sessions.get(str(session_id))
            if dense is None:
                missing_event_count += int(len(selected_events))
                continue
            current_local = np.array([dense.local_by_event_index.get(int(evt), -1) for evt in selected_events], dtype=np.int64)
            mapped = current_local >= 0
            missing_event_count += int((~mapped).sum())
            if not mapped.any():
                continue
            current_local = current_local[mapped]
            selected_side = selected_side[mapped]
            target_local_all, valid_target = target_positions_after_k_book_changes(current_local, dense.top_change_positions, k)
            tail_overflow_count += int((~valid_target).sum())
            if not valid_target.any():
                continue
            target_local = target_local_all[valid_target]
            current_valid = current_local[valid_target]
            side_valid = selected_side[valid_target]
            base = values_for_side(dense, current_valid, side_valid)
            shifted = values_for_side(dense, target_local, side_valid)
            delta = shifted - base
            base_values.append(base)
            shifted_values.append(shifted)
            deltas.append(delta)
            valid_count += int(len(delta))

            orig_quote = original_quote_for_side(dense, current_valid, side_valid)
            base_future = dense.future_mid_h20[current_valid]
            shifted_future = dense.future_mid_h20[target_local]
            finite = np.isfinite(orig_quote) & np.isfinite(base_future) & np.isfinite(shifted_future)
            if finite.any():
                base_markout = raw_quote_markout(orig_quote[finite], base_future[finite], side_valid[finite])
                shifted_markout = raw_quote_markout(orig_quote[finite], shifted_future[finite], side_valid[finite])
                v2_deltas.append(shifted_markout - base_markout)

        base_arr = np.concatenate(base_values) if base_values else np.array([], dtype=np.float64)
        shifted_arr = np.concatenate(shifted_values) if shifted_values else np.array([], dtype=np.float64)
        delta_arr = np.concatenate(deltas) if deltas else np.array([], dtype=np.float64)
        v2_arr = np.concatenate(v2_deltas) if v2_deltas else np.array([], dtype=np.float64)
        ci_lo, ci_hi = bootstrap_mean_ci(delta_arr, BOOTSTRAPS, BOOTSTRAP_SEED + 1000 * K_VALUES.index(k) + 17 * POLICIES.index(policy_id) + SPLITS.index(split))
        rows.append(
            {
                "split": split,
                "policy_id": policy_id,
                "policy_display": POLICY_DISPLAY[policy_id],
                "k_book_changes_after_t": k,
                "selected_action_count": selected_total,
                "paired_valid_count": valid_count,
                "missing_event_count": missing_event_count,
                "tail_overflow_count": tail_overflow_count,
                "excluded_count": int(selected_total - valid_count),
                "excluded_rate": float((selected_total - valid_count) / selected_total) if selected_total else 0.0,
                "paired_base_total_excess_pnl_H20": float(base_arr.sum()) if len(base_arr) else 0.0,
                "shifted_total_excess_pnl_H20": float(shifted_arr.sum()) if len(shifted_arr) else 0.0,
                "total_delta_pnl": float(delta_arr.sum()) if len(delta_arr) else 0.0,
                "mean_delta_pnl_per_decision": float(delta_arr.mean()) if len(delta_arr) else float("nan"),
                "median_delta_pnl_per_decision": float(np.median(delta_arr)) if len(delta_arr) else float("nan"),
                "delta_negative_rate": float((delta_arr < 0).mean()) if len(delta_arr) else float("nan"),
                "bootstrap_mean_delta_ci95_lo": ci_lo,
                "bootstrap_mean_delta_ci95_hi": ci_hi,
                "Ck_cost_pnl_per_decision": float(-delta_arr.mean()) if len(delta_arr) else float("nan"),
                "v2_original_quote_markout_valid_count": int(len(v2_arr)),
                "v2_original_quote_markout_mean_delta": float(v2_arr.mean()) if len(v2_arr) else float("nan"),
                "v2_original_quote_markout_median_delta": float(np.median(v2_arr)) if len(v2_arr) else float("nan"),
                "v2_original_quote_markout_negative_rate": float((v2_arr < 0).mean()) if len(v2_arr) else float("nan"),
                "bootstrap_repetitions": BOOTSTRAPS,
                "decision_source": "frozen_checkpoint_or_model_reconstruction_no_training_no_threshold_selection",
                "label_source": "04b_txf_only_action_universe.excess_pnl_H20",
            }
        )
    return rows


def baseline_rows(by_k: pd.DataFrame, ref: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    k0 = by_k[by_k["k_book_changes_after_t"].eq(0)]
    for split in SPLITS:
        for policy_id in POLICIES:
            row = k0[(k0["split"].eq(split)) & (k0["policy_id"].eq(policy_id))]
            if row.empty:
                raise RuntimeError(f"Missing k=0 row for split={split} policy={policy_id}")
            r = row.iloc[0]
            b = ref[(split, policy_id)]
            pnl_diff = float(r["shifted_total_excess_pnl_H20"]) - float(b["total_excess_pnl_H20"])
            selected_diff = int(r["selected_action_count"]) - int(b["selected_action_count"])
            bid_diff = 0
            ask_diff = 0
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "baseline_source": b["baseline_source"],
                    "reference_total_excess_pnl_H20": b["total_excess_pnl_H20"],
                    "recomputed_k0_total_excess_pnl_H20": float(r["shifted_total_excess_pnl_H20"]),
                    "pnl_abs_diff": abs(pnl_diff),
                    "pnl_exact_match": bool(abs(pnl_diff) <= 1e-9),
                    "reference_selected_action_count": b["selected_action_count"],
                    "recomputed_selected_action_count": int(r["selected_action_count"]),
                    "selected_count_diff": selected_diff,
                    "selected_count_match": bool(selected_diff == 0),
                    "reference_bid_count": b["bid_count"],
                    "reference_ask_count": b["ask_count"],
                    "bid_count_diff_not_recomputed": bid_diff,
                    "ask_count_diff_not_recomputed": ask_diff,
                    "state_count_reference": b["state_count"],
                    "session_count_reference": b["session_count"],
                    "paired_valid_count_k0": int(r["paired_valid_count"]),
                    "missing_event_count_k0": int(r["missing_event_count"]),
                    "tail_overflow_count_k0": int(r["tail_overflow_count"]),
                    "gate_pass": bool(abs(pnl_diff) <= 1e-9 and selected_diff == 0 and int(r["missing_event_count"]) == 0),
                }
            )
    return rows


def strict_gate_or_stop(rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if not bool(row["gate_pass"])]
    if failed:
        write_csv(BASELINE_CSV, rows)
        write_markdown(
            REPORT,
            [
                "# AUDIT: Event-Shift Staleness Cost",
                "",
                "status: `blocked_baseline_consistency_failed`",
                "",
                "The hard k=0 consistency gate failed, so C(k), synthesis estimates, and V2 markout were not reported.",
                "",
                pd.DataFrame(rows).to_markdown(index=False),
                "",
                "No NEW5 data was loaded. No model training, threshold selection, fill simulation, or PnL causal claim was performed.",
            ],
        )
        raise RuntimeError("k=0 baseline consistency gate failed; stopped before reporting C(k)")


def load_stale_delta_06r() -> tuple[float, str]:
    path = PHASE06R / "results" / "stale_exposure_cpu_minus_fpga.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    row = df[df["ticker"].eq("ALL")]
    if row.empty:
        raise RuntimeError(f"Missing ALL row in {path}")
    return float(row["stale_prob_636p7_minus_83p16"].iloc[0]), str(path.relative_to(PROJECT_ROOT))


def synthesis_rows(by_k: pd.DataFrame, ref: dict[tuple[str, str], dict[str, Any]], stale_delta: float, stale_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for policy_id in POLICIES:
            b = ref[(split, policy_id)]
            for k in [1, 2]:
                r = by_k[(by_k["split"].eq(split)) & (by_k["policy_id"].eq(policy_id)) & (by_k["k_book_changes_after_t"].eq(k))]
                if r.empty:
                    continue
                rr = r.iloc[0]
                session_count = int(b["session_count"])
                selected_count = int(b["selected_action_count"])
                daily_selected = selected_count / max(session_count, 1)
                c_signed = float(rr["mean_delta_pnl_per_decision"])
                estimate_daily = stale_delta * c_signed * daily_selected
                estimate_split = stale_delta * c_signed * selected_count
                baseline_total = float(b["total_excess_pnl_H20"])
                rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "policy_display": POLICY_DISPLAY[policy_id],
                        "k_used_as_C": k,
                        "stale_delta_636p7_minus_83p16": stale_delta,
                        "stale_delta_source": stale_source,
                        "Ck_mean_delta_pnl_per_decision": c_signed,
                        "Ck_cost_pnl_per_decision": -c_signed,
                        "selected_action_count_split": selected_count,
                        "session_count_split": session_count,
                        "avg_selected_decisions_per_session": daily_selected,
                        "estimated_daily_delta_pnl": estimate_daily,
                        "estimated_daily_cost_if_delta_negative": -estimate_daily,
                        "estimated_split_delta_pnl": estimate_split,
                        "estimated_split_cost_if_delta_negative": -estimate_split,
                        "baseline_total_excess_pnl_H20": baseline_total,
                        "estimated_split_delta_pct_of_baseline": estimate_split / baseline_total if baseline_total else np.nan,
                        "estimate_status": "hypothetical_scale_estimate_not_causal",
                    }
                )
    return rows


def plot_decay(by_k: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    colors = {
        "side_by_queue_imbalance_abs_0p1": "#1f77b4",
        "ml_hgb_regressor_shallow_convergence_selected": "#ff7f0e",
        "phc_stp_v2_cosine_lr": "#2ca02c",
    }
    markers = {"val": "o", "new3": "s", "new4": "^"}
    for policy_id in POLICIES:
        for split in SPLITS:
            g = by_k[(by_k["policy_id"].eq(policy_id)) & (by_k["split"].eq(split))].sort_values("k_book_changes_after_t")
            if g.empty:
                continue
            plt.plot(
                g["k_book_changes_after_t"],
                g["mean_delta_pnl_per_decision"],
                marker=markers[split],
                color=colors[policy_id],
                alpha=0.85,
                label=f"{POLICY_DISPLAY[policy_id]} {split}",
            )
    plt.axhline(0, color="0.25", linewidth=0.8)
    plt.xlabel("k future top-of-book price changes after decision event t")
    plt.ylabel("Mean paired delta pnl per selected decision")
    plt.title("Event-shift staleness decay curve, V1 repriced label")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=160)
    plt.close()


def fmt_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.6%}"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.6f}"


def compact_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    headers = [str(col) for col in cols]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                text = "N/A"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(by_k: pd.DataFrame, baseline: pd.DataFrame, synthesis: pd.DataFrame, stale_delta: float, stale_source: str) -> None:
    k_table = by_k[by_k["k_book_changes_after_t"].isin([0, 1, 2, 5])].copy()
    k_table["mean_delta"] = k_table["mean_delta_pnl_per_decision"].map(fmt_num)
    k_table["ci95"] = k_table.apply(lambda r: f"[{fmt_num(r['bootstrap_mean_delta_ci95_lo'])}, {fmt_num(r['bootstrap_mean_delta_ci95_hi'])}]", axis=1)
    k_table["neg_rate"] = k_table["delta_negative_rate"].map(fmt_pct)
    k_table["excluded"] = k_table["excluded_count"].astype(int).astype(str)

    synth = synthesis.copy()
    for col in ["stale_delta_636p7_minus_83p16", "estimated_split_delta_pct_of_baseline"]:
        synth[col] = synth[col].map(fmt_pct)
    for col in [
        "Ck_mean_delta_pnl_per_decision",
        "estimated_daily_delta_pnl",
        "estimated_split_delta_pnl",
        "baseline_total_excess_pnl_H20",
    ]:
        synth[col] = synth[col].map(fmt_num)

    lines = [
        "# AUDIT: Event-Shift Staleness Cost",
        "",
        "status: `resolved_with_hypothetical_scale_estimate`",
        "",
        "## Scope Boundary",
        "",
        "- Splits used: `val`, `NEW3`, `NEW4` only.",
        "- NEW5 was not loaded or touched.",
        "- No model was trained, no checkpoint was changed, and no threshold was selected or tuned.",
        "- Frozen PHC/HGB decisions were reconstructed by loading existing frozen model artifacts because no event-level decision sequence artifact is persisted in the existing phases; the k=0 hard gate below proves the reconstruction path matches frozen summaries exactly.",
        "- Artifact compatibility note: loading the frozen HGB joblib under the current `.venv` emitted sklearn `InconsistentVersionWarning` because the model was pickled with sklearn 1.7.2 and loaded with sklearn 1.8.0; the HGB k=0 exact-match gate across val/NEW3/NEW4 is the compatibility check for this diagnostic.",
        "- No fill model, execution simulation, or queue-position model is used. This is label-side paired diagnostics only.",
        "- 本結果不是因果測量;報告中的合成數字只能稱為「假設下的量級估計」。",
        "",
        "## Definition",
        "",
        "- Book change event: an event whose `best_bid` or `best_ask` differs from the immediately previous dense action-universe event in the same session.",
        "- k shift: for a selected decision anchored at event `t`, use the event index of the kth future top-of-book price change strictly after `t`; `k=0` is the original event.",
        "- V1 repriced label: query the frozen same-side `excess_pnl_H20` label at the shifted event. This measures signal freshness decay under repricing and is a lower-bound staleness proxy.",
        "- V2 original-quote markout: keep the original bid/ask quote price at `t`, compare it with the shifted event's `future_mid_H20`, and report paired markout deltas as a coarse adverse-selection-direction upper proxy. It does not pass through a fill model.",
        "",
        "## Baseline Consistency Gate",
        "",
        "The hard gate requires k=0 total `excess_pnl_H20` and selected counts to match the frozen 05c4 summary exactly.",
        "",
        compact_table(
            baseline,
            [
                "split",
                "policy_id",
                "reference_total_excess_pnl_H20",
                "recomputed_k0_total_excess_pnl_H20",
                "pnl_abs_diff",
                "reference_selected_action_count",
                "recomputed_selected_action_count",
                "gate_pass",
            ],
        ),
        "",
        "All k=0 gates passed, so C(k) diagnostics are reported.",
        "",
        "## C(k) V1 Repriced-Label Results",
        "",
        "Negative `mean_delta` means the shifted same-side label is worse than the original decision label. `Ck_cost_pnl_per_decision = -mean_delta`; therefore positive cost means degradation.",
        "",
        compact_table(
            k_table,
            [
                "split",
                "policy_id",
                "k_book_changes_after_t",
                "paired_valid_count",
                "excluded",
                "mean_delta",
                "median_delta_pnl_per_decision",
                "neg_rate",
                "ci95",
            ],
        ),
        "",
        f"Decay curve figure: `figures/{FIGURE.name}`.",
        "",
        "## V2 Original-Quote Markout Extension",
        "",
        "V2 is reported in the same CSV as diagnostic columns prefixed with `v2_original_quote_markout_`. It is a coarse markout sensitivity check, not a fill/PnL simulation.",
        "",
        compact_table(
            by_k[by_k["k_book_changes_after_t"].isin([1, 2, 5])].assign(
                v2_mean=lambda d: d["v2_original_quote_markout_mean_delta"].map(fmt_num),
                v2_neg_rate=lambda d: d["v2_original_quote_markout_negative_rate"].map(fmt_pct),
            ),
            [
                "split",
                "policy_id",
                "k_book_changes_after_t",
                "v2_original_quote_markout_valid_count",
                "v2_mean",
                "v2_neg_rate",
            ],
        ),
        "",
        "## Synthesis Estimate",
        "",
        f"Borrowed 06R stale-exposure delta: `{fmt_pct(stale_delta)}` from `{stale_source}`. This is the LOBSTER pilot `stale_prob(636.7us) - stale_prob(83.16us)` for the five-stock 2012 NASDAQ sample, not TXF.",
        "",
        "Formula: `DeltaPnL_daily ~= DeltaStale_06R * C(k) * average_selected_decisions_per_session`, with signed `C(k)=mean_delta_pnl_per_decision`. k=1 and k=2 form the requested range.",
        "",
        compact_table(
            synth,
            [
                "split",
                "policy_id",
                "k_used_as_C",
                "Ck_mean_delta_pnl_per_decision",
                "avg_selected_decisions_per_session",
                "estimated_daily_delta_pnl",
                "estimated_split_delta_pnl",
                "estimated_split_delta_pct_of_baseline",
            ],
        ),
        "",
        "Required assumptions:",
        "",
        "- (a) stale 深度以 1–2 個事件為主;",
        "- (b) 暴露率借用 2012 NASDAQ pilot,非 TXF 實測;",
        "- (c) fill 處理兩側對稱、未含佇列位置效應。",
        "",
        "## Claim Boundary",
        "",
        "- This is an event-index paired-label diagnostic and a hypothetical scale estimate, not a causal latency-to-PnL measurement.",
        "- 06R is single-day, five-stock, 2012 NASDAQ data; its stale-exposure delta is borrowed only as a scale reference and must not be treated as measured TXF exposure.",
        "- The experiment intentionally excludes the known-bad 06Q latency fill proxy and makes no claim about execution, queue replay, or realized PnL.",
        "",
        "## Outputs",
        "",
        "- `results/staleness_cost_by_k.csv`",
        "- `results/synthesis_estimate.csv`",
        "- `results/baseline_consistency_check.csv`",
        f"- `figures/{FIGURE.name}`",
        "",
        "## Honest Conclusion",
        "",
    ]
    c1 = by_k[by_k["k_book_changes_after_t"].eq(1)]
    negative_count = int((c1["mean_delta_pnl_per_decision"] < 0).sum())
    positive_count = int((c1["mean_delta_pnl_per_decision"] > 0).sum())
    lines.append(
        f"At k=1, `{negative_count}` split-policy cells show negative V1 decay and `{positive_count}` show positive V1 deltas. "
        "The sign is therefore an empirical diagnostic result, not forced into a degradation story. The synthesis table should be read as a signed hypothetical scale estimate under the listed assumptions."
    )
    write_markdown(REPORT, lines)


def main() -> None:
    ensure_dirs()
    ref = read_reference_baselines()
    stp_model, stp_ckpt, stp_selected = selected_model_for_phc()
    by_k_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        if "NEW5" in str(payload.get("split", "")).upper():
            raise RuntimeError(f"Forbidden NEW5 payload split marker: {payload.get('split')}")
        dense_sessions = load_dense_sessions(split)
        decisions = decisions_for_split(split, payload, stp_model, stp_ckpt, stp_selected)
        for policy_id in POLICIES:
            by_k_rows.extend(evaluate_policy_split(split, policy_id, payload, decisions[policy_id], dense_sessions))
    by_k = pd.DataFrame(by_k_rows)
    baseline = pd.DataFrame(baseline_rows(by_k, ref))
    strict_gate_or_stop(baseline.to_dict(orient="records"))
    stale_delta, stale_source = load_stale_delta_06r()
    synthesis = pd.DataFrame(synthesis_rows(by_k, ref, stale_delta, stale_source))
    by_k.to_csv(BY_K_CSV, index=False)
    baseline.to_csv(BASELINE_CSV, index=False)
    synthesis.to_csv(SYNTHESIS_CSV, index=False)
    plot_decay(by_k)
    write_report(by_k, baseline, synthesis, stale_delta, stale_source)
    print(
        json.dumps(
            {
                "baseline_gate_pass": bool(baseline["gate_pass"].all()),
                "rows_by_k": int(len(by_k)),
                "rows_synthesis": int(len(synthesis)),
                "stale_delta_06r": stale_delta,
                "outputs": [str(BY_K_CSV), str(SYNTHESIS_CSV), str(BASELINE_CSV), str(REPORT), str(FIGURE)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
