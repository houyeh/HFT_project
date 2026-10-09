from __future__ import annotations

import argparse
import gc
import importlib.util
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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07e_async_ml_control_plane"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE07D = PROJECT_ROOT / "n2ew_era" / "07d_phc_decision_distillation"

SPLITS = ("val", "new3", "new4")
INTERVALS_S = (1, 2, 5, 10, 30, 60)
MODES = ("enable", "threshold", "direction", "combined")
SOURCES = ("phc", "nonml")
AGGREGATIONS = ("mean", "last")
Q_GRID = (0.25, 0.50, 0.75)
THETA_GRID = (0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40)
FILL_SALT = "05c3_stp_v2_conservative_l2_stress"
MATCHED_RANDOM_SEED = 20261005
BOOTSTRAP_SEED = 20261006
NONML_TRAIN_STATS: dict[str, tuple[float, float]] = {}
PHC_THRESHOLD = 0.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(PHASE05A))
p05a = load_module("phase05a_common_for_07e", PHASE05A / "common_05a.py")


@dataclass(frozen=True)
class Calibration:
    confidence_gate: float
    direction_gate: float


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def require_allowed_split(split: str, payload: dict[str, Any]) -> None:
    if split.lower() == "new5" or bool(payload.get("new5_used", False)):
        raise RuntimeError("NEW5 is forbidden in phase 07E")


def load_payload(split: str) -> dict[str, Any]:
    payload = p05a.load_sequences(split)
    require_allowed_split(split, payload)
    return payload


def load_teacher_scores(split: str, payload: dict[str, Any]) -> np.ndarray:
    path = PHASE07D / "results" / f"teacher_scores_{split}.npy"
    scores = np.load(path).astype(np.float64, copy=False)
    expected = (len(payload["x"]), 2)
    if scores.shape != expected:
        raise RuntimeError(f"Teacher cache mismatch for {split}: {scores.shape} != {expected}")
    return scores


def raw_column(payload: dict[str, Any], name: str) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    return raw[:, p05a.TARGET_RAW_COLUMNS.index(name)].astype(np.float64, copy=False)


def queue_decision(payload: dict[str, Any], theta: float | np.ndarray) -> np.ndarray:
    q = raw_column(payload, "queue_imbalance")
    threshold = np.broadcast_to(np.asarray(theta, dtype=np.float64), q.shape)
    decision = np.zeros(len(q), dtype=np.int8)
    finite = np.isfinite(q) & np.isfinite(threshold)
    decision[finite & (q > threshold)] = p05a.DECISION_BID
    decision[finite & (q < -threshold)] = p05a.DECISION_ASK
    return decision


def block_ids(payload: dict[str, Any], interval_s: int) -> np.ndarray:
    timestamps = payload["timestamp_us"].numpy().astype(np.int64, copy=False)
    sessions = payload["session_code"].numpy().astype(np.int64, copy=False)
    out = np.empty(len(timestamps), dtype=np.int64)
    scale = int(interval_s * 1_000_000)
    for session in np.unique(sessions):
        pos = np.flatnonzero(sessions == session)
        if len(pos) == 0:
            continue
        start = int(timestamps[pos[0]])
        out[pos] = (timestamps[pos] - start) // scale
    return out


def lagged_block_signal(
    payload: dict[str, Any], values: np.ndarray, interval_s: int, aggregation: str
) -> tuple[np.ndarray, np.ndarray]:
    sessions = payload["session_code"].numpy().astype(np.int64, copy=False)
    blocks = block_ids(payload, interval_s)
    values2 = np.asarray(values, dtype=np.float64)
    if values2.ndim == 1:
        values2 = values2[:, None]
    out = np.full_like(values2, np.nan, dtype=np.float64)
    valid = np.zeros(len(values2), dtype=bool)
    for session in np.unique(sessions):
        session_pos = np.flatnonzero(sessions == session)
        session_blocks = blocks[session_pos]
        unique_blocks = np.unique(session_blocks)
        previous: np.ndarray | None = None
        for block in unique_blocks:
            pos = session_pos[session_blocks == block]
            if previous is not None:
                out[pos] = previous
                valid[pos] = True
            current = values2[pos]
            if aggregation == "mean":
                previous = np.nanmean(current, axis=0)
            elif aggregation == "last":
                previous = current[-1]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
    return out, valid


def source_values(payload: dict[str, Any], teacher: np.ndarray, source: str) -> np.ndarray:
    if source == "phc":
        return teacher
    # Volatility/event-rate are non-directional. The third column is the lagged
    # queue imbalance used only by direction-constrained controls.
    vol = raw_column(payload, "volatility_past_50")
    rate = raw_column(payload, "event_rate_past_50")
    queue = raw_column(payload, "queue_imbalance")
    return np.column_stack([vol, rate, queue])


def signal_components(
    payload: dict[str, Any], teacher: np.ndarray, source: str, interval_s: int, aggregation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lagged, valid = lagged_block_signal(payload, source_values(payload, teacher, source), interval_s, aggregation)
    if source == "phc":
        confidence = np.full(len(lagged), np.nan, dtype=np.float64)
        confidence[valid] = np.max(lagged[valid, :2], axis=1)
        direction = lagged[:, 0] - lagged[:, 1]
    else:
        vol = lagged[:, 0]
        rate = lagged[:, 1]
        finite = valid & np.isfinite(vol) & np.isfinite(rate)
        confidence = np.full(len(lagged), np.nan, dtype=np.float64)
        if finite.any():
            if set(NONML_TRAIN_STATS) != {"volatility_past_50", "event_rate_past_50"}:
                raise RuntimeError("Non-ML train normalization statistics were not initialized")
            vol_mu, vol_sigma = NONML_TRAIN_STATS["volatility_past_50"]
            rate_mu, rate_sigma = NONML_TRAIN_STATS["event_rate_past_50"]
            confidence[finite] = 0.5 * (
                (vol[finite] - vol_mu) / vol_sigma + (rate[finite] - rate_mu) / rate_sigma
            )
        direction = lagged[:, 2]
    return confidence, direction, valid


def calibration_from_train(
    confidence: np.ndarray, direction: np.ndarray, valid: np.ndarray, q: float
) -> Calibration:
    c = confidence[valid & np.isfinite(confidence)]
    d = np.abs(direction[valid & np.isfinite(direction)])
    return Calibration(
        confidence_gate=float(np.quantile(c, q)) if len(c) else float("inf"),
        direction_gate=float(np.quantile(d, q)) if len(d) else float("inf"),
    )


def controlled_decision(
    payload: dict[str, Any], base_theta: float, mode: str, confidence: np.ndarray,
    direction: np.ndarray, valid: np.ndarray, calibration: Calibration
) -> np.ndarray:
    active = valid & np.isfinite(confidence) & (confidence >= calibration.confidence_gate)
    theta = np.full(len(valid), base_theta, dtype=np.float64)
    if mode in {"threshold", "combined"}:
        theta[valid & ~active] = base_theta * 1.25
        theta[active] = base_theta * 0.75
    decision = queue_decision(payload, theta)
    if mode in {"enable", "combined"}:
        decision[valid & ~active] = p05a.DECISION_NO_QUOTE
    if mode in {"direction", "combined"}:
        directed = valid & np.isfinite(direction) & (np.abs(direction) >= calibration.direction_gate)
        decision[directed & (direction > 0) & (decision == p05a.DECISION_ASK)] = p05a.DECISION_NO_QUOTE
        decision[directed & (direction < 0) & (decision == p05a.DECISION_BID)] = p05a.DECISION_NO_QUOTE
    return decision


def causal_metrics(split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray, norms: Any) -> dict[str, Any]:
    probs = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (
        p05a.stable_uniform(payload, decision, FILL_SALT) < probs
    )
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(
        payload, decision, fills, pnl, inventory_cap=1
    )
    return {
        "split": split,
        "policy_id": policy_id,
        "sequential_excess_pnl": float(total),
        "fill_count": int(fill_count),
        "avg_pnl_per_fill": float(total / fill_count) if fill_count else 0.0,
        "max_drawdown": float(max_dd),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "fill_win_rate": float(win / fill_count) if fill_count else 0.0,
        "fill_loss_rate": float(loss / fill_count) if fill_count else 0.0,
        "selected_action_count": int((decision != p05a.DECISION_NO_QUOTE).sum()),
        "bid_count": int((decision == p05a.DECISION_BID).sum()),
        "ask_count": int((decision == p05a.DECISION_ASK).sum()),
        "no_quote_count": int((decision == p05a.DECISION_NO_QUOTE).sum()),
    }


def tune_static_theta(val_payload: dict[str, Any], norms: Any) -> tuple[float, pd.DataFrame]:
    rows = []
    for theta in THETA_GRID:
        decision = queue_decision(val_payload, theta)
        row = causal_metrics("val", f"static_queue_theta_{theta:g}", val_payload, decision, norms)
        row["theta"] = theta
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["sequential_excess_pnl", "theta"], ascending=[False, True])
    return float(frame.iloc[0]["theta"]), frame


def train_calibrations(train_payload: dict[str, Any], train_teacher: np.ndarray) -> dict[tuple[str, int, str, float], Calibration]:
    output: dict[tuple[str, int, str, float], Calibration] = {}
    for source in SOURCES:
        for interval_s in INTERVALS_S:
            for aggregation in AGGREGATIONS:
                confidence, direction, valid = signal_components(
                    train_payload, train_teacher, source, interval_s, aggregation
                )
                for q in Q_GRID:
                    output[(source, interval_s, aggregation, q)] = calibration_from_train(
                        confidence, direction, valid, q
                    )
    return output


def evaluate_grid(
    base_theta: float, calibrations: dict[tuple[str, int, str, float], Calibration],
    norms: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, int, str, str], np.ndarray]]:
    selection_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str, int, str, str], np.ndarray] = {}
    selected_q: dict[tuple[str, str, int, str], float] = {}

    val_payload = load_payload("val")
    val_teacher = load_teacher_scores("val", val_payload)
    for source in SOURCES:
        for interval_s in INTERVALS_S:
            for aggregation in AGGREGATIONS:
                confidence, direction, valid = signal_components(
                    val_payload, val_teacher, source, interval_s, aggregation
                )
                for mode in MODES:
                    candidates = []
                    for q in Q_GRID:
                        cal = calibrations[(source, interval_s, aggregation, q)]
                        decision = controlled_decision(
                            val_payload, base_theta, mode, confidence, direction, valid, cal
                        )
                        pid = f"{source}_{mode}_{interval_s}s_{aggregation}_q{q:g}"
                        row = causal_metrics("val", pid, val_payload, decision, norms)
                        row.update({"source": source, "mode": mode, "interval_s": interval_s,
                                    "aggregation": aggregation, "q": q,
                                    "confidence_gate": cal.confidence_gate,
                                    "direction_gate": cal.direction_gate})
                        selection_rows.append(row)
                        candidates.append(row)
                    best = sorted(candidates, key=lambda r: (-r["sequential_excess_pnl"], r["q"]))[0]
                    selected_q[(source, mode, interval_s, aggregation)] = float(best["q"])
    del val_payload, val_teacher
    gc.collect()

    for split in SPLITS:
        payload = load_payload(split)
        teacher = load_teacher_scores(split, payload)
        for source in SOURCES:
            for interval_s in INTERVALS_S:
                for aggregation in AGGREGATIONS:
                    confidence, direction, valid = signal_components(
                        payload, teacher, source, interval_s, aggregation
                    )
                    for mode in MODES:
                        q = selected_q[(source, mode, interval_s, aggregation)]
                        cal = calibrations[(source, interval_s, aggregation, q)]
                        decision = controlled_decision(
                            payload, base_theta, mode, confidence, direction, valid, cal
                        )
                        pid = f"{source}_{mode}_{interval_s}s_{aggregation}"
                        row = causal_metrics(split, pid, payload, decision, norms)
                        row.update({"source": source, "mode": mode, "interval_s": interval_s,
                                    "aggregation": aggregation, "selected_q_on_val": q,
                                    "confidence_gate": cal.confidence_gate,
                                    "direction_gate": cal.direction_gate,
                                    "selection_split": "val", "new3_used_for_selection": False,
                                    "new4_used_for_selection": False, "new5_used": False})
                        result_rows.append(row)
                        decisions[(split, source, interval_s, aggregation, mode)] = decision
        del payload, teacher
        gc.collect()
    return pd.DataFrame(selection_rows), pd.DataFrame(result_rows), decisions


def reference_results(base_theta: float, norms: Any) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    decisions: dict[tuple[str, str], np.ndarray] = {}
    for split in SPLITS:
        payload = load_payload(split)
        teacher = load_teacher_scores(split, payload)
        policies = {
            "queue_theta_0p1": queue_decision(payload, 0.1),
            "queue_theta_val_tuned": queue_decision(payload, base_theta),
            "phc_eventwise_reference": p05a.policy_from_scores(teacher, PHC_THRESHOLD),
        }
        for policy_id, decision in policies.items():
            row = causal_metrics(split, policy_id, payload, decision, norms)
            row.update({"source": "reference", "mode": policy_id, "interval_s": 0,
                        "aggregation": "eventwise_or_static", "selected_q_on_val": np.nan,
                        "selection_split": "val" if policy_id == "queue_theta_val_tuned" else "frozen",
                        "new3_used_for_selection": False, "new4_used_for_selection": False,
                        "new5_used": False})
            rows.append(row)
            decisions[(split, policy_id)] = decision
        del payload, teacher
        gc.collect()
    return pd.DataFrame(rows), decisions


def choose_champions(results: pd.DataFrame) -> pd.DataFrame:
    val = results[(results["split"] == "val") & (results["aggregation"] == "mean")].copy()
    champions = (
        val.sort_values(["sequential_excess_pnl", "interval_s"], ascending=[False, False])
        .groupby(["source", "mode"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return champions


def matched_random_for_champions(
    champions: pd.DataFrame, grid_decisions: dict[tuple[str, str, int, str, str], np.ndarray],
    reference_decisions: dict[tuple[str, str], np.ndarray], n_trials: int
) -> pd.DataFrame:
    rows = []
    targets: list[tuple[str, str, np.ndarray]] = []
    for champion in champions.itertuples(index=False):
        for split in SPLITS:
            key = (split, champion.source, int(champion.interval_s), "mean", champion.mode)
            targets.append((split, champion.policy_id, grid_decisions[key]))
    for split in SPLITS:
        for pid in ("queue_theta_val_tuned", "phc_eventwise_reference"):
            targets.append((split, pid, reference_decisions[(split, pid)]))
    for index, (split, policy_id, decision) in enumerate(targets):
        _, summary = p05a.matched_random_for_split(
            split, decision, n_trials=n_trials, seed=MATCHED_RANDOM_SEED + index
        )
        row = summary.iloc[0].to_dict()
        row["policy_id"] = policy_id
        p_value = float(row["empirical_p_value_fraction_random_ge_model"])
        row["empirical_p_value_wording"] = f"< {1.0 / n_trials:g}" if p_value == 0.0 else f"{p_value:.6f}"
        row["matched_random_metric"] = "action_level_total_excess_pnl_H20"
        rows.append(row)
        print(f"[matched-random] {split} {policy_id} p={row['empirical_p_value_wording']}", flush=True)
    return pd.DataFrame(rows)


def session_causal_values(payload: dict[str, Any], decision: np.ndarray, norms: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    session_codes = payload["session_code"].numpy()
    n = len(decision)
    for code, label in enumerate(payload["session_labels"]):
        mask = session_codes == code
        sub: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and len(value.shape) and value.shape[0] == n:
                sub[key] = value[mask]
            else:
                sub[key] = value
        # Preserve the original session code because stable_uniform hashes it.
        # Renumbering a sliced session to zero would change the deterministic
        # fill draw and make session totals disagree with the full-split run.
        sub["session_code"] = torch.full((int(mask.sum()),), int(code), dtype=torch.int32)
        sub["session_labels"] = list(payload["session_labels"])
        sub_decision = decision[mask]
        output[str(label)] = causal_metrics("session", "session", sub, sub_decision, norms)["sequential_excess_pnl"]
    return output


def paired_session_bootstrap(
    champions: pd.DataFrame, grid_decisions: dict[tuple[str, str, int, str, str], np.ndarray],
    reference_decisions: dict[tuple[str, str], np.ndarray], norms: Any, n_trials: int = 10000
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for champion in champions[champions["source"] == "phc"].itertuples(index=False):
        nonml_match = champions[champions["mode"] == champion.mode]
        nonml_match = nonml_match[nonml_match["source"] == "nonml"].iloc[0]
        for split in SPLITS:
            payload = load_payload(split)
            ml_decision = grid_decisions[(split, "phc", int(champion.interval_s), "mean", champion.mode)]
            nonml_decision = grid_decisions[(split, "nonml", int(nonml_match.interval_s), "mean", champion.mode)]
            static_decision = reference_decisions[(split, "queue_theta_val_tuned")]
            ml = session_causal_values(payload, ml_decision, norms)
            comparators = {
                "queue_theta_val_tuned": session_causal_values(payload, static_decision, norms),
                f"nonml_{champion.mode}": session_causal_values(payload, nonml_decision, norms),
            }
            labels = sorted(ml)
            for comparator, values in comparators.items():
                diffs = np.array([ml[label] - values[label] for label in labels], dtype=np.float64)
                samples = rng.choice(diffs, size=(n_trials, len(diffs)), replace=True).mean(axis=1)
                comparator_decision = static_decision if comparator == "queue_theta_val_tuned" else nonml_decision
                ml_full = causal_metrics(split, "ml_full", payload, ml_decision, norms)["sequential_excess_pnl"]
                comparator_full = causal_metrics(
                    split, "comparator_full", payload, comparator_decision, norms
                )["sequential_excess_pnl"]
                rows.append({
                    "split": split, "ml_policy_id": champion.policy_id,
                    "comparator": comparator, "session_count": len(labels),
                    "full_split_total_difference": float(ml_full - comparator_full),
                    "session_reset_total_difference": float(diffs.sum()),
                    "mean_session_difference": float(diffs.mean()),
                    "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_probability_difference_le_zero": float((samples <= 0).mean()),
                    "n_bootstrap": n_trials,
                })
            del payload
            gc.collect()
    return pd.DataFrame(rows)


def make_figure(all_results: pd.DataFrame, references: pd.DataFrame) -> None:
    main = all_results[(all_results["source"] == "phc") & (all_results["aggregation"] == "mean")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, split in zip(axes, SPLITS):
        subset = main[main["split"] == split]
        for mode in MODES:
            row = subset[subset["mode"] == mode].sort_values("interval_s")
            ax.plot(row["interval_s"], row["sequential_excess_pnl"], marker="o", label=mode)
        tuned = references[(references["split"] == split) & (references["policy_id"] == "queue_theta_val_tuned")]
        phc = references[(references["split"] == split) & (references["policy_id"] == "phc_eventwise_reference")]
        ax.axhline(float(tuned.iloc[0]["sequential_excess_pnl"]), color="black", linestyle="--", label="static tuned")
        ax.axhline(float(phc.iloc[0]["sequential_excess_pnl"]), color="gray", linestyle=":", label="eventwise PHC")
        ax.set_xscale("log")
        ax.set_title(split)
        ax.set_xlabel("Control update interval (s, log scale)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Causal-fill sequential excess PnL (proxy units)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Lagged PHC control-plane sweep (previous complete block mean)")
    fig.tight_layout()
    fig.savefig(FIGURES / "async_control_plane_pnl.png", dpi=180)
    plt.close(fig)


def write_report(
    base_theta: float, static_scan: pd.DataFrame, all_results: pd.DataFrame,
    references: pd.DataFrame, champions: pd.DataFrame, matched: pd.DataFrame,
    bootstrap: pd.DataFrame
) -> None:
    combined = champions[(champions["source"] == "phc") & (champions["mode"] == "combined")].iloc[0]
    nonml = champions[(champions["source"] == "nonml") & (champions["mode"] == "combined")].iloc[0]
    merged = pd.concat([all_results, references], ignore_index=True, sort=False)

    def pnl(split: str, policy_id: str) -> float:
        row = merged[(merged["split"] == split) & (merged["policy_id"] == policy_id)]
        return float(row.iloc[0]["sequential_excess_pnl"])

    checks = []
    for split in SPLITS:
        checks.append({
            "split": split,
            "ml_combined": pnl(split, combined.policy_id),
            "nonml_combined": pnl(split, nonml.policy_id),
            "static_tuned": pnl(split, "queue_theta_val_tuned"),
            "eventwise_phc": pnl(split, "phc_eventwise_reference"),
        })
    check_df = pd.DataFrame(checks)
    val_ok = bool(check_df.loc[check_df["split"] == "val", "ml_combined"].iloc[0] > check_df.loc[check_df["split"] == "val", "static_tuned"].iloc[0])
    beats_nonml = bool(check_df.loc[check_df["split"] == "val", "ml_combined"].iloc[0] > check_df.loc[check_df["split"] == "val", "nonml_combined"].iloc[0])
    oos_ok = bool(np.all(check_df.loc[check_df["split"].isin(["new3", "new4"]), "ml_combined"].to_numpy() > check_df.loc[check_df["split"].isin(["new3", "new4"]), "static_tuned"].to_numpy()))
    status = "resolved_positive" if val_ok and beats_nonml and oos_ok else ("mixed" if val_ok or oos_ok else "resolved_negative")

    display_cols = ["source", "mode", "policy_id", "interval_s", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown"]
    champion_eval = merged[merged["policy_id"].isin(champions["policy_id"])][["split", *display_cols]].sort_values(["source", "mode", "split"])
    refs = references[["split", "policy_id", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown"]]
    best_static = static_scan.iloc[0]
    lines = [
        f"# Async PHC Control-Plane Audit\n\nstatus: `{status}`\n",
        "## Claim boundary\n",
        "This experiment does **not** demonstrate that the recurrent PHC model was physically executed only once per update interval. It replays cached eventwise PHC scores from Phase 07D, aggregates the previous completed time block, and applies that lagged control output to a fast queue rule. The supported claim is therefore: **PHC can be sampled as an asynchronous off-critical-path controller**. True low-frequency recurrent execution with state evolution only at update times remains a separate experiment. NEW3 and NEW4 are post-selection diagnostics; NEW5 was neither loaded nor used.\n",
        "## Pre-registered mechanism\n",
        "- Sessions are blocked independently; no block crosses a session boundary.\n- The main result uses the **mean score from the previous completed block**. The first block of every session falls back to the val-tuned static queue rule.\n- Update intervals are 1, 2, 5, 10, 30, and 60 seconds.\n- `enable` gates quoting, `threshold` switches between 0.75x/1.25x the static threshold, `direction` allows BID-only/ASK-only outside a train-calibrated dead band, and `combined` applies all three.\n- Hyperparameter quantiles are calibrated on train and selected on val only. NEW3/NEW4 do not select intervals or settings.\n- Non-ML controls use lagged volatility plus event rate for activity. Because those variables have no side, the direction control necessarily uses lagged mean queue imbalance; this is explicitly not a volatility-only directional signal.\n",
        "## Static threshold control\n",
        f"The val-selected static queue threshold is `{base_theta:g}` with causal-fill PnL `{best_static['sequential_excess_pnl']:.3f}`. This is the comparator; improvements over the historical theta=0.1 rule alone are not credited to ML.\n",
        "## Reference endpoints\n",
        refs.to_markdown(index=False, floatfmt=".4f") + "\n",
        "`phc_eventwise_reference` is a reference endpoint, not a guaranteed upper bound: a controller may outperform it by trading less often.\n",
        "## Val-selected controller champions\n",
        champions[["source", "mode", "policy_id", "interval_s", "selected_q_on_val", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown"]].to_markdown(index=False, floatfmt=".4f") + "\n",
        "## Champion evaluation\n",
        champion_eval.to_markdown(index=False, floatfmt=".4f") + "\n",
        "## Combined-controller decision test\n",
        check_df.to_markdown(index=False, floatfmt=".4f") + "\n",
        f"The combined PHC controller {'beats' if val_ok else 'does not beat'} the tuned static rule on val, {'beats' if beats_nonml else 'does not beat'} its same-mechanism non-ML control on val, and {'beats the tuned static rule on both OOS diagnostics' if oos_ok else 'does not beat the tuned static rule on both OOS diagnostics'}. This yields status `{status}`.\n",
        "## Statistical checks\n",
        f"Matched-random tests below are action-level tests inherited from Phase 05A. This exploratory run uses `{int(matched['n_trials'].iloc[0])}` trials, so its minimum empirical p-value resolution is `{1.0 / int(matched['n_trials'].iloc[0]):g}`. They test each selected policy against a count/session/side/spread-matched random policy; they do **not** test PHC against the non-ML controller. Pairwise claims use a session-paired bootstrap on causal-fill PnL. Each bootstrap session is simulated independently with inventory reset, whereas the canonical full-split simulator carries its inventory variable across session boundaries. The table therefore reports both `full_split_total_difference` and the session-reset estimand; they need not be identical. With only four to six sessions per split, intervals are wide and must not be over-interpreted.\n",
        matched.to_markdown(index=False, floatfmt=".6f") + "\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Supplementary aggregation check\n",
        "`results/async_control_plane_all_results.csv` includes the supplementary `last` aggregation. It is not used for the headline result because it would make the control depend on one potentially noisy event.\n",
        "## Outputs\n",
        "- `results/static_threshold_scan.csv`\n- `results/val_hyperparameter_selection.csv`\n- `results/async_control_plane_all_results.csv`\n- `results/controller_champions.csv`\n- `results/matched_random_champions.csv`\n- `results/paired_session_bootstrap.csv`\n- `figures/async_control_plane_pnl.png`\n",
    ]
    (ROOT / "AUDIT_async_ml_control_plane.md").write_text("\n".join(lines), encoding="utf-8")


def main(n_random_trials: int, reuse_matched_random: bool = False) -> None:
    global PHC_THRESHOLD
    ensure_dirs()
    selected_policy = json.loads(
        (PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json").read_text(encoding="utf-8")
    )
    PHC_THRESHOLD = float(selected_policy["threshold"])
    norms = p05a.compute_fill_norms_from_train_sequences()
    val_payload = load_payload("val")
    base_theta, static_scan = tune_static_theta(val_payload, norms)
    static_scan.to_csv(RESULTS / "static_threshold_scan.csv", index=False)
    del val_payload
    gc.collect()

    train_payload = load_payload("train")
    train_teacher = load_teacher_scores("train", train_payload)
    for column in ("volatility_past_50", "event_rate_past_50"):
        values = raw_column(train_payload, column)
        finite = values[np.isfinite(values)]
        sigma = float(np.std(finite)) if len(finite) else 1.0
        NONML_TRAIN_STATS[column] = (float(np.mean(finite)) if len(finite) else 0.0, max(sigma, 1e-12))
    calibrations = train_calibrations(train_payload, train_teacher)
    del train_payload, train_teacher
    gc.collect()

    selection, grid_results, grid_decisions = evaluate_grid(base_theta, calibrations, norms)
    references, reference_decisions = reference_results(base_theta, norms)
    selection.to_csv(RESULTS / "val_hyperparameter_selection.csv", index=False)
    grid_results.to_csv(RESULTS / "async_control_plane_all_results.csv", index=False)
    references.to_csv(RESULTS / "reference_results.csv", index=False)

    champions = choose_champions(grid_results)
    champions.to_csv(RESULTS / "controller_champions.csv", index=False)
    matched_path = RESULTS / "matched_random_champions.csv"
    if reuse_matched_random and matched_path.exists():
        matched = pd.read_csv(matched_path)
        print(f"[reuse] matched-random={matched_path}", flush=True)
    else:
        matched = matched_random_for_champions(
            champions, grid_decisions, reference_decisions, n_trials=n_random_trials
        )
        matched.to_csv(matched_path, index=False)
    bootstrap = paired_session_bootstrap(champions, grid_decisions, reference_decisions, norms)
    bootstrap.to_csv(RESULTS / "paired_session_bootstrap.csv", index=False)
    make_figure(grid_results, references)
    write_report(base_theta, static_scan, grid_results, references, champions, matched, bootstrap)
    metadata = {
        "intervals_s": list(INTERVALS_S), "modes": list(MODES), "sources": list(SOURCES),
        "aggregations": list(AGGREGATIONS), "q_grid": list(Q_GRID),
        "static_theta_grid": list(THETA_GRID), "selected_static_theta": base_theta,
        "phc_threshold": PHC_THRESHOLD,
        "nonml_train_normalization": {key: {"mean": value[0], "std": value[1]} for key, value in NONML_TRAIN_STATS.items()},
        "fill_salt": FILL_SALT, "matched_random_trials": n_random_trials,
        "selection_split": "val", "oos_diagnostic_splits": ["new3", "new4"],
        "new5_used": False,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] report={ROOT / 'AUDIT_async_ml_control_plane.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-random-trials", type=int, default=100)
    parser.add_argument("--reuse-matched-random", action="store_true")
    args = parser.parse_args()
    main(args.matched_random_trials, reuse_matched_random=args.reuse_matched_random)
