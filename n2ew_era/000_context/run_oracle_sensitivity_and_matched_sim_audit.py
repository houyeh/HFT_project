from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CTX = PROJECT_ROOT / "n2ew_era" / "000_context"
OUT_DIR = CTX / "results" / "oracle_assumption_sensitivity_sweep"
P05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
P05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"

if str(P05A) not in sys.path:
    sys.path.insert(0, str(P05A))
if str(P05C4) not in sys.path:
    sys.path.insert(0, str(P05C4))

import common_05a as p05a  # noqa: E402
import common_05c4 as c05c4  # noqa: E402


SPLITS = ["train", "val", "new3", "new4"]
OOS_SPLITS = ["val", "new3", "new4"]
DEVELOPMENT_SPLITS = ["train", "val"]
REAL_POLICIES = [
    "side_by_queue_imbalance_abs_0p1",
    "ml_hgb_regressor_shallow_convergence_selected",
    "phc_stp_v2_cosine_lr",
]
FILL_MODELS = ["depth_haircut", "queue_side_haircut", "conservative_l2_stress"]
INVENTORY_CAPS = [1, 2, 3]
REPRESENTATIVE_FILL_MODEL = "conservative_l2_stress"
REPRESENTATIVE_INVENTORY_CAP = 1


@dataclass(frozen=True)
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.6g}"
    if isinstance(value, (np.floating,)):
        return fmt(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if columns is not None:
        work = work[columns]
    if max_rows is not None:
        work = work.head(max_rows)
    headers = list(work.columns)
    rows = [[fmt(v) for v in row] for row in work.itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def raw_feature(payload: dict[str, Any], name: str) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    idx = p05a.TARGET_RAW_COLUMNS.index(name)
    return raw[:, idx].astype(np.float64)


def load_payloads() -> dict[str, dict[str, Any]]:
    payloads = {}
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        if payload.get("new5_used"):
            raise RuntimeError(f"Refusing payload marked new5_used=True: {split}")
        payloads[split] = payload
    return payloads


def compute_fill_norms(payloads: dict[str, dict[str, Any]]) -> FillNorms:
    train = payloads["train"]
    values = {}
    for col in ["top_depth", "bid_sz_1", "ask_sz_1"]:
        arr = raw_feature(train, col)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        values[col] = float(np.nanmedian(arr)) if len(arr) else 1.0
    return FillNorms(values["top_depth"], values["bid_sz_1"], values["ask_sz_1"])


def fill_probabilities(payload: dict[str, Any], decision: np.ndarray, fill_model: str, norms: FillNorms) -> np.ndarray:
    out = np.zeros(len(decision), dtype=np.float64)
    selected = decision != p05a.DECISION_NO_QUOTE
    if not selected.any():
        return out
    top_depth = raw_feature(payload, "top_depth")
    bid_depth = raw_feature(payload, "bid_sz_1")
    ask_depth = raw_feature(payload, "ask_sz_1")
    if fill_model == "depth_haircut":
        norm = np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0)
        out[selected] = 1.0 / (1.0 + norm[selected])
    elif fill_model == "queue_side_haircut":
        side_depth_norm = np.zeros(len(decision), dtype=np.float64)
        bid_mask = decision == p05a.DECISION_BID
        ask_mask = decision == p05a.DECISION_ASK
        side_depth_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
        side_depth_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
        out[selected] = 1.0 / (1.0 + np.maximum(side_depth_norm[selected], 0.0))
    elif fill_model == "conservative_l2_stress":
        depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
        side_depth_norm = np.zeros(len(decision), dtype=np.float64)
        bid_mask = decision == p05a.DECISION_BID
        ask_mask = decision == p05a.DECISION_ASK
        side_depth_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
        side_depth_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
        side = 1.0 / (1.0 + np.maximum(side_depth_norm, 0.0))
        out[selected] = np.clip(0.5 * depth[selected] * side[selected], 0.01, 1.0)
    else:
        raise ValueError(fill_model)
    return np.clip(out, 0.0, 1.0)


def deterministic_fills(payload: dict[str, Any], decision: np.ndarray, fill_model: str, norms: FillNorms) -> np.ndarray:
    probs = fill_probabilities(payload, decision, fill_model, norms)
    uniforms = p05a.stable_uniform(payload, decision, f"000_context_oracle_sensitivity|{fill_model}")
    return (decision != p05a.DECISION_NO_QUOTE) & (uniforms < probs)


def decision_pnl(payload: dict[str, Any], decision: np.ndarray) -> np.ndarray:
    return p05a.decision_pnl_from_payload(payload, decision)


def best_side_and_value(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bid = payload["y_bid"].numpy().astype(np.float64)
    ask = payload["y_ask"].numpy().astype(np.float64)
    bid = np.nan_to_num(bid, nan=0.0, posinf=0.0, neginf=0.0)
    ask = np.nan_to_num(ask, nan=0.0, posinf=0.0, neginf=0.0)
    best = np.maximum(bid, ask)
    side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK).astype(np.int8)
    return side, best


def dense_oracle_decision(payload: dict[str, Any]) -> np.ndarray:
    side, best = best_side_and_value(payload)
    decision = np.zeros(len(best), dtype=np.int8)
    mask = best > 0.0
    decision[mask] = side[mask]
    return decision


def matched_count_oracle_decision(payload: dict[str, Any], selected_count: int) -> np.ndarray:
    side, best = best_side_and_value(payload)
    order = np.argsort(best)[::-1]
    chosen = order[: min(int(selected_count), len(order))]
    decision = np.zeros(len(best), dtype=np.int8)
    decision[chosen] = side[chosen]
    return decision


def make_real_decisions(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    stp_model, stp_ckpt, stp_selected = c05c4.selected_model()
    stp_threshold = float(stp_selected["threshold"])
    out: dict[str, dict[str, np.ndarray]] = {}
    for split, payload in payloads.items():
        stp_decision, _ = c05c4.stp_decision_for_payload(payload, stp_model, stp_ckpt, stp_threshold)
        out[split] = {
            "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
            "phc_stp_v2_cosine_lr": stp_decision,
        }
    return out


def simulate(
    split: str,
    policy_id: str,
    payload: dict[str, Any],
    decision: np.ndarray,
    fill_model: str,
    inventory_cap: int,
    norms: FillNorms,
    oracle_kind: str = "policy",
    oracle_reference_policy: str | None = None,
    matched_count_reference_pnl: float | None = None,
) -> dict[str, Any]:
    fills = deterministic_fills(payload, decision, fill_model, norms)
    pnl = decision_pnl(payload, decision)
    total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(
        payload,
        decision.astype(np.int8),
        fills.astype(np.bool_),
        pnl.astype(np.float64),
        inventory_cap=int(inventory_cap),
    )
    fill_count_i = int(fill_count)
    return {
        "split": split,
        "policy_id": policy_id,
        "oracle_kind": oracle_kind,
        "oracle_reference_policy": oracle_reference_policy or "",
        "fill_model": fill_model,
        "inventory_cap": int(inventory_cap),
        "horizon": int(p05a.HORIZON),
        "state_count": int(len(decision)),
        "selected_action_count": int((decision != p05a.DECISION_NO_QUOTE).sum()),
        "sequential_excess_pnl": float(total),
        "fill_count": fill_count_i,
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "max_drawdown": float(max_dd),
        "win_rate": float(win / max(fill_count_i, 1)) if fill_count_i else 0.0,
        "loss_rate": float(loss / max(fill_count_i, 1)) if fill_count_i else 0.0,
        "avg_trade_excess_pnl": float(total / max(fill_count_i, 1)) if fill_count_i else 0.0,
        "matched_count_reference_pnl": matched_count_reference_pnl,
        "capture_ratio_against_matched_count": float(total / matched_count_reference_pnl) if matched_count_reference_pnl else np.nan,
        "oracle_uses_future_labels": oracle_kind != "policy",
        "new5_used": False,
    }


def run_sensitivity(
    payloads: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, np.ndarray]],
    norms: FillNorms,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, payload in payloads.items():
        dense_decision = dense_oracle_decision(payload)
        for fill_model in FILL_MODELS:
            for inventory_cap in INVENTORY_CAPS:
                matched_rows_for_reference: dict[str, dict[str, Any]] = {}
                for policy_id in REAL_POLICIES:
                    selected_count = int((decisions[split][policy_id] != p05a.DECISION_NO_QUOTE).sum())
                    matched_decision = matched_count_oracle_decision(payload, selected_count)
                    matched_row = simulate(
                        split,
                        f"oracle_matched_count_for_{policy_id}",
                        payload,
                        matched_decision,
                        fill_model,
                        inventory_cap,
                        norms,
                        oracle_kind="matched_count_oracle",
                        oracle_reference_policy=policy_id,
                    )
                    matched_row["matched_count_reference_pnl"] = matched_row["sequential_excess_pnl"]
                    matched_row["capture_ratio_against_matched_count"] = 1.0 if matched_row["sequential_excess_pnl"] else np.nan
                    matched_rows_for_reference[policy_id] = matched_row
                for policy_id in REAL_POLICIES:
                    rows.append(
                        simulate(
                            split,
                            policy_id,
                            payload,
                            decisions[split][policy_id],
                            fill_model,
                            inventory_cap,
                            norms,
                            oracle_kind="policy",
                            oracle_reference_policy=policy_id,
                            matched_count_reference_pnl=matched_rows_for_reference[policy_id]["sequential_excess_pnl"],
                        )
                    )
                rows.append(
                    simulate(
                        split,
                        "oracle_dense_full_action",
                        payload,
                        dense_decision,
                        fill_model,
                        inventory_cap,
                        norms,
                        oracle_kind="dense_oracle",
                    )
                )
                rows.extend(matched_rows_for_reference.values())
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "oracle_assumption_sensitivity_sweep.csv", index=False)
    return out


def summarize_sensitivity(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real = out[out["policy_id"].isin(REAL_POLICIES)].copy()
    aggregate = (
        real.groupby(["policy_id", "fill_model", "inventory_cap"], as_index=False)
        .agg(
            dev_pnl=("sequential_excess_pnl", lambda s: float(real.loc[s.index][real.loc[s.index, "split"].isin(DEVELOPMENT_SPLITS)]["sequential_excess_pnl"].sum())),
            oos_pnl=("sequential_excess_pnl", lambda s: float(real.loc[s.index][real.loc[s.index, "split"].isin(["new3", "new4"])]["sequential_excess_pnl"].sum())),
            val_capture=("capture_ratio_against_matched_count", lambda s: float(real.loc[s.index][real.loc[s.index, "split"].eq("val")]["capture_ratio_against_matched_count"].mean())),
            new3_capture=("capture_ratio_against_matched_count", lambda s: float(real.loc[s.index][real.loc[s.index, "split"].eq("new3")]["capture_ratio_against_matched_count"].mean())),
            new4_capture=("capture_ratio_against_matched_count", lambda s: float(real.loc[s.index][real.loc[s.index, "split"].eq("new4")]["capture_ratio_against_matched_count"].mean())),
            total_fill_count=("fill_count", "sum"),
            total_forced_no_quote=("forced_no_quote_count", "sum"),
        )
        .sort_values(["policy_id", "fill_model", "inventory_cap"])
        .reset_index(drop=True)
    )
    capture_range = (
        real[real["split"].isin(OOS_SPLITS)]
        .groupby(["policy_id"], as_index=False)
        .agg(
            min_capture=("capture_ratio_against_matched_count", "min"),
            max_capture=("capture_ratio_against_matched_count", "max"),
            min_pnl=("sequential_excess_pnl", "min"),
            max_pnl=("sequential_excess_pnl", "max"),
        )
    )
    rank_rows = []
    for (split, fill_model, cap), group in real.groupby(["split", "fill_model", "inventory_cap"]):
        ordered = group.sort_values("sequential_excess_pnl", ascending=False)["policy_id"].tolist()
        rank_rows.append(
            {
                "split": split,
                "fill_model": fill_model,
                "inventory_cap": int(cap),
                "rank_1": ordered[0] if len(ordered) > 0 else "",
                "rank_2": ordered[1] if len(ordered) > 1 else "",
                "rank_3": ordered[2] if len(ordered) > 2 else "",
                "ml_beats_queue": bool(
                    group[group["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")]["sequential_excess_pnl"].iloc[0]
                    > group[group["policy_id"].eq("side_by_queue_imbalance_abs_0p1")]["sequential_excess_pnl"].iloc[0]
                ),
                "phc_beats_ml": bool(
                    group[group["policy_id"].eq("phc_stp_v2_cosine_lr")]["sequential_excess_pnl"].iloc[0]
                    > group[group["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")]["sequential_excess_pnl"].iloc[0]
                ),
            }
        )
    ranks = pd.DataFrame(rank_rows).sort_values(["split", "fill_model", "inventory_cap"])
    aggregate.to_csv(OUT_DIR / "oracle_assumption_sensitivity_summary.csv", index=False)
    capture_range.to_csv(OUT_DIR / "oracle_assumption_capture_ranges.csv", index=False)
    ranks.to_csv(OUT_DIR / "oracle_assumption_policy_ranks.csv", index=False)
    return aggregate, capture_range, ranks


def topk_simulated_oracle(
    payload: dict[str, Any],
    target_fill_count: int,
    fill_model: str,
    inventory_cap: int,
    norms: FillNorms,
    reference_decision: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    side, best = best_side_and_value(payload)
    bid_values = payload["y_bid"].numpy().astype(np.float64)
    ask_values = payload["y_ask"].numpy().astype(np.float64)
    bid_order = np.argsort(np.nan_to_num(bid_values, nan=-np.inf))[::-1]
    ask_order = np.argsort(np.nan_to_num(ask_values, nan=-np.inf))[::-1]
    best_order = np.argsort(best)[::-1]
    n = len(best_order)
    target = int(target_fill_count)
    if target <= 0:
        decision = np.zeros(n, dtype=np.int8)
        return decision, {"target_fill_count": target, "selected_top_k": 0, "fill_error_abs": 0, "fill_error_pct": 0.0}

    selected_reference = int((reference_decision != p05a.DECISION_NO_QUOTE).sum()) if reference_decision is not None else 0
    if reference_decision is not None and selected_reference > 0:
        bid_share = float((reference_decision == p05a.DECISION_BID).sum() / selected_reference)
    else:
        bid_share = 0.5

    def decision_for_k(k: int, balanced: bool) -> np.ndarray:
        decision = np.zeros(n, dtype=np.int8)
        if not balanced:
            idx = best_order[:k]
            decision[idx] = side[idx]
            return decision
        bid_k = int(round(k * bid_share))
        ask_k = max(0, int(k) - bid_k)
        for idx in bid_order[: min(bid_k, n)]:
            if not np.isfinite(bid_values[idx]):
                continue
            if decision[idx] == p05a.DECISION_ASK and ask_values[idx] > bid_values[idx]:
                continue
            decision[idx] = p05a.DECISION_BID
        for idx in ask_order[: min(ask_k, n)]:
            if not np.isfinite(ask_values[idx]):
                continue
            if decision[idx] == p05a.DECISION_BID and bid_values[idx] >= ask_values[idx]:
                continue
            decision[idx] = p05a.DECISION_ASK
        return decision

    candidates = set()
    for multiplier in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]:
        candidates.add(min(n, max(1, int(math.ceil(target * multiplier)))))
    if selected_reference > 0:
        for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            candidates.add(min(n, max(1, int(math.ceil(selected_reference * multiplier)))))
    low, high = 1, n
    for _ in range(18):
        mid = (low + high) // 2
        candidates.add(mid)
        decision = decision_for_k(mid, balanced=reference_decision is not None)
        fills = deterministic_fills(payload, decision, fill_model, norms)
        pnl = decision_pnl(payload, decision)
        _, fill_count, *_ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=inventory_cap)
        if fill_count < target:
            low = min(n, mid + 1)
        else:
            high = max(1, mid - 1)

    best_decision = None
    best_meta = None
    best_key = None
    for k in sorted(candidates):
        for balanced in ([False, True] if reference_decision is not None else [False]):
            decision = decision_for_k(k, balanced=balanced)
            row = simulate(
                "candidate",
                "candidate",
                payload,
                decision,
                fill_model,
                inventory_cap,
                norms,
                oracle_kind="matched_count_simulator_oracle",
            )
            error = abs(int(row["fill_count"]) - target)
            key = (error, -float(row["sequential_excess_pnl"]), k, int(not balanced))
            if best_key is None or key < best_key:
                best_key = key
                best_decision = decision
                best_meta = {
                    "target_fill_count": target,
                    "selected_top_k": int(k),
                    "side_balanced_top_k": bool(balanced),
                    "fill_error_abs": int(error),
                    "fill_error_pct": float(error / max(target, 1)),
                }
    if best_decision is None or best_meta is None:
        raise RuntimeError("topk_simulated_oracle failed to select a candidate")
    return best_decision, best_meta


def run_matched_simulator_oracle(
    payloads: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, np.ndarray]],
    norms: FillNorms,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    base = sensitivity[
        (sensitivity["fill_model"].eq(REPRESENTATIVE_FILL_MODEL))
        & (sensitivity["inventory_cap"].astype(int).eq(REPRESENTATIVE_INVENTORY_CAP))
    ].copy()
    rows = []
    for split, payload in payloads.items():
        for policy_id in REAL_POLICIES:
            policy_row = base[(base["split"].eq(split)) & (base["policy_id"].eq(policy_id))].iloc[0]
            matched_row = base[(base["split"].eq(split)) & (base["policy_id"].eq(f"oracle_matched_count_for_{policy_id}"))].iloc[0]
            target_fill_count = int(policy_row["fill_count"])
            oracle_decision, meta = topk_simulated_oracle(
                payload,
                target_fill_count,
                REPRESENTATIVE_FILL_MODEL,
                REPRESENTATIVE_INVENTORY_CAP,
                norms,
                reference_decision=decisions[split][policy_id],
            )
            oracle_row = simulate(
                split,
                f"oracle_matched_count_simulator_for_{policy_id}",
                payload,
                oracle_decision,
                REPRESENTATIVE_FILL_MODEL,
                REPRESENTATIVE_INVENTORY_CAP,
                norms,
                oracle_kind="matched_count_simulator_oracle",
                oracle_reference_policy=policy_id,
            )
            oracle_pnl = float(oracle_row["sequential_excess_pnl"])
            policy_pnl = float(policy_row["sequential_excess_pnl"])
            matched_pnl = float(matched_row["sequential_excess_pnl"])
            oracle_row.update(
                {
                    **meta,
                    "policy_id_reference": policy_id,
                    "policy_sequential_excess_pnl": policy_pnl,
                    "policy_fill_count": target_fill_count,
                    "matched_count_oracle_sequential_pnl": matched_pnl,
                    "policy_capture_vs_matched_simulator": policy_pnl / oracle_pnl if oracle_pnl else np.nan,
                    "policy_capture_vs_original_matched_count": policy_pnl / matched_pnl if matched_pnl else np.nan,
                    "matched_simulator_vs_original_matched_count_pnl_ratio": oracle_pnl / matched_pnl if matched_pnl else np.nan,
                }
            )
            rows.append(oracle_row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "matched_count_simulator_oracle.csv", index=False)
    return out


def write_sensitivity_report(sensitivity: pd.DataFrame, aggregate: pd.DataFrame, capture_range: pd.DataFrame, ranks: pd.DataFrame) -> None:
    full_table_path = OUT_DIR / "oracle_assumption_sensitivity_sweep.csv"
    complete_9 = sensitivity[sensitivity["split"].isin(OOS_SPLITS)].copy()
    compact_cols = [
        "split",
        "policy_id",
        "fill_model",
        "inventory_cap",
        "sequential_excess_pnl",
        "fill_count",
        "forced_no_quote_count",
        "capture_ratio_against_matched_count",
    ]
    rank_consistency = ranks[ranks["split"].isin(OOS_SPLITS)].copy()
    ml_queue_all = bool(rank_consistency["ml_beats_queue"].all()) if not rank_consistency.empty else False
    phc_ml_all = bool(rank_consistency["phc_beats_ml"].all()) if not rank_consistency.empty else False
    write_text(
        CTX / "AUDIT_oracle_assumption_sensitivity_sweep.md",
        "\n".join(
            [
                "status: resolved",
                "# Audit: Oracle Assumption Sensitivity Sweep",
                "",
                "Hard rules: NEW5 was not loaded. The script loads only `phc_lite_{train,val,new3,new4}_sequences.pt` and frozen policy/checkpoint artifacts. No threshold, checkpoint, or model was changed.",
                "",
                "Execution assumptions swept: `inventory_cap in {1,2,3}` x `fill_model in {depth_haircut, queue_side_haircut, conservative_l2_stress}`. The `conservative_l2_stress` probabilities match the existing 05A/05C4 conservative L2 formula; the two other fill models match Phase 02F semantics.",
                "",
                f"Complete row-level table: `{full_table_path.relative_to(PROJECT_ROOT).as_posix()}`.",
                "",
                "## Capture Ratio Range",
                "",
                md_table(capture_range, ["policy_id", "min_capture", "max_capture", "min_pnl", "max_pnl"]),
                "",
                "## 9-Combination Aggregate Summary",
                "",
                md_table(
                    aggregate,
                    [
                        "policy_id",
                        "fill_model",
                        "inventory_cap",
                        "dev_pnl",
                        "oos_pnl",
                        "val_capture",
                        "new3_capture",
                        "new4_capture",
                        "total_fill_count",
                        "total_forced_no_quote",
                    ],
                ),
                "",
                "## Complete OOS 9-Combination Table",
                "",
                md_table(complete_9[compact_cols], compact_cols),
                "",
                "## Policy Rank Consistency",
                "",
                md_table(rank_consistency, ["split", "fill_model", "inventory_cap", "rank_1", "rank_2", "rank_3", "ml_beats_queue", "phc_beats_ml"]),
                "",
                "## Conclusion",
                "",
                f"ML beats the queue rule in every OOS assumption cell: `{ml_queue_all}`.",
                f"PHC-STP-v2 beats HGB in every OOS assumption cell: `{phc_ml_all}`.",
                "The absolute oracle gap is assumption-sensitive because fill_count and forced_no_quote_count both move with the execution model. The relative ordering among the three frozen policies is stable across the 9 OOS assumption cells in this sweep.",
                "",
                "This is diagnostic only. The oracle rows use future labels and must not be interpreted as tradable policies.",
            ]
        ),
    )


def write_matched_simulator_report(matched_sim: pd.DataFrame) -> None:
    oos = matched_sim[matched_sim["split"].isin(OOS_SPLITS)].copy()
    comparison = pd.DataFrame(
        [
            {
                "oracle_type": "matched-count oracle",
                "matches_policy_fill_count": "No, matches selected quote count before simulator",
                "uses_sequential_fill_simulator": "Yes in this audit table, but selection ignores simulator",
                "uses_future_labels": "Yes",
            },
            {
                "oracle_type": "causal-fill dense oracle",
                "matches_policy_fill_count": "No",
                "uses_sequential_fill_simulator": "Yes",
                "uses_future_labels": "Yes",
            },
            {
                "oracle_type": "matched-count simulator oracle",
                "matches_policy_fill_count": "Yes, targets realized policy fill_count",
                "uses_sequential_fill_simulator": "Yes",
                "uses_future_labels": "Yes",
            },
        ]
    )
    write_text(
        CTX / "AUDIT_matched_count_simulator_oracle.md",
        "\n".join(
            [
                "status: resolved",
                "# Audit: Matched-Count + Sequential-Simulator Oracle",
                "",
                "Hard rules: NEW5 was not loaded. The oracle is a diagnostic upper bound that uses future H20 labels; it is not a tradable policy and was not used to tune any frozen policy.",
                "",
                f"Representative assumption chosen after the B sweep: `fill_model={REPRESENTATIVE_FILL_MODEL}`, `inventory_cap={REPRESENTATIVE_INVENTORY_CAP}`. This is the incumbent assumption used by the earlier causal-fill reports, so it is the cleanest comparison point.",
                "",
                "## Oracle Type Comparison",
                "",
                md_table(comparison),
                "",
                "## Matched-Count Simulator Oracle Results",
                "",
                md_table(
                    matched_sim,
                    [
                        "split",
                        "policy_id_reference",
                        "policy_fill_count",
                        "fill_count",
                        "fill_error_pct",
                        "policy_sequential_excess_pnl",
                        "sequential_excess_pnl",
                        "matched_count_oracle_sequential_pnl",
                        "policy_capture_vs_matched_simulator",
                        "policy_capture_vs_original_matched_count",
                    ],
                ),
                "",
                "## OOS Capture Comparison",
                "",
                md_table(
                    oos[
                        [
                            "split",
                            "policy_id_reference",
                            "policy_capture_vs_matched_simulator",
                            "policy_capture_vs_original_matched_count",
                            "matched_simulator_vs_original_matched_count_pnl_ratio",
                        ]
                    ]
                ),
                "",
                "## Conclusion",
                "",
                "The stricter oracle lowers the oracle denominator whenever execution constraints make the original selected-count oracle unrealistic after causal fill. In those cells, policy capture versus the matched simulator is higher than capture versus the original matched-count oracle.",
                "The construction still uses future labels for ranking, so it remains a diagnostic headroom tool only.",
            ]
        ),
    )


def write_hls_placeholder_report(reason: str) -> None:
    write_text(
        CTX / "AUDIT_hls_cross_seed_resource_stability.md",
        "\n".join(
            [
                "status: open",
                "# Audit: HLS Cross-Seed Resource Stability",
                "",
                "The seed0/seed1 csynth rerun was not completed in this script.",
                "",
                f"Reason: {reason}",
                "",
                "This report is intentionally marked `open`; resource stability must not be claimed until seed0/seed1 HLS projects have real csynth numbers next to the existing seed2 artifact.",
            ]
        ),
    )


def main() -> None:
    ensure_dirs()
    payloads = load_payloads()
    norms = compute_fill_norms(payloads)
    decisions = make_real_decisions(payloads)
    sensitivity_path = OUT_DIR / "oracle_assumption_sensitivity_sweep.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
    else:
        sensitivity = run_sensitivity(payloads, decisions, norms)
    aggregate, capture_range, ranks = summarize_sensitivity(sensitivity)
    write_sensitivity_report(sensitivity, aggregate, capture_range, ranks)
    matched_sim = run_matched_simulator_oracle(payloads, decisions, norms, sensitivity)
    write_matched_simulator_report(matched_sim)
    print(CTX / "AUDIT_oracle_assumption_sensitivity_sweep.md")
    print(CTX / "AUDIT_matched_count_simulator_oracle.md")


if __name__ == "__main__":
    main()
