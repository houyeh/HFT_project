from __future__ import annotations

import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04c1_new5_oracle_headroom_analysis"
RESULTS = ROOT / "results"
PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
PHASE04C = PROJECT_ROOT / "n2ew_era" / "04c_new5_one_shot_final_holdout"
PHASE04B_ACTIONS = PHASE04B / "data" / "txf_only_action_universe.parquet"
ACTION_FILE = PHASE04C / "data" / "new5_txf_only_action_universe.parquet"

ACTION_BID = 0
ACTION_ASK = 1
ACTION_NO_QUOTE = 2
DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2
HORIZON = 20

POLICIES = ["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"]


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


@dataclass
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def read_event_frame(path: Path) -> pd.DataFrame:
    cols = [
        "split",
        "session_id",
        "symbol",
        "event_index",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "excess_pnl_H20",
    ]
    raw = pd.read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(ACTION_ASK)].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int).eq(ACTION_NO_QUOTE)].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet mismatch in {path}")
    for col in ["session_id", "symbol", "event_index"]:
        if not bid[col].reset_index(drop=True).equals(ask[col].reset_index(drop=True)):
            raise RuntimeError(f"Bid/ask {col} mismatch in {path}")
    out = bid.drop(columns=["action_slot", "excess_pnl_H20"]).copy()
    out["bid_excess_pnl_H20"] = pd.to_numeric(bid["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["ask_excess_pnl_H20"] = pd.to_numeric(ask["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return out


def decision_pnl(event: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    out = np.zeros(len(event), dtype=float)
    bid = event["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = event["ask_excess_pnl_H20"].to_numpy(dtype=float)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    out[~np.isfinite(out)] = 0.0
    return out


def update_metric(metric: Metric, decision: np.ndarray, pnl: np.ndarray) -> None:
    selected = decision != DECISION_NO_QUOTE
    metric.state_count += int(len(decision))
    metric.selected_count += int(selected.sum())
    metric.bid_count += int((decision == DECISION_BID).sum())
    metric.ask_count += int((decision == DECISION_ASK).sum())
    selected_pnl = pnl[selected]
    metric.pnl_sum += float(selected_pnl.sum())
    metric.win_count += int((selected_pnl > 0).sum())
    metric.loss_count += int((selected_pnl < 0).sum())
    equity = metric.equity + np.cumsum(pnl)
    if len(equity):
        peak = np.maximum.accumulate(np.maximum(equity, metric.peak))
        dd = peak - equity
        metric.max_drawdown = max(metric.max_drawdown, float(dd.max()))
        metric.equity = float(equity[-1])
        metric.peak = max(metric.peak, float(peak[-1]))


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
        "no_quote_count": metric.state_count - metric.selected_count,
        "total_excess_pnl_H20": metric.pnl_sum,
        "avg_selected_excess": metric.pnl_sum / selected if metric.selected_count else 0.0,
        "win_rate": metric.win_count / selected if metric.selected_count else 0.0,
        "loss_rate": metric.loss_count / selected if metric.selected_count else 0.0,
        "max_drawdown": metric.max_drawdown,
    }


def part_paths(split: str) -> list[Path]:
    base = PHASE04B_ACTIONS / f"split={split}"
    return sorted(base.glob("*.parquet")) if base.exists() else []


def compute_fill_norms() -> FillNorms:
    values = {"top_depth": [], "bid_sz_1": [], "ask_sz_1": []}
    for path in part_paths("train"):
        frame = read_event_frame(path)
        for col in values:
            arr = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(arr):
                values[col].append(arr[arr > 0])
    med = {}
    for col, chunks in values.items():
        arr = np.concatenate([c for c in chunks if len(c)]) if chunks else np.array([1.0])
        med[col] = float(np.nanmedian(arr)) if len(arr) else 1.0
    return FillNorms(med["top_depth"], med["bid_sz_1"], med["ask_sz_1"])


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


def stable_uniform(frame: pd.DataFrame, decision: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str).to_numpy(),
            "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "decision": np.asarray(decision, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little"))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


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
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        if not fills[t]:
            continue
        side_sign = 1 if decision[t] == DECISION_BID else -1
        if abs(inventory + side_sign) > inventory_cap:
            forced_no_quote += 1
            continue
        trade_pnl = float(base_pnl[t])
        inventory += side_sign
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        close_idx = min(t + horizon, n - 1)
        close_side_sum[close_idx] += side_sign
        close_pnl_sum[close_idx] += trade_pnl
        fill_count += 1
        win_count += int(trade_pnl > 0.0)
        loss_count += int(trade_pnl < 0.0)
    return total_pnl, fill_count, forced_no_quote, max_abs_inventory, max_drawdown, win_count, loss_count


class p04b:
    DECISION_NO_QUOTE = DECISION_NO_QUOTE
    DECISION_BID = DECISION_BID
    DECISION_ASK = DECISION_ASK
    HORIZON = HORIZON
    Metric = Metric
    update_metric = staticmethod(update_metric)
    finalize_metric = staticmethod(finalize_metric)
    decision_pnl = staticmethod(decision_pnl)
    compute_fill_norms = staticmethod(compute_fill_norms)
    fill_probabilities = staticmethod(fill_probabilities)
    simulate_scheduled = staticmethod(simulate_scheduled)


class p04c:
    ACTION_FILE = ACTION_FILE
    read_new5_event_frame = staticmethod(lambda: read_event_frame(ACTION_FILE))
    stable_uniform = staticmethod(stable_uniform)


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision_metric(event: pd.DataFrame, decision: np.ndarray, policy_id: str) -> dict[str, Any]:
    metric = p04b.Metric()
    p04b.update_metric(metric, decision, p04b.decision_pnl(event, decision))
    row = p04b.finalize_metric("new5_final_holdout", policy_id, metric)
    row["diagnostic_only"] = True
    row["oracle_uses_future_labels"] = policy_id.startswith("oracle")
    return row


def full_oracle(event: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    bid = event["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = event["ask_excess_pnl_H20"].to_numpy(dtype=float)
    decision = np.zeros(len(event), dtype=np.int8)
    decision[(bid >= ask) & (bid > 0.0)] = p04b.DECISION_BID
    decision[(ask > bid) & (ask > 0.0)] = p04b.DECISION_ASK
    pnl = p04b.decision_pnl(event, decision)
    rows = pd.DataFrame(
        {
            "session_id": event["session_id"].astype(str).to_numpy(),
            "event_index": event["event_index"].to_numpy(),
            "oracle_bid": bid,
            "oracle_ask": ask,
            "oracle_no_quote": 0.0,
            "oracle_action": np.where(decision == p04b.DECISION_BID, "bid", np.where(decision == p04b.DECISION_ASK, "ask", "no_quote")),
            "oracle_pnl": pnl,
        }
    )
    return rows, decision


def top_best_state_oracle(event: pd.DataFrame, selected_count: int, positive_only: bool = False) -> np.ndarray:
    bid = event["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = event["ask_excess_pnl_H20"].to_numpy(dtype=float)
    best = np.maximum(bid, ask)
    side = np.where(bid >= ask, p04b.DECISION_BID, p04b.DECISION_ASK)
    candidates = np.where(best > 0.0)[0] if positive_only else np.arange(len(event))
    order = candidates[np.argsort(best[candidates])[::-1]]
    chosen = order[: min(int(selected_count), len(order))]
    decision = np.zeros(len(event), dtype=np.int8)
    decision[chosen] = side[chosen].astype(np.int8)
    return decision


def greedy_count_side_oracle(event: pd.DataFrame, bid_count: int, ask_count: int, positive_only: bool = False) -> np.ndarray:
    bid = event["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = event["ask_excess_pnl_H20"].to_numpy(dtype=float)
    candidates: list[tuple[float, int, int]] = []
    for idx, value in enumerate(bid):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p04b.DECISION_BID))
    for idx, value in enumerate(ask):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p04b.DECISION_ASK))
    candidates.sort(reverse=True, key=lambda x: x[0])
    remaining = {p04b.DECISION_BID: int(bid_count), p04b.DECISION_ASK: int(ask_count)}
    used = np.zeros(len(event), dtype=bool)
    decision = np.zeros(len(event), dtype=np.int8)
    for _, idx, side in candidates:
        if used[idx] or remaining[side] <= 0:
            continue
        decision[idx] = side
        used[idx] = True
        remaining[side] -= 1
        if remaining[p04b.DECISION_BID] <= 0 and remaining[p04b.DECISION_ASK] <= 0:
            break
    return decision


def load_policy_metrics() -> pd.DataFrame:
    path = PHASE04C / "results" / "new5_frozen_policy_eval.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def matched_oracles(event: pd.DataFrame, policy_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows = []
    side_rows = []
    for policy_id in POLICIES:
        policy = policy_eval[policy_eval["policy_id"].eq(policy_id)].iloc[0]
        policy_pnl = float(policy["total_excess_pnl_H20"])
        selected_count = int(policy["selected_action_count"])
        positive_decision = top_best_state_oracle(event, selected_count, positive_only=True)
        count_decision = top_best_state_oracle(event, selected_count, positive_only=False)
        count_metric = decision_metric(event, count_decision, f"oracle_matched_count_for_{policy_id}")
        positive_metric = decision_metric(event, positive_decision, f"oracle_positive_only_matched_count_for_{policy_id}")
        count_oracle_pnl = float(count_metric["total_excess_pnl_H20"])
        positive_oracle_pnl = float(positive_metric["total_excess_pnl_H20"])
        count_rows.append(
            {
                "policy_id": policy_id,
                "policy_selected_count": selected_count,
                "policy_pnl": policy_pnl,
                "matched_count_oracle_selected_count": int((count_decision != p04b.DECISION_NO_QUOTE).sum()),
                "matched_count_oracle_pnl": count_oracle_pnl,
                "positive_only_oracle_selected_count": int((positive_decision != p04b.DECISION_NO_QUOTE).sum()),
                "positive_only_oracle_pnl": positive_oracle_pnl,
                "forced_nonpositive_actions_for_exact_count": int((count_decision != p04b.DECISION_NO_QUOTE).sum() - (positive_decision != p04b.DECISION_NO_QUOTE).sum()),
                "capture_ratio": policy_pnl / count_oracle_pnl if count_oracle_pnl else np.nan,
                "opportunity_gap": count_oracle_pnl - policy_pnl,
                "oracle_uses_future_labels": True,
                "diagnostic_only": True,
            }
        )
        side_positive_decision = greedy_count_side_oracle(event, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=True)
        side_decision = greedy_count_side_oracle(event, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=False)
        side_metric = decision_metric(event, side_decision, f"oracle_matched_count_side_for_{policy_id}")
        side_positive_metric = decision_metric(event, side_positive_decision, f"oracle_positive_only_matched_count_side_for_{policy_id}")
        side_pnl = float(side_metric["total_excess_pnl_H20"])
        side_positive_pnl = float(side_positive_metric["total_excess_pnl_H20"])
        side_rows.append(
            {
                "policy_id": policy_id,
                "policy_selected_count": selected_count,
                "policy_bid_count": int(policy["bid_count"]),
                "policy_ask_count": int(policy["ask_count"]),
                "policy_pnl": policy_pnl,
                "oracle_selected_count": int((side_decision != p04b.DECISION_NO_QUOTE).sum()),
                "oracle_bid_count": int((side_decision == p04b.DECISION_BID).sum()),
                "oracle_ask_count": int((side_decision == p04b.DECISION_ASK).sum()),
                "matched_count_side_oracle_pnl": side_pnl,
                "positive_only_oracle_selected_count": int((side_positive_decision != p04b.DECISION_NO_QUOTE).sum()),
                "positive_only_oracle_bid_count": int((side_positive_decision == p04b.DECISION_BID).sum()),
                "positive_only_oracle_ask_count": int((side_positive_decision == p04b.DECISION_ASK).sum()),
                "positive_only_oracle_pnl": side_positive_pnl,
                "forced_nonpositive_actions_for_exact_count": int((side_decision != p04b.DECISION_NO_QUOTE).sum() - (side_positive_decision != p04b.DECISION_NO_QUOTE).sum()),
                "capture_ratio": policy_pnl / side_pnl if side_pnl else np.nan,
                "opportunity_gap": side_pnl - policy_pnl,
                "oracle_uses_future_labels": True,
                "diagnostic_only": True,
            }
        )
    count_df = pd.DataFrame(count_rows)
    side_df = pd.DataFrame(side_rows)
    count_df.to_csv(RESULTS / "new5_matched_count_oracle.csv", index=False)
    side_df.to_csv(RESULTS / "new5_matched_count_side_oracle.csv", index=False)
    return count_df, side_df


def causal_fill_oracle(event: pd.DataFrame, full_decision: np.ndarray, count_df: pd.DataFrame, side_df: pd.DataFrame) -> pd.DataFrame:
    norms = p04b.compute_fill_norms()
    policy_eval = load_policy_metrics()
    decisions = {"oracle_full_action": full_decision}
    for policy_id in POLICIES:
        policy = policy_eval[policy_eval["policy_id"].eq(policy_id)].iloc[0]
        decisions[f"oracle_matched_count_for_{policy_id}"] = top_best_state_oracle(event, int(policy["selected_action_count"]), positive_only=False)
        decisions[f"oracle_matched_count_side_for_{policy_id}"] = greedy_count_side_oracle(event, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=False)
    rows = []
    for oracle_id, decision in decisions.items():
        probs = p04b.fill_probabilities(event, decision, norms)
        fills = (decision != p04b.DECISION_NO_QUOTE) & (p04c.stable_uniform(event, decision, f"04c1_{oracle_id}") < probs)
        result = p04b.simulate_scheduled(
            decision.astype(np.int8),
            fills.astype(np.bool_),
            p04b.decision_pnl(event, decision).astype(np.float64),
            p04b.HORIZON,
            1,
        )
        fill_count = max(int(result[1]), 1)
        rows.append(
            {
                "oracle_id": oracle_id,
                "split": "new5_final_holdout",
                "fill_model": "conservative_l2_stress",
                "inventory_cap": 1,
                "horizon": p04b.HORIZON,
                "sequential_excess_pnl": float(result[0]),
                "fill_count": int(result[1]),
                "forced_no_quote_count": int(result[2]),
                "max_abs_inventory": int(result[3]),
                "max_drawdown": float(result[4]),
                "win_rate": int(result[5]) / fill_count if int(result[1]) else 0.0,
                "loss_rate": int(result[6]) / fill_count if int(result[1]) else 0.0,
                "oracle_uses_future_labels": True,
                "diagnostic_only": True,
                "true_queue_replay": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new5_causal_fill_oracle.csv", index=False)
    return out


def write_reports(full_rows: pd.DataFrame, full_decision: np.ndarray, count_df: pd.DataFrame, side_df: pd.DataFrame, causal_df: pd.DataFrame) -> None:
    event = p04c.read_new5_event_frame()
    full_metric = pd.DataFrame([decision_metric(event, full_decision, "oracle_full_action")])
    full_metric.to_csv(RESULTS / "new5_full_action_oracle_summary.csv", index=False)
    write_markdown(
        ROOT / "NEW5_FULL_ACTION_ORACLE_REPORT.md",
        [
            "# NEW5 Full Action Oracle Report",
            "",
            "Oracle action is `argmax(excess_bid, excess_ask, 0)` for every NEW5 state. It uses future H20 labels and is non-causal.",
            "",
            full_metric.to_markdown(index=False),
            "",
            "This oracle is diagnostic only and cannot be used to tune policy.",
        ],
    )
    hgb_count = count_df[count_df["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].iloc[0]
    hgb_side = side_df[side_df["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].iloc[0]
    queue_count = count_df[count_df["policy_id"].eq("side_by_queue_imbalance_abs_0p1")].iloc[0]
    write_markdown(
        ROOT / "NEW5_ORACLE_HEADROOM_REPORT.md",
        [
            "# NEW5 Oracle Headroom Report",
            "",
            "Oracle uses future labels and is non-causal. Oracle is diagnostic only. Oracle cannot be used to tune policy. Oracle does not imply real-world profitability.",
            "",
            "## Unconstrained Oracle Upper Bound",
            "",
            full_metric.to_markdown(index=False),
            "",
            "## Matched-Count Oracle",
            "",
            count_df.to_markdown(index=False),
            "",
            "## Matched-Count + Side Oracle",
            "",
            side_df.to_markdown(index=False),
            "",
            "## Causal-Fill Sequential Oracle",
            "",
            causal_df.to_markdown(index=False),
            "",
            "## Required Answers",
            "",
            f"Unconstrained full-action oracle PnL: `{float(full_metric.iloc[0]['total_excess_pnl_H20']):,.2f}`.",
            f"HGB matched-count oracle PnL: `{float(hgb_count['matched_count_oracle_pnl']):,.2f}`; HGB capture ratio: `{float(hgb_count['capture_ratio']):.4f}`; gap: `{float(hgb_count['opportunity_gap']):,.2f}`.",
            f"HGB matched-count+side oracle PnL: `{float(hgb_side['matched_count_side_oracle_pnl']):,.2f}`; HGB capture ratio: `{float(hgb_side['capture_ratio']):.4f}`; gap: `{float(hgb_side['opportunity_gap']):,.2f}`.",
            f"Queue matched-count oracle PnL: `{float(queue_count['matched_count_oracle_pnl']):,.2f}`; queue capture ratio: `{float(queue_count['capture_ratio']):.4f}`.",
            "Because the full no-quote-allowed oracle finds fewer positive states than the frozen policies selected, the CSV also reports `positive_only_oracle_*` and `forced_nonpositive_actions_for_exact_count`. The main matched-count columns enforce the exact policy count by including the least-bad remaining actions when necessary.",
            "The gap between HGB/queue and oracle indicates remaining theoretical headroom for sequence models such as PHC-lite, but only fresh unseen data can support a formal claim.",
            "",
            "Why oracle is not tradable: it directly selects using `excess_pnl_H20_bid` and `excess_pnl_H20_ask`, which are future outcome labels unavailable at quote decision time.",
        ],
    )


def main() -> None:
    ensure_dirs()
    event = p04c.read_new5_event_frame()
    full_rows, full_decision = full_oracle(event)
    full_rows.to_csv(RESULTS / "new5_full_action_oracle.csv", index=False)
    policy_eval = load_policy_metrics()
    count_df, side_df = matched_oracles(event, policy_eval)
    causal_df = causal_fill_oracle(event, full_decision, count_df, side_df)
    write_reports(full_rows, full_decision, count_df, side_df, causal_df)
    print(ROOT / "NEW5_ORACLE_HEADROOM_REPORT.md")


if __name__ == "__main__":
    main()
