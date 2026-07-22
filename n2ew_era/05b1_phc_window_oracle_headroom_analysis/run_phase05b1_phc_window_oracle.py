from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05b1_phc_window_oracle_headroom_analysis"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05B = PROJECT_ROOT / "n2ew_era" / "05b_phc_lite_new5_posthoc_diagnostic"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402


SPLIT = "post_hoc_diagnostic_only_not_fresh_holdout"
POLICIES = [
    "side_by_queue_imbalance_abs_0p1",
    "ml_hgb_regressor_shallow_convergence_selected",
    "phc_lite_posthoc_new5",
]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_payload() -> dict[str, Any]:
    path = PHASE05B / "data" / "phc_new5_posthoc_sequences.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 05B NEW5 PHC sequence payload: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("seq_len", -1)) != 128 or int(payload.get("stride", -1)) != 32:
        raise RuntimeError("PHC-window oracle requires the frozen 05B seq_len=128 / stride=32 payload.")
    if not bool(payload.get("post_hoc_diagnostic_only", False)):
        raise RuntimeError("Expected Phase 05B payload to be marked post_hoc_diagnostic_only.")
    return payload


def load_policy_eval() -> pd.DataFrame:
    path = PHASE05B / "results" / "phc_new5_posthoc_policy_eval.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def oracle_full_decision(payload: dict[str, Any]) -> np.ndarray:
    bid = payload["y_bid"].numpy().astype(float)
    ask = payload["y_ask"].numpy().astype(float)
    decision = np.zeros(len(bid), dtype=np.int8)
    decision[(bid >= ask) & (bid > 0.0)] = p05a.DECISION_BID
    decision[(ask > bid) & (ask > 0.0)] = p05a.DECISION_ASK
    return decision


def top_best_state_oracle(payload: dict[str, Any], selected_count: int, positive_only: bool = False) -> np.ndarray:
    bid = payload["y_bid"].numpy().astype(float)
    ask = payload["y_ask"].numpy().astype(float)
    best = np.maximum(bid, ask)
    side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
    candidates = np.where(best > 0.0)[0] if positive_only else np.arange(len(best))
    order = candidates[np.argsort(best[candidates])[::-1]]
    chosen = order[: min(int(selected_count), len(order))]
    decision = np.zeros(len(best), dtype=np.int8)
    decision[chosen] = side[chosen].astype(np.int8)
    return decision


def greedy_count_side_oracle(payload: dict[str, Any], bid_count: int, ask_count: int, positive_only: bool = False) -> np.ndarray:
    bid = payload["y_bid"].numpy().astype(float)
    ask = payload["y_ask"].numpy().astype(float)
    candidates: list[tuple[float, int, int]] = []
    for idx, value in enumerate(bid):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p05a.DECISION_BID))
    for idx, value in enumerate(ask):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p05a.DECISION_ASK))
    candidates.sort(reverse=True, key=lambda x: x[0])
    remaining = {p05a.DECISION_BID: int(bid_count), p05a.DECISION_ASK: int(ask_count)}
    used = np.zeros(len(bid), dtype=bool)
    decision = np.zeros(len(bid), dtype=np.int8)
    for _, idx, side in candidates:
        if used[idx] or remaining[side] <= 0:
            continue
        decision[idx] = side
        used[idx] = True
        remaining[side] -= 1
        if remaining[p05a.DECISION_BID] <= 0 and remaining[p05a.DECISION_ASK] <= 0:
            break
    return decision


def oracle_rows(payload: dict[str, Any], decision: np.ndarray) -> pd.DataFrame:
    bid = payload["y_bid"].numpy().astype(float)
    ask = payload["y_ask"].numpy().astype(float)
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    return pd.DataFrame(
        {
            "split": SPLIT,
            "window_id": np.arange(len(decision), dtype=np.int64),
            "event_index": payload["event_index"].numpy().astype(np.int64),
            "oracle_bid": bid,
            "oracle_ask": ask,
            "oracle_no_quote": 0.0,
            "oracle_action": np.where(decision == p05a.DECISION_BID, "bid", np.where(decision == p05a.DECISION_ASK, "ask", "no_quote")),
            "oracle_pnl": pnl,
            "oracle_uses_future_labels": True,
            "post_hoc_diagnostic_only": True,
        }
    )


def metric_row(payload: dict[str, Any], decision: np.ndarray, policy_id: str) -> dict[str, Any]:
    row = p05a.metric_from_decision(SPLIT, policy_id, payload, decision)
    row["oracle_uses_future_labels"] = policy_id.startswith("oracle")
    row["post_hoc_diagnostic_only"] = True
    row["new5_used_for_tuning"] = False
    return row


def full_oracle(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    decision = oracle_full_decision(payload)
    detail = oracle_rows(payload, decision)
    detail.to_csv(RESULTS / "phc_new5_window_full_oracle.csv", index=False)
    summary = pd.DataFrame([metric_row(payload, decision, "oracle_phc_window_full_action")])
    summary.to_csv(RESULTS / "phc_new5_window_full_oracle_summary.csv", index=False)
    write_markdown(
        ROOT / "PHC_NEW5_WINDOW_FULL_ORACLE_REPORT.md",
        [
            "# PHC NEW5 Window Full Oracle Report",
            "",
            "For each PHC sequence terminal state, oracle action is `argmax(excess_bid, excess_ask, 0)`. Oracle uses future labels and is non-causal.",
            "",
            summary.to_markdown(index=False),
            "",
            "This is post-hoc diagnostic because NEW5 was already used in Phase 04C.",
        ],
    )
    return detail, summary, decision


def matched_oracles(payload: dict[str, Any], policy_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    for policy_id in POLICIES:
        policy = policy_eval[policy_eval["policy_id"].eq(policy_id)].iloc[0]
        policy_pnl = float(policy["total_excess_pnl_H20"])
        selected_count = int(policy["selected_action_count"])
        count_decision = top_best_state_oracle(payload, selected_count, positive_only=False)
        pos_count_decision = top_best_state_oracle(payload, selected_count, positive_only=True)
        count_metric = metric_row(payload, count_decision, f"oracle_matched_count_for_{policy_id}")
        pos_count_metric = metric_row(payload, pos_count_decision, f"oracle_positive_only_matched_count_for_{policy_id}")
        count_oracle_pnl = float(count_metric["total_excess_pnl_H20"])
        pos_oracle_pnl = float(pos_count_metric["total_excess_pnl_H20"])
        count_rows.append(
            {
                "policy_id": policy_id,
                "policy_selected_count": selected_count,
                "policy_pnl": policy_pnl,
                "matched_count_oracle_selected_count": int((count_decision != p05a.DECISION_NO_QUOTE).sum()),
                "matched_count_oracle_pnl": count_oracle_pnl,
                "positive_only_oracle_selected_count": int((pos_count_decision != p05a.DECISION_NO_QUOTE).sum()),
                "positive_only_oracle_pnl": pos_oracle_pnl,
                "forced_nonpositive_actions_for_exact_count": int((count_decision != p05a.DECISION_NO_QUOTE).sum() - (pos_count_decision != p05a.DECISION_NO_QUOTE).sum()),
                "capture_ratio": policy_pnl / count_oracle_pnl if count_oracle_pnl else np.nan,
                "opportunity_gap": count_oracle_pnl - policy_pnl,
                "oracle_uses_future_labels": True,
                "post_hoc_diagnostic_only": True,
            }
        )
        side_decision = greedy_count_side_oracle(payload, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=False)
        pos_side_decision = greedy_count_side_oracle(payload, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=True)
        side_metric = metric_row(payload, side_decision, f"oracle_matched_count_side_for_{policy_id}")
        pos_side_metric = metric_row(payload, pos_side_decision, f"oracle_positive_only_matched_count_side_for_{policy_id}")
        side_pnl = float(side_metric["total_excess_pnl_H20"])
        pos_side_pnl = float(pos_side_metric["total_excess_pnl_H20"])
        side_rows.append(
            {
                "policy_id": policy_id,
                "policy_selected_count": selected_count,
                "policy_bid_count": int(policy["bid_count"]),
                "policy_ask_count": int(policy["ask_count"]),
                "policy_pnl": policy_pnl,
                "oracle_selected_count": int((side_decision != p05a.DECISION_NO_QUOTE).sum()),
                "oracle_bid_count": int((side_decision == p05a.DECISION_BID).sum()),
                "oracle_ask_count": int((side_decision == p05a.DECISION_ASK).sum()),
                "matched_count_side_oracle_pnl": side_pnl,
                "positive_only_oracle_selected_count": int((pos_side_decision != p05a.DECISION_NO_QUOTE).sum()),
                "positive_only_oracle_bid_count": int((pos_side_decision == p05a.DECISION_BID).sum()),
                "positive_only_oracle_ask_count": int((pos_side_decision == p05a.DECISION_ASK).sum()),
                "positive_only_oracle_pnl": pos_side_pnl,
                "forced_nonpositive_actions_for_exact_count": int((side_decision != p05a.DECISION_NO_QUOTE).sum() - (pos_side_decision != p05a.DECISION_NO_QUOTE).sum()),
                "capture_ratio": policy_pnl / side_pnl if side_pnl else np.nan,
                "opportunity_gap": side_pnl - policy_pnl,
                "oracle_uses_future_labels": True,
                "post_hoc_diagnostic_only": True,
            }
        )
    count_df = pd.DataFrame(count_rows)
    side_df = pd.DataFrame(side_rows)
    count_df.to_csv(RESULTS / "phc_new5_window_matched_count_oracle.csv", index=False)
    side_df.to_csv(RESULTS / "phc_new5_window_matched_count_side_oracle.csv", index=False)
    return count_df, side_df


def causal_fill_oracles(payload: dict[str, Any], full_decision: np.ndarray, policy_eval: pd.DataFrame) -> pd.DataFrame:
    norms = p05a.compute_fill_norms_from_train_sequences()
    decisions = {"oracle_phc_window_full_action": full_decision}
    for policy_id in POLICIES:
        policy = policy_eval[policy_eval["policy_id"].eq(policy_id)].iloc[0]
        decisions[f"oracle_matched_count_for_{policy_id}"] = top_best_state_oracle(payload, int(policy["selected_action_count"]), positive_only=False)
        decisions[f"oracle_matched_count_side_for_{policy_id}"] = greedy_count_side_oracle(payload, int(policy["bid_count"]), int(policy["ask_count"]), positive_only=False)
    rows: list[dict[str, Any]] = []
    for oracle_id, decision in decisions.items():
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, f"05b1_{oracle_id}") < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
        rows.append(
            {
                "split": SPLIT,
                "oracle_id": oracle_id,
                "horizon": p05a.HORIZON,
                "fill_model": "conservative_l2_stress",
                "inventory_cap": 1,
                "fee_multiplier": 1.0,
                "sequential_excess_pnl": float(total),
                "fill_count": int(fill_count),
                "forced_no_quote_count": int(forced),
                "max_abs_inventory": int(max_inv),
                "max_drawdown": float(max_dd),
                "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
                "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
                "oracle_uses_future_labels": True,
                "post_hoc_diagnostic_only": True,
                "true_queue_replay": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_new5_window_causal_fill_oracle.csv", index=False)
    return out


def write_headroom_report(full_summary: pd.DataFrame, count_df: pd.DataFrame, side_df: pd.DataFrame, causal_df: pd.DataFrame) -> None:
    phc_count = count_df[count_df["policy_id"].eq("phc_lite_posthoc_new5")].iloc[0]
    phc_side = side_df[side_df["policy_id"].eq("phc_lite_posthoc_new5")].iloc[0]
    hgb_count = count_df[count_df["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].iloc[0]
    queue_count = count_df[count_df["policy_id"].eq("side_by_queue_imbalance_abs_0p1")].iloc[0]
    reduced_gap_vs_hgb = float(phc_count["opportunity_gap"]) < float(hgb_count["opportunity_gap"])
    write_markdown(
        ROOT / "PHC_NEW5_WINDOW_ORACLE_HEADROOM_REPORT.md",
        [
            "# PHC NEW5 Window Oracle Headroom Report",
            "",
            "Oracle uses future labels. Oracle is not tradable. Oracle cannot tune PHC. This is post-hoc diagnostic because NEW5 was already used in Phase 04C.",
            "",
            "## PHC-window Full Oracle",
            "",
            full_summary.to_markdown(index=False),
            "",
            "## Matched-count Oracle",
            "",
            count_df.to_markdown(index=False),
            "",
            "## Matched-count + Side Oracle",
            "",
            side_df.to_markdown(index=False),
            "",
            "## Causal-fill Sequential Oracle",
            "",
            causal_df.to_markdown(index=False),
            "",
            "## Required Answers",
            "",
            f"PHC-window full oracle upper bound: `{float(full_summary.iloc[0]['total_excess_pnl_H20']):,.2f}` across `{int(full_summary.iloc[0]['state_count'])}` sequence terminal states.",
            f"PHC-lite matched-count capture ratio: `{float(phc_count['capture_ratio']):.4f}`; opportunity gap: `{float(phc_count['opportunity_gap']):,.2f}`.",
            f"PHC-lite matched-count+side capture ratio: `{float(phc_side['capture_ratio']):.4f}`; opportunity gap: `{float(phc_side['opportunity_gap']):,.2f}`.",
            f"HGB matched-count capture ratio on the same PHC-window universe: `{float(hgb_count['capture_ratio']):.4f}`.",
            f"Queue matched-count capture ratio on the same PHC-window universe: `{float(queue_count['capture_ratio']):.4f}`.",
            f"Does PHC reduce the matched-count opportunity gap relative to HGB? `{bool(reduced_gap_vs_hgb)}`.",
            "There is still large headroom because the oracle directly observes future `excess_pnl_H20_bid/ask` labels unavailable at decision time.",
            "",
            "This oracle is diagnostic only and cannot be used to modify PHC threshold, seq_len, architecture, feature set, or selection claims.",
        ],
    )


def main() -> None:
    ensure_dirs()
    payload = load_payload()
    policy_eval = load_policy_eval()
    _, full_summary, full_decision = full_oracle(payload)
    count_df, side_df = matched_oracles(payload, policy_eval)
    causal_df = causal_fill_oracles(payload, full_decision, policy_eval)
    write_headroom_report(full_summary, count_df, side_df, causal_df)
    print(ROOT / "PHC_NEW5_WINDOW_ORACLE_HEADROOM_REPORT.md")


if __name__ == "__main__":
    main()
