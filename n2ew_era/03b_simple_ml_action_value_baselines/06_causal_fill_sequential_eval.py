from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase03b import (
    ACTION_BID,
    ACTION_ASK,
    DECISION_BID,
    DECISION_ASK,
    FROZEN_SPLITS,
    HORIZON,
    RESULTS,
    TARGET,
    assert_bid_ask_pairing,
    build_feature_frame,
    compute_train_fill_norms,
    conservative_l2_fill_mask,
    ensure_dirs,
    part_paths,
    read_json,
    selected_actions_path,
    simulate_scheduled_excess,
)


def part_state_pnl(matrix: pd.DataFrame, decision: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    states = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    bid = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = matrix[matrix["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    assert_bid_ask_pairing(bid, ask, "part_state_pnl")
    pnl = np.zeros(len(states), dtype=np.float32)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    pnl[bid_mask] = bid[TARGET].to_numpy(dtype=np.float32)[bid_mask]
    pnl[ask_mask] = ask[TARGET].to_numpy(dtype=np.float32)[ask_mask]
    return states, np.nan_to_num(pnl, nan=0.0)


def main() -> None:
    ensure_dirs()
    selected_policy = read_json(RESULTS / "selected_phase03b_policy.json")
    norms = compute_train_fill_norms()
    rows = []
    curve_rows = []
    for split in FROZEN_SPLITS:
        selected = pd.read_parquet(selected_actions_path(split))
        selected_by_part = {str(k): g.sort_values("event_index").reset_index(drop=True) for k, g in selected.groupby("session_id")}
        split_equity = 0.0
        point = 0
        totals = {
            "sequential_excess_pnl": 0.0,
            "fill_count": 0,
            "forced_no_quote_count": 0,
            "max_abs_inventory": 0,
            "max_drawdown": 0.0,
            "wins": 0.0,
            "losses": 0.0,
        }
        for path in part_paths(split):
            matrix = build_feature_frame(path, include_targets=True)
            session_id = str(matrix["session_id"].iloc[0])
            sel = selected_by_part[session_id]
            decision = sel["decision"].astype(np.int8).to_numpy()
            states, pnl = part_state_pnl(matrix, decision)
            fills, _ = conservative_l2_fill_mask(states, decision, norms)
            result = simulate_scheduled_excess(decision, fills, pnl, horizon=HORIZON, inventory_cap=1)
            totals["sequential_excess_pnl"] += result["sequential_excess_pnl"]
            totals["fill_count"] += result["fill_count"]
            totals["forced_no_quote_count"] += result["forced_no_quote_count"]
            totals["max_abs_inventory"] = max(totals["max_abs_inventory"], result["max_abs_inventory"])
            totals["max_drawdown"] = max(totals["max_drawdown"], result["max_drawdown"])
            totals["wins"] += result["win_rate"] * result["fill_count"]
            totals["losses"] += result["loss_rate"] * result["fill_count"]
            split_equity += result["sequential_excess_pnl"]
            curve_rows.append({"split": split, "point": point, "equity": split_equity, "session_id": session_id})
            point += 1
            print(f"sequential eval split={split} session={session_id}")
        fill_count = int(totals["fill_count"])
        rows.append(
            {
                "split": split,
                "policy_id": selected_policy["policy_id"],
                "model_id": selected_policy["model_id"],
                "horizon": HORIZON,
                "fill_model": "conservative_l2_stress",
                "inventory_cap": 1,
                "fee_multiplier": 1.0,
                "sequential_excess_pnl": float(totals["sequential_excess_pnl"]),
                "fill_count": fill_count,
                "forced_no_quote_count": int(totals["forced_no_quote_count"]),
                "max_abs_inventory": int(totals["max_abs_inventory"]),
                "max_drawdown": float(totals["max_drawdown"]),
                "win_rate": float(totals["wins"] / max(fill_count, 1)) if fill_count else 0.0,
                "loss_rate": float(totals["losses"] / max(fill_count, 1)) if fill_count else 0.0,
                "avg_trade_excess_pnl": float(totals["sequential_excess_pnl"] / max(fill_count, 1)) if fill_count else 0.0,
                "fill_uses_future_pnl": False,
                "test_new3_new4_used_for_selection": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "causal_fill_sequential_eval.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RESULTS / "causal_fill_sequential_equity_curve.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
