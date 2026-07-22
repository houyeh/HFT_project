from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02f import (
    EVAL_SPLITS,
    HORIZONS,
    OLD_BID_POLICY_ID,
    RESULTS,
    SELECTED_POLICY_ID,
    decision_pnl,
    ensure_dirs,
    iter_excess_event_frames,
    matched_random_hurdle,
    policy_decision,
    policy_set,
)


def evaluate_policy(split: str, policy: dict, horizon: int) -> dict:
    state_count = 0
    selected_count = 0
    bid_count = 0
    ask_count = 0
    total = 0.0
    wins = 0
    losses = 0
    equity_offset = 0.0
    max_drawdown = 0.0
    peak = 0.0
    for frame in iter_excess_event_frames(split, horizons=[horizon]):
        decision = policy_decision(policy, frame)
        pnl = decision_pnl(frame, decision, horizon)
        selected = decision != 0
        selected_values = pnl[selected]
        state_count += int(len(frame))
        selected_count += int(selected.sum())
        bid_count += int((decision == 1).sum())
        ask_count += int((decision == 2).sum())
        total += float(pnl.sum())
        wins += int((selected_values > 0.0).sum())
        losses += int((selected_values < 0.0).sum())
        if len(pnl):
            eq = equity_offset + np.cumsum(pnl)
            running_peak = np.maximum.accumulate(np.maximum(eq, peak))
            dd = running_peak - eq
            max_drawdown = max(max_drawdown, float(dd.max()))
            peak = max(peak, float(running_peak[-1]))
            equity_offset = float(eq[-1])
    denom = max(selected_count, 1)
    return {
        "split": split,
        "policy_id": policy["policy_id"],
        "base_policy_id": policy["policy_id"],
        "horizon": int(horizon),
        "state_count": int(state_count),
        "selected_action_count": int(selected_count),
        "selection_rate": float(selected_count / state_count) if state_count else 0.0,
        "bid_count": int(bid_count),
        "ask_count": int(ask_count),
        "total_excess_pnl": float(total),
        "win_rate": float(wins / denom) if selected_count else 0.0,
        "loss_rate": float(losses / denom) if selected_count else 0.0,
        "max_drawdown": float(max_drawdown),
    }


def main() -> None:
    ensure_dirs()
    policies = policy_set()
    rows = []
    random_rows = []
    for split in EVAL_SPLITS:
        for horizon in HORIZONS:
            for policy_id in ["no_quote", OLD_BID_POLICY_ID, SELECTED_POLICY_ID]:
                rows.append(evaluate_policy(split, policies[policy_id], horizon))
            for policy_id in [OLD_BID_POLICY_ID, SELECTED_POLICY_ID]:
                hurdle = matched_random_hurdle(split, policies[policy_id], horizon)
                random_rows.append(hurdle)
                rows.append(
                    {
                        "split": split,
                        "policy_id": "matched_random_95p",
                        "base_policy_id": policy_id,
                        "horizon": int(horizon),
                        "state_count": np.nan,
                        "selected_action_count": int(hurdle["selected_action_count"]),
                        "selection_rate": np.nan,
                        "bid_count": np.nan,
                        "ask_count": np.nan,
                        "total_excess_pnl": float(hurdle["matched_random_95p"]),
                        "win_rate": np.nan,
                        "loss_rate": np.nan,
                        "max_drawdown": np.nan,
                    }
                )
            print(f"excess policy evaluation complete split={split} H{horizon}")
    out = pd.DataFrame(rows)
    random = pd.DataFrame(random_rows)
    out = out.merge(
        random[["split", "base_policy_id", "horizon", "matched_random_mean", "matched_random_95p", "matched_random_std_normal_approx", "random_method"]],
        on=["split", "base_policy_id", "horizon"],
        how="left",
    )
    out["beats_matched_random_95p"] = out["total_excess_pnl"].astype(float) > out["matched_random_95p"].astype(float)
    out["selection_split"] = "val"
    out["test_new3_new4_used_for_selection"] = False
    out.to_csv(RESULTS / "excess_policy_evaluation.csv", index=False)
    random.to_csv(RESULTS / "excess_matched_random_hurdles.csv", index=False)
    h20 = out[out["horizon"].astype(int) == 20].copy()
    print("Excess policy evaluation complete.")
    print(h20[["split", "policy_id", "base_policy_id", "total_excess_pnl", "matched_random_95p", "beats_matched_random_95p"]].to_string(index=False))


if __name__ == "__main__":
    main()

