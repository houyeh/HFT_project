from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common_phase02e import (
    ADVERSE_MULTIPLIERS,
    DELAY_EVENTS,
    EVAL_SPLITS,
    FEE_MULTIPLIERS,
    OLD_BID_POLICY_ID,
    POSITIVE_FILL_KEEP_RATES,
    RESULTS,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    delayed_decision,
    ensure_dirs,
    iter_event_frames,
    part_paths,
    pnl_from_decision,
    policy_decision,
    policy_set,
    stress_pnl,
)


@dataclass
class StressState:
    state_count: int = 0
    selected_action_count: int = 0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    cum_pnl: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0


def update_state(state: StressState, pnl: np.ndarray, decision: np.ndarray) -> None:
    selected = decision != 0
    selected_pnl = pnl[selected]
    state.state_count += int(len(pnl))
    state.selected_action_count += int(selected.sum())
    state.total_pnl += float(pnl.sum())
    state.win_count += int((selected_pnl > 0.0).sum())
    state.loss_count += int((selected_pnl < 0.0).sum())
    # The stress grid is a large Cartesian diagnostic; keep drawdown as a
    # conservative single-step proxy to avoid thousands of full equity curves.
    if len(selected_pnl):
        state.max_drawdown = max(state.max_drawdown, float(max(0.0, -np.nanmin(selected_pnl))))


def main() -> None:
    ensure_dirs()
    policies = policy_set()
    policy_ids = [OLD_BID_POLICY_ID, SELECTED_POLICY_ID]
    rows = []
    for split in EVAL_SPLITS:
        if not part_paths(split):
            continue
        for policy_id in policy_ids:
            policy = policies[policy_id]
            states = {
                (fee, keep, adv, delay): StressState()
                for fee in FEE_MULTIPLIERS
                for keep in POSITIVE_FILL_KEEP_RATES
                for adv in ADVERSE_MULTIPLIERS
                for delay in DELAY_EVENTS
            }
            for frame in iter_event_frames(split, horizons=[SELECTED_HORIZON]):
                base_decision = policy_decision(policy, frame)
                for delay in DELAY_EVENTS:
                    decision = delayed_decision(base_decision, delay)
                    for fee in FEE_MULTIPLIERS:
                        base = pnl_from_decision(frame, decision, horizon=SELECTED_HORIZON, fee_multiplier=fee)
                        for keep in POSITIVE_FILL_KEEP_RATES:
                            for adv in ADVERSE_MULTIPLIERS:
                                update_state(states[(fee, keep, adv, delay)], stress_pnl(base, keep, adv), decision)
            for (fee, keep, adv, delay), state in states.items():
                denom = max(state.selected_action_count, 1)
                rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "horizon": SELECTED_HORIZON,
                        "fee_multiplier": float(fee),
                        "positive_fill_keep_rate": float(keep),
                        "adverse_multiplier": float(adv),
                        "delay_events": int(delay),
                        "state_count": int(state.state_count),
                        "selected_action_count": int(state.selected_action_count),
                        "selection_rate": float(state.selected_action_count / state.state_count) if state.state_count else 0.0,
                        "stress_total_pnl": float(state.total_pnl),
                        "win_rate": float(state.win_count / denom) if state.selected_action_count else 0.0,
                        "loss_rate": float(state.loss_count / denom) if state.selected_action_count else 0.0,
                        "max_drawdown": float(state.max_drawdown),
                        "positive_under_stress": bool(state.total_pnl > 0.0),
                        "moderate_stress": bool(fee == 1.0 and keep == 0.5 and adv == 1.5 and delay == 0),
                        "strict_fee5_adverse3_keep01_delay10": bool(fee == 5.0 and keep == 0.1 and adv == 3.0 and delay == 10),
                    }
                )
            print(f"strict stress complete split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "strict_execution_stress_comparison.csv", index=False)
    moderate = out[out["moderate_stress"].astype(bool)].copy()
    print("Strict execution stress comparison complete.")
    print(moderate[["split", "policy_id", "stress_total_pnl", "positive_under_stress"]].to_string(index=False))


if __name__ == "__main__":
    main()
