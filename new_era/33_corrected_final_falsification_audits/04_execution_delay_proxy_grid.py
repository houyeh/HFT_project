from __future__ import annotations

import pandas as pd

from common_033 import DELAY_TICKS, RESULTS, baseline_mask, delayed_entry_values, ensure_dirs, load_all_eval_actions


def main() -> None:
    ensure_dirs()
    actions = load_all_eval_actions()
    rows = []
    for eval_group, group in actions.groupby("eval_group", sort=True):
        group = group.sort_values(["symbol", "session_id", "local_row_id", "side"]).reset_index(drop=True)
        decision_selected = baseline_mask(group)
        base_values = group["quote_ev_moderate"].astype(float).to_numpy()
        base_pnl = float(base_values[decision_selected].sum())
        for delay in DELAY_TICKS:
            delayed_values, missing = delayed_entry_values(group, decision_selected, delay)
            proxy_pnl = float(delayed_values[decision_selected].sum())
            rows.append(
                {
                    "eval_group": eval_group,
                    "delay_ticks": delay,
                    "decision_time": "row_t",
                    "entry_time": f"row_t_plus_{delay}",
                    "decision_rule": "side_t == bid and spread_t >= 1",
                    "quote_value_source": "recomputed_from_delayed_entry_row_proxy_ev",
                    "decision_selected_count": int(decision_selected.sum()),
                    "missing_delayed_entry_count": int(missing),
                    "effective_entry_count": int(decision_selected.sum() - missing),
                    "proxy_entry_pnl": proxy_pnl,
                    "gap_vs_delay0": proxy_pnl - base_pnl,
                    "delay0_reference_pnl": base_pnl,
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "execution_delay_proxy_grid.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote execution-delay proxy grid to {RESULTS}")
