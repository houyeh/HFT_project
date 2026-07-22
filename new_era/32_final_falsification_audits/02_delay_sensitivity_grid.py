from __future__ import annotations

import pandas as pd

from common_final32 import RESULTS, attach_lagged_spreads, baseline_decision, ensure_dirs, eval_selected, load_eval_actions


def main() -> None:
    ensure_dirs()
    delays = [0, 1, 2, 3, 5, 10]
    actions = attach_lagged_spreads(load_eval_actions(), delays)
    rows = []
    for split, group in actions.groupby("phase031_split", sort=True):
        base_pnl = None
        for delay in delays:
            selected = baseline_decision(group, f"spread_lag_{delay}")
            row = eval_selected(group, selected, f"bid_only_spread_ge_1_delay_{delay}")
            if delay == 0:
                base_pnl = row["selected_quote_pnl"]
            row.update(
                {
                    "phase031_split": split,
                    "delay_ticks": delay,
                    "gap_vs_delay0": row["selected_quote_pnl"] - float(base_pnl if base_pnl is not None else row["selected_quote_pnl"]),
                    "boundary_suppressed_count": int(group[f"spread_lag_{delay}"].isna().sum()),
                }
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "delay_sensitivity_grid.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote delay sensitivity grid to {RESULTS}")

