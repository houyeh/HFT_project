from __future__ import annotations

import numpy as np
import pandas as pd

from common_033 import DELAY_TICKS, RESULTS, add_lagged_spreads, ensure_dirs, load_all_eval_actions


def main() -> None:
    ensure_dirs()
    actions = add_lagged_spreads(load_all_eval_actions(), DELAY_TICKS)
    rows = []
    for eval_group, group in actions.groupby("eval_group", sort=True):
        group = group.reset_index(drop=True)
        values = group["quote_ev_moderate"].astype(float).to_numpy()
        current = ((group["side"].astype(str) == "bid") & (group["spread_lag_0"].astype(float) >= 1.0)).to_numpy(dtype=bool)
        current_pnl = float(values[current].sum())
        for lag in DELAY_TICKS:
            spread = group[f"spread_lag_{lag}"].astype(float)
            selected = ((group["side"].astype(str) == "bid") & (spread >= 1.0)).to_numpy(dtype=bool)
            boundary_missing = int(spread.isna().sum())
            rows.append(
                {
                    "eval_group": eval_group,
                    "lag_ticks": lag,
                    "stress_type": "lagged_book_timing_artifact_stress",
                    "selected_quote_count": int(selected.sum()),
                    "selected_quote_pnl": float(values[selected].sum()),
                    "gap_vs_current_row_pnl": float(values[selected].sum() - current_pnl),
                    "decision_mismatch_count_vs_current": int(np.sum(selected != current)),
                    "boundary_or_missing_lag_rows": boundary_missing,
                    "interpretation": "Lagged spread changes only decision features; it does not recalculate execution value from a delayed entry row.",
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "lagged_book_decision_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote lagged-book sensitivity to {RESULTS}")
