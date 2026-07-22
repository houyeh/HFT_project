from __future__ import annotations

import pandas as pd

from common_final32 import RESULTS, ensure_dirs, load_eval_actions, stressed_quote_values


def main() -> None:
    ensure_dirs()
    actions = load_eval_actions()
    selected_all = actions["selected_by_baseline"].astype(bool).to_numpy()
    rows = []
    for split, group in actions.groupby("phase031_split", sort=True):
        selected = group["selected_by_baseline"].astype(bool).to_numpy()
        for fee_mult in [1.0, 2.0, 5.0]:
            for keep_rate in [1.0, 0.75, 0.5, 0.25]:
                for adverse_mult in [1.0, 1.5, 2.0, 3.0]:
                    values = stressed_quote_values(
                        group,
                        fee_multiplier=fee_mult,
                        winning_fill_keep_rate=keep_rate,
                        adverse_multiplier=adverse_mult,
                    )
                    rows.append(
                        {
                            "phase031_split": split,
                            "fee_multiplier": fee_mult,
                            "winning_fill_keep_rate": keep_rate,
                            "adverse_multiplier": adverse_mult,
                            "selected_count": int(selected.sum()),
                            "stress_pnl": float(values[selected].sum()),
                            "avg_selected_stress_pnl": float(values[selected].mean()) if selected.any() else 0.0,
                            "positive_selected_rate": float((values[selected] > 0.0).mean()) if selected.any() else 0.0,
                            "negative_selected_loss": float(-values[selected & (values < 0.0)].sum()),
                        }
                    )
    pd.DataFrame(rows).to_csv(RESULTS / "adverse_stress_grid.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote adverse stress grid to {RESULTS}")

