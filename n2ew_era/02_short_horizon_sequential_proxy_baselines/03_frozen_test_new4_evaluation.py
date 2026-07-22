from __future__ import annotations

import pandas as pd

from common_phase02 import FROZEN_EVAL_SPLITS, RESULTS_DIR, ensure_dirs, selected_policy_entries


def main() -> None:
    ensure_dirs()
    grid = pd.read_csv(RESULTS_DIR / "baseline_grid_all_splits.csv")
    selected = selected_policy_entries()
    rows: list[dict] = []
    for policy in selected:
        policy_id = policy["policy_id"]
        horizon = int(policy["selected_horizon"])
        for split in FROZEN_EVAL_SPLITS:
            row = grid[
                (grid["split"].astype(str) == split)
                & (grid["policy_id"].astype(str) == policy_id)
                & (grid["horizon"].astype(int) == horizon)
            ]
            if row.empty:
                continue
            r = row.iloc[0].to_dict()
            old = grid[
                (grid["split"].astype(str) == split)
                & (grid["policy_id"].astype(str) == "old_bid_only_spread_ge_1")
                & (grid["horizon"].astype(int) == horizon)
            ].iloc[0]
            no_quote = grid[
                (grid["split"].astype(str) == split)
                & (grid["policy_id"].astype(str) == "no_quote")
                & (grid["horizon"].astype(int) == horizon)
            ].iloc[0]
            rows.append(
                {
                    "role": policy.get("role", ""),
                    "policy_id": policy_id,
                    "horizon": horizon,
                    "split": split,
                    "selection_split": "val",
                    "test_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "total_pnl": float(r["total_pnl"]),
                    "selected_action_count": int(r["selected_action_count"]),
                    "selection_rate": float(r["selection_rate"]),
                    "win_rate": float(r["win_rate"]),
                    "avg_pnl_per_selected_action": float(r["avg_pnl_per_selected_action"]),
                    "max_drawdown": float(r["max_drawdown"]),
                    "no_quote_pnl": float(no_quote["total_pnl"]),
                    "old_bid_only_pnl": float(old["total_pnl"]),
                    "beats_no_quote": bool(float(r["total_pnl"]) > float(no_quote["total_pnl"])),
                    "beats_old_bid_only": bool(float(r["total_pnl"]) > float(old["total_pnl"])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "frozen_baseline_test_new4.csv", index=False)
    print("Frozen selected policy evaluation:")
    print(out.to_string(index=False))
    print(f"Wrote {RESULTS_DIR / 'frozen_baseline_test_new4.csv'}")


if __name__ == "__main__":
    main()
