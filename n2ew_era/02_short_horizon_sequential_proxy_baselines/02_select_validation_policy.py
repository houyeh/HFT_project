from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02 import RESULTS_DIR, VAL_SELECTION_SPLIT, ensure_dirs, read_json, write_json


MIN_ACTION_COUNT = 1_000
MAX_DRAWDOWN_LIMIT = 2_000_000.0
BACKUP_COUNT = 2


def main() -> None:
    ensure_dirs()
    grid = pd.read_csv(RESULTS_DIR / "baseline_grid_all_splits.csv")
    catalog = read_json(RESULTS_DIR / "baseline_policy_catalog.json")
    val = grid[grid["split"].astype(str) == VAL_SELECTION_SPLIT].copy()
    val["passes_positive_pnl"] = val["total_pnl"].astype(float) > 0.0
    val["passes_min_action_count"] = val["selected_action_count"].astype(int) >= MIN_ACTION_COUNT
    val["passes_drawdown_limit"] = val["max_drawdown"].astype(float) <= MAX_DRAWDOWN_LIMIT
    val["eligible_for_selection"] = val["passes_positive_pnl"] & val["passes_min_action_count"] & val["passes_drawdown_limit"]
    val["selection_score"] = val["total_pnl"].astype(float) - 0.10 * val["max_drawdown"].astype(float)

    eligible = val[val["eligible_for_selection"]].copy()
    if eligible.empty:
        eligible = val[val["passes_positive_pnl"] & val["passes_min_action_count"]].copy()
    if eligible.empty:
        eligible = val.copy()

    ranked = eligible.sort_values(
        ["selection_score", "total_pnl", "complexity_rank", "selected_action_count"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    selected_rows = ranked.head(1 + BACKUP_COUNT).copy()

    policies_by_id = {p["policy_id"]: p for p in catalog["policies"]}

    def pack(row: pd.Series, role: str) -> dict:
        policy = dict(policies_by_id[str(row["policy_id"])])
        policy.update(
            {
                "role": role,
                "selected_horizon": int(row["horizon"]),
                "selection_split": VAL_SELECTION_SPLIT,
                "test_used_for_selection": False,
                "new4_used_for_selection": False,
                "validation_metrics": {
                    "total_pnl": float(row["total_pnl"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "selected_action_count": int(row["selected_action_count"]),
                    "selection_rate": float(row["selection_rate"]),
                    "win_rate": float(row["win_rate"]),
                    "avg_pnl_per_selected_action": float(row["avg_pnl_per_selected_action"]),
                    "selection_score": float(row["selection_score"]),
                    "eligible_for_selection": bool(row["eligible_for_selection"]),
                },
            }
        )
        return policy

    payload = {
        "selection_split": VAL_SELECTION_SPLIT,
        "test_used_for_selection": False,
        "new4_used_for_selection": False,
        "selection_constraints": {
            "min_action_count": MIN_ACTION_COUNT,
            "max_drawdown_limit": MAX_DRAWDOWN_LIMIT,
            "required_total_pnl_positive": True,
            "score": "total_pnl - 0.10 * max_drawdown, tie-break toward simpler HLS-friendly policies",
        },
        "main_policy": pack(selected_rows.iloc[0], "main"),
        "backup_policies": [pack(r, f"backup_{i}") for i, (_, r) in enumerate(selected_rows.iloc[1:].iterrows(), start=1)],
        "validation_ranked_top10": ranked.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }
    write_json(RESULTS_DIR / "selected_phase02_policy.json", payload)
    selected_rows.to_csv(RESULTS_DIR / "selected_phase02_policy_val_rows.csv", index=False)
    print("Selected Phase 02 policies:")
    print(selected_rows[["policy_id", "horizon", "total_pnl", "max_drawdown", "selected_action_count", "win_rate"]].to_string(index=False))
    print(f"Wrote {RESULTS_DIR / 'selected_phase02_policy.json'}")


if __name__ == "__main__":
    main()
