from __future__ import annotations

import pandas as pd

from baseline_utils import RESULTS_DIR, candidate_policies, ensure_dirs, evaluate_non_ml, policy_from_row, select_non_ml_policy, val_test_actions, write_json


def run() -> None:
    ensure_dirs()
    val_actions = val_test_actions("val")
    rows = [evaluate_non_ml(val_actions, policy, split="validation") for policy in candidate_policies()]
    grid = pd.DataFrame(rows)
    selected_row = select_non_ml_policy(grid)
    grid["is_selected_policy"] = grid["policy_id"].astype(str) == str(selected_row["policy_id"])
    grid.to_csv(RESULTS_DIR / "non_ml_policy_search_val.csv", index=False)
    payload = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selected_policy": policy_from_row(selected_row),
        "validation_metrics": selected_row.to_dict(),
    }
    write_json(RESULTS_DIR / "non_ml_selected_policy.json", payload)


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'non_ml_policy_search_val.csv'}")

