from __future__ import annotations

import pandas as pd

from common_027 import (
    RESULTS,
    apply_non_ml_policy,
    candidate_spread_policies,
    ensure_dirs,
    evaluate_policy_on_stress,
    load_actions_for_split,
    load_selected_non_ml_policy,
    matched_random_quote_pnls,
    random_summary_row,
)


def main() -> None:
    ensure_dirs()
    policies = candidate_spread_policies()
    splits = ["val", "test", "new4"]
    actions_by_split = {split: load_actions_for_split(split) for split in splits}

    rows: list[dict] = []
    for split, actions in actions_by_split.items():
        for policy in policies:
            row = evaluate_policy_on_stress(actions, policy, split, "moderate")
            row["dataset_role"] = "validation_selection_evidence" if split == "val" else "frozen_or_external_evaluation"
            rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "non_ml_policy_robustness_grid.csv", index=False)

    selected_policy = load_selected_non_ml_policy()
    random_rows: list[dict] = []
    audit_rows: list[pd.DataFrame] = []
    for split, actions in actions_by_split.items():
        selected = apply_non_ml_policy(actions, selected_policy)
        for match_level, group_cols in [
            ("session_count_matched_main", ["session_id"]),
            ("side_session_count_matched_diagnostic", ["side", "session_id"]),
        ]:
            random_pnls, audit = matched_random_quote_pnls(actions, selected, trials=5000, seed=20260627, group_cols=group_cols)
            random_rows.append(
                random_summary_row(
                    split,
                    actions,
                    selected,
                    random_pnls,
                    match_keys=",".join(group_cols),
                    random_match_level=match_level,
                )
            )
            audit.insert(0, "random_match_level", match_level)
            audit.insert(0, "split", split)
            audit_rows.append(audit)
            pd.DataFrame({"trial": range(len(random_pnls)), "matched_random_quote_pnl": random_pnls}).to_csv(
                RESULTS / f"matched_random_quote_trials_{split}_{match_level}.csv",
                index=False,
            )
            if match_level == "session_count_matched_main":
                pd.DataFrame({"trial": range(len(random_pnls)), "matched_random_quote_pnl": random_pnls}).to_csv(
                    RESULTS / f"matched_random_quote_trials_{split}.csv",
                    index=False,
                )
    pd.DataFrame(random_rows).to_csv(RESULTS / "matched_random_quote_summary.csv", index=False)
    pd.concat(audit_rows, ignore_index=True).to_csv(RESULTS / "matched_random_quote_matching_audit.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote robustness grid and matched random summary to {RESULTS}")
