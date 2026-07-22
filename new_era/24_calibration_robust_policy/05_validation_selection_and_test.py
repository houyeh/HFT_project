from __future__ import annotations

import pandas as pd

from calibration_utils import (
    RESULTS_DIR,
    ROOT,
    apply_policy,
    ensure_dirs,
    evaluate_candidate,
    load_context,
    policy_from_row,
    select_validation_policy,
    selected_policy_detail,
    write_json,
)


GRID_FILES = [
    "rolling_calibration_validation_grid.csv",
    "spread_conditional_quantile_validation_grid.csv",
    "side_specific_calibration_validation_grid.csv",
]


def run() -> None:
    ensure_dirs()
    grids = [pd.read_csv(RESULTS_DIR / name) for name in GRID_FILES]
    validation_grid = pd.concat(grids, ignore_index=True)
    selected_row = select_validation_policy(validation_grid)
    validation_grid["is_selected_policy"] = validation_grid["policy_id"].astype(str) == str(selected_row["policy_id"])
    validation_grid.to_csv(RESULTS_DIR / "combined_validation_selection_grid.csv", index=False)

    policy = policy_from_row(selected_row)
    payload = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "anti_overfit_rule": "All windows, quote fractions, score choices, and grouping choices are selected on validation only. The burned diagnostic test is evaluated exactly once after freezing this JSON.",
        "selected_policy": policy,
        "validation_metrics": selected_row.to_dict(),
    }
    write_json(ROOT / "selected_calibration_robust_policy.json", payload)

    _, test_actions = load_context("test")
    selected, thresholds = apply_policy(test_actions, policy)
    test_metrics = evaluate_candidate(
        test_actions,
        selected,
        policy_id=policy["policy_id"],
        family=policy["family"],
        score_name=policy["score_name"],
        score_col=policy["score_column"],
        window=policy["window"],
        quote_fraction=policy.get("quote_fraction"),
        group_cols=policy["group_cols"],
        higher_is_better=policy["higher_is_better"],
        split="test_diagnostic_burned",
        extra={
            "frozen_test_evaluation_count": 1,
            "used_for_model_selection": False,
            "thresholds_are_past_only": True,
        },
    )
    pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / "frozen_calibration_robust_test.csv", index=False)
    selected_policy_detail(test_actions, selected, policy["policy_id"], "test_diagnostic_burned").to_csv(
        RESULTS_DIR / "frozen_calibration_robust_test_by_session_side_spread.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "action_id": test_actions["action_id"].to_numpy(),
            "session_id": test_actions["session_id"].to_numpy(),
            "row_index": test_actions["row_index"].to_numpy(),
            "side": test_actions["side"].to_numpy(),
            "spread_bucket": test_actions["spread_bucket"].to_numpy(),
            "score": test_actions[policy["score_column"]].to_numpy(dtype=float),
            "past_only_threshold": thresholds,
            "selected_quote": selected,
            "quote_value_moderate": test_actions["quote_value_moderate"].to_numpy(dtype=float),
        }
    ).to_csv(RESULTS_DIR / "frozen_calibration_robust_test_actions.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote selected policy and frozen test results under {ROOT}")

