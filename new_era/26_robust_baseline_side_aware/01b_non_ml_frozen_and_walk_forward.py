from __future__ import annotations

import pandas as pd

from baseline_utils import (
    RESULTS_DIR,
    actions_from_frame,
    candidate_policies,
    ensure_dirs,
    evaluate_non_ml,
    load_label_frame,
    load_selected_non_ml_policy,
    policy_from_row,
    select_non_ml_policy,
    val_test_actions,
    write_markdown,
)


def run() -> None:
    ensure_dirs()
    policy = load_selected_non_ml_policy()
    test_actions = val_test_actions("test")
    pd.DataFrame([evaluate_non_ml(test_actions, policy, split="test_diagnostic_burned", extra={"policy_role": "fixed_026_non_ml"})]).to_csv(
        RESULTS_DIR / "non_ml_frozen_test.csv",
        index=False,
    )

    labels = load_label_frame()
    sessions = [s for s in labels["session_id"].drop_duplicates().tolist() if not str(s).startswith("20260127")]
    folds = [
        {"fold": 0, "val": sessions[3], "test": sessions[4]},
        {"fold": 1, "val": sessions[4], "test": sessions[5]},
        {"fold": 2, "val": sessions[5], "test": sessions[6]},
    ]
    rows = []
    for fold in folds:
        val = labels[labels["session_id"].astype(str) == str(fold["val"])].copy()
        test = labels[labels["session_id"].astype(str) == str(fold["test"])].copy()
        val_actions = actions_from_frame(val, salt=f"fold{fold['fold']}_val")
        test_actions_fold = actions_from_frame(test, salt=f"fold{fold['fold']}_test")
        fixed = evaluate_non_ml(test_actions_fold, policy, split="fresh_test_fold", extra={**fold, "policy_role": "fixed_026_selected"})
        rows.append(fixed)
        grid = pd.DataFrame([evaluate_non_ml(val_actions, p, split="fold_validation") for p in candidate_policies()])
        fold_policy = policy_from_row(select_non_ml_policy(grid))
        fold_eval = evaluate_non_ml(test_actions_fold, fold_policy, split="fresh_test_fold", extra={**fold, "policy_role": "fold_val_selected_non_ml"})
        rows.append(fold_eval)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "non_ml_walk_forward.csv", index=False)
    write_markdown(
        RESULTS_DIR / "hls_non_ml_controller_spec.md",
        [
            "# HLS Non-ML Controller Spec",
            "",
            "Inputs: side bit, spread_ticks, spread_bucket.",
            f"Selected validation policy: `{policy['policy_id']}`.",
            "Output: quote/keep when the side-aware spread rule or side x spread bucket LUT returns true; otherwise block quote.",
            "No ML score, floating-point model, online retraining, or future information is required.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote non-ML frozen and walk-forward outputs to {RESULTS_DIR}")

