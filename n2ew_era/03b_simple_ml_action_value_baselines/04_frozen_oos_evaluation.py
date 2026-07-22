from __future__ import annotations

import joblib
import pandas as pd

from common_phase03b import (
    FROZEN_SPLITS,
    PREDICTIONS,
    RESULTS,
    build_feature_frame,
    baseline_policy_summary,
    ensure_dirs,
    evaluate_decision,
    model_path,
    part_paths,
    predict_model,
    read_json,
    select_best_quote,
    selected_actions_frame,
    selected_actions_path,
)


def evaluate_split(split: str, model_id: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = joblib.load(model_path(model_id))
    rows = []
    selected_frames = []
    baseline_frames = []
    for path in part_paths(split):
        matrix = build_feature_frame(path, include_targets=True)
        score = predict_model(model, matrix, model_id)
        decision = select_best_quote(matrix, score, threshold)
        rows.append(evaluate_decision(matrix, decision, split, "phase03b_selected_simple_ml_action_value", model_id, threshold))
        selected_frames.append(selected_actions_frame(matrix, decision, model_id, threshold))
        baseline_frames.append(baseline_policy_summary(matrix, split))
        print(f"evaluated split={split} part={path.name}")
    ml_part = pd.DataFrame(rows)
    ml_summary = ml_part.groupby(["split", "policy_id", "model_id", "horizon", "threshold"], as_index=False).agg(
        state_count=("state_count", "sum"),
        selected_action_count=("selected_action_count", "sum"),
        bid_count=("bid_count", "sum"),
        ask_count=("ask_count", "sum"),
        total_excess_pnl=("total_excess_pnl", "sum"),
        max_drawdown=("max_drawdown", "max"),
    )
    ml_summary["selection_rate"] = ml_summary["selected_action_count"] / ml_summary["state_count"].clip(lower=1)
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    baseline_all = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    baseline_summary = baseline_all.groupby(["split", "policy_id", "horizon"], as_index=False).agg(
        state_count=("state_count", "sum"),
        selected_action_count=("selected_action_count", "sum"),
        bid_count=("bid_count", "sum"),
        ask_count=("ask_count", "sum"),
        total_excess_pnl=("total_excess_pnl", "sum"),
        max_drawdown=("max_drawdown", "max"),
    )
    baseline_summary["selection_rate"] = baseline_summary["selected_action_count"] / baseline_summary["state_count"].clip(lower=1)
    selected_all.to_parquet(selected_actions_path(split), index=False, compression="zstd")
    return ml_summary, baseline_summary


def main() -> None:
    ensure_dirs()
    selected = read_json(RESULTS / "selected_phase03b_policy.json")
    model_id = selected["model_id"]
    threshold = float(selected["threshold"])
    rows = []
    baselines = []
    for split in FROZEN_SPLITS:
        ml_summary, baseline_summary = evaluate_split(split, model_id, threshold)
        rows.append(ml_summary)
        baselines.append(baseline_summary)
    ml = pd.concat(rows, ignore_index=True)
    base = pd.concat(baselines, ignore_index=True)
    ml.to_csv(RESULTS / "frozen_oos_evaluation.csv", index=False)
    base.to_csv(RESULTS / "frozen_oos_baseline_controls.csv", index=False)
    print("Frozen OOS evaluation complete.")
    print(ml.to_string(index=False))


if __name__ == "__main__":
    main()

