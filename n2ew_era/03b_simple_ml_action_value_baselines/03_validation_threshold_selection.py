from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from common_phase03b import (
    PREDICTIONS,
    RESULTS,
    TARGET,
    ensure_dirs,
    evaluate_decision,
    feature_matrix_columns,
    load_val_matrix,
    model_path,
    model_registry,
    prediction_path,
    predict_model,
    select_best_quote,
    write_json,
)


def threshold_grid(scores: np.ndarray, score_kind: str) -> np.ndarray:
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([np.inf])
    quantiles = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 1.0])
    grid = np.unique(np.nanquantile(scores, quantiles))
    extras = [0.0]
    if score_kind == "probability":
        extras.extend([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    grid = np.unique(np.concatenate([grid, np.asarray(extras, dtype=float)]))
    return np.sort(grid)


def main() -> None:
    ensure_dirs()
    val = load_val_matrix()
    rows = []
    best = None
    for model_id, info in model_registry().items():
        pred_file = prediction_path("val", model_id)
        if pred_file.exists():
            pred = pd.read_parquet(pred_file)
            score = pred["score"].to_numpy(dtype=np.float32)
        else:
            model = joblib.load(model_path(model_id))
            score = predict_model(model, val, model_id)
        quote_mask = val["action_slot"].astype(int).isin([0, 1]).to_numpy()
        for threshold in threshold_grid(score[quote_mask], info["score_kind"]):
            decision = select_best_quote(val, score, float(threshold))
            metrics = evaluate_decision(val, decision, "val", "ml_action_value_threshold", model_id, float(threshold))
            rows.append(metrics)
            if best is None or metrics["total_excess_pnl"] > best["total_excess_pnl"]:
                best = dict(metrics)
    if best is None:
        raise RuntimeError("No validation threshold candidates were evaluated.")
    out = pd.DataFrame(rows).sort_values(["total_excess_pnl"], ascending=False)
    out.to_csv(RESULTS / "validation_threshold_selection.csv", index=False)
    selected = {
        "policy_id": "phase03b_selected_simple_ml_action_value",
        "model_id": best["model_id"],
        "threshold": float(best["threshold"]),
        "selection_split": "val",
        "selection_metric": "validation_total_excess_pnl_H20",
        "validation_metrics": best,
        "feature_columns": feature_matrix_columns(),
        "test_new3_new4_used_for_selection": False,
    }
    write_json(RESULTS / "selected_phase03b_policy.json", selected)
    print("Selected Phase 03B policy:")
    print(pd.DataFrame([best]).to_string(index=False))
    print(f"wrote {RESULTS / 'selected_phase03b_policy.json'}")


if __name__ == "__main__":
    main()

