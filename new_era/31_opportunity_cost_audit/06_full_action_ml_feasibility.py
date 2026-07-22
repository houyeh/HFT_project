from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_031 import (
    FULL_ACTION_FEATURE_COLUMNS,
    MODELS,
    RANDOM_SEED,
    RANDOM_TRIALS,
    RESULTS,
    ensure_dirs,
    eval_selection,
    load_full_actions,
    matched_random_selected_pnls,
    model_matrix,
    regression_metrics,
    safe_pr_auc,
    safe_roc_auc,
    save_pickle,
    split_df,
    threshold_curve_for_selector,
    write_json,
)


RANDOM_STATE = 20260701


def calibrated_logistic():
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE)),
        ]
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def model_specs() -> list[dict]:
    return [
        {"model_name": "full_calibrated_logistic", "objective": "positive_ev_classification", "model": calibrated_logistic(), "complexity": 1},
        {
            "model_name": "full_hist_gradient_boosting_classifier",
            "objective": "positive_ev_classification",
            "model": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_STATE),
            "complexity": 3,
        },
        {
            "model_name": "full_small_mlp_classifier",
            "objective": "positive_ev_classification",
            "model": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1500, early_stopping=True, random_state=RANDOM_STATE)),
                ]
            ),
            "complexity": 4,
        },
        {"model_name": "full_ridge_ev_regressor", "objective": "quote_ev_regression", "model": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]), "complexity": 1},
        {
            "model_name": "full_hist_gradient_boosting_regressor",
            "objective": "quote_ev_regression",
            "model": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_STATE),
            "complexity": 3,
        },
    ]


def predict_score(spec: dict, model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, _, _ = model_matrix(df)
    if spec["objective"].endswith("classification"):
        score = model.predict_proba(x)[:, 1]
        pred_ev = np.full(len(df), np.nan)
    else:
        pred_ev = model.predict(x)
        score = np.asarray(pred_ev, dtype=float)
    return np.asarray(score, dtype=float), np.asarray(pred_ev, dtype=float)


def select_model_on_val(val_predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    for (model_name, objective, complexity), group in val_predictions.groupby(["model_name", "objective", "complexity"], observed=False):
        curve = threshold_curve_for_selector(group, "score")
        curve["model_name"] = model_name
        curve["objective"] = objective
        curve["complexity"] = complexity
        curves.append(curve)
        eligible = curve[(curve["selected_count"].astype(int) >= 5) & (curve["selection_rate"].astype(float) <= 0.85)].copy()
        if eligible.empty:
            eligible = curve.copy()
        best = eligible.sort_values(["selected_quote_pnl", "selected_count", "complexity"], ascending=[False, False, True]).iloc[0].to_dict()
        rows.append(best)
    selection = pd.DataFrame(rows)
    best = selection.sort_values(["selected_quote_pnl", "selected_count", "complexity"], ascending=[False, False, True]).iloc[0].to_dict()
    return best, pd.concat(curves, ignore_index=True)


def evaluate_policy(df: pd.DataFrame, pred: pd.DataFrame, selected_spec: dict, split: str) -> dict:
    score = pred["score"].astype(float).to_numpy()
    selected = score >= float(selected_spec["threshold"])
    baseline = df["selected_by_baseline"].astype(bool).to_numpy()
    eval_ml = eval_selection(df, selected, "full_action_ml_selector")
    eval_base = eval_selection(df, baseline, "bid_only_spread_ge_1")
    random_pnls = matched_random_selected_pnls(
        df,
        selected,
        group_cols=["side", "session_id", "spread_bucket"],
        trials=RANDOM_TRIALS,
        seed=RANDOM_SEED,
    )
    row = {
        "phase031_split": split,
        "model_name": selected_spec["model_name"],
        "objective": selected_spec["objective"],
        "threshold": float(selected_spec["threshold"]),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "baseline_pnl": eval_base["selected_quote_pnl"],
        "ml_selected_pnl": eval_ml["selected_quote_pnl"],
        "gap_vs_baseline": eval_ml["selected_quote_pnl"] - eval_base["selected_quote_pnl"],
        "ml_selected_count": eval_ml["selected_count"],
        "ml_selection_rate": eval_ml["selection_rate"],
        "ml_oracle_recall": eval_ml["oracle_recall"],
        "ml_oracle_precision": eval_ml["oracle_precision"],
        "matched_random_mean": float(np.mean(random_pnls)),
        "matched_random_95p": float(np.quantile(random_pnls, 0.95)),
        "matched_random_99p": float(np.quantile(random_pnls, 0.99)),
        "beats_baseline": bool(eval_ml["selected_quote_pnl"] > eval_base["selected_quote_pnl"]),
        "beats_matched_random_95p": bool(eval_ml["selected_quote_pnl"] > float(np.quantile(random_pnls, 0.95))),
    }
    return row


def main() -> None:
    ensure_dirs()
    actions = load_full_actions()
    frames = {split: split_df(actions, split) for split in ["train", "val", "test", "new4"]}
    train = frames["train"]
    x_train, y_cls, y_ev = model_matrix(train)

    metric_rows: list[dict] = []
    pred_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in frames}
    for spec in model_specs():
        model = spec["model"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spec["objective"].endswith("classification"):
                model.fit(x_train, y_cls)
            else:
                model.fit(x_train, y_ev)
        save_pickle(MODELS / f"{spec['model_name']}.pkl", model)
        for split, df in frames.items():
            score, pred_ev = predict_score(spec, model, df)
            x, y_split, y_ev_split = model_matrix(df)
            pred = df[["sample_id", "phase031_split", "dataset", "symbol", "session_id", "side", "spread_bucket", "quote_ev_moderate", "positive_ev_moderate"]].copy()
            pred["model_name"] = spec["model_name"]
            pred["objective"] = spec["objective"]
            pred["complexity"] = spec["complexity"]
            pred["score"] = score
            pred["predicted_quote_ev"] = pred_ev
            pred_parts[split].append(pred)
            row = {
                "model_name": spec["model_name"],
                "objective": spec["objective"],
                "split": split,
                "sample_count": int(len(df)),
                "positive_ev_rate": float(y_split.mean()) if len(y_split) else np.nan,
                "pr_auc_positive_ev": safe_pr_auc(y_split, score),
                "roc_auc_positive_ev": safe_roc_auc(y_split, score),
                "spearman_score_vs_positive_ev": float(pd.Series(score).corr(pd.Series(y_split), method="spearman")) if len(y_split) else np.nan,
            }
            if spec["objective"].endswith("regression"):
                row.update(regression_metrics(y_ev_split, pred_ev))
            metric_rows.append(row)

    for split, parts in pred_parts.items():
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"full_action_ml_predictions_{split}.csv", index=False)
    val_predictions = pd.concat(pred_parts["val"], ignore_index=True)
    selected, curves = select_model_on_val(val_predictions)
    curves.to_csv(RESULTS / "full_action_ml_validation_threshold_curves.csv", index=False)
    write_json(RESULTS / "selected_full_action_ml_policy.json", {"selection_split": "validation", "test_used_for_selection": False, "selected_policy": selected, "feature_columns": FULL_ACTION_FEATURE_COLUMNS})
    pd.DataFrame(metric_rows).to_csv(RESULTS / "full_action_ml_metrics.csv", index=False)

    result_rows: list[dict] = []
    for split in ["val", "test", "new4"]:
        pred = pd.concat(pred_parts[split], ignore_index=True)
        pred = pred[pred["model_name"] == selected["model_name"]].reset_index(drop=True)
        result_rows.append(evaluate_policy(frames[split], pred, selected, split))
    out = pd.DataFrame(result_rows)
    out["viable_full_action_ml"] = out["beats_baseline"].astype(bool) & out["beats_matched_random_95p"].astype(bool)
    out.to_csv(RESULTS / "full_action_ml_policy_results.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote full-action ML feasibility outputs to {RESULTS}")
