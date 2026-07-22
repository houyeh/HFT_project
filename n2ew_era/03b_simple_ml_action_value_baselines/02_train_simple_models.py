from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_phase03b import (
    MODELS,
    POSITIVE_TARGET,
    PREDICTIONS,
    RESULTS,
    TARGET,
    ensure_dirs,
    feature_matrix_columns,
    load_train_matrix,
    load_val_matrix,
    model_path,
    predict_model,
    prediction_path,
    safe_spearman,
)


def build_models() -> dict[str, object]:
    return {
        "ridge_l2": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "lasso_l1": make_pipeline(StandardScaler(), Lasso(alpha=0.0005, max_iter=2500, random_state=20260601)),
        "logistic_l2": make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=0.5, solver="saga", max_iter=120, n_jobs=-1, random_state=20260601)),
        "logistic_l1": make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=0.25, solver="saga", max_iter=120, n_jobs=-1, random_state=20260601)),
        "hgb_regressor_shallow": HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.1, random_state=20260601),
        "hgb_classifier_shallow": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.1, random_state=20260601),
    }


def main() -> None:
    ensure_dirs()
    warnings.filterwarnings("ignore", category=UserWarning)
    train = load_train_matrix()
    val = load_val_matrix()
    features = feature_matrix_columns()
    x_train = train[features].to_numpy(dtype=np.float32)
    y_reg = train[TARGET].to_numpy(dtype=np.float32)
    y_cls = train[POSITIVE_TARGET].astype(int).to_numpy()
    y_val_reg = val[TARGET].to_numpy(dtype=np.float32)
    y_val_cls = val[POSITIVE_TARGET].astype(int).to_numpy()
    x_val = val[features].to_numpy(dtype=np.float32)

    rows = []
    for model_id, model in build_models().items():
        is_classifier = "classifier" in model_id or model_id.startswith("logistic")
        y_train = y_cls if is_classifier else y_reg
        model.fit(x_train, y_train)
        joblib.dump(model, model_path(model_id))
        if is_classifier and hasattr(model, "predict_proba"):
            score = model.predict_proba(x_val)[:, 1].astype(np.float32)
        elif is_classifier:
            score = model.decision_function(x_val).astype(np.float32)
        else:
            score = model.predict(x_val).astype(np.float32)
        pred_frame = val[["split", "session_id", "symbol", "event_index", "timestamp", "action_side", "action_slot", TARGET, POSITIVE_TARGET]].copy()
        pred_frame["score"] = score
        pred_frame["model_id"] = model_id
        pred_frame.to_parquet(prediction_path("val", model_id), index=False, compression="zstd")

        rmse = float(np.sqrt(mean_squared_error(y_val_reg, score))) if not is_classifier else float("nan")
        mae = float(mean_absolute_error(y_val_reg, score)) if not is_classifier else float("nan")
        pr_auc = float(average_precision_score(y_val_cls, score)) if len(np.unique(y_val_cls)) > 1 else float("nan")
        roc_auc = float(roc_auc_score(y_val_cls, score)) if len(np.unique(y_val_cls)) > 1 else float("nan")
        spearman = safe_spearman(score, y_val_reg)
        rows.append(
            {
                "model_id": model_id,
                "model_family": "classifier" if is_classifier else "regressor",
                "train_rows": len(train),
                "val_rows": len(val),
                "feature_count": len(features),
                "val_rmse_vs_excess": rmse,
                "val_mae_vs_excess": mae,
                "val_pr_auc_positive_excess": pr_auc,
                "val_roc_auc_positive_excess": roc_auc,
                "val_spearman_score_vs_excess": spearman,
                "test_new3_new4_used_for_training": False,
            }
        )
        print(f"trained {model_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "simple_model_training_metrics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

