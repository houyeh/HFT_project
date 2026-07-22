from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_028 import (
    FEATURE_COLUMNS,
    MODELS,
    RESULTS,
    economic_lift_rows,
    ensure_dirs,
    load_baseline_dataset,
    save_pickle,
)


def _safe_auc(metric_fn, y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(metric_fn(y, score))


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    return float(pd.Series(score).corr(pd.Series(y), method="spearman"))


def _best_val_policy_pnl(df: pd.DataFrame, score_col: str) -> tuple[float, float, int]:
    scores = df[score_col].astype(float).to_numpy()
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    finite = scores[np.isfinite(scores)]
    no_veto_threshold = float(np.max(finite) + 1e-6) if len(finite) else 1.000001
    thresholds = [no_veto_threshold]
    if len(finite):
        thresholds.extend(np.unique(np.quantile(finite, np.linspace(0.50, 0.99, 50))).tolist())
    best = (-np.inf, np.inf, 0)
    base = float(values.sum())
    for thr in thresholds:
        veto = scores >= float(thr)
        pnl = base - float(values[veto].sum())
        if pnl > best[0] or (pnl == best[0] and int(veto.sum()) < best[2]):
            best = (pnl, float(thr), int(veto.sum()))
    return best


def main() -> None:
    ensure_dirs()
    train = load_baseline_dataset("train")
    val = load_baseline_dataset("val")
    test = load_baseline_dataset("test")
    new4 = load_baseline_dataset("new4")

    x_train = train[FEATURE_COLUMNS].astype(float).to_numpy()
    y_train = train["negative_ev_moderate"].astype(int).to_numpy()

    models = {
        "logistic_l2_balanced": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=20260628)),
            ]
        ),
        "small_mlp_8": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        activation="relu",
                        alpha=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=20260628,
                    ),
                ),
            ]
        ),
    }

    rows: list[dict] = []
    lift_rows: list[dict] = []
    prediction_frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": [], "new4": []}
    split_frames = {"train": train, "val": val, "test": test, "new4": new4}

    for model_name, model in models.items():
        if len(np.unique(y_train)) < 2:
            raise RuntimeError("Training baseline universe has only one EV class; residual EV model cannot be trained.")
        model.fit(x_train, y_train)
        save_pickle(MODELS / f"{model_name}.pkl", model)

        for split, df in split_frames.items():
            x = df[FEATURE_COLUMNS].astype(float).to_numpy()
            y = df["negative_ev_moderate"].astype(int).to_numpy()
            score = model.predict_proba(x)[:, 1]
            pred = df[["split", "dataset", "symbol", "session_id", "side", "spread_bucket", "quote_ev_moderate", "negative_ev_moderate"]].copy()
            pred["model_name"] = model_name
            pred["negative_ev_score"] = score
            prediction_frames[split].append(pred)
            pr_auc = _safe_auc(average_precision_score, y, score)
            roc_auc = _safe_auc(roc_auc_score, y, score)
            spearman = _spearman(y, score)
            val_pnl, val_thr, val_veto_count = _best_val_policy_pnl(val.assign(negative_ev_score=score[: len(val)] if split == "val" else np.nan), "negative_ev_score") if split == "val" else (np.nan, np.nan, np.nan)
            if split == "val":
                best_pnl, best_thr, best_veto_count = _best_val_policy_pnl(pred, "negative_ev_score")
            else:
                best_pnl, best_thr, best_veto_count = (np.nan, np.nan, np.nan)
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "sample_count": int(len(df)),
                    "negative_ev_rate": float(y.mean()) if len(y) else np.nan,
                    "pr_auc_negative_ev": pr_auc,
                    "roc_auc_negative_ev": roc_auc,
                    "spearman_score_vs_negative_ev": spearman,
                    "validation_selected_policy_pnl": best_pnl if split == "val" else np.nan,
                    "validation_selected_threshold": best_thr if split == "val" else np.nan,
                    "validation_selected_veto_count": best_veto_count if split == "val" else np.nan,
                    "feature_set": ",".join(FEATURE_COLUMNS),
                }
            )
            lift_rows.extend(economic_lift_rows(pred, "negative_ev_score", model_name, split))

    for split, parts in prediction_frames.items():
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"residual_ev_predictions_{split}.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "residual_ev_model_metrics.csv", index=False)
    pd.DataFrame(lift_rows).to_csv(RESULTS / "residual_ev_economic_lift.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote residual EV model metrics and predictions to {RESULTS}")
