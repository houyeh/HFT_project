from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ev_utils import MODELS_DIR, RESULTS_DIR, ensure_dirs, make_quote_ev_labels, quote_rows_for_split, select_threshold_for_val_pnl


FEATURE_SETS = {
    "spread_only": ["spread_ticks"],
    "spread_queue_imbalance": ["spread_ticks", "queue_imbalance_z"],
    "spread_event_rate": ["spread_ticks", "trade_intensity_short_z"],
    "spread_cancel_pressure": ["spread_ticks", "cancel_pressure_short_z"],
    "all_hls": ["spread_ticks", "queue_imbalance_z", "trade_intensity_short_z", "cancel_pressure_short_z", "normalized_spread_z"],
}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank(), method="pearson"))


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _eval_scores(split: str, df: pd.DataFrame, scores: np.ndarray, model_name: str, feature_set: str, val_threshold: float | None) -> list[dict]:
    y = df["quote_positive_ev_moderate"].to_numpy(dtype=int)
    pnl = df["realized_quote_pnl_moderate"].to_numpy(dtype=float)
    rows = []
    try:
        pr_auc = float(average_precision_score(y, scores))
    except ValueError:
        pr_auc = np.nan
    rows.append(
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "split": split,
            "metric_type": "overall",
            "pr_auc": pr_auc,
            "spearman_score_vs_pnl": _rank_corr(scores, pnl),
            "positive_ev_rate": float(np.mean(y)),
            "avg_stress_quote_pnl": float(np.mean(pnl)),
            "validation_selected_threshold": val_threshold,
            "validation_selected_policy_pnl": np.nan,
        }
    )
    for frac in [0.01, 0.05, 0.10]:
        thr = np.sort(scores)[::-1][max(0, min(len(scores) - 1, int(np.ceil(frac * len(scores))) - 1))]
        mask = scores >= thr
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split,
                "metric_type": f"top_{int(frac*100)}pct_lift",
                "pr_auc": pr_auc,
                "spearman_score_vs_pnl": _rank_corr(scores, pnl),
                "positive_ev_rate": float(np.mean(y[mask])),
                "avg_stress_quote_pnl": float(np.mean(pnl[mask])),
                "economic_lift_vs_base": float(np.mean(pnl[mask]) - np.mean(pnl)),
                "validation_selected_threshold": val_threshold,
                "validation_selected_policy_pnl": np.nan,
            }
        )
    if val_threshold is not None:
        quote = scores >= float(val_threshold)
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split,
                "metric_type": "validation_selected_policy",
                "pr_auc": pr_auc,
                "spearman_score_vs_pnl": _rank_corr(scores, pnl),
                "positive_ev_rate": float(np.mean(y[quote])) if quote.any() else np.nan,
                "avg_stress_quote_pnl": float(np.mean(pnl[quote])) if quote.any() else np.nan,
                "economic_lift_vs_base": float(np.mean(pnl[quote]) - np.mean(pnl)) if quote.any() else np.nan,
                "quote_rate": float(np.mean(quote)),
                "validation_selected_threshold": val_threshold,
                "validation_selected_policy_pnl": float(np.sum(pnl[quote])),
            }
        )
    return rows


def run() -> None:
    ensure_dirs()
    labels = make_quote_ev_labels()
    train = _sample(labels[labels["split"] == "train"].copy(), 400_000, 20260525)
    val = quote_rows_for_split("val")
    test = quote_rows_for_split("test")
    rows = []
    model_manifest = []

    specs = [("logistic_sgd", name, cols) for name, cols in FEATURE_SETS.items()]
    specs.append(("small_mlp", "all_hls", FEATURE_SETS["all_hls"]))
    for model_name, feature_set, cols in specs:
        X_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y_train = train["quote_positive_ev_moderate"].to_numpy(dtype=int)
        if model_name == "logistic_sgd":
            model = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, class_weight="balanced", random_state=20260525))
        else:
            model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=30, alpha=1e-4, random_state=20260525, early_stopping=True))
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(val[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float))[:, 1]
        thr, val_pnl, quote_rate = select_threshold_for_val_pnl(val_scores, val["realized_quote_pnl_moderate"].to_numpy(dtype=float), quote_high_scores=True)
        for split, df, scores in [
            ("val", val, val_scores),
            ("test_diagnostic_burned", test, model.predict_proba(test[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float))[:, 1]),
        ]:
            rows.extend(_eval_scores(split, df, scores, model_name, feature_set, thr))
        model_path = MODELS_DIR / f"{model_name}_{feature_set}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        model_manifest.append({"model_name": model_name, "feature_set": feature_set, "features": cols, "model_path": str(model_path), "val_policy_pnl": val_pnl, "val_quote_rate": quote_rate})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "quote_ev_model_metrics.csv", index=False)
    (MODELS_DIR / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'quote_ev_model_metrics.csv'}")
