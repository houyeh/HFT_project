from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_030 import (
    FEATURE_COLUMNS,
    MODELS,
    RESULTS,
    ensure_dirs,
    evaluate_veto,
    load_all_splits,
    model_matrix,
    regression_metrics,
    safe_pr_auc,
    safe_roc_auc,
    save_pickle,
    select_threshold_on_val,
    threshold_curve,
    write_json,
)


RANDOM_STATE = 20260630
TORCH_AVAILABLE = bool(importlib.util.find_spec("torch"))
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


if TORCH_AVAILABLE:

    class _TorchMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            return self.net(x)


    class TorchMLPBinaryClassifier:
        def __init__(self, epochs: int = 350, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4):
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.lr = float(lr)
            self.weight_decay = float(weight_decay)
            self.x_scaler = StandardScaler()
            self.model: _TorchMLP | None = None

        def fit(self, x: np.ndarray, y: np.ndarray):
            torch.manual_seed(RANDOM_STATE)
            x_scaled = self.x_scaler.fit_transform(x).astype(np.float32)
            y_arr = y.astype(np.float32).reshape(-1, 1)
            self.model = _TorchMLP(x_scaled.shape[1], 1)
            pos = float(y_arr.sum())
            neg = float(len(y_arr) - pos)
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loader = DataLoader(TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_arr)), batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(self.model(xb), yb)
                    loss.backward()
                    opt.step()
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            if self.model is None:
                raise RuntimeError("Model is not fitted")
            x_scaled = self.x_scaler.transform(x).astype(np.float32)
            self.model.eval()
            with torch.no_grad():
                p = torch.sigmoid(self.model(torch.from_numpy(x_scaled))).numpy().reshape(-1)
            return np.column_stack([1.0 - p, p])


    class TorchMLPRegressor:
        def __init__(self, epochs: int = 450, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4):
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.lr = float(lr)
            self.weight_decay = float(weight_decay)
            self.x_scaler = StandardScaler()
            self.y_mean = 0.0
            self.y_std = 1.0
            self.model: _TorchMLP | None = None

        def fit(self, x: np.ndarray, y: np.ndarray):
            torch.manual_seed(RANDOM_STATE)
            x_scaled = self.x_scaler.fit_transform(x).astype(np.float32)
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y) if np.std(y) > 1e-9 else 1.0)
            y_scaled = ((y.astype(float) - self.y_mean) / self.y_std).astype(np.float32).reshape(-1, 1)
            self.model = _TorchMLP(x_scaled.shape[1], 1)
            loss_fn = nn.MSELoss()
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loader = DataLoader(TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_scaled)), batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(self.model(xb), yb)
                    loss.backward()
                    opt.step()
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            if self.model is None:
                raise RuntimeError("Model is not fitted")
            x_scaled = self.x_scaler.transform(x).astype(np.float32)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch.from_numpy(x_scaled)).numpy().reshape(-1)
            return pred * self.y_std + self.y_mean


def calibrated_logistic() -> CalibratedClassifierCV:
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
    specs = [
        {
            "model_name": "calibrated_logistic_sigmoid",
            "objective": "classification_negative_ev",
            "model": calibrated_logistic(),
            "hls_complexity_rank": 1,
        },
        {
            "model_name": "hist_gradient_boosting_classifier",
            "objective": "classification_negative_ev",
            "model": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_STATE),
            "hls_complexity_rank": 3,
        },
        {
            "model_name": "sklearn_mlp_64_32_classifier",
            "objective": "classification_negative_ev",
            "model": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-3,
                            batch_size=64,
                            max_iter=1500,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "hls_complexity_rank": 4,
        },
        {
            "model_name": "ridge_quote_ev_regressor",
            "objective": "regression_quote_ev",
            "model": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
            "hls_complexity_rank": 1,
        },
        {
            "model_name": "hist_gradient_boosting_regressor",
            "objective": "regression_quote_ev",
            "model": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_STATE),
            "hls_complexity_rank": 3,
        },
        {
            "model_name": "sklearn_mlp_64_32_regressor",
            "objective": "regression_quote_ev",
            "model": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-3,
                            batch_size=64,
                            max_iter=1500,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "hls_complexity_rank": 4,
        },
    ]
    if TORCH_AVAILABLE:
        specs.extend(
            [
                {
                    "model_name": "torch_mlp_64_32_classifier_cpu",
                    "objective": "classification_negative_ev",
                    "model": TorchMLPBinaryClassifier(),
                    "hls_complexity_rank": 4,
                },
                {
                    "model_name": "torch_mlp_64_32_regressor_cpu",
                    "objective": "regression_quote_ev",
                    "model": TorchMLPRegressor(),
                    "hls_complexity_rank": 4,
                },
            ]
        )
    return specs


def predict_scores(spec: dict, model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, _, _ = model_matrix(df)
    if spec["objective"].startswith("classification"):
        score = model.predict_proba(x)[:, 1]
        pred_ev = np.full(len(df), np.nan)
    else:
        pred_ev = model.predict(x)
        score = -np.asarray(pred_ev, dtype=float)
    return np.asarray(score, dtype=float), np.asarray(pred_ev, dtype=float)


def metric_row(spec: dict, split: str, df: pd.DataFrame, score: np.ndarray, pred_ev: np.ndarray, selected_val: dict | None) -> dict:
    y_cls = df["negative_ev_moderate"].astype(int).to_numpy()
    y_ev = df["quote_ev_moderate"].astype(float).to_numpy()
    row = {
        "model_name": spec["model_name"],
        "objective": spec["objective"],
        "split": split,
        "sample_count": int(len(df)),
        "negative_ev_rate": float(y_cls.mean()) if len(y_cls) else np.nan,
        "pr_auc_negative_ev": safe_pr_auc(y_cls, score),
        "roc_auc_negative_ev": safe_roc_auc(y_cls, score),
        "spearman_score_vs_negative_ev": float(pd.Series(score).corr(pd.Series(y_cls), method="spearman")) if len(y_cls) else np.nan,
        "hls_complexity_rank": spec["hls_complexity_rank"],
        "torch_available_for_pytorch_mlp": TORCH_AVAILABLE,
        "notes": "PyTorch MLP trained on local CPU; external pack supports RTX GPU training" if spec["model_name"].startswith("torch_mlp") else "",
    }
    if spec["objective"].startswith("regression"):
        row.update(regression_metrics(y_ev, pred_ev))
    else:
        row.update({"rmse_quote_ev": np.nan, "mae_quote_ev": np.nan, "spearman_pred_ev_vs_ev": np.nan})
    if split == "val" and selected_val is not None:
        row.update({f"val_selected_{k}": v for k, v in selected_val.items() if k in {"threshold", "veto_count", "veto_rate", "after_veto_pnl", "incremental_vs_baseline", "negative_ev_rate_vetoed"}})
    return row


def main() -> None:
    ensure_dirs()
    frames = load_all_splits()
    train = frames["train"]
    x_train, y_cls_train, y_ev_train = model_matrix(train)

    metrics: list[dict] = []
    pred_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in frames}
    curve_parts: list[pd.DataFrame] = []
    selection_rows: list[dict] = []

    for spec in model_specs():
        model = spec["model"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spec["objective"].startswith("classification"):
                if len(np.unique(y_cls_train)) < 2:
                    continue
                model.fit(x_train, y_cls_train)
            else:
                model.fit(x_train, y_ev_train)
        save_pickle(MODELS / f"{spec['model_name']}.pkl", model)

        split_scores: dict[str, np.ndarray] = {}
        split_pred_ev: dict[str, np.ndarray] = {}
        for split, df in frames.items():
            score, pred_ev = predict_scores(spec, model, df)
            split_scores[split] = score
            split_pred_ev[split] = pred_ev
            pred = df[
                [
                    "phase030_split" if "phase030_split" in df.columns else "split",
                    "dataset",
                    "symbol",
                    "session_id",
                    "side",
                    "spread_bucket",
                    "quote_ev_moderate",
                    "negative_ev_moderate",
                ]
            ].copy()
            pred = pred.rename(columns={"phase030_split": "split"})
            pred["model_name"] = spec["model_name"]
            pred["objective"] = spec["objective"]
            pred["negative_ev_score"] = score
            pred["predicted_quote_ev"] = pred_ev
            pred_parts[split].append(pred)

        val_with_score = frames["val"].copy()
        val_with_score["negative_ev_score"] = split_scores["val"]
        selected_val = select_threshold_on_val(val_with_score, "negative_ev_score")
        selected_val.update({"model_name": spec["model_name"], "objective": spec["objective"], "hls_complexity_rank": spec["hls_complexity_rank"]})
        selection_rows.append(selected_val)
        curve = threshold_curve(val_with_score, "negative_ev_score")
        curve.insert(0, "model_name", spec["model_name"])
        curve.insert(1, "objective", spec["objective"])
        curve_parts.append(curve)

        for split, df in frames.items():
            metrics.append(metric_row(spec, split, df, split_scores[split], split_pred_ev[split], selected_val if split == "val" else None))

    for split, parts in pred_parts.items():
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"model_family_predictions_{split}.csv", index=False)
    pd.concat(curve_parts, ignore_index=True).to_csv(RESULTS / "model_family_validation_threshold_curves.csv", index=False)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(RESULTS / "model_family_validation_selection.csv", index=False)
    pd.DataFrame(metrics).to_csv(RESULTS / "model_family_metrics.csv", index=False)

    eligible = selection[selection["constraints_pass"].astype(bool)].copy()
    if eligible.empty:
        eligible = selection.copy()
    best = eligible.sort_values(["after_veto_pnl", "veto_count", "hls_complexity_rank"], ascending=[False, True, True]).iloc[0].to_dict()
    write_json(
        RESULTS / "selected_model_family.json",
        {
            "selection_split": "validation",
            "test_used_for_selection": False,
            "max_veto_rate": 0.05,
            "selected_model": best,
            "feature_columns": FEATURE_COLUMNS,
            "local_torch_available": bool(importlib.util.find_spec("torch")),
        },
    )


if __name__ == "__main__":
    main()
    print(f"Wrote expanded model-family metrics to {RESULTS}")
