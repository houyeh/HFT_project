from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))


import torch
import torch.nn as nn

from src.models.tcn import HLSFriendlyTCNClassifier


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs"
DEFAULT_INPUT = DEFAULT_OUT_DIR / "binary_labels_v2.parquet"

SELECTED_FEATURES = [
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
]
Z_FEATURES = [f"{name}_z" for name in SELECTED_FEATURES]
SIDES = ["bid", "ask"]


@dataclass
class SplitData:
    features_by_code: list[np.ndarray]
    session_codes: np.ndarray
    positions: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame

    def subset(self, selected: np.ndarray) -> "SplitData":
        selected = np.asarray(selected, dtype=np.int64)
        return SplitData(
            features_by_code=self.features_by_code,
            session_codes=self.session_codes[selected],
            positions=self.positions[selected],
            y=self.y[selected],
            meta=self.meta.iloc[selected].reset_index(drop=True),
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_split_data(
    labels: pd.DataFrame,
    *,
    split: str,
    side: str,
    stride: int,
    window_size: int,
) -> SplitData:
    side_label = f"{side}_toxic"
    side_fill = f"{side}_fill_proxy"
    side_trade = f"{side}_trade_hits_quote"
    side_cross = f"{side}_price_cross_hits_quote"

    split_df = labels[labels["split"] == split].copy()
    if split_df.empty:
        raise RuntimeError(f"No rows found for split={split}")

    features_by_code: list[np.ndarray] = []
    session_codes: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    meta_cols = [
        "event_id",
        "dataset",
        "symbol",
        "session_id",
        "split",
        "local_row_id",
        "bid_px_1",
        "ask_px_1",
        "mid_price",
        "tick_size",
        "future_mid_return_H_ticks",
        "label_valid_horizon",
        side_fill,
        side_trade,
        side_cross,
        side_label,
    ]

    for _, (session_id, group) in enumerate(split_df.groupby("session_id", sort=True)):
        group = group.sort_values("local_row_id").reset_index(drop=True)
        feature_values = group[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
        label_values = group[side_label].to_numpy(dtype=np.int64, copy=True)
        valid_values = group["label_valid_horizon"].to_numpy(dtype=bool, copy=True)

        candidate = np.arange(int(window_size) - 1, len(group), int(stride), dtype=np.int64)
        candidate = candidate[valid_values[candidate]]
        if candidate.size == 0:
            continue

        code = len(features_by_code)
        features_by_code.append(feature_values)
        session_codes.append(np.full(candidate.size, code, dtype=np.int32))
        positions.append(candidate.astype(np.int32))
        y_parts.append(label_values[candidate].astype(np.int64))
        meta = group.iloc[candidate][meta_cols].copy()
        meta["side"] = side
        meta_parts.append(meta)

    if not y_parts:
        raise RuntimeError(f"No windowed examples for split={split}, side={side}")

    return SplitData(
        features_by_code=features_by_code,
        session_codes=np.concatenate(session_codes, axis=0),
        positions=np.concatenate(positions, axis=0),
        y=np.concatenate(y_parts, axis=0),
        meta=pd.concat(meta_parts, ignore_index=True),
    )


def _downsample_train(
    train_data: SplitData,
    *,
    neg_pos_ratio: float,
    seed: int,
) -> tuple[SplitData, dict[str, Any]]:
    y = train_data.y.astype(np.int64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    target_neg = min(len(neg_idx), int(np.ceil(len(pos_idx) * float(neg_pos_ratio))))
    if target_neg < len(neg_idx):
        neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
    selected = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(selected)
    sampled = train_data.subset(selected)
    return sampled, {
        "original_train_examples": int(len(y)),
        "original_positive_examples": int(len(pos_idx)),
        "original_negative_examples": int(np.sum(y == 0)),
        "sampled_train_examples": int(len(sampled.y)),
        "sampled_positive_examples": int(np.sum(sampled.y == 1)),
        "sampled_negative_examples": int(np.sum(sampled.y == 0)),
        "neg_pos_ratio": float(neg_pos_ratio),
    }


def _window_views(data: SplitData, window_size: int) -> list[np.ndarray]:
    return [
        np.lib.stride_tricks.sliding_window_view(features, window_shape=int(window_size), axis=0)
        for features in data.features_by_code
    ]


def _iter_batches(
    n_rows: int,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    indices = np.arange(int(n_rows), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _make_window_batch(
    data: SplitData,
    views: list[np.ndarray],
    batch_idx: np.ndarray,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = data.session_codes[batch_idx].astype(np.int64, copy=False)
    starts = data.positions[batch_idx].astype(np.int64, copy=False) - int(window_size) + 1
    x_np = np.empty((len(batch_idx), len(Z_FEATURES), int(window_size)), dtype=np.float32)
    for code in np.unique(codes):
        out_idx = np.flatnonzero(codes == code)
        x_np[out_idx] = views[int(code)][starts[out_idx]]
    y_np = data.y[batch_idx].astype(np.float32, copy=False)
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


@torch.no_grad()
def _predict(
    model: nn.Module,
    data: SplitData,
    *,
    criterion: nn.Module,
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    probs: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    views = _window_views(data, window_size)
    rng = np.random.default_rng(0)

    for batch_idx in _iter_batches(len(data.y), batch_size=batch_size, shuffle=False, rng=rng):
        batch_x, batch_y = _make_window_batch(data, views, batch_idx, window_size)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).squeeze(1)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * int(len(batch_y))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        y_true.append(batch_y.detach().cpu().numpy().astype(np.int64))

    p = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    y = np.concatenate(y_true, axis=0) if y_true else np.zeros((0,), dtype=np.int64)
    return {
        "loss": float(total_loss / max(len(y), 1)),
        "probs": p,
        "y_true": y,
    }


def _metrics_from_probs(
    *,
    side: str,
    split: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    loss: float,
    sampling_mode: str,
) -> dict[str, Any]:
    pred = (probs >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        roc_auc = float("nan")

    return {
        "side": side,
        "split": split,
        "sampling_mode": sampling_mode,
        "examples": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "loss": float(loss),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_0p5": float(precision),
        "recall_at_0p5": float(recall),
        "f1_at_0p5": float(f1),
        "predicted_positive_rate_at_0p5": float(np.mean(pred)) if len(pred) else float("nan"),
    }


def _plot_pr_curve(y_true: np.ndarray, probs: np.ndarray, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
    ax.axhline(float(np.mean(y_true)), color="gray", linestyle="--", linewidth=1.0, label="base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_loss_curve(history_by_side: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for side, history in history_by_side.items():
        ax.plot(history["epoch"], history["train_loss"], label=f"{side} train")
        ax.plot(history["epoch"], history["val_loss"], label=f"{side} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Binary Model Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_val_predictions(
    *,
    side: str,
    val_data: SplitData,
    probs: np.ndarray,
    out_path: Path,
) -> None:
    pred = val_data.meta.copy()
    pred["prob_toxic"] = probs.astype(np.float32)
    pred["y_true"] = val_data.y.astype(np.int8)
    pred.rename(
        columns={
            f"{side}_fill_proxy": "fill_proxy",
            f"{side}_trade_hits_quote": "trade_hits_quote",
            f"{side}_price_cross_hits_quote": "price_cross_hits_quote",
            f"{side}_toxic": "toxic",
        },
        inplace=True,
    )
    pred.to_parquet(out_path, index=False)


def _train_one_side(
    *,
    labels: pd.DataFrame,
    side: str,
    out_dir: Path,
    run_dir: Path,
    sampling_mode: str,
    window_size: int,
    train_stride: int,
    eval_stride: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    downsample_neg_pos_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    side_dir = run_dir / f"{side}_model"
    side_dir.mkdir(parents=True, exist_ok=True)

    train_natural = _build_split_data(labels, split="train", side=side, stride=train_stride, window_size=window_size)
    val_data = _build_split_data(labels, split="val", side=side, stride=eval_stride, window_size=window_size)

    sample_audit: dict[str, Any] = {
        "sampling_mode": sampling_mode,
        "natural_train_examples": int(len(train_natural.y)),
        "natural_train_positive_examples": int(np.sum(train_natural.y == 1)),
        "natural_train_positive_rate": float(np.mean(train_natural.y)),
    }

    if sampling_mode == "train_only_downsample":
        train_for_fit, downsample_audit = _downsample_train(
            train_natural,
            neg_pos_ratio=downsample_neg_pos_ratio,
            seed=seed,
        )
        sample_audit.update(downsample_audit)
    else:
        train_for_fit = train_natural
        sample_audit.update(
            {
                "sampled_train_examples": int(len(train_for_fit.y)),
                "sampled_positive_examples": int(np.sum(train_for_fit.y == 1)),
                "sampled_negative_examples": int(np.sum(train_for_fit.y == 0)),
            }
        )

    pos_count = int(np.sum(train_for_fit.y == 1))
    neg_count = int(np.sum(train_for_fit.y == 0))
    if pos_count == 0:
        raise RuntimeError(f"No positive training labels for side={side}")

    model = HLSFriendlyTCNClassifier(
        num_inputs=len(Z_FEATURES),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=int(window_size),
    ).to(device)

    if sampling_mode == "weighted_bce":
        pos_weight_value = float(neg_count / max(pos_count, 1))
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        )
    else:
        pos_weight_value = 1.0
    criterion = nn.BCEWithLogitsLoss()

    eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    train_views = _window_views(train_for_fit, window_size)

    best_val_pr_auc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        rng = np.random.default_rng(int(seed) + int(epoch))
        for batch_idx in _iter_batches(len(train_for_fit.y), batch_size=batch_size, shuffle=True, rng=rng):
            batch_x, batch_y = _make_window_batch(train_for_fit, train_views, batch_idx, window_size)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(len(batch_y))
            seen += int(len(batch_y))

        val_pred = _predict(
            model,
            val_data,
            criterion=eval_criterion,
            device=device,
            window_size=window_size,
            batch_size=batch_size,
        )
        try:
            val_pr_auc = float(average_precision_score(val_pred["y_true"], val_pred["probs"]))
        except ValueError:
            val_pr_auc = float("nan")

        train_loss = float(total_loss / max(seen, 1))
        val_loss = float(val_pred["loss"])
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "train_examples_for_fit": int(len(train_for_fit.y)),
                "val_examples": int(len(val_data.y)),
            }
        )
        print(
            f"[{side}] epoch={epoch:02d}/{int(epochs)} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_pr_auc={val_pr_auc:.6f}",
            flush=True,
        )

        improved = False
        if np.isfinite(val_pr_auc) and val_pr_auc > best_val_pr_auc:
            improved = True
        elif np.isclose(val_pr_auc, best_val_pr_auc, equal_nan=True) and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_pr_auc = val_pr_auc
            best_val_loss = val_loss
            best_epoch = int(epoch)
            torch.save(model.state_dict(), side_dir / "best_model.pt")

    history = pd.DataFrame(history_rows)
    history.to_csv(side_dir / "train_history.csv", index=False)

    if not (side_dir / "best_model.pt").exists():
        torch.save(model.state_dict(), side_dir / "best_model.pt")
        best_epoch = int(epochs)

    best_model = HLSFriendlyTCNClassifier(
        num_inputs=len(Z_FEATURES),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=int(window_size),
    ).to(device)
    best_model.load_state_dict(torch.load(side_dir / "best_model.pt", map_location=device))
    best_model.eval()

    train_eval = _predict(
        best_model,
        train_natural,
        criterion=eval_criterion,
        device=device,
        window_size=window_size,
        batch_size=batch_size,
    )
    val_eval = _predict(
        best_model,
        val_data,
        criterion=eval_criterion,
        device=device,
        window_size=window_size,
        batch_size=batch_size,
    )

    metrics = pd.DataFrame(
        [
            _metrics_from_probs(
                side=side,
                split="train_natural",
                y_true=train_eval["y_true"],
                probs=train_eval["probs"],
                loss=float(train_eval["loss"]),
                sampling_mode=sampling_mode,
            ),
            _metrics_from_probs(
                side=side,
                split="val",
                y_true=val_eval["y_true"],
                probs=val_eval["probs"],
                loss=float(val_eval["loss"]),
                sampling_mode=sampling_mode,
            ),
        ]
    )

    metrics_path = out_dir / f"metrics_{side}.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics.to_csv(side_dir / f"metrics_{side}.csv", index=False)

    _plot_pr_curve(
        val_eval["y_true"],
        val_eval["probs"],
        out_dir / f"pr_curve_{side}.png",
        title=f"{side.capitalize()} Toxic Model PR Curve (Val)",
    )
    _plot_pr_curve(
        val_eval["y_true"],
        val_eval["probs"],
        side_dir / f"pr_curve_{side}.png",
        title=f"{side.capitalize()} Toxic Model PR Curve (Val)",
    )

    _save_val_predictions(
        side=side,
        val_data=val_data,
        probs=val_eval["probs"],
        out_path=run_dir / f"val_predictions_{side}.parquet",
    )

    model_config = {
        "side": side,
        "sampling_mode": sampling_mode,
        "window_size": int(window_size),
        "train_stride": int(train_stride),
        "eval_stride": int(eval_stride),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(device),
        "features": Z_FEATURES,
        "best_epoch": int(best_epoch),
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_val_loss": float(best_val_loss),
        "pos_weight": float(pos_weight_value),
        "sample_audit": sample_audit,
        "test_evaluated": False,
        "test_policy": "Test split is intentionally not evaluated in Task 4/5.",
    }
    (side_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    return metrics, history.assign(side=side)


def train_binary_models(
    *,
    input_path: Path,
    out_dir: Path,
    sampling_mode: str,
    window_size: int,
    train_stride: int,
    eval_stride: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device_arg: str,
    downsample_neg_pos_ratio: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"model_{sampling_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)
    device = _resolve_device(device_arg)

    labels = pd.read_parquet(input_path)
    labels = labels.reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)

    histories: dict[str, pd.DataFrame] = {}
    metrics_all: list[pd.DataFrame] = []
    for side in SIDES:
        metrics, history = _train_one_side(
            labels=labels,
            side=side,
            out_dir=out_dir,
            run_dir=run_dir,
            sampling_mode=sampling_mode,
            window_size=window_size,
            train_stride=train_stride,
            eval_stride=eval_stride,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed,
            device=device,
            downsample_neg_pos_ratio=downsample_neg_pos_ratio,
        )
        metrics_all.append(metrics)
        histories[side] = history

    _plot_loss_curve(histories, out_dir / "loss_curve.png")
    pd.concat(metrics_all, ignore_index=True).to_csv(run_dir / "metrics_all.csv", index=False)

    latest = {
        "model_dir": str(run_dir),
        "sampling_mode": sampling_mode,
        "input_path": str(input_path),
        "created_by": str(Path(__file__).resolve()),
        "test_evaluated": False,
    }
    (out_dir / "latest_training_run.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train side-specific binary cancel-risk models.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sampling_mode",
        choices=["natural", "weighted_bce", "train_only_downsample"],
        default="weighted_bce",
    )
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--downsample_neg_pos_ratio", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = train_binary_models(
        input_path=args.input_path,
        out_dir=args.out_dir,
        sampling_mode=args.sampling_mode,
        window_size=args.window_size,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device_arg=args.device,
        downsample_neg_pos_ratio=args.downsample_neg_pos_ratio,
    )
    print("Binary model training complete:")
    print(f"- {run_dir}")
    print("- Test split was not evaluated.")


if __name__ == "__main__":
    main()
