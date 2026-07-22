from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Ensure torch is importable from project-local .venv packages.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.tcn import HLSFriendlyTCNClassifier

from common import DATASETS_DIR, MODELS_DIR, REPORTS_DIR, WINDOW_SIZE, ensure_output_dirs, save_json


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SIDES = ["bid", "ask"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_npz(side: str) -> dict[str, np.ndarray]:
    path = DATASETS_DIR / f"{side}_primary_fw1_h10.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(xt, yt)


@torch.no_grad()
def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).squeeze(1)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item())
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(batch_y.detach().cpu().numpy())

    if not logits_all:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
            "positive_rate_pred": float("nan"),
            "positive_rate_true": float("nan"),
            "logits": np.zeros((0,), dtype=np.float32),
            "y_true": np.zeros((0,), dtype=np.int64),
            "probs": np.zeros((0,), dtype=np.float32),
        }

    logits_np = np.concatenate(logits_all, axis=0)
    y_np = np.concatenate(y_all, axis=0).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    pred = (probs >= 0.5).astype(np.int64)
    acc = float((pred == y_np).mean()) if y_np.size else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_np, pred, average="binary", zero_division=0)
    pos_true = float(y_np.mean()) if y_np.size else float("nan")
    pos_pred = float(pred.mean()) if pred.size else float("nan")

    try:
        auc_roc = float(roc_auc_score(y_np, probs))
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_np, probs))
    except ValueError:
        auc_pr = float("nan")

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "positive_rate_pred": pos_pred,
        "positive_rate_true": pos_true,
        "logits": logits_np,
        "y_true": y_np,
        "probs": probs,
    }


def _threshold_diag(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in THRESHOLDS:
        pred = (probs >= thr).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        spec = tn / max(tn + fp, 1)
        rows.append(
            {
                "threshold": float(thr),
                "signal_rate": float(pred.mean()),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "specificity": float(spec),
                "predicted_positive_count": int(pred.sum()),
                "true_positive_count": int(tp),
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
            }
        )
    return pd.DataFrame(rows)


def _save_classification_report(path: Path, y_true: np.ndarray, probs: np.ndarray) -> None:
    pred = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    p, r, f1, sup = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    rows = [
        {"class": "non_toxic", "precision": p[0], "recall": r[0], "f1_score": f1[0], "support": sup[0]},
        {"class": "toxic", "precision": p[1], "recall": r[1], "f1_score": f1[1], "support": sup[1]},
    ]
    rows.append(
        {
            "class": "binary_avg",
            "precision": float(np.mean(p)),
            "recall": float(np.mean(r)),
            "f1_score": float(np.mean(f1)),
            "support": int(np.sum(sup)),
        }
    )
    rows.append({"class": "tp", "precision": tp, "recall": np.nan, "f1_score": np.nan, "support": np.nan})
    rows.append({"class": "fp", "precision": fp, "recall": np.nan, "f1_score": np.nan, "support": np.nan})
    rows.append({"class": "tn", "precision": tn, "recall": np.nan, "f1_score": np.nan, "support": np.nan})
    rows.append({"class": "fn", "precision": fn, "recall": np.nan, "f1_score": np.nan, "support": np.nan})
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_training(history_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    best_idx = history_df["val_loss"].idxmin()
    axes[0].scatter([history_df.loc[best_idx, "epoch"]], [history_df.loc[best_idx, "val_loss"]], color="red", zorder=3)
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="val_f1")
    axes[1].set_title("Val F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["val_auc_pr"], label="val_auc_pr")
    axes[2].set_title("Val PR-AUC")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    pred = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_0", "pred_1"])
    ax.set_yticks([0, 1], labels=["true_0", "true_1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix (Test)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pr_roc_hist(y_true: np.ndarray, probs: np.ndarray, out_dir: Path) -> None:
    # PR curve
    p, r, _ = precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(r, p)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve (Test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_test.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr)
    except ValueError:
        ax.text(0.1, 0.5, "ROC unavailable (single class)", transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve (Test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve_test.png", dpi=150)
    plt.close(fig)

    # Probability histogram
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.hist(probs[y_true == 0], bins=40, alpha=0.6, label="true_0")
    ax.hist(probs[y_true == 1], bins=40, alpha=0.6, label="true_1")
    ax.set_title("Probability Histogram (Test)")
    ax.set_xlabel("P(toxic)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probability_histogram_test.png", dpi=150)
    plt.close(fig)


def train_one_side(side: str, config_name: str = "primary_fw1_h10_posw") -> dict[str, object]:
    out_dir = MODELS_DIR / f"{side}_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_npz(side)
    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(_tensor_dataset(x_train, y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(_tensor_dataset(x_val, y_val), batch_size=2048, shuffle=False)
    test_loader = DataLoader(_tensor_dataset(x_test, y_test), batch_size=2048, shuffle=False)

    model = HLSFriendlyTCNClassifier(
        num_inputs=x_train.shape[1],
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    pos_weight_val = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history_rows: list[dict[str, float]] = []
    train_logs: list[str] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        logits_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            logits_all.append(logits.detach().cpu().numpy())
            y_all.append(batch_y.detach().cpu().numpy())

        train_logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0,), dtype=np.float32)
        train_true = np.concatenate(y_all, axis=0).astype(np.int64) if y_all else np.zeros((0,), dtype=np.int64)
        train_probs = 1.0 / (1.0 + np.exp(-train_logits))
        train_pred = (train_probs >= 0.5).astype(np.int64)
        train_acc = float((train_pred == train_true).mean()) if train_true.size else float("nan")
        p, r, f1, _ = precision_recall_fscore_support(train_true, train_pred, average="binary", zero_division=0)
        try:
            train_auc_roc = float(roc_auc_score(train_true, train_probs))
        except ValueError:
            train_auc_roc = float("nan")
        try:
            train_auc_pr = float(average_precision_score(train_true, train_probs))
        except ValueError:
            train_auc_pr = float("nan")

        val_metrics = _eval_model(model, val_loader, criterion, device)

        row = {
            "epoch": float(epoch),
            "train_loss": float(total_loss / max(len(train_loader), 1)),
            "train_acc": float(train_acc),
            "train_precision": float(p),
            "train_recall": float(r),
            "train_f1": float(f1),
            "train_auc_roc": float(train_auc_roc),
            "train_auc_pr": float(train_auc_pr),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auc_roc": float(val_metrics["auc_roc"]),
            "val_auc_pr": float(val_metrics["auc_pr"]),
            "val_positive_rate_pred": float(val_metrics["positive_rate_pred"]),
            "val_positive_rate_true": float(val_metrics["positive_rate_true"]),
        }
        history_rows.append(row)
        log_line = (
            f"epoch={epoch:02d} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_f1={row['val_f1']:.6f} val_auc_pr={row['val_auc_pr']:.6f}"
        )
        train_logs.append(log_line)
        print(f"[{side}] {log_line}")

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "train_history.csv", index=False)
    _plot_training(history_df, out_dir / "train_history_plot.png")

    # Load best and evaluate.
    best_model = HLSFriendlyTCNClassifier(
        num_inputs=x_train.shape[1],
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    val_metrics = _eval_model(best_model, val_loader, criterion, device)
    test_metrics = _eval_model(best_model, test_loader, criterion, device)
    y_val_true = val_metrics["y_true"]
    p_val = val_metrics["probs"]
    y_test_true = test_metrics["y_true"]
    p_test = test_metrics["probs"]

    _save_classification_report(out_dir / "classification_report_val.csv", y_val_true, p_val)
    _save_classification_report(out_dir / "classification_report_test.csv", y_test_true, p_test)
    _plot_confusion(y_test_true, p_test, out_dir / "confusion_matrix_test.png")
    _plot_pr_roc_hist(y_test_true, p_test, out_dir)

    tdiag = _threshold_diag(y_test_true, p_test)
    tdiag.to_csv(out_dir / "threshold_diagnostics_test.csv", index=False)

    model_cfg = {
        "side": side,
        "config_name": config_name,
        "window_size": WINDOW_SIZE,
        "feature_count": int(x_train.shape[1]),
        "channel_list": [16, 16, 16],
        "kernel_size": 3,
        "dropout": 0.1,
        "epochs": 30,
        "batch_size": 1024,
        "lr": 0.001,
        "seed": 42,
        "device": str(device),
        "pos_weight": float(pos_weight_val),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else float("nan"),
        "positive_rate_val": float(y_val.mean()) if len(y_val) else float("nan"),
        "positive_rate_test": float(y_test.mean()) if len(y_test) else float("nan"),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_pr_auc": float(val_metrics["auc_pr"]),
        "test_pr_auc": float(test_metrics["auc_pr"]),
        "test_roc_auc": float(test_metrics["auc_roc"]),
    }
    save_json(out_dir / "model_config.json", model_cfg)
    (out_dir / "training_log.txt").write_text("\n".join(train_logs), encoding="utf-8")

    summary = {
        "side": side,
        "config_name": config_name,
        "model_dir": str(out_dir),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "val_auc_pr": float(val_metrics["auc_pr"]),
        "test_auc_pr": float(test_metrics["auc_pr"]),
        "test_auc_roc": float(test_metrics["auc_roc"]),
        "test_f1_at_0p5": float(
            precision_recall_fscore_support(y_test_true, (p_test >= 0.5).astype(np.int64), average="binary", zero_division=0)[2]
        ),
        "best_threshold_by_f1": float(tdiag.sort_values("f1", ascending=False).iloc[0]["threshold"]) if not tdiag.empty else float("nan"),
    }
    return summary


def main() -> None:
    ensure_output_dirs()
    summaries = []
    for side in SIDES:
        summaries.append(train_one_side(side))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "binary_training_summary.csv", index=False)

    # Reliability report
    lines = []
    lines.append("# Training Reliability Report")
    lines.append("")
    for _, row in summary_df.iterrows():
        lines.append(f"## {row['side']} model")
        lines.append(f"- converged (best_val_loss): {row['best_val_loss']:.6f}")
        lines.append(f"- val PR-AUC: {row['val_auc_pr']:.6f}")
        lines.append(f"- test PR-AUC: {row['test_auc_pr']:.6f}")
        lines.append(f"- test ROC-AUC: {row['test_auc_roc']:.6f}")
        lines.append(f"- best threshold by test F1: {row['best_threshold_by_f1']:.2f}")
        lines.append("")
    lines.append("Notes:")
    lines.append("- This report is for the pos_weight training configuration only.")
    lines.append("- PR-AUC should be compared against positive base rate from model_config.json.")
    (REPORTS_DIR / "training_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Binary training done.")
    print(f"Summary: {MODELS_DIR / 'binary_training_summary.csv'}")


if __name__ == "__main__":
    main()
