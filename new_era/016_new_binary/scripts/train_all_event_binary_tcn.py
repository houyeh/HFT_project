from __future__ import annotations

import json
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

from fix_common import MODELS_DIR, REPORTS_DIR, WINDOW_SIZE, ensure_fix_dirs

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


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SIDES = ["bid", "ask"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_npz(side: str) -> dict[str, np.ndarray]:
    path = MODELS_DIR.parent / "datasets" / f"{side}_all_event_fw1_h10.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing all-event dataset: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _tensor_ds(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float | np.ndarray]:
    model.eval()
    losses = []
    logits_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
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
            "y_true": np.zeros((0,), dtype=np.int64),
            "probs": np.zeros((0,), dtype=np.float32),
        }
    logits = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(y_all, axis=0).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    try:
        auc_roc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, probs))
    except ValueError:
        auc_pr = float("nan")
    return {
        "loss": float(np.mean(losses)),
        "acc": float((pred == y_true).mean()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_true": float(y_true.mean()),
        "y_true": y_true,
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


def _save_class_report(path: Path, y_true: np.ndarray, probs: np.ndarray) -> None:
    pred = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    p, r, f1, sup = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    rows = [
        {"class": "non_toxic", "precision": p[0], "recall": r[0], "f1_score": f1[0], "support": sup[0]},
        {"class": "toxic", "precision": p[1], "recall": r[1], "f1_score": f1[1], "support": sup[1]},
        {"class": "binary_avg", "precision": float(np.mean(p)), "recall": float(np.mean(r)), "f1_score": float(np.mean(f1)), "support": int(np.sum(sup))},
        {"class": "tp", "precision": tp, "recall": np.nan, "f1_score": np.nan, "support": np.nan},
        {"class": "fp", "precision": fp, "recall": np.nan, "f1_score": np.nan, "support": np.nan},
        {"class": "tn", "precision": tn, "recall": np.nan, "f1_score": np.nan, "support": np.nan},
        {"class": "fn", "precision": fn, "recall": np.nan, "f1_score": np.nan, "support": np.nan},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_train(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Loss")
    axes[1].plot(history["epoch"], history["val_f1"], label="val_f1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Val F1")
    axes[2].plot(history["epoch"], history["val_auc_pr"], label="val_auc_pr")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Val PR-AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    pred = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
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

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr)
    except ValueError:
        ax.text(0.15, 0.5, "ROC unavailable", transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve (Test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve_test.png", dpi=150)
    plt.close(fig)

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


def train_side(side: str) -> dict[str, object]:
    config_name = "all_event_fw1_h10_posw"
    out_dir = MODELS_DIR / f"{side}_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _load_npz(side)
    x_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    x_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(np.int64)
    x_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int64)

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(_tensor_ds(x_train, y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(_tensor_ds(x_val, y_val), batch_size=2048, shuffle=False)
    test_loader = DataLoader(_tensor_ds(x_test, y_test), batch_size=2048, shuffle=False)

    model = HLSFriendlyTCNClassifier(
        num_inputs=x_train.shape[1],
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    pos_weight_val = neg / max(pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float32, device=device))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    history_rows = []
    logs = []
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, 31):
        model.train()
        train_logits = []
        train_true = []
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            train_logits.append(logits.detach().cpu().numpy())
            train_true.append(y.detach().cpu().numpy())
        tl = np.concatenate(train_logits, axis=0)
        ty = np.concatenate(train_true, axis=0).astype(np.int64)
        tp = 1.0 / (1.0 + np.exp(-tl))
        tpred = (tp >= 0.5).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(ty, tpred, average="binary", zero_division=0)
        try:
            t_auc_roc = float(roc_auc_score(ty, tp))
        except ValueError:
            t_auc_roc = float("nan")
        try:
            t_auc_pr = float(average_precision_score(ty, tp))
        except ValueError:
            t_auc_pr = float("nan")
        vm = _eval(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "train_acc": float((tpred == ty).mean()),
            "train_precision": float(p),
            "train_recall": float(r),
            "train_f1": float(f1),
            "train_auc_roc": float(t_auc_roc),
            "train_auc_pr": float(t_auc_pr),
            "val_loss": float(vm["loss"]),
            "val_acc": float(vm["acc"]),
            "val_precision": float(vm["precision"]),
            "val_recall": float(vm["recall"]),
            "val_f1": float(vm["f1"]),
            "val_auc_roc": float(vm["auc_roc"]),
            "val_auc_pr": float(vm["auc_pr"]),
            "val_positive_rate_pred": float(vm["positive_rate_pred"]),
            "val_positive_rate_true": float(vm["positive_rate_true"]),
        }
        history_rows.append(row)
        logs.append(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_f1={row['val_f1']:.6f} val_auc_pr={row['val_auc_pr']:.6f}"
        )
        print(f"[{side}-all_event] {logs[-1]}")
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    history = pd.DataFrame(history_rows)
    history.to_csv(out_dir / "train_history.csv", index=False)
    _plot_train(history, out_dir / "train_history_plot.png")

    best_model = HLSFriendlyTCNClassifier(
        num_inputs=x_train.shape[1],
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    vm = _eval(best_model, val_loader, criterion, device)
    tm = _eval(best_model, test_loader, criterion, device)

    yv = vm["y_true"]
    pv = vm["probs"]
    yt = tm["y_true"]
    pt = tm["probs"]
    _save_class_report(out_dir / "classification_report_val.csv", yv, pv)
    _save_class_report(out_dir / "classification_report_test.csv", yt, pt)
    _plot_confusion(yt, pt, out_dir / "confusion_matrix_test.png")
    _plot_pr_roc_hist(yt, pt, out_dir)
    _threshold_diag(yv, pv).to_csv(out_dir / "threshold_diagnostics_val.csv", index=False)
    _threshold_diag(yt, pt).to_csv(out_dir / "threshold_diagnostics_test.csv", index=False)
    (out_dir / "training_log.txt").write_text("\n".join(logs), encoding="utf-8")

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
        "best_val_pr_auc": float(vm["auc_pr"]),
        "test_pr_auc": float(tm["auc_pr"]),
        "test_roc_auc": float(tm["auc_roc"]),
    }
    (out_dir / "model_config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")

    diag = _threshold_diag(yt, pt)
    best_thr = float(diag.sort_values("f1", ascending=False).iloc[0]["threshold"]) if not diag.empty else float("nan")
    return {
        "side": side,
        "config_name": config_name,
        "model_dir": str(out_dir),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "val_auc_pr": float(vm["auc_pr"]),
        "test_auc_pr": float(tm["auc_pr"]),
        "test_auc_roc": float(tm["auc_roc"]),
        "test_f1_at_0p5": float(precision_recall_fscore_support(yt, (pt >= 0.5).astype(np.int64), average="binary", zero_division=0)[2]),
        "best_threshold_by_f1": best_thr,
        "positive_rate_test": float(y_test.mean()) if len(y_test) else float("nan"),
    }


def main() -> None:
    ensure_fix_dirs()
    summaries = [train_side("bid"), train_side("ask")]
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(MODELS_DIR / "all_event_binary_training_summary.csv", index=False)

    lines = ["# All-event Training Reliability Report", ""]
    for _, row in sdf.iterrows():
        lines.append(f"## {row['side']} model")
        lines.append(f"- best_val_loss={row['best_val_loss']:.6f}")
        lines.append(f"- val_pr_auc={row['val_auc_pr']:.6f}")
        lines.append(f"- test_pr_auc={row['test_auc_pr']:.6f}")
        lines.append(f"- base_rate_test={row['positive_rate_test']:.6f}")
        lines.append("")
    (REPORTS_DIR / "all_event_training_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:")
    print(MODELS_DIR / "all_event_binary_training_summary.csv")


if __name__ == "__main__":
    main()
