from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import concatenate_records, load_sequence_records, split_records_by_session
from src.models.tcn import HLSFriendlyTCNClassifier


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _label_names(cfg: ExperimentConfig) -> list[str]:
    if cfg.num_classes == 3:
        return ["Down", "Stationary", "Up"]
    mapping = {
        "toxic_bid_binary": ["Non-toxic", "Toxic"],
        "toxic_ask_binary": ["Non-toxic", "Toxic"],
        "adverse_bid_fill_binary": ["Non-adverse", "Adverse"],
        "adverse_ask_fill_binary": ["Non-adverse", "Adverse"],
        "active_buy_profitable_binary": ["Not-profitable", "Profitable"],
        "active_sell_profitable_binary": ["Not-profitable", "Profitable"],
    }
    return mapping.get(cfg.labels.label_type, ["Negative", "Positive"])


def _ece_binary(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            in_bin = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            in_bin = (prob_pos >= lo) & (prob_pos < hi)
        if not np.any(in_bin):
            continue
        bin_frac = float(np.mean(in_bin))
        avg_conf = float(np.mean(prob_pos[in_bin]))
        avg_acc = float(np.mean(y_true[in_bin]))
        ece += bin_frac * abs(avg_acc - avg_conf)
    return float(ece)


def _save_confusion_matrix_plots(
    cm_counts: np.ndarray,
    class_names: list[str],
    title: str,
    out_count_path: Path,
    out_percent_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    cm_counts = np.asarray(cm_counts, dtype=np.int64)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_counts, np.maximum(row_sums, 1), dtype=np.float64)

    def _plot(matrix_for_color: np.ndarray, color_title: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix_for_color, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} | {color_title}")

        vmax = float(np.max(matrix_for_color)) if matrix_for_color.size else 1.0
        text_threshold = vmax * 0.6
        for i in range(cm_counts.shape[0]):
            for j in range(cm_counts.shape[1]):
                count = int(cm_counts[i, j])
                pct = float(cm_percent[i, j] * 100.0)
                color = "white" if float(matrix_for_color[i, j]) >= text_threshold else "black"
                ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", color=color, fontsize=10)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    _plot(cm_counts.astype(np.float64), "Color = count", out_count_path)
    _plot(cm_percent.astype(np.float64), "Color = row-normalized %", out_percent_path)


def _save_prediction_distribution_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    num_classes = len(class_names)

    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    x = np.arange(num_classes)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.4
    ax.bar(x - width / 2, true_counts, width=width, label="True", color="#4C78A8")
    ax.bar(x + width / 2, pred_counts, width=width, label="Pred", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _threshold_grid() -> list[float]:
    return [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]


def _threshold_diagnostics_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    total = float(len(y_true)) if len(y_true) else 1.0

    for thr in _threshold_grid():
        pred_pos = prob_pos >= thr
        num_signals = float(np.sum(pred_pos))
        signal_rate = num_signals / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred_pos.astype(np.int64),
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(thr),
                "num_signals": float(num_signals),
                "signal_rate": float(signal_rate),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "stationary_rate_if_3class": float("nan"),
                "positive_rate_if_binary": float(precision),
            }
        )
    return pd.DataFrame(rows)


def _threshold_diagnostics_3class(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    p_down = prob[:, 0]
    p_up = prob[:, 2]
    total = float(len(y_true)) if len(y_true) else 1.0
    is_nonstationary_true = y_true != 1

    for thr in _threshold_grid():
        p_move = np.maximum(p_down, p_up)
        signal = p_move >= thr
        num_signals = float(np.sum(signal))
        signal_rate = num_signals / total

        tp = float(np.sum(signal & is_nonstationary_true))
        fp = float(np.sum(signal & ~is_nonstationary_true))
        fn = float(np.sum((~signal) & is_nonstationary_true))

        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        rows.append(
            {
                "threshold": float(thr),
                "num_signals": float(num_signals),
                "signal_rate": float(signal_rate),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "stationary_rate_if_3class": float(1.0 - signal_rate),
                "positive_rate_if_binary": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _save_threshold_diagnostics_plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df["threshold"], df["precision"], marker="o", label="precision")
    ax1.plot(df["threshold"], df["recall"], marker="o", label="recall")
    ax1.plot(df["threshold"], df["f1"], marker="o", label="f1")
    for _, row in df.iterrows():
        if round(row["threshold"], 2) in [0.5, 0.7, 0.9]:
            ax1.text(row["threshold"], row["precision"] + 0.02, f"{row['precision']:.2f}", fontsize=8, ha="center")
            ax1.text(row["threshold"], row["recall"] + 0.02, f"{row['recall']:.2f}", fontsize=8, ha="center")
            ax1.text(row["threshold"], row["f1"] + 0.02, f"{row['f1']:.2f}", fontsize=8, ha="center")

    ax1.set_ylabel("Metric")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.2)

    ax2.plot(df["threshold"], df["num_signals"], marker="o", label="num_signals", color="#4C78A8")
    for _, row in df.iterrows():
        if round(row["threshold"], 2) in [0.5, 0.7, 0.9]:
            ax2.text(row["threshold"], row["num_signals"] * 1.05, f"{int(row['num_signals'])}", fontsize=8, ha="center")

    ax2.set_ylabel("Signals")
    ax2.set_xlabel("threshold")
    ax2.grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def _predict_logits(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch_x = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        logits = model(batch_x).detach().cpu().numpy()
        chunks.append(logits)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def _split_metrics(cfg: ExperimentConfig, split_name: str, y_true: np.ndarray, prob: np.ndarray, loss: float) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    y_pred = prob.argmax(axis=1) if len(prob) else np.zeros_like(y_true)

    metrics: dict[str, object] = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "num_samples": int(len(y_true)),
    }

    if cfg.num_classes == 2 and len(y_true):
        prob_pos = prob[:, 1]
        ap = float(average_precision_score(y_true, prob_pos))
        metrics["average_precision"] = ap

        precision, recall, thresholds = precision_recall_curve(y_true, prob_pos)
        f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-12))
        if len(thresholds):
            best_idx = int(np.nanargmax(f1[:-1]))
            metrics["best_threshold_by_f1"] = float(thresholds[best_idx])
        else:
            metrics["best_threshold_by_f1"] = float("nan")

        min_recall = 0.20
        valid = recall[:-1] >= min_recall if len(thresholds) else np.array([], dtype=bool)
        if len(thresholds) and np.any(valid):
            candidates = np.where(valid)[0]
            best_prec_idx = int(candidates[np.nanargmax(precision[candidates])])
            metrics["best_threshold_by_precision_at_min_recall"] = float(thresholds[best_prec_idx])
            metrics["precision_at_min_recall"] = float(precision[best_prec_idx])
            metrics["min_recall"] = float(min_recall)
        else:
            metrics["best_threshold_by_precision_at_min_recall"] = float("nan")
            metrics["precision_at_min_recall"] = float("nan")
            metrics["min_recall"] = float(min_recall)

    if cfg.num_classes == 3 and len(y_true):
        prob_down = prob[:, 0]
        prob_up = prob[:, 2]
        down_mask = prob_down >= 0.70
        up_mask = prob_up >= 0.70
        metrics["p_down_ge_0p70_actual_down_rate"] = float(np.mean(y_true[down_mask] == 0)) if np.any(down_mask) else float("nan")
        metrics["p_up_ge_0p70_actual_up_rate"] = float(np.mean(y_true[up_mask] == 2)) if np.any(up_mask) else float("nan")

    if cfg.num_classes == 2 and len(y_true):
        prob_pos = prob[:, 1]
        pos_mask = prob_pos >= 0.70
        metrics["p_pos_ge_0p70_actual_pos_rate"] = float(np.mean(y_true[pos_mask] == 1)) if np.any(pos_mask) else float("nan")

    return metrics


def evaluate_full(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)
    paths.ensure_dirs()
    save_experiment_config(paths, cfg)

    if not paths.dataset_npz.exists():
        raise FileNotFoundError(f"Dataset not found: {paths.dataset_npz}")
    if not paths.model_weights_pth.exists():
        raise FileNotFoundError(f"Model weights not found: {paths.model_weights_pth}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _label_names(cfg)

    records = load_sequence_records(paths.dataset_npz)
    _, val_records, test_records = split_records_by_session(
        records,
        train_frac=cfg.split.train_session_frac,
        val_frac=cfg.split.val_session_frac,
    )
    if not val_records:
        raise RuntimeError("No validation sessions available; cannot evaluate.")

    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    model.load_state_dict(torch.load(paths.model_weights_pth, map_location=device, weights_only=True))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    summary: dict[str, object] = {
        "device": str(device),
        "label_type": cfg.labels.label_type,
        "class_names": class_names,
        "splits": {},
    }

    for split_name, split_records in (("val", val_records), ("test", test_records)):
        if not split_records:
            continue

        x_split, y_split = concatenate_records(split_records, channel_first=True)
        logits = _predict_logits(model, x_split, device=device)
        if len(logits) != len(y_split):
            raise RuntimeError(f"{split_name}: logits and labels length mismatch ({len(logits)} vs {len(y_split)})")

        prob = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_pred = prob.argmax(axis=1) if len(prob) else np.zeros_like(y_split)

        loss = float(criterion(torch.tensor(logits, dtype=torch.float32), torch.tensor(y_split, dtype=torch.long)).item())

        split_metrics = _split_metrics(cfg, split_name, y_split, prob, loss=loss)
        summary["splits"][split_name] = split_metrics

        cm = confusion_matrix(y_split, y_pred, labels=list(range(cfg.num_classes)))
        _save_confusion_matrix_plots(
            cm_counts=cm,
            class_names=class_names,
            title=f"{split_name} | loss={loss:.4f} | acc={split_metrics['accuracy']:.4f} | bal_acc={split_metrics['balanced_accuracy']:.4f}",
            out_count_path=paths.eval_dir / f"confusion_matrix_count_{split_name}.png",
            out_percent_path=paths.eval_dir / f"confusion_matrix_percent_{split_name}.png",
        )

        report = classification_report(
            y_split,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report).transpose().to_csv(paths.eval_dir / f"classification_report_{split_name}.csv", index=True)

        _save_prediction_distribution_plot(
            y_true=y_split,
            y_pred=y_pred,
            class_names=class_names,
            title=f"{split_name} prediction distribution",
            out_path=paths.eval_dir / f"prediction_distribution_{split_name}.png",
        )

        if cfg.num_classes == 2:
            prob_pos = prob[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_split, prob_pos)
            ap = float(average_precision_score(y_split, prob_pos))
            fpr, tpr, _ = roc_curve(y_split, prob_pos)
            roc_auc = float(roc_auc_score(y_split, prob_pos))

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, label=f"AP={ap:.4f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{split_name} PR curve")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"precision_recall_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{split_name} ROC curve")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"roc_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            frac_pos, mean_pred = calibration_curve(y_split, prob_pos, n_bins=10, strategy="uniform")
            brier = float(brier_score_loss(y_split, prob_pos))
            ece = float(_ece_binary(prob_pos, y_split, n_bins=10))
            summary["splits"][split_name]["brier_score"] = brier
            summary["splits"][split_name]["expected_calibration_error"] = ece

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(mean_pred, frac_pos, marker="o", label="model")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"{split_name} calibration | brier={brier:.4f} ece={ece:.4f}")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"calibration_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            diag_df = _threshold_diagnostics_binary(y_split, prob_pos)
            diag_df.to_csv(paths.eval_dir / f"threshold_diagnostics_{split_name}.csv", index=False)
            _save_threshold_diagnostics_plot(
                diag_df,
                title=f"{split_name} threshold diagnostics (binary)",
                out_path=paths.eval_dir / f"threshold_diagnostics_plot_{split_name}.png",
            )

        if cfg.num_classes == 3:
            prob_down = prob[:, 0]
            prob_up = prob[:, 2]

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            for cls, name in ((0, "Down vs rest"), (2, "Up vs rest")):
                y_bin = (y_split == cls).astype(np.int64)
                p = prob[:, cls]
                prec, rec, thresholds = precision_recall_curve(y_bin, p)
                ap = float(average_precision_score(y_bin, p))
                line = ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})")[0]
                key_thresholds = [0.5, 0.7, 0.9]
                indices = []
                for kt in key_thresholds:
                    if len(thresholds) > 0:
                        idx = int(np.abs(thresholds - kt).argmin())
                        if idx not in indices:
                            indices.append(idx)
                for i in indices:
                    ax.plot(rec[i], prec[i], marker="o", color=line.get_color())
                    ax.text(rec[i], prec[i], f"Thr={thresholds[i]:.2f}\n({rec[i]:.2f},{prec[i]:.2f})", fontsize=7, color=line.get_color())
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{split_name} PR curves (one-vs-rest)")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"precision_recall_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            for cls, name in ((0, "Down vs rest"), (2, "Up vs rest")):
                y_bin = (y_split == cls).astype(np.int64)
                p = prob[:, cls]
                fpr, tpr, thresholds = roc_curve(y_bin, p)
                auc = float(roc_auc_score(y_bin, p))
                line = ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")[0]
                key_thresholds = [0.5, 0.7, 0.9]
                indices = []
                for kt in key_thresholds:
                    if len(thresholds) > 0:
                        idx = int(np.abs(thresholds - kt).argmin())
                        if idx not in indices:
                            indices.append(idx)
                for i in indices:
                    ax.plot(fpr[i], tpr[i], marker="o", color=line.get_color())
                    ax.text(fpr[i], tpr[i], f"Thr={thresholds[i]:.2f}\n({fpr[i]:.2f},{tpr[i]:.2f})", fontsize=7, color=line.get_color())
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{split_name} ROC curves (one-vs-rest)")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"roc_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            for cls, name in ((0, "Down"), (2, "Up")):
                y_bin = (y_split == cls).astype(np.int64)
                p = prob[:, cls]
                frac_pos, mean_pred = calibration_curve(y_bin, p, n_bins=10, strategy="uniform")
                brier = float(brier_score_loss(y_bin, p))
                ece = float(_ece_binary(p, y_bin, n_bins=10))
                line = ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (brier={brier:.4f}, ece={ece:.4f})")[0]
                for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
                    if i in [0, len(mean_pred)//2, len(mean_pred)-1]:
                        ax.text(mp, fp, f"({mp:.2f},{fp:.2f})", fontsize=7, color=line.get_color())
                summary["splits"][split_name][f"brier_{name.lower()}"] = brier
                summary["splits"][split_name][f"ece_{name.lower()}"] = ece
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"{split_name} calibration (Down/Up one-vs-rest)")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(paths.eval_dir / f"calibration_curve_{split_name}.png", dpi=200)
            plt.close(fig)

            diag_df = _threshold_diagnostics_3class(y_split, prob)
            diag_df.to_csv(paths.eval_dir / f"threshold_diagnostics_{split_name}.csv", index=False)
            _save_threshold_diagnostics_plot(
                diag_df,
                title=f"{split_name} threshold diagnostics (3-class: signal=Down/Up)",
                out_path=paths.eval_dir / f"threshold_diagnostics_plot_{split_name}.png",
            )

    write_json(paths.eval_dir / "evaluation_summary.json", summary)
    return summary


def build_default_config() -> ExperimentConfig:
    project_root = get_project_root()
    return ExperimentConfig(
        data=DataConfig(
            data_dir=project_root / "data" / "NEW",
            symbols=("TXFB6",),
            zscore_window=2000,
        )
    )


def main() -> None:
    cfg = build_default_config()
    paths = build_experiment_paths(cfg)
    print(f"[eval] dataset: {paths.dataset_npz}")
    print(f"[eval] weights: {paths.model_weights_pth}")
    summary = evaluate_full(cfg, paths)
    for split_name, split_metrics in summary["splits"].items():  # type: ignore[union-attr]
        print(f"[eval] {split_name} acc={split_metrics['accuracy']:.4f} macro_f1={split_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

