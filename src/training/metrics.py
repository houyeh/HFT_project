"""Small classification-metric helpers shared by training and evaluation scripts."""

from __future__ import annotations

import numpy as np


LABEL_NAMES = {0: "Down", 1: "Stationary", 2: "Up"}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        matrix[truth, pred] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class = {}
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for class_idx in range(num_classes):
        tp = float(matrix[class_idx, class_idx])
        fp = float(matrix[:, class_idx].sum() - tp)
        fn = float(matrix[class_idx, :].sum() - tp)
        support = int(matrix[class_idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        per_class[LABEL_NAMES.get(class_idx, str(class_idx))] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "macro_f1": float(np.mean(f1_scores)),
        "support": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
