from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from fi2010_phc_lite import PHCLiteFI2010Classifier, count_trainable_parameters as count_phc_parameters
from fi2010_tcn import TCNFI2010Classifier, count_trainable_parameters as count_tcn_parameters


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZONS = [10, 20, 30, 50, 100]
CLASS_LABELS = [0, 1, 2]
DEGENERATE_PRED_SHARE = 0.95


class LegacyZeroDecayPHCLiteFI2010Classifier(nn.Module):
    """Compatibility loader for v1/v2 checkpoints with decay/alpha buffers."""

    def __init__(self, feature_dim: int = 40, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.enc_linear = nn.Linear(self.feature_dim, self.hidden_dim)
        self.enc_norm = nn.LayerNorm(self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.candidate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_encoded = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_prev_state = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.synapse = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.loop_norm = nn.LayerNorm(self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, self.num_classes)

        self.register_buffer("decay", torch.zeros(self.hidden_dim))
        self.register_buffer("alpha_scalar", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        if x_window.ndim != 3:
            raise ValueError(f"Expected (batch, time, feature), got shape {tuple(x_window.shape)}")
        batch_size, seq_len, _ = x_window.shape
        state = x_window.new_zeros((batch_size, self.hidden_dim))
        logits = x_window.new_zeros((batch_size, self.num_classes))
        decay = self.decay.view(1, -1)

        flat = x_window.reshape(batch_size * seq_len, self.feature_dim)
        encoded_pre = self.enc_norm(self.enc_linear(flat))
        encoded_flat = encoded_pre * torch.sigmoid(encoded_pre)
        encoded = encoded_flat.reshape(batch_size, seq_len, self.hidden_dim)
        gates = torch.sigmoid(self.gate(encoded_flat)).reshape(batch_size, seq_len, self.hidden_dim)
        candidates = torch.tanh(self.candidate(encoded_flat)).reshape(batch_size, seq_len, self.hidden_dim)

        for t in range(seq_len):
            encoded_t = encoded[:, t, :]
            state_temporal = decay * state + gates[:, t, :] * candidates[:, t, :]
            gain_t = torch.sigmoid(self.gain_from_encoded(encoded_t) + self.gain_from_prev_state(state))
            message_pre = self.synapse(state_temporal)
            message = message_pre * torch.sigmoid(message_pre)
            msg_gated = gain_t * message
            state = self.loop_norm(state_temporal + self.alpha_scalar * msg_gated)
            logits = self.readout(state)
        return logits


def load_cache() -> dict[str, np.ndarray]:
    with np.load(CACHE / "train_cf7.npz") as z:
        train_features = z["features"].astype(np.float32, copy=False)
        train_labels = z["labels"].astype(np.int64, copy=False) - 1
    test_features_parts = []
    test_label_parts = []
    for name in ["test_cf7", "test_cf8", "test_cf9"]:
        with np.load(CACHE / f"{name}.npz") as z:
            test_features_parts.append(z["features"].astype(np.float32, copy=False))
            test_label_parts.append(z["labels"].astype(np.int64, copy=False) - 1)
    cut = int(train_features.shape[0] * TRAIN_FRACTION)
    return {
        "val_features": train_features[cut:],
        "val_labels": train_labels[cut:],
        "test_features": np.concatenate(test_features_parts, axis=0),
        "test_labels": np.concatenate(test_label_parts, axis=0),
    }


def batch_windows(features: np.ndarray, starts: np.ndarray) -> torch.Tensor:
    idx = starts[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
    return torch.from_numpy(np.ascontiguousarray(features[idx])).to(dtype=torch.float32)


def batch_labels(labels: np.ndarray, starts: np.ndarray, horizon_idx: int) -> torch.Tensor:
    y = labels[starts + WINDOW - 1, horizon_idx]
    return torch.from_numpy(np.ascontiguousarray(y)).to(dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    horizon_idx: int,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    starts_all = np.arange(0, features.shape[0] - WINDOW + 1, dtype=np.int64)
    y_true_parts = []
    y_pred_parts = []
    total_loss = 0.0
    total_n = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for pos in range(0, len(starts_all), batch_size):
        starts = starts_all[pos : pos + batch_size]
        x = batch_windows(features, starts)
        y = batch_labels(labels, starts, horizon_idx)
        logits = model(x)
        loss = loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.detach().cpu())
        total_n += int(y.numel())
        y_true_parts.append(y.detach().cpu().numpy())
        y_pred_parts.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    pred_counts = np.bincount(y_pred, minlength=3)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "weighted_recall_accuracy_abs_diff": abs(float(weighted_recall) - accuracy),
        "support_0_up": int(support[0]),
        "support_1_stationary": int(support[1]),
        "support_2_down": int(support[2]),
        "pred_0_up": int(pred_counts[0]),
        "pred_1_stationary": int(pred_counts[1]),
        "pred_2_down": int(pred_counts[2]),
    }


def jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_rows(path: Path, rows: list[dict[str, object]], allow_overwrite: bool = False) -> None:
    if not rows:
        return
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_keys = ["model_type", "architecture_variant", "loss_mode", "horizon", "hidden_dim", "channels"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in metrics_rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)

    for key in sorted(groups):
        subset = groups[key]
        group_values = dict(zip(group_keys, key))
        for split in ["val", "test"]:
            acc = np.array([float(r[f"{split}_accuracy"]) for r in subset])
            macro_pre = np.array([float(r[f"{split}_macro_precision"]) for r in subset])
            macro_rec = np.array([float(r[f"{split}_macro_recall"]) for r in subset])
            macro_f1 = np.array([float(r[f"{split}_macro_f1"]) for r in subset])
            weighted_pre = np.array([float(r[f"{split}_weighted_precision"]) for r in subset])
            weighted_rec = np.array([float(r[f"{split}_weighted_recall"]) for r in subset])
            weighted_f1 = np.array([float(r[f"{split}_weighted_f1"]) for r in subset])
            rows.append(
                {
                    "split": split,
                    **group_values,
                    "n_seeds": len(subset),
                    "accuracy_mean": float(acc.mean()),
                    "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "macro_precision_mean": float(macro_pre.mean()),
                    "macro_precision_std": float(macro_pre.std(ddof=1)) if len(macro_pre) > 1 else 0.0,
                    "macro_recall_mean": float(macro_rec.mean()),
                    "macro_recall_std": float(macro_rec.std(ddof=1)) if len(macro_rec) > 1 else 0.0,
                    "macro_f1_mean": float(macro_f1.mean()),
                    "macro_f1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
                    "weighted_precision_mean": float(weighted_pre.mean()),
                    "weighted_precision_std": float(weighted_pre.std(ddof=1)) if len(weighted_pre) > 1 else 0.0,
                    "weighted_recall_mean": float(weighted_rec.mean()),
                    "weighted_recall_std": float(weighted_rec.std(ddof=1)) if len(weighted_rec) > 1 else 0.0,
                    "weighted_f1_mean": float(weighted_f1.mean()),
                    "weighted_f1_std": float(weighted_f1.std(ddof=1)) if len(weighted_f1) > 1 else 0.0,
                    "max_weighted_recall_accuracy_abs_diff": float(
                        max(float(r[f"{split}_weighted_recall_accuracy_abs_diff"]) for r in subset)
                    ),
                }
            )
    return rows


def infer_model_type(path: Path) -> str:
    if path.name.startswith("tcn_"):
        return "tcn"
    return "phc"


def build_model(path: Path, payload: dict[str, object]) -> tuple[nn.Module, str, int]:
    model_type = infer_model_type(path)
    if model_type == "tcn":
        model = TCNFI2010Classifier(
            feature_dim=int(payload.get("feature_dim", 40)),
            channels=int(payload.get("channels", 16)),
            num_classes=int(payload.get("num_classes", 3)),
            kernel_size=int(payload.get("kernel_size", 3)),
            dropout=float(payload.get("dropout", 0.1)),
            input_output_bias=bool(payload.get("input_output_bias", True)),
        )
        model.load_state_dict(payload["model_state_dict"])
        return model, "tcn_dilated_causal_conv", count_tcn_parameters(model)

    state_dict = payload["model_state_dict"]
    if "decay" in state_dict and "decay_param" not in state_dict:
        model = LegacyZeroDecayPHCLiteFI2010Classifier(
            feature_dim=int(payload["feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_classes=int(payload["num_classes"]),
        )
        model.load_state_dict(state_dict)
        return model, "legacy_zero_decay_alpha_buffers", count_phc_parameters(model)

    model = PHCLiteFI2010Classifier(
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_classes=int(payload["num_classes"]),
    )
    model.load_state_dict(state_dict)
    return model, "v3_trainable_decay_alpha", count_phc_parameters(model)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    metrics_path = Path(args.metrics_output) if args.metrics_output is not None else RESULTS / f"stage1_metrics{args.output_suffix}.csv"
    summary_path = Path(args.summary_output) if args.summary_output is not None else RESULTS / f"stage1_summary{args.output_suffix}.csv"
    return metrics_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute FI-2010 checkpoint metrics without retraining.")
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=None,
        help="Checkpoint filename glob under models/. Repeat to include multiple patterns.",
    )
    parser.add_argument("--output-suffix", default="_weighted_recompute")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--weighted-recall-atol", type=float, default=1e-12)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[str, int, int, int, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    horizon = int(payload.get("horizon", 0))
    horizon_rank = HORIZONS.index(horizon) if horizon in HORIZONS else len(HORIZONS)
    loss_rank = {
        "raw": 0,
        "raw_cross_entropy": 0,
        "weighted": 1,
        "class_weighted_cross_entropy": 1,
    }.get(str(payload.get("loss_mode", "")), 2)
    return (infer_model_type(path), horizon_rank, int(payload.get("seed", 0)), loss_rank, path.name)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    data = load_cache()
    checkpoint_globs = args.checkpoint_glob or ["phc_fi2010_h*_seed*_*.pt"]
    checkpoint_paths = []
    for pattern in checkpoint_globs:
        checkpoint_paths.extend(MODELS.glob(pattern))
    checkpoint_paths = sorted(set(checkpoint_paths), key=checkpoint_sort_key)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched: {checkpoint_globs}")

    rows: list[dict[str, object]] = []
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        horizon = int(payload["horizon"])
        horizon_idx = HORIZONS.index(horizon)
        model, architecture_variant, trainable_parameters = build_model(path, payload)
        val = evaluate(model, data["val_features"], data["val_labels"], horizon_idx, batch_size=args.batch_size)
        test = evaluate(model, data["test_features"], data["test_labels"], horizon_idx, batch_size=args.batch_size)
        if val["weighted_recall_accuracy_abs_diff"] > args.weighted_recall_atol:
            raise AssertionError(
                f"val weighted_recall != accuracy for {path.name}: {val['weighted_recall_accuracy_abs_diff']}"
            )
        if test["weighted_recall_accuracy_abs_diff"] > args.weighted_recall_atol:
            raise AssertionError(
                f"test weighted_recall != accuracy for {path.name}: {test['weighted_recall_accuracy_abs_diff']}"
            )
        max_val_pred_share = max(val["pred_0_up"], val["pred_1_stationary"], val["pred_2_down"]) / max(
            val["pred_0_up"] + val["pred_1_stationary"] + val["pred_2_down"], 1
        )
        max_test_pred_share = max(test["pred_0_up"], test["pred_1_stationary"], test["pred_2_down"]) / max(
            test["pred_0_up"] + test["pred_1_stationary"] + test["pred_2_down"], 1
        )
        row = {
            "model_type": infer_model_type(path),
            "architecture_variant": architecture_variant,
            "horizon": horizon,
            "seed": int(payload["seed"]),
            "loss_mode": payload.get("loss_mode", ""),
            "best_epoch": int(payload.get("best_epoch", -1)),
            "trainable_parameters": trainable_parameters,
            "hidden_dim": payload.get("hidden_dim", ""),
            "channels": payload.get("channels", ""),
            "input_output_bias": payload.get("input_output_bias", ""),
            "class_weights": json.dumps(jsonable(payload.get("class_weights", []))),
            "device": "cpu",
            "val_accuracy": val["accuracy"],
            "val_macro_precision": val["macro_precision"],
            "val_macro_recall": val["macro_recall"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_precision": val["weighted_precision"],
            "val_weighted_recall": val["weighted_recall"],
            "val_weighted_f1": val["weighted_f1"],
            "val_weighted_recall_accuracy_abs_diff": val["weighted_recall_accuracy_abs_diff"],
            "val_pred_0_up": val["pred_0_up"],
            "val_pred_1_stationary": val["pred_1_stationary"],
            "val_pred_2_down": val["pred_2_down"],
            "val_support_0_up": val["support_0_up"],
            "val_support_1_stationary": val["support_1_stationary"],
            "val_support_2_down": val["support_2_down"],
            "test_accuracy": test["accuracy"],
            "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall"],
            "test_macro_f1": test["macro_f1"],
            "test_weighted_precision": test["weighted_precision"],
            "test_weighted_recall": test["weighted_recall"],
            "test_weighted_f1": test["weighted_f1"],
            "test_weighted_recall_accuracy_abs_diff": test["weighted_recall_accuracy_abs_diff"],
            "test_pred_0_up": test["pred_0_up"],
            "test_pred_1_stationary": test["pred_1_stationary"],
            "test_pred_2_down": test["pred_2_down"],
            "test_support_0_up": test["support_0_up"],
            "test_support_1_stationary": test["support_1_stationary"],
            "test_support_2_down": test["support_2_down"],
            "max_val_pred_share": max_val_pred_share,
            "max_test_pred_share": max_test_pred_share,
            "degenerate_prediction": bool(max_val_pred_share >= DEGENERATE_PRED_SHARE),
            "checkpoint": str(path.relative_to(ROOT)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    metrics_path, summary_path = output_paths(args)
    write_rows(metrics_path, rows, allow_overwrite=args.allow_overwrite)
    write_rows(summary_path, summarize(rows), allow_overwrite=args.allow_overwrite)
    print(json.dumps({"summary": summarize(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
