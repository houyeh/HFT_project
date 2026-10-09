from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fi2010_tcn import TCNFI2010Classifier, count_trainable_parameters


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZONS = [10, 20, 30, 50, 100]
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZONS)}
CLASS_LABELS = [0, 1, 2]

CURVES_CSV = RESULTS / "tcn_fi2010_training_curves.csv"
METRICS_CSV = RESULTS / "tcn_fi2010_metrics.csv"
SUMMARY_CSV = RESULTS / "tcn_fi2010_summary.csv"
TIMING_CSV = RESULTS / "tcn_fi2010_timing_estimate.csv"
RUN_STATE_JSON = RESULTS / "tcn_fi2010_run_state.json"
REPORT_MD = ROOT / "TCN_FI2010_REPORT.md"


def configure_output_suffix(suffix: str) -> None:
    """Route optional experiment variants to separate tcn-prefixed artifacts."""
    global CURVES_CSV, METRICS_CSV, SUMMARY_CSV, TIMING_CSV, RUN_STATE_JSON, REPORT_MD
    clean = str(suffix or "")
    if clean and not clean.startswith("_"):
        clean = "_" + clean
    CURVES_CSV = RESULTS / f"tcn_fi2010_training_curves{clean}.csv"
    METRICS_CSV = RESULTS / f"tcn_fi2010_metrics{clean}.csv"
    SUMMARY_CSV = RESULTS / f"tcn_fi2010_summary{clean}.csv"
    TIMING_CSV = RESULTS / f"tcn_fi2010_timing_estimate{clean}.csv"
    RUN_STATE_JSON = RESULTS / f"tcn_fi2010_run_state{clean}.json"
    report_stem = "TCN_FI2010_REPORT" if not clean else f"TCN_FI2010{clean.upper()}_REPORT"
    REPORT_MD = ROOT / f"{report_stem}.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    existing = read_rows(path)
    all_rows: list[dict[str, object]] = [*existing, *rows]
    write_rows(path, all_rows)


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
    return {
        "train_features_all": train_features,
        "train_labels_all": train_labels,
        "test_features_all": np.concatenate(test_features_parts, axis=0),
        "test_labels_all": np.concatenate(test_label_parts, axis=0),
    }


def split_train_val(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    cut = int(math.floor(features.shape[0] * TRAIN_FRACTION))
    return {
        "train_features": features[:cut],
        "train_labels": labels[:cut],
        "val_features": features[cut:],
        "val_labels": labels[cut:],
    }


def batch_windows(features: np.ndarray, starts: np.ndarray, device: torch.device) -> torch.Tensor:
    offsets = np.arange(WINDOW, dtype=np.int64)
    idx = starts[:, None] + offsets[None, :]
    x = torch.from_numpy(np.ascontiguousarray(features[idx]))
    return x.to(device=device, dtype=torch.float32)


def batch_labels(labels: np.ndarray, starts: np.ndarray, horizon_idx: int, device: torch.device) -> torch.Tensor:
    y = labels[starts + WINDOW - 1, horizon_idx]
    return torch.from_numpy(np.ascontiguousarray(y)).to(device=device, dtype=torch.long)


def compute_class_weights(labels: np.ndarray, horizon_idx: int) -> torch.Tensor:
    window_labels = labels[WINDOW - 1 :, horizon_idx]
    counts = np.bincount(window_labels, minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: TCNFI2010Classifier,
    features: np.ndarray,
    labels: np.ndarray,
    horizon_idx: int,
    batch_size: int,
    device: torch.device,
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
        y = batch_labels(labels, starts, horizon_idx, device)
        x = batch_windows(features, starts, device)
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
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    pred_counts = np.bincount(y_pred, minlength=3)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "support_0_up": int(support[0]),
        "support_1_stationary": int(support[1]),
        "support_2_down": int(support[2]),
        "pred_0_up": int(pred_counts[0]),
        "pred_1_stationary": int(pred_counts[1]),
        "pred_2_down": int(pred_counts[2]),
        "per_class_precision": [float(x) for x in precision],
        "per_class_recall": [float(x) for x in recall],
        "per_class_f1": [float(x) for x in f1],
    }


def checkpoint_paths(horizon: int, seed: int, suffix: str = "") -> tuple[Path, Path]:
    clean = str(suffix or "")
    if clean and not clean.startswith("_"):
        clean = "_" + clean
    stem = f"tcn_fi2010_h{horizon}_seed{seed}{clean}_scratch"
    return MODELS / f"{stem}_best.pt", MODELS / f"{stem}_last.pt"


def metric_done(horizon: int, seed: int) -> bool:
    for row in read_rows(METRICS_CSV):
        if int(row["horizon"]) == horizon and int(row["seed"]) == seed:
            return True
    return False


def build_model(args: argparse.Namespace) -> TCNFI2010Classifier:
    model = TCNFI2010Classifier(
        feature_dim=40,
        channels=args.channels,
        num_classes=3,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        input_output_bias=not args.no_input_output_bias,
    )
    if model.receptive_field() < WINDOW:
        raise RuntimeError(f"TCN receptive field {model.receptive_field()} does not cover WINDOW={WINDOW}")
    return model


def loss_mode_name(args: argparse.Namespace) -> str:
    if args.loss_mode == "weighted":
        return "class_weighted_cross_entropy"
    return "raw_cross_entropy"


def make_loss_fn(args: argparse.Namespace, class_weights: torch.Tensor) -> torch.nn.Module:
    if args.loss_mode == "weighted":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    return torch.nn.CrossEntropyLoss()


def timing_probe(data: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    set_seed(args.seeds[0])
    split = split_train_val(data["train_features_all"], data["train_labels_all"])
    horizon_idx = HORIZON_TO_IDX[args.horizons[0]]
    model = build_model(args).to(device)
    model.train()
    weights = compute_class_weights(split["train_labels"], horizon_idx).to(device)
    loss_fn = make_loss_fn(args, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    starts_all = np.arange(0, split["train_features"].shape[0] - WINDOW + 1, dtype=np.int64)
    np.random.shuffle(starts_all)

    batches = max(1, int(args.timing_batches))
    train_samples = 0
    t0 = time.perf_counter()
    for pos in range(0, min(len(starts_all), batches * args.batch_size), args.batch_size):
        starts = starts_all[pos : pos + args.batch_size]
        x = batch_windows(split["train_features"], starts, device)
        y = batch_labels(split["train_labels"], starts, horizon_idx, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        train_samples += int(y.numel())
    train_seconds = time.perf_counter() - t0

    model.eval()
    val_starts = np.arange(0, split["val_features"].shape[0] - WINDOW + 1, dtype=np.int64)
    eval_batches = max(1, int(args.eval_timing_batches))
    eval_samples = 0
    t1 = time.perf_counter()
    with torch.no_grad():
        for pos in range(0, min(len(val_starts), eval_batches * args.eval_batch_size), args.eval_batch_size):
            starts = val_starts[pos : pos + args.eval_batch_size]
            x = batch_windows(split["val_features"], starts, device)
            _ = model(x)
            eval_samples += int(len(starts))
    eval_seconds = time.perf_counter() - t1

    train_sps = train_samples / max(train_seconds, 1e-9)
    eval_sps = eval_samples / max(eval_seconds, 1e-9)
    n_train = int(split["train_features"].shape[0] - WINDOW + 1)
    n_val = int(split["val_features"].shape[0] - WINDOW + 1)
    n_test = int(data["test_features_all"].shape[0] - WINDOW + 1)
    estimated_epoch_seconds = n_train / max(train_sps, 1e-9) + n_val / max(eval_sps, 1e-9)
    estimated_combo_seconds = args.max_epochs * estimated_epoch_seconds + n_test / max(eval_sps, 1e-9)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "timing_horizon": args.horizons[0],
        "timing_seed": args.seeds[0],
        "train_samples": train_samples,
        "train_seconds": train_seconds,
        "train_samples_per_second": train_sps,
        "eval_samples": eval_samples,
        "eval_seconds": eval_seconds,
        "eval_samples_per_second": eval_sps,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "estimated_combo_seconds_max_epochs": estimated_combo_seconds,
        "estimated_all_15_seconds_max_epochs": estimated_combo_seconds * 15,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "channels": args.channels,
        "loss_mode": loss_mode_name(args),
        "input_output_bias": not args.no_input_output_bias,
    }
    write_rows(TIMING_CSV, [row])
    return row


def load_last_if_available(
    model: TCNFI2010Classifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    last_path: Path,
) -> tuple[int, float, int, bool]:
    if not last_path.exists():
        return 0, -1.0, 0, False
    ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"]), float(ckpt["best_val_macro_f1"]), int(ckpt["best_epoch"]), True


def train_one(
    data: dict[str, np.ndarray],
    horizon: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    split = split_train_val(data["train_features_all"], data["train_labels_all"])
    horizon_idx = HORIZON_TO_IDX[horizon]
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr_min)
    best_path, last_path = checkpoint_paths(horizon, seed, args.output_suffix)
    start_epoch, best_val_f1, best_epoch, resumed = load_last_if_available(model, optimizer, scheduler, last_path)
    patience_left = args.patience

    class_weights = compute_class_weights(split["train_labels"], horizon_idx).to(device)
    loss_fn = make_loss_fn(args, class_weights)
    starts_all = np.arange(0, split["train_features"].shape[0] - WINDOW + 1, dtype=np.int64)
    curves: list[dict[str, object]] = []
    combo_t0 = time.time()
    final_epoch = start_epoch
    early_stop = False
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        final_epoch = epoch
        model.train()
        np.random.shuffle(starts_all)
        train_loss_sum = 0.0
        train_seen = 0
        for pos in range(0, len(starts_all), args.batch_size):
            starts = starts_all[pos : pos + args.batch_size]
            x = batch_windows(split["train_features"], starts, device)
            y = batch_labels(split["train_labels"], starts, horizon_idx, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(y.numel())
            train_seen += int(y.numel())

        val_metrics = evaluate(model, split["val_features"], split["val_labels"], horizon_idx, args.eval_batch_size, device)
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        val_f1 = float(val_metrics["macro_f1"])
        improved = val_f1 > best_val_f1 + args.min_delta
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "phase": "08a_tcn_scratch",
                    "model_type": "TCNFI2010Classifier",
                    "horizon": horizon,
                    "seed": seed,
                    "feature_dim": 40,
                    "channels": args.channels,
                    "num_classes": 3,
                    "window": WINDOW,
                    "train_fraction": TRAIN_FRACTION,
                    "dilations": list(model.dilations),
                    "kernel_size": args.kernel_size,
                    "dropout": args.dropout,
                    "input_output_bias": not args.no_input_output_bias,
                    "receptive_field": model.receptive_field(),
                    "output_mode": "final_timestep",
                    "loss_mode": loss_mode_name(args),
                    "class_weights": [float(x) for x in class_weights.detach().cpu().numpy()],
                    "output_suffix": args.output_suffix,
                    "model_state_dict": model.state_dict(),
                    "trainable_parameters": count_trainable_parameters(model),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "new5_used": False,
                },
                best_path,
            )
        else:
            patience_left -= 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "horizon": horizon,
                "seed": seed,
                "channels": args.channels,
                "input_output_bias": not args.no_input_output_bias,
                "loss_mode": loss_mode_name(args),
                "output_suffix": args.output_suffix,
                "new5_used": False,
            },
            last_path,
        )
        row = {
            "phase": "08a_tcn_scratch",
            "horizon": horizon,
            "seed": seed,
            "channels": args.channels,
            "loss_mode": loss_mode_name(args),
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_val_macro_f1_so_far": best_val_f1,
            "best_epoch_so_far": best_epoch,
            "current_lr": current_lr,
            "patience_left": patience_left,
            "epoch_seconds": time.time() - combo_t0,
            "resumed_from_last": resumed,
            "torch_threads": torch.get_num_threads(),
        }
        curves.append(row)
        print(json.dumps(row), flush=True)
        if patience_left <= 0:
            early_stop = True
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was written for horizon={horizon}, seed={seed}")
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_final = evaluate(model, split["val_features"], split["val_labels"], horizon_idx, args.eval_batch_size, device)
    test_final = evaluate(model, data["test_features_all"], data["test_labels_all"], horizon_idx, args.eval_batch_size, device)
    metrics = {
        "phase": "08a_tcn_scratch",
        "model": "TCNFI2010Classifier",
        "horizon": horizon,
        "seed": seed,
        "channels": args.channels,
        "input_output_bias": not args.no_input_output_bias,
        "loss_mode": loss_mode_name(args),
        "output_suffix": args.output_suffix,
        "best_epoch": int(best_ckpt["best_epoch"]),
        "final_epoch": final_epoch,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stop": early_stop,
        "max_epoch_hit": final_epoch >= args.max_epochs and not early_stop,
        "trainable_parameters": int(best_ckpt["trainable_parameters"]),
        "receptive_field": int(best_ckpt["receptive_field"]),
        "output_mode": "final_timestep",
        "window": WINDOW,
        "class_weights": json.dumps(best_ckpt["class_weights"]),
        "torch_threads": torch.get_num_threads(),
        "combo_seconds": time.time() - combo_t0,
        "checkpoint": str(best_path.relative_to(ROOT)),
        "val_accuracy": val_final["accuracy"],
        "val_macro_precision": val_final["macro_precision"],
        "val_macro_recall": val_final["macro_recall"],
        "val_macro_f1": val_final["macro_f1"],
        "val_pred_0_up": val_final["pred_0_up"],
        "val_pred_1_stationary": val_final["pred_1_stationary"],
        "val_pred_2_down": val_final["pred_2_down"],
        "test_accuracy": test_final["accuracy"],
        "test_macro_precision": test_final["macro_precision"],
        "test_macro_recall": test_final["macro_recall"],
        "test_macro_f1": test_final["macro_f1"],
        "test_pred_0_up": test_final["pred_0_up"],
        "test_pred_1_stationary": test_final["pred_1_stationary"],
        "test_pred_2_down": test_final["pred_2_down"],
        "new5_used": False,
    }
    return metrics, curves


def summarize() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(METRICS_CSV)
    rows = []
    for horizon, group in metrics.groupby("horizon", sort=True):
        rows.append(
            {
                "model": "TCNFI2010Classifier",
                "phase": "08a_tcn_scratch",
                "split": "test",
                "horizon": int(horizon),
                "n_seeds_completed": int(group["seed"].nunique()),
                "test_macro_f1_mean": float(group["test_macro_f1"].mean()),
                "test_macro_f1_std": float(group["test_macro_f1"].std(ddof=0)),
                "test_accuracy_mean": float(group["test_accuracy"].mean()),
                "val_macro_f1_mean": float(group["val_macro_f1"].mean()),
                "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
                "verification_level": "stage1_tcn_scratch_software_measured",
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def load_phc_v2_table() -> pd.DataFrame:
    path = RESULTS / "stage1_primary_metrics_v2.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["split"] == "test"].copy()


def write_report(run_state: dict[str, object] | None = None) -> None:
    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()
    summary = summarize()
    phc = load_phc_v2_table()
    phc_map = {int(r.horizon): float(r.macro_f1_mean) for r in phc.itertuples()} if not phc.empty else {}
    param_phc = int(phc["trainable_parameters"].iloc[0]) if not phc.empty else 2115
    rows = []
    if not summary.empty:
        for r in summary.itertuples():
            phc_f1 = phc_map.get(int(r.horizon), float("nan"))
            rows.append(
                {
                    "horizon": int(r.horizon),
                    "TCN_completed_seeds": int(r.n_seeds_completed),
                    "TCN_test_macro_F1": f"{100.0 * float(r.test_macro_f1_mean):.2f} +/- {100.0 * float(r.test_macro_f1_std):.2f}",
                    "PHC_v2_test_macro_F1": f"{100.0 * phc_f1:.2f}" if not math.isnan(phc_f1) else "n/a",
                    "TCN_minus_PHC_pct_pt": f"{100.0 * (float(r.test_macro_f1_mean) - phc_f1):+.2f}" if not math.isnan(phc_f1) else "n/a",
                }
            )
    metric_rows = []
    if not metrics.empty:
        for r in metrics.sort_values(["horizon", "seed"]).itertuples():
            metric_rows.append(
                {
                    "horizon": int(r.horizon),
                    "seed": int(r.seed),
                    "best_epoch": int(r.best_epoch),
                    "final_epoch": int(r.final_epoch),
                    "early_stop": bool(r.early_stop),
                    "val_macro_F1": f"{100.0 * float(r.val_macro_f1):.2f}",
                    "test_macro_F1": f"{100.0 * float(r.test_macro_f1):.2f}",
                    "combo_seconds": f"{float(r.combo_seconds):.1f}",
                }
            )

    def md_table(table_rows: list[dict[str, object]], cols: list[str]) -> str:
        if not table_rows:
            return "_No rows yet._"
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in table_rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    timing = pd.read_csv(TIMING_CSV).iloc[0].to_dict() if TIMING_CSV.exists() else {}
    state = run_state or (json.loads(RUN_STATE_JSON.read_text(encoding="utf-8")) if RUN_STATE_JSON.exists() else {})
    tcn_params = int(metrics["trainable_parameters"].iloc[0]) if not metrics.empty else count_trainable_parameters(TCNFI2010Classifier())
    tcn_channels = int(metrics["channels"].iloc[0]) if (not metrics.empty and "channels" in metrics.columns) else int(state.get("channels", 16) or 16)
    input_output_bias = (
        bool(metrics["input_output_bias"].iloc[0])
        if (not metrics.empty and "input_output_bias" in metrics.columns)
        else bool(state.get("input_output_bias", True))
    )
    requested_combo_count = len(state.get("horizons_order", [])) * len(state.get("seeds", []))
    completed_combo_count = int(len(metrics)) if not metrics.empty else 0
    all_15_seconds = float(timing.get("estimated_all_15_seconds_max_epochs", float("nan")))
    all_15_hours = all_15_seconds / 3600.0 if math.isfinite(all_15_seconds) else float("nan")
    budget_hours = float(state.get("wall_clock_budget_hours", 0.0) or 0.0)
    full_15_feasible = bool(math.isfinite(all_15_hours) and budget_hours > 0.0 and all_15_hours <= budget_hours)
    report = f"""# FI-2010 TCN Scratch Report

status: tcn_scratch_partial_or_complete

## Scope

This report covers only the FI-2010 scratch TCN baseline. It does not modify PHC-lite files, PHC v2/v3 checkpoints, TXF phases, or NEW1-NEW5 data.

## Architecture

- Model file: `fi2010_tcn.py`
- Input: `(batch, time=100, feature=40)`
- Projection: per-timestep `Linear(40 -> {tcn_channels})`; for `tcn_small`, this shrinks width while preserving the same causal TCN topology.
- TCN blocks: 5 residual causal-conv blocks, kernel=3, dilations `[1, 2, 4, 8, 16]`, ReLU, dropout.
- Receptive field: `1 + 2 * (3 - 1) * sum(dilations) = 125`, covering the full 100-event FI-2010 window.
- Output: final-timestep readout. This matches the benchmark label at the terminal event and avoids pooling future/early-window context into a decision that should be anchored at the last event.
- Input/readout bias enabled: `{input_output_bias}`.

| model | trainable parameters |
| --- | ---: |
| TCN scratch | {tcn_params} |
| PHC-lite FI-2010 v2 | {param_phc} |

## Runtime Discipline

| item | value |
| --- | --- |
| start_time | {state.get('start_time', '')} |
| wall_clock_budget_hours | {state.get('wall_clock_budget_hours', '')} |
| cpu_count | {state.get('cpu_count', timing.get('cpu_count', ''))} |
| torch_threads | {state.get('torch_threads', timing.get('torch_threads', ''))} |
| timing_train_samples_per_second | {timing.get('train_samples_per_second', '')} |
| timing_eval_samples_per_second | {timing.get('eval_samples_per_second', '')} |
| estimated_combo_seconds_max_epochs | {timing.get('estimated_combo_seconds_max_epochs', '')} |
| estimated_all_15_hours_max_epochs | {all_15_hours:.2f} |
| full_15_combo_scan_feasible_under_budget | {full_15_feasible} |
| requested_combos_this_run | {requested_combo_count} |
| completed_combos_total | {completed_combo_count} |
| budget_exhausted | {state.get('budget_exhausted', False)} |
| stopped_before_next_combo | {state.get('stopped_before_next_combo', '')} |

The timing probe is deliberately recorded before training. On this CPU-only run, the full 15-combination scan is not feasible under the 6-hour wall-clock budget, so k=50 was prioritized first.

## Per-Seed Results

{md_table(metric_rows, ["horizon", "seed", "best_epoch", "final_epoch", "early_stop", "val_macro_F1", "test_macro_F1", "combo_seconds"])}

## TCN vs PHC-lite v2

PHC-lite v2 is used as the comparison baseline because it is the closed Stage 1 convergence baseline. PHC v3 is intentionally not used here.

{md_table(rows, ["horizon", "TCN_completed_seeds", "TCN_test_macro_F1", "PHC_v2_test_macro_F1", "TCN_minus_PHC_pct_pt"])}

## Files

- Metrics: `{METRICS_CSV.relative_to(ROOT)}`
- Curves: `{CURVES_CSV.relative_to(ROOT)}`
- Summary: `{SUMMARY_CSV.relative_to(ROOT)}`
- Timing estimate: `{TIMING_CSV.relative_to(ROOT)}`
- Checkpoints: `models/tcn_fi2010_h{{horizon}}_seed{{seed}}_scratch_best.pt`

## Claim Boundary

- Scratch means direct label training; no distillation was used.
- This is software-only FI-2010 evidence, not HLS, P&R, or board validation.
- This does not touch TXF NEW1-NEW5.
- If fewer than 15 horizon/seed combinations are complete, the report is a partial time-budgeted run, not a full benchmark.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def configure_threads(args: argparse.Namespace) -> int:
    cpu = os.cpu_count() or 1
    if args.torch_threads == "auto":
        threads = max(1, cpu // 3)
    else:
        threads = max(1, int(args.torch_threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[50, 10, 20, 30, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss-mode", choices=["raw", "weighted"], default="weighted")
    parser.add_argument("--no-input-output-bias", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", default="auto")
    parser.add_argument("--time-budget-hours", type=float, default=6.0)
    parser.add_argument("--timing-batches", type=int, default=6)
    parser.add_argument("--eval-timing-batches", type=int, default=3)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_output_suffix(args.output_suffix)
    ensure_dirs()
    threads = configure_threads(args)
    device = torch.device(args.device)
    data = load_cache()
    run_start = time.time()
    state: dict[str, object] = {
        "phase": "08a_tcn_scratch",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start)),
        "wall_clock_budget_hours": args.time_budget_hours,
        "cpu_count": os.cpu_count(),
        "torch_threads": threads,
        "device": str(device),
        "horizons_order": args.horizons,
        "seeds": args.seeds,
        "channels": args.channels,
        "loss_mode": loss_mode_name(args),
        "input_output_bias": not args.no_input_output_bias,
        "output_suffix": args.output_suffix,
        "budget_exhausted": False,
        "stopped_before_next_combo": "",
        "new5_used": False,
    }
    RUN_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
    timing = timing_probe(data, args, device)
    print(json.dumps({"timing": timing}, indent=2), flush=True)
    if args.timing_only:
        state["timing_only"] = True
        RUN_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        write_report(state)
        return

    estimated_next = float(timing["estimated_combo_seconds_max_epochs"])
    for horizon in args.horizons:
        if horizon not in HORIZON_TO_IDX:
            raise ValueError(f"Unsupported horizon: {horizon}")
        for seed in args.seeds:
            combo = f"h{horizon}_seed{seed}"
            if args.resume_existing and metric_done(horizon, seed):
                print(f"skip existing {combo}", flush=True)
                continue
            elapsed = time.time() - run_start
            remaining = args.time_budget_hours * 3600.0 - elapsed
            if remaining < max(300.0, estimated_next * 0.8):
                state["budget_exhausted"] = True
                state["stopped_before_next_combo"] = combo
                RUN_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
                write_report(state)
                print(f"stop before {combo}: remaining={remaining:.1f}s estimated_next={estimated_next:.1f}s", flush=True)
                return
            metrics, curves = train_one(data, horizon, seed, args, device)
            append_rows(CURVES_CSV, curves)
            append_rows(METRICS_CSV, [metrics])
            summarize()
            state["last_completed_combo"] = combo
            state["last_completed_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            RUN_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
            write_report(state)
    state["completed_all_requested"] = True
    RUN_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
    write_report(state)


if __name__ == "__main__":
    main()
