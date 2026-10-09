from __future__ import annotations

import argparse
import csv
import ctypes
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

from spatial_conv_front_end import (
    SpatialPHCLiteFI2010Classifier,
    SpatialTCNFI2010Classifier,
    count_trainable_parameters,
)


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZONS = [10, 20, 30, 50, 100]
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZONS)}
CLASS_LABELS = [0, 1, 2]

METRICS_CSV = RESULTS / "spatial_frontend_probe_metrics.csv"
CURVES_CSV = RESULTS / "spatial_frontend_probe_training_curves.csv"
PREFLIGHT_CSV = RESULTS / "spatial_frontend_probe_resource_preflight.csv"
PARAMS_CSV = RESULTS / "spatial_frontend_probe_parameter_count.csv"
HISTORY_CSV = RESULTS / "spatial_frontend_probe_metrics_history.csv"
REPORT_MD = ROOT / "SPATIAL_CONV_FRONT_END_PROBE.md"

PHC_BASELINE_PARAMS = 2132
PHC_BASELINE_TEST_MACRO_F1 = 0.6231065899322604
PHC_BASELINE_LABEL = "PHC v3 hidden16 fair512 seed0 raw"
TCN_BASELINE_PARAMS = 2344
TCN_BASELINE_TEST_MACRO_F1 = 0.612344017719343
TCN_BASELINE_LABEL = "TCN small seed0 raw"


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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_or_replace(path: Path, new_rows: list[dict[str, object]], key_fields: list[str]) -> list[dict[str, object]]:
    existing: list[dict[str, object]] = [dict(row) for row in read_rows(path)]
    new_keys = {tuple(str(row.get(k, "")) for k in key_fields) for row in new_rows}
    kept = [row for row in existing if tuple(str(row.get(k, "")) for k in key_fields) not in new_keys]
    out = [*kept, *new_rows]
    write_rows(path, out)
    return out


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    existing: list[dict[str, object]] = [dict(row) for row in read_rows(path)]
    write_rows(path, [*existing, *rows])


def archive_matching_rows(
    path: Path,
    key_fields: list[str],
    key_values: dict[str, object],
    reason: str,
) -> None:
    existing: list[dict[str, object]] = [dict(row) for row in read_rows(path)]
    if not existing:
        return
    matched = [
        row
        for row in existing
        if all(str(row.get(field, "")) == str(key_values.get(field, "")) for field in key_fields)
    ]
    if not matched:
        return
    archived_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    archived_existing = [dict(row) for row in read_rows(HISTORY_CSV)]
    archive_identity_fields = [
        *key_fields,
        "best_epoch",
        "final_epoch",
        "slow_abort",
        "combo_seconds",
        "archive_reason",
    ]
    deduped: list[dict[str, object]] = []
    for row in matched:
        row["archived_at"] = archived_at
        row["archive_reason"] = reason
        identity = tuple(str(row.get(field, "")) for field in archive_identity_fields)
        exists = any(
            tuple(str(old.get(field, "")) for field in archive_identity_fields) == identity
            for old in archived_existing
        )
        if not exists:
            deduped.append(row)
    append_rows(HISTORY_CSV, deduped)


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
    model: torch.nn.Module,
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
        x = batch_windows(features, starts, device)
        y = batch_labels(labels, starts, horizon_idx, device)
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
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="weighted",
        zero_division=0,
    )
    pred_counts = np.bincount(y_pred, minlength=3)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "weighted_recall_accuracy_abs_diff": abs(float(weighted_recall) - float(accuracy_score(y_true, y_pred))),
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


def available_memory_mb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(MemoryStatusEx)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    if not ok:
        return None
    return float(stat.ullAvailPhys) / (1024.0 * 1024.0)


def run_preflight(sample_seconds: float = 5.0) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    suspicious: list[dict[str, object]] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    mem_mb = available_memory_mb()
    try:
        import psutil  # type: ignore
    except ImportError:
        rows.append(
            {
                "preflight_time": timestamp,
                "available_memory_mb": mem_mb,
                "id": os.getpid(),
                "process_name": "psutil_unavailable",
                "cpu_delta_5s": "",
                "working_set_mb": "",
                "path": "",
            }
        )
        return rows, suspicious

    before: dict[int, float] = {}
    processes: dict[int, Any] = {}
    for proc in psutil.process_iter(["pid", "name", "exe", "memory_info"]):
        try:
            cpu = proc.cpu_times()
            before[int(proc.info["pid"])] = float(cpu.user + cpu.system)
            processes[int(proc.info["pid"])] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(float(sample_seconds))
    for proc in psutil.process_iter(["pid", "name", "exe", "memory_info"]):
        try:
            pid = int(proc.info["pid"])
            cpu = proc.cpu_times()
            now_cpu = float(cpu.user + cpu.system)
            delta = now_cpu - before.get(pid, now_cpu)
            mem_info = proc.info.get("memory_info")
            working_mb = float(mem_info.rss) / (1024.0 * 1024.0) if mem_info is not None else ""
            path = proc.info.get("exe") or ""
            name = proc.info.get("name") or ""
            row = {
                "preflight_time": timestamp,
                "available_memory_mb": round(mem_mb, 1) if mem_mb is not None else "",
                "id": pid,
                "process_name": name,
                "cpu_delta_5s": round(delta, 3),
                "working_set_mb": round(working_mb, 1) if working_mb != "" else "",
                "path": path,
            }
            rows.append(row)
            lower = f"{name} {path}".lower()
            is_suspicious_name = any(token in lower for token in ["anaconda", "conda", "onedrive"])
            is_current = pid == os.getpid()
            if is_suspicious_name and not is_current and (delta >= 0.5 or (working_mb != "" and working_mb >= 1000.0)):
                suspicious.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda r: float(r["cpu_delta_5s"]) if r["cpu_delta_5s"] != "" else -1.0, reverse=True)
    return rows[:40], suspicious


def build_model(model_kind: str, args: argparse.Namespace) -> torch.nn.Module:
    if model_kind == "phc":
        return SpatialPHCLiteFI2010Classifier(
            front_channels=args.front_channels,
            hidden_dim=args.hidden_dim,
            num_classes=3,
        )
    if model_kind == "tcn":
        return SpatialTCNFI2010Classifier(
            front_channels=args.front_channels,
            channels=args.tcn_channels,
            num_classes=3,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            input_output_bias=not args.no_input_output_bias,
        )
    raise ValueError(f"unknown model_kind: {model_kind}")


def checkpoint_path(model_kind: str, horizon: int, seed: int, suffix: str) -> Path:
    return MODELS / f"{model_kind}_fi2010_h{horizon}_seed{seed}_raw{suffix}.pt"


def loss_fn_for(loss_mode: str, class_weights: torch.Tensor) -> torch.nn.Module:
    if loss_mode == "weighted":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    return torch.nn.CrossEntropyLoss()


def train_one(
    data: dict[str, np.ndarray],
    model_kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(args.seed)
    horizon_idx = HORIZON_TO_IDX[args.horizon]
    split = split_train_val(data["train_features_all"], data["train_labels_all"])
    model = build_model(model_kind, args).to(device)
    params = count_trainable_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if model_kind == "tcn":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr_min)
    class_weights = compute_class_weights(split["train_labels"], horizon_idx).to(device)
    loss_fn = loss_fn_for(args.loss_mode, class_weights)
    starts_all = np.arange(0, split["train_features"].shape[0] - WINDOW + 1, dtype=np.int64)
    best_path = checkpoint_path(model_kind, args.horizon, args.seed, args.output_suffix)

    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = args.patience
    start_epoch = 1
    resumed_from_checkpoint = False
    resume_source_best_epoch = ""
    resume_source_best_val_macro_f1 = ""
    if args.resume_from_existing:
        if not best_path.exists():
            raise FileNotFoundError(f"--resume-from-existing requested but checkpoint is missing: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_f1 = float(ckpt["best_val_macro_f1"])
        best_epoch = int(ckpt["best_epoch"])
        patience_left = args.resume_patience_left if args.resume_patience_left >= 0 else args.patience
        start_epoch = max(1, int(args.resume_start_epoch))
        resumed_from_checkpoint = True
        resume_source_best_epoch = str(best_epoch)
        resume_source_best_val_macro_f1 = str(best_val_f1)
    early_stop = False
    slow_abort = False
    curves: list[dict[str, object]] = []
    combo_t0 = time.time()
    previous_epoch_end: float | None = None
    final_epoch = 0

    for epoch in range(start_epoch, args.max_epochs + 1):
        final_epoch = epoch
        epoch_start = time.time()
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
        if scheduler is not None:
            scheduler.step()
            current_lr = float(scheduler.get_last_lr()[0])
        else:
            current_lr = float(args.lr)
        epoch_end = time.time()
        epoch_seconds = epoch_end - epoch_start
        inter_epoch_gap = epoch_seconds if previous_epoch_end is None else epoch_end - previous_epoch_end
        previous_epoch_end = epoch_end
        val_f1 = float(val_metrics["macro_f1"])
        improved = val_f1 > best_val_f1 + args.min_delta
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "phase": "08a_spatial_frontend_probe",
                    "model_kind": model_kind,
                    "horizon": args.horizon,
                    "seed": args.seed,
                    "front_channels": args.front_channels,
                    "front_output_dim": int(model.front_end.output_dim),  # type: ignore[attr-defined]
                    "hidden_dim": args.hidden_dim if model_kind == "phc" else "",
                    "tcn_channels": args.tcn_channels if model_kind == "tcn" else "",
                    "loss_mode": args.loss_mode,
                    "window": WINDOW,
                    "train_fraction": TRAIN_FRACTION,
                    "model_state_dict": model.state_dict(),
                    "trainable_parameters": params,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "class_weights": [float(x) for x in class_weights.detach().cpu().numpy()],
                    "new5_used": False,
                },
                best_path,
            )
        else:
            patience_left -= 1
        row = {
            "phase": "08a_spatial_frontend_probe",
            "model_kind": model_kind,
            "horizon": args.horizon,
            "seed": args.seed,
            "front_channels": args.front_channels,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "best_val_macro_f1_so_far": best_val_f1,
            "best_epoch_so_far": best_epoch,
            "current_lr": current_lr,
            "patience_left": patience_left,
            "epoch_wall_seconds": epoch_seconds,
            "inter_epoch_gap_seconds": inter_epoch_gap,
            "elapsed_seconds": epoch_end - combo_t0,
            "torch_threads": torch.get_num_threads(),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_source_best_epoch": resume_source_best_epoch,
            "resume_source_best_val_macro_f1": resume_source_best_val_macro_f1,
        }
        curves.append(row)
        print(json.dumps(row), flush=True)
        if epoch_seconds > args.max_expected_epoch_seconds:
            slow_abort = True
            print(
                json.dumps(
                    {
                        "slow_epoch_abort": True,
                        "model_kind": model_kind,
                        "epoch": epoch,
                        "epoch_wall_seconds": epoch_seconds,
                        "max_expected_epoch_seconds": args.max_expected_epoch_seconds,
                    }
                ),
                flush=True,
            )
            break
        if patience_left <= 0:
            early_stop = True
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was written for {model_kind}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_final = evaluate(model, split["val_features"], split["val_labels"], horizon_idx, args.eval_batch_size, device)
    test_final = evaluate(model, data["test_features_all"], data["test_labels_all"], horizon_idx, args.eval_batch_size, device)
    combo_seconds = time.time() - combo_t0
    baseline_params = PHC_BASELINE_PARAMS if model_kind == "phc" else TCN_BASELINE_PARAMS
    baseline_macro = PHC_BASELINE_TEST_MACRO_F1 if model_kind == "phc" else TCN_BASELINE_TEST_MACRO_F1
    baseline_label = PHC_BASELINE_LABEL if model_kind == "phc" else TCN_BASELINE_LABEL
    metric = {
        "phase": "08a_spatial_frontend_probe",
        "model_kind": model_kind,
        "architecture_variant": f"{model_kind}_spatial_conv_c{args.front_channels}",
        "horizon": args.horizon,
        "seed": args.seed,
        "loss_mode": args.loss_mode,
        "front_channels": args.front_channels,
        "front_output_dim": int(model.front_end.output_dim),  # type: ignore[attr-defined]
        "hidden_dim": args.hidden_dim if model_kind == "phc" else "",
        "tcn_channels": args.tcn_channels if model_kind == "tcn" else "",
        "input_output_bias": "" if model_kind == "phc" else (not args.no_input_output_bias),
        "best_epoch": int(ckpt["best_epoch"]),
        "final_epoch": final_epoch,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stop": early_stop,
        "slow_abort": slow_abort,
        "max_epoch_hit": (final_epoch >= args.max_epochs and not early_stop and not slow_abort),
        "trainable_parameters": params,
        "baseline_trainable_parameters": baseline_params,
        "parameter_delta": params - baseline_params,
        "parameter_delta_pct": (params - baseline_params) / baseline_params,
        "baseline_label": baseline_label,
        "baseline_test_macro_f1": baseline_macro,
        "test_macro_f1_delta_vs_baseline": float(test_final["macro_f1"]) - baseline_macro,
        "class_weights": json.dumps(ckpt["class_weights"]),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_source_best_epoch": resume_source_best_epoch,
        "resume_source_best_val_macro_f1": resume_source_best_val_macro_f1,
        "combo_seconds": combo_seconds,
        "checkpoint": str(best_path.relative_to(ROOT)),
        "val_accuracy": val_final["accuracy"],
        "val_macro_precision": val_final["macro_precision"],
        "val_macro_recall": val_final["macro_recall"],
        "val_macro_f1": val_final["macro_f1"],
        "val_weighted_precision": val_final["weighted_precision"],
        "val_weighted_recall": val_final["weighted_recall"],
        "val_weighted_f1": val_final["weighted_f1"],
        "val_weighted_recall_accuracy_abs_diff": val_final["weighted_recall_accuracy_abs_diff"],
        "val_pred_0_up": val_final["pred_0_up"],
        "val_pred_1_stationary": val_final["pred_1_stationary"],
        "val_pred_2_down": val_final["pred_2_down"],
        "test_accuracy": test_final["accuracy"],
        "test_macro_precision": test_final["macro_precision"],
        "test_macro_recall": test_final["macro_recall"],
        "test_macro_f1": test_final["macro_f1"],
        "test_weighted_precision": test_final["weighted_precision"],
        "test_weighted_recall": test_final["weighted_recall"],
        "test_weighted_f1": test_final["weighted_f1"],
        "test_weighted_recall_accuracy_abs_diff": test_final["weighted_recall_accuracy_abs_diff"],
        "test_pred_0_up": test_final["pred_0_up"],
        "test_pred_1_stationary": test_final["pred_1_stationary"],
        "test_pred_2_down": test_final["pred_2_down"],
        "new5_used": False,
        "verification_level": "fi2010_software_training_single_seed_no_hls",
    }
    return metric, curves


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def seconds(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def md_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    if not rows:
        return "_No rows yet._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_report() -> None:
    def truthy(value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes"}

    def fmt_pct(value: Any) -> str:
        try:
            return f"{100.0 * float(value):.2f}%"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_pct_plain(value: Any) -> str:
        try:
            return f"{100.0 * float(value):.2f}"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_delta(value: Any) -> str:
        try:
            return f"{100.0 * float(value):+.2f} pp"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_mean_std(mean_value: Any, std_value: Any, n_value: int) -> str:
        if n_value <= 1 or pd.isna(std_value):
            return f"{fmt_pct(mean_value)} +/- n/a"
        return f"{fmt_pct(mean_value)} +/- {fmt_pct_plain(std_value)}"

    def first_float(frame: pd.DataFrame, col: str, default: float | None = None) -> float | None:
        if frame.empty or col not in frame.columns:
            return default
        value = frame.iloc[0][col]
        if pd.isna(value):
            return default
        return float(value)

    def first_int(frame: pd.DataFrame, col: str, default: int | None = None) -> int | None:
        if frame.empty or col not in frame.columns:
            return default
        value = frame.iloc[0][col]
        if pd.isna(value):
            return default
        return int(value)

    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()
    history = pd.read_csv(HISTORY_CSV) if HISTORY_CSV.exists() else pd.DataFrame()
    params = pd.read_csv(PARAMS_CSV) if PARAMS_CSV.exists() else pd.DataFrame()
    preflight = pd.read_csv(PREFLIGHT_CSV) if PREFLIGHT_CSV.exists() else pd.DataFrame()
    if not metrics.empty:
        metrics = metrics[
            (metrics["horizon"].astype(int) == 50)
            & (metrics["front_channels"].astype(int) == 8)
        ].copy()

    baseline_by_kind: dict[str, dict[str, object]] = {
        "phc": {
            "label": "PHC v3 hidden16 fair512 raw",
            "n": 1,
            "params": PHC_BASELINE_PARAMS,
            "test_macro_mean": PHC_BASELINE_TEST_MACRO_F1,
            "test_macro_std": np.nan,
            "test_weighted_mean": np.nan,
            "test_weighted_std": np.nan,
            "source": "fallback constants in this script",
        },
        "tcn": {
            "label": "TCN small raw",
            "n": 1,
            "params": TCN_BASELINE_PARAMS,
            "test_macro_mean": TCN_BASELINE_TEST_MACRO_F1,
            "test_macro_std": np.nan,
            "test_weighted_mean": np.nan,
            "test_weighted_std": np.nan,
            "source": "fallback constants in this script",
        },
    }

    phc_primary_path = RESULTS / "stage1_primary_metrics_v3_fair512.csv"
    phc_weighted_path = RESULTS / "stage1_summary_v3_fair512_weighted_recompute.csv"
    if phc_primary_path.exists():
        phc_primary = pd.read_csv(phc_primary_path)
        phc_test = phc_primary[
            (phc_primary["split"].astype(str) == "test")
            & (phc_primary["horizon"].astype(int) == 50)
        ]
        if not phc_test.empty:
            baseline_by_kind["phc"].update(
                {
                    "n": first_int(phc_test, "n_seeds", 1),
                    "params": first_int(phc_test, "trainable_parameters", PHC_BASELINE_PARAMS),
                    "test_macro_mean": first_float(phc_test, "macro_f1_mean", PHC_BASELINE_TEST_MACRO_F1),
                    "test_macro_std": first_float(phc_test, "macro_f1_std", np.nan),
                    "source": "results/stage1_primary_metrics_v3_fair512.csv",
                }
            )
    if phc_weighted_path.exists():
        phc_weighted = pd.read_csv(phc_weighted_path)
        phc_weighted_test = phc_weighted[
            (phc_weighted["split"].astype(str) == "test")
            & (phc_weighted["horizon"].astype(int) == 50)
        ]
        if not phc_weighted_test.empty:
            baseline_by_kind["phc"].update(
                {
                    "test_weighted_mean": first_float(phc_weighted_test, "weighted_f1_mean", np.nan),
                    "test_weighted_std": first_float(phc_weighted_test, "weighted_f1_std", np.nan),
                }
            )

    tcn_metrics_path = RESULTS / "tcn_fi2010_metrics_small_raw.csv"
    tcn_weighted_path = RESULTS / "tcn_fi2010_metrics_weighted_recompute.csv"
    if tcn_metrics_path.exists():
        tcn_metrics = pd.read_csv(tcn_metrics_path)
        tcn_seed0 = tcn_metrics[(tcn_metrics["horizon"].astype(int) == 50)].head(1)
        if not tcn_seed0.empty:
            baseline_by_kind["tcn"].update(
                {
                    "n": 1,
                    "params": first_int(tcn_seed0, "trainable_parameters", TCN_BASELINE_PARAMS),
                    "test_macro_mean": first_float(tcn_seed0, "test_macro_f1", TCN_BASELINE_TEST_MACRO_F1),
                    "test_macro_std": np.nan,
                    "source": "results/tcn_fi2010_metrics_small_raw.csv (seed0 only)",
                }
            )
    if tcn_weighted_path.exists():
        tcn_weighted = pd.read_csv(tcn_weighted_path)
        tcn_weighted_seed0 = tcn_weighted[
            (tcn_weighted["model_type"].astype(str) == "tcn")
            & (tcn_weighted["architecture_variant"].astype(str) == "tcn_dilated_causal_conv")
            & (tcn_weighted["horizon"].astype(int) == 50)
            & (tcn_weighted["channels"].astype(int) == 8)
            & (tcn_weighted["loss_mode"].astype(str) == "raw_cross_entropy")
        ].head(1)
        if not tcn_weighted_seed0.empty:
            baseline_by_kind["tcn"].update(
                {
                    "test_weighted_mean": first_float(tcn_weighted_seed0, "test_weighted_f1", np.nan),
                    "test_weighted_std": np.nan,
                }
            )

    baseline_rows: list[dict[str, object]] = []
    for kind in ["phc", "tcn"]:
        b = baseline_by_kind[kind]
        baseline_rows.append(
            {
                "baseline": b["label"],
                "n": b["n"],
                "params": b["params"],
                "test_macro_F1": fmt_mean_std(b["test_macro_mean"], b["test_macro_std"], int(b["n"])),
                "test_weighted_F1": fmt_mean_std(b["test_weighted_mean"], b["test_weighted_std"], int(b["n"])),
                "source": b["source"],
            }
        )

    per_seed_rows: list[dict[str, object]] = []
    if not metrics.empty:
        for r in metrics.sort_values(["model_kind", "seed"]).itertuples():
            per_seed_rows.append(
                {
                    "model": "PHC+spatial" if r.model_kind == "phc" else "TCN+spatial",
                    "seed": int(r.seed),
                    "batch": int(r.batch_size),
                    "best": int(r.best_epoch),
                    "final": int(r.final_epoch),
                    "early_stop": truthy(r.early_stop),
                    "slow_abort": truthy(r.slow_abort),
                    "max_epoch": truthy(r.max_epoch_hit),
                    "resumed": truthy(getattr(r, "resumed_from_checkpoint", False)),
                    "val_macro": fmt_pct(r.val_macro_f1),
                    "val_weighted": fmt_pct(r.val_weighted_f1),
                    "test_macro": fmt_pct(r.test_macro_f1),
                    "test_weighted": fmt_pct(r.test_weighted_f1),
                    "seconds": seconds(r.combo_seconds),
                    "checkpoint": r.checkpoint,
                }
            )

    summary_rows: list[dict[str, object]] = []
    summary_by_kind: dict[str, dict[str, object]] = {}
    for kind in ["phc", "tcn"]:
        subset = metrics[metrics["model_kind"].astype(str) == kind].copy() if not metrics.empty else pd.DataFrame()
        if subset.empty:
            continue
        n = int(subset["seed"].nunique())
        macro_mean = float(subset["test_macro_f1"].mean())
        macro_std = float(subset["test_macro_f1"].std()) if n > 1 else np.nan
        weighted_mean = float(subset["test_weighted_f1"].mean())
        weighted_std = float(subset["test_weighted_f1"].std()) if n > 1 else np.nan
        val_macro_mean = float(subset["val_macro_f1"].mean())
        val_macro_std = float(subset["val_macro_f1"].std()) if n > 1 else np.nan
        val_weighted_mean = float(subset["val_weighted_f1"].mean())
        val_weighted_std = float(subset["val_weighted_f1"].std()) if n > 1 else np.nan
        b = baseline_by_kind[kind]
        macro_delta = macro_mean - float(b["test_macro_mean"])
        weighted_delta = weighted_mean - float(b["test_weighted_mean"]) if not pd.isna(b["test_weighted_mean"]) else np.nan
        all_early = bool(subset["early_stop"].map(truthy).all())
        any_slow = bool(subset["slow_abort"].map(truthy).any())
        any_max = bool(subset["max_epoch_hit"].map(truthy).any())
        batch_sizes = ",".join(str(int(x)) for x in sorted(subset["batch_size"].astype(int).unique()))
        summary_by_kind[kind] = {
            "n": n,
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std,
            "macro_delta": macro_delta,
            "weighted_delta": weighted_delta,
            "all_early": all_early,
            "any_slow": any_slow,
            "any_max": any_max,
            "batch_sizes": batch_sizes,
        }
        summary_rows.append(
            {
                "model": "PHC+spatial" if kind == "phc" else "TCN+spatial",
                "n": n,
                "batch_sizes": batch_sizes,
                "params": int(subset["trainable_parameters"].iloc[0]),
                "val_macro_F1": fmt_mean_std(val_macro_mean, val_macro_std, n),
                "val_weighted_F1": fmt_mean_std(val_weighted_mean, val_weighted_std, n),
                "test_macro_F1": fmt_mean_std(macro_mean, macro_std, n),
                "test_weighted_F1": fmt_mean_std(weighted_mean, weighted_std, n),
                "baseline": b["label"],
                "delta_macro": fmt_delta(macro_delta),
                "delta_weighted": fmt_delta(weighted_delta),
                "all_early_stop": all_early,
                "any_slow_abort": any_slow,
                "any_max_epoch": any_max,
                "total_hours": f"{float(subset['combo_seconds'].sum()) / 3600.0:.2f}",
            }
        )

    history_rows: list[dict[str, object]] = []
    if not history.empty:
        history = history[
            (history["horizon"].astype(int) == 50)
            & (history["front_channels"].astype(int) == 8)
        ].copy()
        for r in history.sort_values(["model_kind", "archived_at"]).itertuples():
            history_rows.append(
                {
                    "model": "PHC+spatial" if r.model_kind == "phc" else "TCN+spatial",
                    "seed": int(r.seed),
                    "batch": int(getattr(r, "batch_size", 0)),
                    "archived_at": getattr(r, "archived_at", ""),
                    "reason": getattr(r, "archive_reason", ""),
                    "best": int(r.best_epoch),
                    "final": int(r.final_epoch),
                    "slow_abort": truthy(r.slow_abort),
                    "test_macro": fmt_pct(r.test_macro_f1),
                    "test_weighted": fmt_pct(getattr(r, "test_weighted_f1", np.nan)),
                }
            )

    param_rows: list[dict[str, object]] = []
    if not params.empty:
        for r in params.sort_values(["model_kind"]).itertuples():
            param_rows.append(
                {
                    "model": r.model_kind,
                    "front_channels": int(r.front_channels),
                    "front_output_dim": int(r.front_output_dim),
                    "trainable_parameters": int(r.trainable_parameters),
                    "baseline_parameters": int(r.baseline_trainable_parameters),
                    "delta": f"{int(r.parameter_delta):+d}",
                }
            )

    preflight_rows: list[dict[str, object]] = []
    preflight_note = "No preflight rows recorded."
    if not preflight.empty:
        clean_labels = ["phc_pre", "phc_post", "tcn_pre", "tcn_post"]
        proc_names = preflight["process_name"].astype(str)
        labeled = preflight[preflight.get("preflight_label", pd.Series(dtype=str)).isin(clean_labels)].copy()
        if not labeled.empty:
            latest_snapshots = []
            for label in clean_labels:
                label_rows = labeled[labeled["preflight_label"] == label]
                if label_rows.empty:
                    continue
                latest_time = label_rows["preflight_time"].astype(str).max()
                latest_snapshots.append(label_rows[label_rows["preflight_time"].astype(str) == latest_time])
            latest = pd.concat(latest_snapshots, ignore_index=False) if latest_snapshots else labeled
            latest_proc_names = latest["process_name"].astype(str)
            top = (
                latest[latest_proc_names != "System Idle Process"]
                .sort_values(["preflight_label", "cpu_delta_5s"], ascending=[True, False])
                .groupby("preflight_label", as_index=False, group_keys=False)
                .head(3)
            )
            preflight_note = (
                "The table shows the latest `phc_pre/phc_post/tcn_pre/tcn_post` snapshots. "
                "Training-time process checks also found no active Vivado/HLS/xsim process names. "
                "Repeated labels can correspond to later seed runs; the full audit trail is in the CSV."
            )
        else:
            top = preflight[proc_names != "System Idle Process"].sort_values("cpu_delta_5s", ascending=False).head(8)
            preflight_note = "Only legacy preflight rows were available; interpret wall-clock timing with caution."
        for r in top.itertuples():
            label = getattr(r, "preflight_label", "")
            if pd.isna(label):
                label = ""
            preflight_rows.append(
                {
                    "label": label,
                    "time": getattr(r, "preflight_time"),
                    "avail_MB": getattr(r, "available_memory_mb"),
                    "pid": getattr(r, "id"),
                    "process": getattr(r, "process_name"),
                    "cpu_delta_5s": getattr(r, "cpu_delta_5s"),
                    "working_MB": getattr(r, "working_set_mb"),
                }
            )

    required_kinds = {"phc", "tcn"}
    completed_kinds = set(summary_by_kind)
    all_three = required_kinds.issubset(completed_kinds) and all(
        int(summary_by_kind[k]["n"]) == 3 for k in required_kinds
    )
    no_slow = all(not bool(summary_by_kind.get(k, {}).get("any_slow", True)) for k in required_kinds)
    no_max = all(not bool(summary_by_kind.get(k, {}).get("any_max", True)) for k in required_kinds)
    all_early = all(bool(summary_by_kind.get(k, {}).get("all_early", False)) for k in required_kinds)
    all_batch512 = all(str(summary_by_kind.get(k, {}).get("batch_sizes", "")) == "512" for k in required_kinds)
    phc_positive = "phc" in summary_by_kind and float(summary_by_kind["phc"]["macro_delta"]) > 0.0
    tcn_positive = "tcn" in summary_by_kind and float(summary_by_kind["tcn"]["macro_delta"]) > 0.0

    if all_three and no_slow and no_max and all_early and all_batch512:
        status = "software_probe_3seed_batch512_natural_early_stop_no_slow_abort"
        if phc_positive and tcn_positive and int(baseline_by_kind["tcn"]["n"]) >= 3:
            decision = (
                "3-seed confirmed positive, ready to cite as a robust result in the paper. "
                "Both spatial front-end variants beat their corresponding 3-seed baselines."
            )
        elif phc_positive and tcn_positive:
            decision = (
                "PHC+spatial is positive in a 3-seed vs 3-seed comparison. TCN+spatial is also "
                "positive against the existing TCN-small baseline, but that baseline is seed0 only "
                "in the current repo, so do not label the TCN comparison as 3-seed-vs-3-seed robust "
                "until TCN-small seed1/seed2 baselines exist."
            )
        elif phc_positive or tcn_positive:
            decision = (
                "3-seed result is mixed: one model improved and the other did not. Do not claim a "
                "stable cross-model spatial front-end improvement."
            )
        else:
            decision = "3-seed result is not positive; record as negative evidence and stop here."
    elif all_three:
        status = "software_probe_3seed_batch512_not_fully_clean"
        decision = (
            "All six seed/model combinations exist, but at least one run was not a clean natural "
            "early-stop batch512 run. Inspect the per-seed table before using the result."
        )
    else:
        status = "software_probe_3seed_incomplete"
        decision = "Partial: one or more requested seed/model combinations are still pending."

    report = f"""# Spatial Conv Front-End Probe

status: {status}

## Scope

This is a k=50 FI-2010 software training probe for PHC+spatial and TCN+spatial, expanded to seed0/seed1/seed2. It does not touch TXF NEW1-NEW5, does not modify `fi2010_phc_lite.py` or `fi2010_tcn.py`, and does not run HLS, csynth, implementation, or board work.

## Front-End Design

- Input reshape: `40 -> 10 levels x 4 values`.
- Pair convolution: `Conv1d(1 -> C, kernel_size=2, stride=2)` within each level, pairing `(ask_price, ask_volume)` and `(bid_price, bid_volume)`.
- Adjacent-level convolution: reshape the two side-pair outputs into `2C` channels over 10 levels, then apply `Conv1d(2C -> C, kernel_size=2, stride=2)` across adjacent price levels.
- Activation: `SiLU` after each convolution.
- Probe setting: `C=8`, so the front end returns `5*C=40` features and the downstream PHC/TCN entry width stays comparable to the existing baselines.

## Preflight

The preflight samples process CPU deltas over 5 seconds and records available memory before training. It is intended to catch Anaconda/OneDrive style background-load contamination before the run starts.

{md_table(preflight_rows, ["label", "time", "avail_MB", "pid", "process", "cpu_delta_5s", "working_MB"])}

{preflight_note}

## Parameter Counts

{md_table(param_rows, ["model", "front_channels", "front_output_dim", "trainable_parameters", "baseline_parameters", "delta"])}

## Baselines

{md_table(baseline_rows, ["baseline", "n", "params", "test_macro_F1", "test_weighted_F1", "source"])}

## Per-Seed Results

{md_table(per_seed_rows, ["model", "seed", "batch", "best", "final", "early_stop", "slow_abort", "max_epoch", "resumed", "val_macro", "val_weighted", "test_macro", "test_weighted", "seconds", "checkpoint"])}

## 3-Seed Summary

{md_table(summary_rows, ["model", "n", "batch_sizes", "params", "val_macro_F1", "val_weighted_F1", "test_macro_F1", "test_weighted_F1", "baseline", "delta_macro", "delta_weighted", "all_early_stop", "any_slow_abort", "any_max_epoch", "total_hours"])}

## Archived Superseded Runs

These rows are retained only as provenance. The old slow-abort rows were interrupted by `slow_epoch_abort=True`; the old TCN seed0 clean row used batch_size=2048 and was superseded by the batch512 consistency rerun.

{md_table(history_rows, ["model", "seed", "batch", "archived_at", "reason", "best", "final", "slow_abort", "test_macro", "test_weighted"])}

## Decision Gate

{decision}

## Files

- Module: `spatial_conv_front_end.py`
- Training script: `14_train_spatial_frontend_probe.py`
- Metrics: `results/spatial_frontend_probe_metrics.csv`
- Archived metrics: `results/spatial_frontend_probe_metrics_history.csv`
- Curves: `results/spatial_frontend_probe_training_curves.csv`
- Preflight: `results/spatial_frontend_probe_resource_preflight.csv`
- Parameter counts: `results/spatial_frontend_probe_parameter_count.csv`

## Claim Boundary

- These numbers are FI-2010 PyTorch/software training results only.
- Weighted F1 uses the reconciled sklearn weighted-average definition: `precision_recall_fscore_support(..., average="weighted")`.
- No HLS/csynth/resource implementation claim is made here; "parameters" is the only resource-like quantity recorded in this probe.
- PHC baseline has a 3-seed fair512 source in this repo. TCN-small raw baseline currently has only seed0 in this repo, so any TCN 3-seed-vs-baseline statement must carry that limitation.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["phc", "tcn", "both"], default="both")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--front-channels", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--tcn-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-input-output-bias", action="store_true", default=True)
    parser.add_argument("--loss-mode", choices=["raw", "weighted"], default="raw")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-suffix", default="_spatial_c8_probe")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--archive-reason",
        default="superseded_by_clean_rerun_after_vivado_overlap_slow_abort",
        help="Reason recorded when --force-rerun archives an existing metrics row.",
    )
    parser.add_argument("--resume-from-existing", action="store_true")
    parser.add_argument("--resume-start-epoch", type=int, default=1)
    parser.add_argument("--resume-patience-left", type=int, default=-1)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--abort-on-suspicious-preflight", action="store_true", default=True)
    parser.add_argument("--max-expected-epoch-seconds", type=float, default=240.0)
    return parser.parse_args()


def model_kinds_from_args(args: argparse.Namespace) -> list[str]:
    if args.model_kind == "both":
        return ["phc", "tcn"]
    return [args.model_kind]


def metric_exists(model_kind: str, args: argparse.Namespace) -> bool:
    for row in read_rows(METRICS_CSV):
        if (
            str(row.get("model_kind")) == model_kind
            and int(row.get("horizon", -1)) == args.horizon
            and int(row.get("seed", -1)) == args.seed
            and str(row.get("front_channels")) == str(args.front_channels)
        ):
            return True
    return False


def write_parameter_rows(args: argparse.Namespace) -> None:
    rows = []
    for kind in ["phc", "tcn"]:
        model = build_model(kind, args)
        params = count_trainable_parameters(model)
        baseline = PHC_BASELINE_PARAMS if kind == "phc" else TCN_BASELINE_PARAMS
        rows.append(
            {
                "phase": "08a_spatial_frontend_probe",
                "model_kind": kind,
                "front_channels": args.front_channels,
                "front_output_dim": int(model.front_end.output_dim),  # type: ignore[attr-defined]
                "trainable_parameters": params,
                "baseline_trainable_parameters": baseline,
                "parameter_delta": params - baseline,
                "parameter_delta_pct": (params - baseline) / baseline,
            }
        )
    write_rows(PARAMS_CSV, rows)


def record_preflight(label: str) -> list[dict[str, object]]:
    rows, suspicious = run_preflight()
    for row in rows:
        row["preflight_label"] = label
    append_rows(PREFLIGHT_CSV, rows)
    if suspicious:
        for row in suspicious:
            row["preflight_label"] = label
    return suspicious


def main() -> None:
    args = parse_args()
    if args.horizon not in HORIZON_TO_IDX:
        raise ValueError(f"Unsupported horizon: {args.horizon}")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    ensure_dirs()
    if not args.skip_preflight:
        suspicious = record_preflight(f"{args.model_kind}_pre")
        if suspicious and args.abort_on_suspicious_preflight:
            write_report()
            raise RuntimeError(f"Suspicious Anaconda/OneDrive background load detected: {suspicious}")
    write_parameter_rows(args)
    data = load_cache()
    device = torch.device(args.device)
    for kind in model_kinds_from_args(args):
        if metric_exists(kind, args) and not args.force_rerun and not args.resume_from_existing:
            print(json.dumps({"skip_existing": kind, "metrics": str(METRICS_CSV)}), flush=True)
            continue
        if args.force_rerun:
            archive_matching_rows(
                METRICS_CSV,
                ["model_kind", "horizon", "seed", "front_channels"],
                {
                    "model_kind": kind,
                    "horizon": args.horizon,
                    "seed": args.seed,
                    "front_channels": args.front_channels,
                },
                args.archive_reason,
            )
        metric, curves = train_one(data, kind, args, device)
        append_or_replace(METRICS_CSV, [metric], ["model_kind", "horizon", "seed", "front_channels"])
        curve_key = ["model_kind", "horizon", "seed", "front_channels", "epoch"]
        append_or_replace(CURVES_CSV, curves, curve_key)
        if not args.skip_preflight:
            record_preflight(f"{kind}_post")
        write_report()
    write_report()
    print(json.dumps({"metrics_csv": str(METRICS_CSV), "report": str(REPORT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
