from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TRAIN_SCRIPT = ROOT / "02_train_phc_lite_stage1.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_phc_lite_stage1", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_json_list(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(x) for x in json.loads(text)]


def metric_key(row: dict[str, object]) -> tuple[int, int, str]:
    return (int(row["horizon"]), int(row["seed"]), str(row["loss_mode"]))


def best_curve_state(curves: list[dict[str, Any]], seed: int) -> tuple[int, float, int, float]:
    seed_rows = [
        row
        for row in curves
        if int(row["horizon"]) == 50 and int(row["seed"]) == seed and str(row["loss_mode"]) == "raw"
    ]
    if not seed_rows:
        return 0, -1.0, 0, 0.0
    final_epoch = max(int(row["epoch"]) for row in seed_rows)
    final_elapsed = max(float(row.get("elapsed_seconds", 0.0)) for row in seed_rows)
    best = max(seed_rows, key=lambda row: float(row["val_macro_f1"]))
    return int(best["epoch"]), float(best["val_macro_f1"]), final_epoch, final_elapsed


def checkpoint_path(seed: int, suffix: str) -> Path:
    return MODELS / f"phc_fi2010_h50_seed{seed}_raw{suffix}.pt"


def save_checkpoint(
    train_mod: Any,
    model: torch.nn.Module,
    path: Path,
    args: argparse.Namespace,
    seed: int,
    best_epoch: int,
    best_val_f1: float,
    class_weights: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": args.phase_label,
            "horizon": 50,
            "seed": seed,
            "loss_mode": "raw",
            "feature_dim": 40,
            "hidden_dim": args.hidden_dim,
            "num_classes": 3,
            "window": train_mod.WINDOW,
            "train_fraction": train_mod.TRAIN_FRACTION,
            "model_state_dict": model.state_dict(),
            "trainable_parameters": train_mod.count_trainable_parameters(model),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "class_weights": class_weights,
            "new5_used": False,
            "fast_exact_final_step": False,
            "fast_equiv_max_abs_diff": 0.0,
            "decay_effective_final": model.decay.detach().cpu().tolist(),
            "alpha_scalar_final": float(model.alpha_scalar.detach().cpu()),
            "resume_note": "v3 fair512 continuation helper; optimizer state is reinitialized because prior checkpoints did not store optimizer state",
        },
        path,
    )


def make_metric_row(
    train_mod: Any,
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    best_epoch: int,
    final_epoch: int,
    early_stop: bool,
    max_epoch_hit: bool,
    class_weights: list[float],
    checkpoint: Path,
    resumed_from_epoch: int,
    wallclock_start_iso: str,
    wallclock_end_iso: str,
) -> dict[str, Any]:
    horizon_idx = train_mod.HORIZONS.index(50)
    val_metrics = train_mod.evaluate(model, split["val_features"], split["val_labels"], horizon_idx, args.eval_batch_size, device, False)
    test_metrics = train_mod.evaluate(model, data["test_features_all"], data["test_labels_all"], horizon_idx, args.eval_batch_size, device, False)
    val_total = int(val_metrics["pred_0_up"]) + int(val_metrics["pred_1_stationary"]) + int(val_metrics["pred_2_down"])
    test_total = int(test_metrics["pred_0_up"]) + int(test_metrics["pred_1_stationary"]) + int(test_metrics["pred_2_down"])
    decay = model.decay.detach().cpu().numpy().astype(np.float64)
    best_epoch_distance_to_max = int(args.max_epochs - best_epoch)
    convergence_closed = bool(early_stop and best_epoch < (args.max_epochs - args.patience))
    return {
        "phase": args.phase_label,
        "horizon": 50,
        "seed": seed,
        "loss_mode": "raw",
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "torch_threads": args.torch_threads,
        "resumed_from_epoch": resumed_from_epoch,
        "decay_effective_final_json": json.dumps([float(x) for x in decay]),
        "decay_effective_final_mean": float(decay.mean()),
        "decay_effective_final_min": float(decay.min()),
        "decay_effective_final_max": float(decay.max()),
        "alpha_scalar_final": float(model.alpha_scalar.detach().cpu()),
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stop": early_stop,
        "max_epoch_hit": max_epoch_hit,
        "best_epoch_distance_to_max": best_epoch_distance_to_max,
        "convergence_closed": convergence_closed,
        "trainable_parameters": train_mod.count_trainable_parameters(model),
        "fast_exact_final_step": False,
        "fast_equiv_max_abs_diff": 0.0,
        "class_weights": json.dumps(class_weights),
        "device": str(device),
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_pred_0_up": val_metrics["pred_0_up"],
        "val_pred_1_stationary": val_metrics["pred_1_stationary"],
        "val_pred_2_down": val_metrics["pred_2_down"],
        "val_support_0_up": val_metrics["support_0_up"],
        "val_support_1_stationary": val_metrics["support_1_stationary"],
        "val_support_2_down": val_metrics["support_2_down"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_pred_0_up": test_metrics["pred_0_up"],
        "test_pred_1_stationary": test_metrics["pred_1_stationary"],
        "test_pred_2_down": test_metrics["pred_2_down"],
        "test_support_0_up": test_metrics["support_0_up"],
        "test_support_1_stationary": test_metrics["support_1_stationary"],
        "test_support_2_down": test_metrics["support_2_down"],
        "max_val_pred_share": max(int(val_metrics["pred_0_up"]), int(val_metrics["pred_1_stationary"]), int(val_metrics["pred_2_down"])) / max(val_total, 1),
        "max_test_pred_share": max(int(test_metrics["pred_0_up"]), int(test_metrics["pred_1_stationary"]), int(test_metrics["pred_2_down"])) / max(test_total, 1),
        "degenerate_prediction": bool(max(int(val_metrics["pred_0_up"]), int(val_metrics["pred_1_stationary"]), int(val_metrics["pred_2_down"])) / max(val_total, 1) >= args.degenerate_pred_share),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "wallclock_start_iso": wallclock_start_iso,
        "wallclock_end_iso": wallclock_end_iso,
        "verification_level": "stage1_v3_true_recurrent_software_measured_natural_patience_stop" if early_stop else "stage1_v3_true_recurrent_software_measured_max_epoch_boundary",
    }


def train_seed(
    train_mod: Any,
    data: dict[str, np.ndarray],
    existing_curves: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    split = train_mod.split_train_val(data["train_features_all"], data["train_labels_all"])
    horizon_idx = train_mod.HORIZONS.index(50)
    model = train_mod.PHCLiteFI2010Classifier(feature_dim=40, hidden_dim=args.hidden_dim, num_classes=3).to(device)
    class_weights = [1.0, 1.0, 1.0]
    loss_fn = torch.nn.CrossEntropyLoss()
    ckpt = checkpoint_path(seed, args.output_suffix)
    curve_best_epoch, curve_best_f1, final_logged_epoch, base_elapsed = best_curve_state(existing_curves, seed)
    resumed_from_epoch = 0
    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = args.patience
    if ckpt.exists() and final_logged_epoch > 0:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if payload.get("new5_used", False):
            raise RuntimeError(f"Hard rule violation: {ckpt} has new5_used=True")
        model.load_state_dict(payload["model_state_dict"])
        resumed_from_epoch = final_logged_epoch
        start_epoch = final_logged_epoch + 1
        best_epoch = int(payload.get("best_epoch", curve_best_epoch))
        best_val_f1 = float(payload.get("best_val_macro_f1", curve_best_f1))
        non_improve_epochs = max(0, final_logged_epoch - best_epoch)
        patience_left = max(1, args.patience - non_improve_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    starts_all = np.arange(0, split["train_features"].shape[0] - train_mod.WINDOW + 1, dtype=np.int64)
    curves: list[dict[str, Any]] = []
    heartbeats: list[dict[str, Any]] = []
    early_stop = False
    final_epoch = start_epoch - 1
    run_wall_start = datetime.now().isoformat(timespec="seconds")
    run_t0 = time.time()
    prev_epoch_end_monotonic: float | None = None
    prev_epoch_end_wall: str | None = None
    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start_monotonic = time.time()
        epoch_start_wall = datetime.now().isoformat(timespec="seconds")
        final_epoch = epoch
        model.train()
        np.random.shuffle(starts_all)
        train_loss_sum = 0.0
        train_seen = 0
        try:
            for pos in range(0, len(starts_all), args.batch_size):
                starts = starts_all[pos : pos + args.batch_size]
                y = train_mod.batch_labels(split["train_labels"], starts, horizon_idx, device)
                optimizer.zero_grad(set_to_none=True)
                logits = train_mod.logits_for_starts(model, split["train_features"], starts, device, False)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                train_loss_sum += float(loss.detach().cpu()) * int(y.numel())
                train_seen += int(y.numel())
        except (RuntimeError, OSError) as exc:
            error_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "seed": seed,
                "epoch": epoch,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            write_rows(RESULTS / "v3_fair512_runtime_errors.csv", [error_row])
            raise

        val_metrics = train_mod.evaluate(model, split["val_features"], split["val_labels"], horizon_idx, args.eval_batch_size, device, False)
        epoch_end_monotonic = time.time()
        epoch_end_wall = datetime.now().isoformat(timespec="seconds")
        inter_epoch_gap = 0.0 if prev_epoch_end_monotonic is None else epoch_end_monotonic - prev_epoch_end_monotonic
        elapsed_seconds = base_elapsed + (epoch_end_monotonic - run_t0)
        row = {
            "horizon": 50,
            "seed": seed,
            "loss_mode": "raw",
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_seconds": elapsed_seconds,
            "phase": args.phase_label,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "current_lr": args.lr,
            "fast_exact_final_step": False,
            "fast_equiv_max_abs_diff": 0.0,
            "wallclock_epoch_start_iso": epoch_start_wall,
            "wallclock_epoch_end_iso": epoch_end_wall,
            "epoch_wall_seconds": epoch_end_monotonic - epoch_start_monotonic,
            "inter_epoch_gap_seconds": inter_epoch_gap,
            "patience_left_before_update": patience_left,
        }
        curves.append(row)
        heartbeat = {
            "phase": args.phase_label,
            "seed": seed,
            "epoch": epoch,
            "wallclock_epoch_start_iso": epoch_start_wall,
            "wallclock_epoch_end_iso": epoch_end_wall,
            "epoch_wall_seconds": epoch_end_monotonic - epoch_start_monotonic,
            "inter_epoch_gap_seconds": inter_epoch_gap,
            "prev_epoch_end_iso": prev_epoch_end_wall or "",
            "val_macro_f1": val_metrics["macro_f1"],
        }
        heartbeats.append(heartbeat)
        print(json.dumps(row), flush=True)

        if float(val_metrics["macro_f1"]) > best_val_f1 + args.min_delta:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(train_mod, model, ckpt, args, seed, best_epoch, best_val_f1, class_weights)
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_stop = True
                break
        prev_epoch_end_monotonic = epoch_end_monotonic
        prev_epoch_end_wall = epoch_end_wall

    if not ckpt.exists():
        save_checkpoint(train_mod, model, ckpt, args, seed, best_epoch, best_val_f1, class_weights)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    best_epoch = int(payload.get("best_epoch", best_epoch))
    max_epoch_hit = bool((not early_stop) and final_epoch >= args.max_epochs)
    metric = make_metric_row(
        train_mod,
        model,
        data,
        split,
        device,
        args,
        seed,
        best_epoch,
        final_epoch,
        early_stop,
        max_epoch_hit,
        class_weights,
        ckpt,
        resumed_from_epoch,
        run_wall_start,
        datetime.now().isoformat(timespec="seconds"),
    )
    return metric, curves, heartbeats


def annotate_anomalies(heartbeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(heartbeats)
    intervals = [float(row["epoch_wall_seconds"]) for row in rows if float(row["epoch_wall_seconds"]) > 0.0]
    if not intervals:
        return rows
    median = float(np.median(np.array(intervals, dtype=np.float64)))
    normal = [x for x in intervals if x <= 3.0 * median]
    baseline = float(np.mean(normal)) if normal else float(np.mean(intervals))
    threshold = 3.0 * baseline
    for row in rows:
        seconds = float(row["epoch_wall_seconds"])
        row["baseline_epoch_seconds"] = baseline
        row["anomaly_threshold_seconds"] = threshold
        row["duration_anomaly_gt_3x_baseline"] = bool(seconds > threshold)
    return rows


def aggregate_primary(train_mod: Any, metrics: list[dict[str, Any]], phase_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary = train_mod.primary_rows(metrics, phase_label)
    summary = train_mod.summarize(metrics)
    convergence = train_mod.convergence_rows(metrics, phase_label)
    return primary, summary, convergence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--degenerate-pred-share", type=float, default=0.95)
    parser.add_argument("--output-suffix", default="_v3_fair512")
    parser.add_argument("--phase-label", default="08a_stage1_v3_fair512")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    train_mod = load_training_module()
    data = train_mod.load_cache()
    existing_curves = [dict(row) for row in read_rows(RESULTS / "training_curves_v3_fair512.csv")]
    all_curves = list(existing_curves)
    existing_metrics = [dict(row) for row in read_rows(RESULTS / "stage1_metrics_v3_fair512.csv")]
    metrics_by_key = {metric_key(row): dict(row) for row in existing_metrics}
    all_heartbeats = read_rows(RESULTS / "training_heartbeat_v3_fair512.csv")

    for seed in args.seeds:
        key = (50, seed, "raw")
        old = metrics_by_key.get(key)
        if old and truthy(old.get("convergence_closed", False)):
            print(json.dumps({"skip_converged": {"horizon": 50, "seed": seed, "loss_mode": "raw"}}), flush=True)
            continue
        metric, curves, heartbeats = train_seed(train_mod, data, all_curves, args, seed)
        all_curves.extend(curves)
        all_heartbeats.extend(heartbeats)
        metrics_by_key[key] = metric
        metrics = [metrics_by_key[k] for k in sorted(metrics_by_key)]
        write_rows(RESULTS / "stage1_metrics_v3_fair512.csv", metrics)
        write_rows(RESULTS / "training_curves_v3_fair512.csv", all_curves)
        write_rows(RESULTS / "training_heartbeat_v3_fair512.csv", annotate_anomalies([dict(row) for row in all_heartbeats]))
        primary, summary, convergence = aggregate_primary(train_mod, metrics, args.phase_label)
        write_rows(RESULTS / "stage1_primary_metrics_v3_fair512.csv", primary)
        write_rows(RESULTS / "stage1_summary_v3_fair512.csv", summary)
        write_rows(RESULTS / "stage1_convergence_v3_fair512.csv", convergence)

    metrics = [metrics_by_key[k] for k in sorted(metrics_by_key)]
    primary, summary, convergence = aggregate_primary(train_mod, metrics, args.phase_label)
    write_rows(RESULTS / "stage1_metrics_v3_fair512.csv", metrics)
    write_rows(RESULTS / "training_curves_v3_fair512.csv", all_curves)
    write_rows(RESULTS / "training_heartbeat_v3_fair512.csv", annotate_anomalies([dict(row) for row in all_heartbeats]))
    write_rows(RESULTS / "stage1_primary_metrics_v3_fair512.csv", primary)
    write_rows(RESULTS / "stage1_summary_v3_fair512.csv", summary)
    write_rows(RESULTS / "stage1_convergence_v3_fair512.csv", convergence)
    print(json.dumps({"metrics": metrics, "primary": primary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
