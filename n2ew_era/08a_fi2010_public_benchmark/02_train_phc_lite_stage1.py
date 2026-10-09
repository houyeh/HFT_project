from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fi2010_phc_lite import PHCLiteFI2010Classifier, count_trainable_parameters


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZONS = [10, 20, 30, 50, 100]
CLASS_LABELS = [0, 1, 2]
PUBLISHED_COMPARISONS = RESULTS / "published_comparisons.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def batch_final_features(features: np.ndarray, starts: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = starts + WINDOW - 1
    x = torch.from_numpy(np.ascontiguousarray(features[idx]))
    return x.to(device=device, dtype=torch.float32)


def batch_labels(labels: np.ndarray, starts: np.ndarray, horizon_idx: int, device: torch.device) -> torch.Tensor:
    y = labels[starts + WINDOW - 1, horizon_idx]
    return torch.from_numpy(np.ascontiguousarray(y)).to(device=device, dtype=torch.long)


def assert_fast_final_step_safe(model: PHCLiteFI2010Classifier) -> None:
    decay = getattr(model, "decay", None)
    alpha_scalar = getattr(model, "alpha_scalar", None)
    if decay is None or alpha_scalar is None:
        raise RuntimeError("fast final-step path requires decay and alpha_scalar buffers")
    if float(decay.detach().abs().max().cpu()) != 0.0:
        raise RuntimeError("fast final-step path is invalid because decay is not all zero")
    if float(alpha_scalar.detach().abs().max().cpu()) != 0.0:
        raise RuntimeError("fast final-step path is invalid because alpha_scalar is not zero")


def logits_for_starts(
    model: PHCLiteFI2010Classifier,
    features: np.ndarray,
    starts: np.ndarray,
    device: torch.device,
    fast_final_step: bool,
) -> torch.Tensor:
    if not fast_final_step:
        x = batch_windows(features, starts, device)
        return model(x)
    x_last = batch_final_features(features, starts, device)
    state = x_last.new_zeros((x_last.shape[0], model.hidden_dim))
    logits, _ = model.forward_step(x_last, state)
    return logits


@torch.no_grad()
def verify_fast_final_step_equivalence(
    model: PHCLiteFI2010Classifier,
    features: np.ndarray,
    device: torch.device,
    sample_count: int = 128,
) -> float:
    assert_fast_final_step_safe(model)
    n = min(sample_count, features.shape[0] - WINDOW + 1)
    starts = np.linspace(0, features.shape[0] - WINDOW, num=n, dtype=np.int64)
    model.eval()
    full = logits_for_starts(model, features, starts, device, fast_final_step=False)
    fast = logits_for_starts(model, features, starts, device, fast_final_step=True)
    return float((full - fast).detach().abs().max().cpu())


def compute_class_weights(labels: np.ndarray, horizon_idx: int) -> torch.Tensor:
    window_labels = labels[WINDOW - 1 :, horizon_idx]
    counts = np.bincount(window_labels, minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: PHCLiteFI2010Classifier,
    features: np.ndarray,
    labels: np.ndarray,
    horizon_idx: int,
    batch_size: int,
    device: torch.device,
    fast_final_step: bool = False,
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
        logits = logits_for_starts(model, features, starts, device, fast_final_step)
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


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_key(row: dict[str, object]) -> tuple[int, int, str]:
    return (int(row["horizon"]), int(row["seed"]), str(row["loss_mode"]))


def output_path(base_name: str, suffix: str) -> Path:
    stem, ext = base_name.rsplit(".", 1)
    return RESULTS / f"{stem}{suffix}.{ext}"


def phase_version(phase_label: str) -> str:
    if phase_label.endswith("_v3") or "v3" in phase_label:
        return "v3"
    if phase_label.endswith("_v2") or "v2" in phase_label:
        return "v2"
    return "v1"


def verification_level(phase_label: str) -> str:
    version = phase_version(phase_label)
    if version == "v3":
        return "stage1_v3_true_recurrent_software_measured"
    if version == "v2":
        return "stage1_v2_closed_recursion_software_measured"
    return "stage1_v1_software_measured"


def checkpoint_path(horizon: int, seed: int, loss_mode: str, suffix: str) -> Path:
    return MODELS / f"phc_fi2010_h{horizon}_seed{seed}_{loss_mode}{suffix}.pt"


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def train_one(
    data: dict[str, np.ndarray],
    horizon: int,
    seed: int,
    loss_mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    set_seed(seed)
    horizon_idx = HORIZONS.index(horizon)
    split = split_train_val(data["train_features_all"], data["train_labels_all"])
    device = torch.device(args.device)
    model = PHCLiteFI2010Classifier(feature_dim=40, hidden_dim=args.hidden_dim, num_classes=3).to(device)
    if args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    fast_equiv_max_abs_diff = 0.0
    if args.fast_exact_final_step:
        fast_equiv_max_abs_diff = verify_fast_final_step_equivalence(
            base_model,
            split_train_val(data["train_features_all"], data["train_labels_all"])["train_features"],
            device,
        )
        if fast_equiv_max_abs_diff > args.fast_equiv_atol:
            raise RuntimeError(
                f"fast final-step path failed equivalence check: max_abs_diff={fast_equiv_max_abs_diff}"
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if loss_mode == "weighted":
        weights = compute_class_weights(split["train_labels"], horizon_idx).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        class_weights = [float(x) for x in weights.detach().cpu().numpy()]
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        class_weights = [1.0, 1.0, 1.0]

    starts_all = np.arange(0, split["train_features"].shape[0] - WINDOW + 1, dtype=np.int64)
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = checkpoint_path(horizon, seed, loss_mode, args.output_suffix)
    curves: list[dict[str, object]] = []
    patience_left = args.patience
    early_stop = False
    final_epoch = 0
    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        final_epoch = epoch
        model.train()
        np.random.shuffle(starts_all)
        train_loss_sum = 0.0
        train_seen = 0
        for pos in range(0, len(starts_all), args.batch_size):
            starts = starts_all[pos : pos + args.batch_size]
            y = batch_labels(split["train_labels"], starts, horizon_idx, device)
            optimizer.zero_grad(set_to_none=True)
            logits = logits_for_starts(model, split["train_features"], starts, device, args.fast_exact_final_step)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(y.numel())
            train_seen += int(y.numel())

        val_metrics = evaluate(
            model,
            split["val_features"],
            split["val_labels"],
            horizon_idx,
            args.eval_batch_size,
            device,
            args.fast_exact_final_step,
        )
        row = {
            "horizon": horizon,
            "seed": seed,
            "loss_mode": loss_mode,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_seconds": time.time() - t0,
            "phase": args.phase_label,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "current_lr": args.lr,
            "fast_exact_final_step": args.fast_exact_final_step,
            "fast_equiv_max_abs_diff": fast_equiv_max_abs_diff,
        }
        curves.append(row)
        print(json.dumps(row), flush=True)

        if float(val_metrics["macro_f1"]) > best_val_f1 + args.min_delta:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            patience_left = args.patience
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "phase": args.phase_label,
                    "horizon": horizon,
                    "seed": seed,
                    "loss_mode": loss_mode,
                    "feature_dim": 40,
                    "hidden_dim": args.hidden_dim,
                    "num_classes": 3,
                    "window": WINDOW,
                    "train_fraction": TRAIN_FRACTION,
                    "model_state_dict": base_model.state_dict(),
                    "trainable_parameters": count_trainable_parameters(base_model),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "class_weights": class_weights,
                    "new5_used": False,
                    "fast_exact_final_step": args.fast_exact_final_step,
                    "fast_equiv_max_abs_diff": fast_equiv_max_abs_diff,
                    "decay_effective_final": base_model.decay.detach().cpu().tolist(),
                    "alpha_scalar_final": float(base_model.alpha_scalar.detach().cpu()),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_stop = True
                break

    payload = torch.load(best_path, map_location=device, weights_only=False)
    base_model.load_state_dict(payload["model_state_dict"])
    val_metrics = evaluate(
        model,
        split["val_features"],
        split["val_labels"],
        horizon_idx,
        args.eval_batch_size,
        device,
        args.fast_exact_final_step,
    )
    test_metrics = evaluate(
        model,
        data["test_features_all"],
        data["test_labels_all"],
        horizon_idx,
        args.eval_batch_size,
        device,
        args.fast_exact_final_step,
    )
    pred_total = (
        int(test_metrics["pred_0_up"])
        + int(test_metrics["pred_1_stationary"])
        + int(test_metrics["pred_2_down"])
    )
    max_pred_share = max(
        int(test_metrics["pred_0_up"]),
        int(test_metrics["pred_1_stationary"]),
        int(test_metrics["pred_2_down"]),
    ) / max(pred_total, 1)
    best_epoch_distance_to_max = int(args.max_epochs - best_epoch)
    max_epoch_hit = bool((not early_stop) and final_epoch >= args.max_epochs)
    convergence_closed = bool(early_stop and best_epoch < (args.max_epochs - args.patience))
    decay_effective_final = base_model.decay.detach().cpu().tolist()
    alpha_scalar_final = float(base_model.alpha_scalar.detach().cpu())
    summary = {
        "phase": args.phase_label,
        "horizon": horizon,
        "seed": seed,
        "loss_mode": loss_mode,
        "hidden_dim": args.hidden_dim,
        "decay_effective_final_json": json.dumps(decay_effective_final),
        "decay_effective_final_mean": float(np.mean(decay_effective_final)),
        "decay_effective_final_min": float(np.min(decay_effective_final)),
        "decay_effective_final_max": float(np.max(decay_effective_final)),
        "alpha_scalar_final": alpha_scalar_final,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stop": early_stop,
        "max_epoch_hit": max_epoch_hit,
        "best_epoch_distance_to_max": best_epoch_distance_to_max,
        "convergence_closed": convergence_closed,
        "trainable_parameters": count_trainable_parameters(base_model),
        "fast_exact_final_step": args.fast_exact_final_step,
        "fast_equiv_max_abs_diff": fast_equiv_max_abs_diff,
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
        "max_val_pred_share": max(
            int(val_metrics["pred_0_up"]),
            int(val_metrics["pred_1_stationary"]),
            int(val_metrics["pred_2_down"]),
        )
        / max(
            int(val_metrics["pred_0_up"])
            + int(val_metrics["pred_1_stationary"])
            + int(val_metrics["pred_2_down"]),
            1,
        ),
        "max_test_pred_share": max_pred_share,
        "degenerate_prediction": bool(
            max(
                int(val_metrics["pred_0_up"]),
                int(val_metrics["pred_1_stationary"]),
                int(val_metrics["pred_2_down"]),
            )
            / max(
                int(val_metrics["pred_0_up"])
                + int(val_metrics["pred_1_stationary"])
                + int(val_metrics["pred_2_down"]),
                1,
            )
            >= args.degenerate_pred_share
        ),
        "checkpoint": str(best_path.relative_to(ROOT)),
    }
    return summary, curves, best_path


def summarize(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for loss_mode in sorted({str(r["loss_mode"]) for r in metrics_rows}):
        for horizon in HORIZONS:
            subset = [r for r in metrics_rows if int(r["horizon"]) == horizon and str(r["loss_mode"]) == loss_mode]
            if not subset:
                continue
            for prefix in ["val", "test"]:
                f1s = np.array([float(r[f"{prefix}_macro_f1"]) for r in subset], dtype=np.float64)
                accs = np.array([float(r[f"{prefix}_accuracy"]) for r in subset], dtype=np.float64)
                ps = np.array([float(r[f"{prefix}_macro_precision"]) for r in subset], dtype=np.float64)
                rs = np.array([float(r[f"{prefix}_macro_recall"]) for r in subset], dtype=np.float64)
                rows.append(
                    {
                        "split": prefix,
                        "loss_mode": loss_mode,
                        "horizon": horizon,
                        "n_seeds": len(subset),
                        "accuracy_mean": float(accs.mean()),
                        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                        "macro_precision_mean": float(ps.mean()),
                        "macro_precision_std": float(ps.std(ddof=1)) if len(ps) > 1 else 0.0,
                        "macro_recall_mean": float(rs.mean()),
                        "macro_recall_std": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
                        "macro_f1_mean": float(f1s.mean()),
                        "macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                    }
                )
    return rows


def primary_rows(metrics_rows: list[dict[str, object]], phase_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        chosen: list[dict[str, object]] = []
        modes: list[str] = []
        for seed in sorted({int(r["seed"]) for r in metrics_rows if int(r["horizon"]) == horizon}):
            candidates = [
                r for r in metrics_rows if int(r["horizon"]) == horizon and int(r["seed"]) == seed
            ]
            weighted = [r for r in candidates if str(r["loss_mode"]) == "weighted"]
            raw = [r for r in candidates if str(r["loss_mode"]) == "raw"]
            if weighted:
                chosen.append(weighted[-1])
                modes.append("weighted")
            elif raw:
                chosen.append(raw[-1])
                modes.append("raw")
        if not chosen:
            continue
        primary_loss_mode = modes[0] if len(set(modes)) == 1 else "mixed_raw_weighted"
        for split in ["val", "test"]:
            acc = np.array([float(r[f"{split}_accuracy"]) for r in chosen], dtype=np.float64)
            pre = np.array([float(r[f"{split}_macro_precision"]) for r in chosen], dtype=np.float64)
            rec = np.array([float(r[f"{split}_macro_recall"]) for r in chosen], dtype=np.float64)
            f1 = np.array([float(r[f"{split}_macro_f1"]) for r in chosen], dtype=np.float64)
            rows.append(
                {
                    "model": "PHC-lite FI-2010",
                    "phase": phase_label,
                    "split": split,
                    "horizon": horizon,
                    "primary_loss_mode": primary_loss_mode,
                    "n_seeds": len(chosen),
                    "accuracy_mean": float(acc.mean()),
                    "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "macro_precision_mean": float(pre.mean()),
                    "macro_precision_std": float(pre.std(ddof=1)) if len(pre) > 1 else 0.0,
                    "macro_recall_mean": float(rec.mean()),
                    "macro_recall_std": float(rec.std(ddof=1)) if len(rec) > 1 else 0.0,
                    "macro_f1_mean": float(f1.mean()),
                    "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                    "trainable_parameters": int(chosen[0]["trainable_parameters"]),
                    "verification_level": verification_level(phase_label),
                }
            )
    return rows


def convergence_rows(metrics_rows: list[dict[str, object]], phase_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in sorted(metrics_rows, key=metric_key):
        convergence_closed = truthy(row.get("convergence_closed", False))
        max_epoch_hit = truthy(row.get("max_epoch_hit", False))
        early_stop = truthy(row.get("early_stop", False))
        if convergence_closed:
            status = "convergence_closed"
        elif max_epoch_hit:
            status = "max_epoch_boundary_not_closed"
        elif early_stop:
            status = "early_stop_but_best_epoch_near_boundary"
        else:
            status = "not_closed"
        rows.append(
            {
                "phase": phase_label,
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "loss_mode": str(row["loss_mode"]),
                "best_epoch": int(row["best_epoch"]),
                "final_epoch": int(row.get("final_epoch", row["best_epoch"])),
                "max_epochs": int(row.get("max_epochs", 0)),
                "patience": int(row.get("patience", 0)),
                "best_epoch_distance_to_max": int(row.get("best_epoch_distance_to_max", 0)),
                "early_stop": early_stop,
                "max_epoch_hit": max_epoch_hit,
                "convergence_closed": convergence_closed,
                "convergence_status": status,
                "checkpoint": str(row["checkpoint"]),
            }
        )
    return rows


def comparison_table(primary: list[dict[str, object]], phase_label: str, suffix: str) -> list[dict[str, object]]:
    test_primary = [r for r in primary if str(r["split"]) == "test"]
    f1_by_horizon = {int(r["horizon"]): 100.0 * float(r["macro_f1_mean"]) for r in test_primary}
    version = phase_version(phase_label)
    rows: list[dict[str, object]] = [
        {
            "model": f"PHC-lite FI-2010 {version}",
            "parameters": int(test_primary[0]["trainable_parameters"]) if test_primary else "",
            "input": "100x40 official NoAuction/Zscore LOB rows",
            "protocol": f"Setup 2, Stage 1 {version} software validation",
            "f1_k10": f1_by_horizon.get(10, ""),
            "f1_k20": f1_by_horizon.get(20, ""),
            "f1_k50": f1_by_horizon.get(50, ""),
            "source": f"This experiment; results/{output_path('stage1_primary_metrics.csv', suffix).name}",
            "source_url": "",
            "verification_level": verification_level(phase_label),
        }
    ]
    if PUBLISHED_COMPARISONS.exists():
        with PUBLISHED_COMPARISONS.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out = dict(row)
                out.setdefault("verification_level", "published")
                rows.append(out)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--loss-mode", choices=["raw", "weighted"], default="raw")
    parser.add_argument("--auto-weight-if-degenerate", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--degenerate-pred-share", type=float, default=0.95)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--fast-exact-final-step", action="store_true")
    parser.add_argument("--fast-equiv-atol", type=float, default=1e-7)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--phase-label", default="08a_stage1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast_exact_final_step:
        raise RuntimeError(
            "--fast-exact-final-step is invalid from v3 onward. It relied on "
            "decay and alpha_scalar being hardcoded-zero buffers (v2 "
            "architecture), which made the full-window forward and the "
            "final-timestep-only shortcut mathematically identical. v3 makes "
            "decay_param/alpha_scalar trainable and initializes them away "
            "from zero (initial_decay=0.90), so that equivalence no longer "
            "holds and this flag must not be used with v3 checkpoints. "
            "Train v3 without this flag; it will run the true step-wise "
            "recurrent forward over the full 100-step window."
        )
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = load_cache()
    base_model = PHCLiteFI2010Classifier(hidden_dim=args.hidden_dim)
    parameter_row = {
        "model": "PHC-lite FI-2010 classifier",
        "feature_dim": 40,
        "hidden_dim": args.hidden_dim,
        "num_classes": 3,
        "trainable_parameters": count_trainable_parameters(base_model),
    }
    write_rows(output_path("parameter_count.csv", args.output_suffix), [parameter_row])

    metrics_path = output_path("stage1_metrics.csv", args.output_suffix)
    curves_path = output_path("training_curves.csv", args.output_suffix)
    summary_path = output_path("stage1_summary.csv", args.output_suffix)
    primary_path = output_path("stage1_primary_metrics.csv", args.output_suffix)
    comparison_path = output_path("stage1_comparison_table.csv", args.output_suffix)
    convergence_path = output_path("stage1_convergence.csv", args.output_suffix)
    all_metrics: list[dict[str, object]] = read_existing_metrics(metrics_path) if args.resume_existing else []
    all_curves: list[dict[str, object]] = read_existing_metrics(curves_path) if args.resume_existing else []
    if args.force_rerun:
        target_keys = {(horizon, seed, args.loss_mode) for horizon in args.horizons for seed in args.seeds}
        all_metrics = [row for row in all_metrics if metric_key(row) not in target_keys]
        all_curves = [
            row
            for row in all_curves
            if (int(row["horizon"]), int(row["seed"]), str(row["loss_mode"])) not in target_keys
        ]
    completed = {metric_key(row) for row in all_metrics}
    for horizon in args.horizons:
        for seed in args.seeds:
            key = (horizon, seed, args.loss_mode)
            if key in completed:
                print(json.dumps({"skip_existing": {"horizon": horizon, "seed": seed, "loss_mode": args.loss_mode}}))
                summary = next(row for row in all_metrics if metric_key(row) == key)
            else:
                summary, curves, _ = train_one(data, horizon, seed, args.loss_mode, args)
                all_metrics.append(summary)
                all_curves.extend(curves)
                completed.add(key)
                write_rows(metrics_path, all_metrics)
                write_rows(curves_path, all_curves)
                write_rows(summary_path, summarize(all_metrics))
                write_rows(primary_path, primary_rows(all_metrics, args.phase_label))
                write_rows(comparison_path, comparison_table(primary_rows(all_metrics, args.phase_label), args.phase_label, args.output_suffix))
                write_rows(convergence_path, convergence_rows(all_metrics, args.phase_label))
            if args.auto_weight_if_degenerate and args.loss_mode == "raw" and str(summary["degenerate_prediction"]).lower() == "true":
                weighted_key = (horizon, seed, "weighted")
                if weighted_key in completed:
                    print(json.dumps({"skip_existing": {"horizon": horizon, "seed": seed, "loss_mode": "weighted"}}))
                    continue
                weighted_summary, weighted_curves, _ = train_one(data, horizon, seed, "weighted", args)
                all_metrics.append(weighted_summary)
                all_curves.extend(weighted_curves)
                completed.add(weighted_key)
                write_rows(metrics_path, all_metrics)
                write_rows(curves_path, all_curves)
                write_rows(summary_path, summarize(all_metrics))
                write_rows(primary_path, primary_rows(all_metrics, args.phase_label))
                write_rows(comparison_path, comparison_table(primary_rows(all_metrics, args.phase_label), args.phase_label, args.output_suffix))
                write_rows(convergence_path, convergence_rows(all_metrics, args.phase_label))

    write_rows(metrics_path, all_metrics)
    write_rows(curves_path, all_curves)
    write_rows(summary_path, summarize(all_metrics))
    primary = primary_rows(all_metrics, args.phase_label)
    write_rows(primary_path, primary)
    write_rows(comparison_path, comparison_table(primary, args.phase_label, args.output_suffix))
    write_rows(convergence_path, convergence_rows(all_metrics, args.phase_label))
    print(json.dumps({"metrics": all_metrics, "summary": summarize(all_metrics)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
