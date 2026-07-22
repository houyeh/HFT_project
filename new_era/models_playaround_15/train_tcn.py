from __future__ import annotations

import json
import random
import sys
from csv import DictWriter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import (
    build_loader,
    compute_class_weight_tensor,
    concatenate_records,
    load_sequence_records,
    split_records_by_session,
)
from src.models.tcn import HLSFriendlyTCNClassifier


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def _evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        total += int(batch_y.numel())
        correct += int((preds == batch_y).sum().item())
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": correct / max(total, 1),
    }


def train_tcn(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)
    paths.ensure_dirs()
    save_experiment_config(paths, cfg)

    if not paths.dataset_npz.exists():
        raise FileNotFoundError(f"Dataset not found: {paths.dataset_npz}")

    _set_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)

    records = load_sequence_records(paths.dataset_npz)
    train_records, val_records, test_records = split_records_by_session(
        records,
        train_frac=cfg.split.train_session_frac,
        val_frac=cfg.split.val_session_frac,
    )
    if not val_records:
        raise RuntimeError("No validation sessions available; add more sessions or adjust split fractions.")

    x_train, y_train = concatenate_records(train_records, channel_first=True)
    x_val, y_val = concatenate_records(val_records, channel_first=True)

    weights_tensor = compute_class_weight_tensor(y_train, num_classes=cfg.num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    train_loader = build_loader(x_train, y_train, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = build_loader(x_val, y_val, batch_size=cfg.train.batch_size, shuffle=False)

    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)

    num_parameters = int(model.count_parameters())
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total += int(batch_y.numel())
            correct += int((preds == batch_y).sum().item())

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        val_metrics = _evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_acc = float(val_metrics["acc"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            torch.save(model.state_dict(), paths.model_weights_pth)

    def _dist(records) -> dict[str, int]:
        if not records:
            return {}
        y = np.concatenate([r.y for r in records], axis=0).astype(np.int64)
        counts = np.bincount(y, minlength=cfg.num_classes).tolist()
        return {str(i): int(counts[i]) for i in range(len(counts))}

    summary = {
        "device": str(device),
        "num_inputs": cfg.num_inputs,
        "num_classes": cfg.num_classes,
        "num_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "num_train_sessions": len(train_records),
        "num_val_sessions": len(val_records),
        "num_test_sessions": len(test_records),
        "num_train_samples": int(len(y_train)),
        "num_val_samples": int(len(y_val)),
        "num_test_samples": int(sum(len(r.y) for r in test_records)),
        "class_distribution_train": _dist(train_records),
        "class_distribution_val": _dist(val_records),
        "class_distribution_test": _dist(test_records),
    }

    write_json(paths.train_dir / "train_summary.json", summary)
    (paths.train_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (paths.train_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)

    checkpoint_meta = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "weights_path": str(paths.model_weights_pth),
        "num_inputs": cfg.num_inputs,
        "num_classes": cfg.num_classes,
        "num_parameters": num_parameters,
        "label_type": cfg.labels.label_type,
        "feature_count": cfg.features.feature_count,
        "selected_features": cfg.features.selected_features,
        "z_features": cfg.features.z_features,
        "window_size": cfg.window.window_size,
        "horizon": cfg.window.horizon,
        "threshold_ticks": cfg.window.threshold_ticks,
    }
    write_json(paths.checkpoints_dir / "checkpoint_metadata.json", checkpoint_meta)
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
    print(f"[train] reading from: {paths.dataset_npz}")
    summary = train_tcn(cfg, paths)
    print(f"[train] best_epoch={summary['best_epoch']} best_val_acc={summary['best_val_acc']:.4f}")
    print(f"[train] weights: {paths.model_weights_pth}")


if __name__ == "__main__":
    main()
