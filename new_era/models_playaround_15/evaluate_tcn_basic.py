from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import concatenate_records, load_sequence_records, split_records_by_session
from src.models.tcn import HLSFriendlyTCNClassifier


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


@torch.no_grad()
def _predict_proba(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch_x = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        logits = model(batch_x)
        batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(batch_probs)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def evaluate_basic(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)
    paths.ensure_dirs()
    save_experiment_config(paths, cfg)

    if not paths.dataset_npz.exists():
        raise FileNotFoundError(f"Dataset not found: {paths.dataset_npz}")
    if not paths.model_weights_pth.exists():
        raise FileNotFoundError(f"Model weights not found: {paths.model_weights_pth}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    metrics: dict[str, object] = {"device": str(device), "splits": {}}
    for split_name, split_records in (("val", val_records), ("test", test_records)):
        if not split_records:
            continue

        x_split, y_split = concatenate_records(split_records, channel_first=True)
        prob = _predict_proba(model, x_split, device=device)
        pred = prob.argmax(axis=1) if len(prob) else np.zeros_like(y_split)

        metrics["splits"][split_name] = {
            "num_samples": int(len(y_split)),
            "accuracy": float(accuracy_score(y_split, pred)),
            "confusion_matrix": confusion_matrix(y_split, pred).tolist(),
            "classification_report": classification_report(y_split, pred, zero_division=0, output_dict=True),
        }

    write_json(paths.eval_dir / "basic_eval_metrics.json", metrics)
    return metrics


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
    metrics = evaluate_basic(cfg, paths)
    for split_name, split_metrics in metrics["splits"].items():  # type: ignore[union-attr]
        print(f"[eval] {split_name} accuracy={split_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

