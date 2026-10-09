from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "02_train_phc_lite_stage1.py"
HORIZON = 50
WEIGHTED_RECALL_ATOL = 1e-9


def load_training_module():
    spec = importlib.util.spec_from_file_location("fi2010_stage1_train_gpu_sweep", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training module: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment_record(seed: int, concurrent_processes: int) -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(props.total_memory),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "python_random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "torch_cuda_random_seed_all": seed,
        "concurrent_processes_declared": concurrent_processes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID", ""),
    }


def start_watchdog(run_dir: Path, limit_seconds: float, started_monotonic: float) -> threading.Thread:
    def watch() -> None:
        remaining = max(0.0, limit_seconds - (time.monotonic() - started_monotonic))
        time.sleep(remaining)
        elapsed = time.monotonic() - started_monotonic
        write_json(
            run_dir / "watchdog_abort.json",
            {
                "status": "self_aborted_wall_limit",
                "wall_time_seconds": elapsed,
                "wall_time_limit_seconds": limit_seconds,
            },
        )
        os._exit(124)

    thread = threading.Thread(target=watch, name="run-wall-limit-watchdog", daemon=True)
    thread.start()
    return thread


@torch.no_grad()
def weighted_metrics(train_mod, model, features, labels, horizon_idx: int, batch_size: int, device):
    model.eval()
    starts_all = np.arange(0, features.shape[0] - train_mod.WINDOW + 1, dtype=np.int64)
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for pos in range(0, len(starts_all), batch_size):
        starts = starts_all[pos : pos + batch_size]
        x = train_mod.batch_windows(features, starts, device)
        y = train_mod.batch_labels(labels, starts, horizon_idx, device)
        pred = torch.argmax(model(x), dim=1)
        true_parts.append(y.cpu().numpy())
        pred_parts.append(pred.cpu().numpy())
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=train_mod.CLASS_LABELS,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "weighted_recall_accuracy_abs_diff": abs(float(recall) - accuracy),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One deterministic FI-2010 GPU capacity run.")
    parser.add_argument("--mode", choices=["probe", "sweep"], required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--wall-limit-seconds", type=float, required=True)
    parser.add_argument("--concurrent-processes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU.")
    if args.mode == "probe" and args.max_epochs != 3:
        raise RuntimeError("The approved probe must use exactly max_epochs=3.")
    if args.batch_size != 512 or args.patience != 10 or args.lr != 1e-3:
        raise RuntimeError("Fixed-recipe violation: require batch=512, patience=10, lr=1e-3.")
    if args.max_epochs <= 0 or args.wall_limit_seconds <= 0:
        raise ValueError("max_epochs and wall_limit_seconds must be positive.")

    torch.set_num_threads(args.torch_threads)
    configure_determinism(args.seed)

    family = "gpu_probe" if args.mode == "probe" else "gpu_sweep"
    run_name = f"cap{args.hidden_dim}_seed{args.seed}"
    run_dir = ROOT / "results" / family / run_name
    model_dir = ROOT / "models" / family
    if run_dir.exists() or (model_dir / f"phc_fi2010_h50_cap{args.hidden_dim}_seed{args.seed}_gpu.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_name}")
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=True)

    started_wall = time.time()
    started_monotonic = time.monotonic()
    env = environment_record(args.seed, args.concurrent_processes)
    env.update(
        {
            "mode": args.mode,
            "hidden_dim": args.hidden_dim,
            "horizon": HORIZON,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "lr_scheduler": "none",
            "loss": "raw CrossEntropyLoss",
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "wall_time_limit_seconds": args.wall_limit_seconds,
            "started_unix_seconds": started_wall,
        }
    )
    write_json(run_dir / "environment.json", env)
    write_json(run_dir / "run_status.json", {"status": "running", **env})
    start_watchdog(run_dir, args.wall_limit_seconds, started_monotonic)

    train_mod = load_training_module()
    configure_determinism(args.seed)
    train_mod.CACHE = ROOT / "data" / "cache"
    train_mod.RESULTS = run_dir
    train_mod.MODELS = model_dir
    checkpoint = model_dir / f"phc_fi2010_h50_cap{args.hidden_dim}_seed{args.seed}_gpu.pt"
    train_mod.checkpoint_path = lambda horizon, seed, loss_mode, suffix: checkpoint
    train_mod.set_seed = configure_determinism

    train_args = argparse.Namespace(
        hidden_dim=args.hidden_dim,
        device="cuda:0",
        compile_model=False,
        lr=args.lr,
        weight_decay=0.0,
        grad_clip=5.0,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=1e-5,
        phase_label=f"08a_capacity_{family}",
        output_suffix="",
        degenerate_pred_share=0.95,
        fast_exact_final_step=False,
        fast_equiv_atol=1e-7,
    )
    data = train_mod.load_cache()
    metric, curves, saved_checkpoint = train_mod.train_one(data, HORIZON, args.seed, "raw", train_args)

    payload = torch.load(saved_checkpoint, map_location="cuda:0", weights_only=False)
    model = train_mod.PHCLiteFI2010Classifier(
        feature_dim=40,
        hidden_dim=args.hidden_dim,
        num_classes=3,
    ).to("cuda:0")
    model.load_state_dict(payload["model_state_dict"])
    split = train_mod.split_train_val(data["train_features_all"], data["train_labels_all"])
    horizon_idx = train_mod.HORIZONS.index(HORIZON)
    val_weighted = weighted_metrics(
        train_mod,
        model,
        split["val_features"],
        split["val_labels"],
        horizon_idx,
        args.eval_batch_size,
        torch.device("cuda:0"),
    )
    test_weighted = weighted_metrics(
        train_mod,
        model,
        data["test_features_all"],
        data["test_labels_all"],
        horizon_idx,
        args.eval_batch_size,
        torch.device("cuda:0"),
    )
    max_identity_error = max(
        val_weighted["weighted_recall_accuracy_abs_diff"],
        test_weighted["weighted_recall_accuracy_abs_diff"],
    )
    if max_identity_error > WEIGHTED_RECALL_ATOL:
        raise RuntimeError(
            f"weighted recall != accuracy: max_abs_diff={max_identity_error} > {WEIGHTED_RECALL_ATOL}"
        )

    elapsed = time.monotonic() - started_monotonic
    epoch_rows: list[dict[str, object]] = []
    previous_elapsed = 0.0
    for curve in curves:
        epoch_elapsed = float(curve["elapsed_seconds"])
        epoch_rows.append(
            {
                "horizon": int(curve["horizon"]),
                "hidden_dim": args.hidden_dim,
                "seed": int(curve["seed"]),
                "epoch": int(curve["epoch"]),
                "train_loss": float(curve["train_loss"]),
                "val_loss": float(curve["val_loss"]),
                "val_accuracy": float(curve["val_accuracy"]),
                "val_macro_precision": float(curve["val_macro_precision"]),
                "val_macro_recall": float(curve["val_macro_recall"]),
                "val_macro_f1": float(curve["val_macro_f1"]),
                "epoch_wall_seconds": epoch_elapsed - previous_elapsed,
                "elapsed_seconds": epoch_elapsed,
                "concurrent_processes_declared": args.concurrent_processes,
            }
        )
        previous_elapsed = epoch_elapsed
    metric.update(
        {
            "wall_time_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(int(metric["final_epoch"]), 1),
            "concurrent_processes_declared": args.concurrent_processes,
            "gpu_name": env["gpu_name"],
            "torch_version": env["torch_version"],
            "cuda_runtime_version": env["cuda_runtime_version"],
            "cudnn_version": env["cudnn_version"],
            "deterministic_algorithms": env["deterministic_algorithms"],
            "val_weighted_precision": val_weighted["weighted_precision"],
            "val_weighted_recall": val_weighted["weighted_recall"],
            "val_weighted_f1": val_weighted["weighted_f1"],
            "val_weighted_recall_accuracy_abs_diff": val_weighted[
                "weighted_recall_accuracy_abs_diff"
            ],
            "test_weighted_precision": test_weighted["weighted_precision"],
            "test_weighted_recall": test_weighted["weighted_recall"],
            "test_weighted_f1": test_weighted["weighted_f1"],
            "test_weighted_recall_accuracy_abs_diff": test_weighted[
                "weighted_recall_accuracy_abs_diff"
            ],
        }
    )
    write_csv(run_dir / "run_metrics.csv", [metric])
    write_csv(run_dir / "training_curves.csv", curves)
    write_csv(run_dir / "probe_epoch_metrics.csv", epoch_rows)
    write_json(
        run_dir / "run_status.json",
        {
            "status": "complete",
            "wall_time_seconds": elapsed,
            "seconds_per_epoch": metric["seconds_per_epoch"],
            "checkpoint": str(saved_checkpoint.relative_to(ROOT)),
            "weighted_recall_accuracy_max_abs_diff": max_identity_error,
            **env,
        },
    )
    print(json.dumps({"status": "complete", "run": run_name, "wall_time_seconds": elapsed}))


if __name__ == "__main__":
    main()
