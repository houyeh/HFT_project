from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def load_training_module():
    path = ROOT / "02_train_phc_lite_stage1.py"
    spec = importlib.util.spec_from_file_location("train_phc_lite_stage1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def fmt_pct(x: object) -> str:
    return f"{100.0 * float(x):.2f}%"


def md_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    train_mod = load_training_module()
    device = torch.device("cpu")
    data = train_mod.load_cache()
    split = train_mod.split_train_val(data["train_features_all"], data["train_labels_all"])
    checkpoint = MODELS / "phc_fi2010_h50_seed0_raw_v3_fair512.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("new5_used", False):
        raise RuntimeError("Hard rule violation: checkpoint has new5_used=True")

    model = train_mod.PHCLiteFI2010Classifier(feature_dim=40, hidden_dim=16, num_classes=3).to(device)
    model.load_state_dict(payload["model_state_dict"])
    horizon_idx = train_mod.HORIZONS.index(50)
    val_metrics = train_mod.evaluate(model, split["val_features"], split["val_labels"], horizon_idx, 1024, device, False)
    test_metrics = train_mod.evaluate(model, data["test_features_all"], data["test_labels_all"], horizon_idx, 1024, device, False)
    decay = model.decay.detach().cpu().numpy().astype(np.float64)
    alpha = float(model.alpha_scalar.detach().cpu())

    curves = read_jsonl(RESULTS / "train_v3_fair512_seed0.log")
    curve_rows = []
    for row in curves:
        curve_rows.append(
            {
                "horizon": row["horizon"],
                "seed": row["seed"],
                "loss_mode": row["loss_mode"],
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "val_accuracy": row["val_accuracy"],
                "val_macro_precision": row["val_macro_precision"],
                "val_macro_recall": row["val_macro_recall"],
                "val_macro_f1": row["val_macro_f1"],
                "elapsed_seconds": row["elapsed_seconds"],
                "phase": row["phase"],
                "max_epochs": row["max_epochs"],
                "patience": row["patience"],
                "current_lr": row["current_lr"],
                "fast_exact_final_step": row["fast_exact_final_step"],
                "fast_equiv_max_abs_diff": row["fast_equiv_max_abs_diff"],
            }
        )
    write_csv(RESULTS / "training_curves_v3_fair512.csv", curve_rows)
    write_csv(RESULTS / "training_curves_v3_fair512_partial_from_log.csv", curve_rows)

    best_curve = max(curves, key=lambda r: float(r["val_macro_f1"])) if curves else {}
    final_curve = curves[-1] if curves else {}
    metric_row = {
        "phase": "08a_stage1_v3_fair512_partial",
        "horizon": 50,
        "seed": 0,
        "loss_mode": "raw",
        "hidden_dim": 16,
        "batch_size": 512,
        "torch_threads": 2,
        "max_epochs": 60,
        "patience": 10,
        "completed_epochs_logged": final_curve.get("epoch", ""),
        "best_epoch_from_checkpoint": payload["best_epoch"],
        "best_epoch_from_curve": best_curve.get("epoch", ""),
        "stopped_by_wall_clock": True,
        "convergence_closed": False,
        "early_stop": False,
        "fast_exact_final_step": False,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "decay_effective_final_json": json.dumps([float(x) for x in decay]),
        "decay_effective_final_mean": float(decay.mean()),
        "decay_effective_final_min": float(decay.min()),
        "decay_effective_final_max": float(decay.max()),
        "alpha_scalar_final": alpha,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "curve_log": "results/train_v3_fair512_seed0.log",
        "verification_level": "stage1_v3_true_recurrent_software_measured_partial_wallclock_stop",
    }
    write_csv(RESULTS / "stage1_metrics_v3_fair512.csv", [metric_row])
    write_csv(RESULTS / "stage1_metrics_v3_fair512_partial.csv", [metric_row])

    v2_rows = read_csv(RESULTS / "stage1_metrics_v2.csv")
    speed_rows = read_csv(RESULTS / "stage1_metrics_v3_k50_speedprobe.csv")
    v2 = next(r for r in v2_rows if r["horizon"] == "50" and r["seed"] == "0" and r["loss_mode"] == "raw")
    speed = next(r for r in speed_rows if r["horizon"] == "50" and r["seed"] == "0" and r["loss_mode"] == "raw")

    compare_rows = [
        {
            "run": "v2 k=50 seed0 raw baseline",
            "batch": "512",
            "epochs": f"{v2['best_epoch']} / {v2['final_epoch']}",
            "status": "convergence_closed",
            "val macro-F1": fmt_pct(v2["val_macro_f1"]),
            "test macro-F1": fmt_pct(v2["test_macro_f1"]),
            "source": "results/stage1_metrics_v2.csv",
        },
        {
            "run": "v3 speedprobe k=50 seed0 raw",
            "batch": "128",
            "epochs": f"{speed['best_epoch']} / {speed['final_epoch']}",
            "status": "1_epoch_speedprobe_not_converged",
            "val macro-F1": fmt_pct(speed["val_macro_f1"]),
            "test macro-F1": fmt_pct(speed["test_macro_f1"]),
            "source": "results/stage1_metrics_v3_k50_speedprobe.csv",
        },
        {
            "run": "v3 fair512 k=50 seed0 raw",
            "batch": "512",
            "epochs": f"{payload['best_epoch']} / {final_curve.get('epoch', '')}",
            "status": "partial_wall_clock_stop_not_converged",
            "val macro-F1": fmt_pct(metric_row["val_macro_f1"]),
            "test macro-F1": fmt_pct(metric_row["test_macro_f1"]),
            "source": "results/stage1_metrics_v3_fair512.csv",
        },
    ]
    write_csv(RESULTS / "v3_fair512_comparison_seed0.csv", compare_rows)

    curve_focus = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": f"{float(row['train_loss']):.6f}",
            "val_loss": f"{float(row['val_loss']):.6f}",
            "val_macro_f1": fmt_pct(row["val_macro_f1"]),
            "elapsed_min": f"{float(row['elapsed_seconds']) / 60.0:.2f}",
        }
        for row in curves
        if int(row["epoch"]) <= 6 or int(row["epoch"]) in {10, 11, 16, 18, 22, 26, 30}
    ]
    decay_rows = [{"channel": i, "sigmoid_decay": f"{x:.9f}"} for i, x in enumerate(decay)]

    report = [
        "status: partial",
        "finding: v3_fair512_seed0_removes_batch_size_confound_and_beats_v2_but_did_not_converge_within_wallclock_budget",
        "# V3 Fair512 Report",
        "",
        "## Scope",
        "",
        "This report covers the FI-2010 PHC v3 fair batch-size control requested for `k=50`, `seed=0`, `loss_mode=raw`.",
        "",
        "- Architecture/training code was not modified.",
        "- TXF NEW1-NEW5 data was not used or touched.",
        "- The run used `batch_size=512`, matching the v2 baseline.",
        "- The run used `patience=10`, `max_epochs=60`, `torch_threads=2`, and no `--fast-exact-final-step`.",
        "- The process was stopped after exceeding the requested wall-clock budget; it did not naturally early stop.",
        "- Seeds 1 and 2 were not started because seed0 alone exceeded the wall-clock budget.",
        "",
        "## Preflight",
        "",
        "| check | result |",
        "| --- | --- |",
        "| Python environment | `.venv/Scripts/python.exe`, Torch 2.10.0+cpu |",
        "| visible logical cores | 8 via `[Environment]::ProcessorCount` |",
        "| torch thread limit | 2 |",
        "| `decay_param` is `nn.Parameter` | True |",
        "| `alpha_scalar` is `nn.Parameter` | True |",
        "| `named_buffers()` | empty |",
        "| initial effective decay | 0.899999976 on all 16 channels |",
        "| initial alpha_scalar | 0.25 |",
        "| `--fast-exact-final-step` | rejected at startup with RuntimeError, as expected for v3 |",
        "",
        "## Seed0 Result",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| completed logged epochs | {final_curve.get('epoch', '')} |",
        f"| best epoch | {payload['best_epoch']} |",
        f"| val macro-F1 at best checkpoint | {fmt_pct(metric_row['val_macro_f1'])} |",
        f"| test macro-F1 at best checkpoint | {fmt_pct(metric_row['test_macro_f1'])} |",
        f"| decay mean/min/max | {metric_row['decay_effective_final_mean']:.9f} / {metric_row['decay_effective_final_min']:.9f} / {metric_row['decay_effective_final_max']:.9f} |",
        f"| alpha_scalar final | {alpha:.9f} |",
        f"| checkpoint | `{metric_row['checkpoint']}` |",
        f"| curve log | `{metric_row['curve_log']}` |",
        "",
        "## Three-Way Comparison",
        "",
        md_table(compare_rows, ["run", "batch", "epochs", "status", "val macro-F1", "test macro-F1", "source"]),
        "",
        "## Curve Snapshot",
        "",
        md_table(curve_focus, ["epoch", "train_loss", "val_loss", "val_macro_f1", "elapsed_min"]),
        "",
        "## Decay Channels",
        "",
        md_table(decay_rows, ["channel", "sigmoid_decay"]),
        "",
        "## Honest Conclusion",
        "",
        "The fair512 control removes the original batch-size confound for seed0: with the same batch size as v2, the v3 recurrent architecture reaches a much higher best validation macro-F1 than both the v2 k=50/seed0/raw baseline and the old one-epoch batch128 speedprobe. However, the run did not close by early stopping within the wall-clock budget. The correct status is therefore partial but already strongly positive for seed0, not a fully converged 3-seed claim.",
        "",
    ]
    (ROOT / "V3_FAIR512_REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
