#!/usr/bin/env python3
"""
空間前端規模掃描：隔離執行單一 (front_channels, seed) 組合。

目的
----
目前已知：
  容量軸（hidden 16 -> 64，+21,744 參數）  : macro-F1 -0.09 pp，可排除 >+2.76 pp
  表徵軸（spatial C=8，+288 參數）         : macro-F1 +1.17 pp（3v3，不顯著）
兩軸都只在一個尺度上測過。本掃描把表徵軸放大，並刻意安排一個
**參數配對點**：

  spatial C=64  約 23,252 參數   vs   hidden64  23,876 參數

兩者參數量幾乎相同，可直接回答「同樣的參數預算，花在加寬 vs 加表徵，哪個有用」。

為什麼需要這個 wrapper
--------------------
14_train_spatial_frontend_probe.py 寫入共用的 results/ 與 models/。
多個組合並行會互相覆寫，故把 RESULTS / MODELS 改指到 per-(C,seed) 目錄。

固定配方（與既有 C=8 探針一致，不可更動）
--------------------------------------
  batch 512 / raw CE / Adam / lr 1e-3 / patience 10 / hidden 16 / horizon 50
  model-kind = phc（TCN 基準只有 1 seed，不納入本次比較）

用法
----
  python run_spatial_scaling.py --channels 32 --seed 0
  python run_spatial_scaling.py --channels 32 --seed 1
  ...
或直接用 start_spatial_scaling.cmd 批次啟動。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "14_train_spatial_frontend_probe.py"

FIXED = {
    "model_kind": "phc",
    "horizon": 50,
    "hidden_dim": 16,
    "kernel_size": 3,
    "dropout": 0.1,
    "loss_mode": "raw",
    "max_epochs": 60,
    "patience": 10,
    "min_delta": 1e-5,
    "batch_size": 512,
    "eval_batch_size": 1024,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "weight_decay": 0.0,
    "grad_clip": 5.0,
    "torch_threads": 2,
}
ALLOWED_CHANNELS = {16, 32, 64}   # C=8 已存在，不重跑


def expected_params(c: int) -> int:
    front = (1 * c * 2 + c) + (2 * c * c * 2 + c)
    proj_delta = max((c * 5 - 40) * 16, 0)
    return 2132 + front + proj_delta


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated single (channels, seed) spatial scaling run.")
    ap.add_argument("--channels", type=int, required=True, choices=sorted(ALLOWED_CHANNELS))
    ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TARGET.exists():
        sys.exit(f"FATAL: 找不到 {TARGET}")

    tag = f"c{args.channels}_seed{args.seed}"
    out_results = ROOT / f"results_spatial_{tag}"
    out_models = ROOT / f"models_spatial_{tag}"

    # 失敗的 run 會留下 banner 與 status 兩個檔，兩者都不是訓練產物
    IGNORABLE = {"runner_banner.json", "runner_status.json"}
    if out_results.exists():
        leftovers = [f.name for f in out_results.iterdir() if f.name not in IGNORABLE]
        if leftovers:
            sys.exit(f"FATAL: {out_results} 已含訓練產物 {leftovers[:5]}。要重跑請先改名。")

    spec = importlib.util.spec_from_file_location("spatial_probe", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS = out_results
    mod.MODELS = out_models

    argv = [
        str(TARGET),
        "--model-kind", FIXED["model_kind"],
        "--horizon", str(FIXED["horizon"]),
        "--seed", str(args.seed),
        "--front-channels", str(args.channels),
        "--hidden-dim", str(FIXED["hidden_dim"]),
        "--kernel-size", str(FIXED["kernel_size"]),
        "--dropout", str(FIXED["dropout"]),
        "--loss-mode", FIXED["loss_mode"],
        "--max-epochs", str(FIXED["max_epochs"]),
        "--patience", str(FIXED["patience"]),
        "--min-delta", str(FIXED["min_delta"]),
        "--batch-size", str(FIXED["batch_size"]),
        "--eval-batch-size", str(FIXED["eval_batch_size"]),
        "--lr", str(FIXED["lr"]),
        "--lr-min", str(FIXED["lr_min"]),
        "--weight-decay", str(FIXED["weight_decay"]),
        "--grad-clip", str(FIXED["grad_clip"]),
        "--torch-threads", str(FIXED["torch_threads"]),
    ]

    banner = {
        "front_channels": args.channels,
        "seed": args.seed,
        "expected_trainable_parameters_estimate": expected_params(args.channels),
        "matched_comparison": ("hidden64 = 23,876 params" if args.channels == 64 else
                               "TCN scratch = 8,547 params" if args.channels == 32 else None),
        "resolved_RESULTS": str(mod.RESULTS),
        "resolved_MODELS": str(mod.MODELS),
        "shared_results_untouched": str(ROOT / "results"),
        "fixed_recipe": FIXED,
        "argv": argv[1:],
        "started_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    print("=" * 72, flush=True)
    print(json.dumps(banner, indent=2, ensure_ascii=False), flush=True)
    print("=" * 72, flush=True)

    if args.dry_run:
        print("DRY RUN：未啟動訓練，且未寫入任何檔案。", flush=True)
        return

    out_results.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)
    (out_results / "runner_banner.json").write_text(
        json.dumps(banner, indent=2, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    sys.argv = argv
    status = "completed"
    try:
        mod.main()
    except BaseException as exc:  # noqa: BLE001
        status = f"aborted: {type(exc).__name__}: {exc}"
        raise
    finally:
        (out_results / "runner_status.json").write_text(json.dumps({
            "front_channels": args.channels, "seed": args.seed, "status": status,
            "wall_seconds": round(time.time() - t0, 1),
            "ended_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
