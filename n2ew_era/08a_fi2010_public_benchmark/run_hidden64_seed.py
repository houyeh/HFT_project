#!/usr/bin/env python3
"""
hidden64 容量對照：單一 seed 的隔離執行器。

為什麼需要這個 wrapper：
  13_train_hidden64_capacity.py 會對 results/ 底下的共用 CSV 做
  「讀取 -> 修改 -> 寫入」。兩個 seed 若同時以原腳本執行，會互相覆寫並
  把訓練曲線寫壞。本 wrapper 把該腳本的 RESULTS / MODELS 兩個模組層級
  全域變數改指到 per-seed 目錄，使兩個 process 可安全並行。

配方必須與既有 hidden64 seed0 完全一致（否則三個 seed 不可比）：
  batch 512 / raw CE / Adam / lr 1e-3 / 無 scheduler / patience 10
  max_epochs 120 / hidden 64 / horizon k=50 / eval_batch 1024
  torch_threads 2   <-- 注意：seed0 是用 2 執行緒訓練的，不要改

用法（兩個 process 並行，各 2 執行緒 = 共 4 核心）：
  python run_hidden64_seed.py --seed 1
  python run_hidden64_seed.py --seed 2
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
TARGET = ROOT / "13_train_hidden64_capacity.py"

# 與 seed0 相同的配方；這些值不接受命令列覆寫，避免手滑破壞可比性。
FIXED = {
    "max_epochs": 120,
    "patience": 10,
    "batch_size": 512,
    "eval_batch_size": 1024,
    "lr": 1e-3,
    "hidden_dim": 64,
    "torch_threads": 2,
    "device": "cpu",
    "output_suffix": "_v3_hidden64_capacity",
    "phase_label": "08a_stage1_v3_hidden64_capacity",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated single-seed runner for hidden64 capacity.")
    ap.add_argument("--seed", type=int, required=True, choices=[1, 2],
                    help="只允許 1 或 2；seed0 已存在，不得重跑覆寫")
    ap.add_argument("--dry-run", action="store_true", help="只印出解析後的路徑與參數，不實際訓練")
    args = ap.parse_args()

    if not TARGET.exists():
        sys.exit(f"FATAL: 找不到 {TARGET}")

    out_results = ROOT / f"results_seed{args.seed}"
    out_models = ROOT / f"models_seed{args.seed}"

    # 防呆：不得覆寫既有「訓練產物」。
    # 注意：dry-run 本身不寫檔，故此處不會被自己觸發；仍白名單化以防萬一。
    IGNORABLE = {"runner_banner.json"}
    if out_results.exists():
        leftovers = [f.name for f in out_results.iterdir() if f.name not in IGNORABLE]
        if leftovers:
            sys.exit(
                f"FATAL: {out_results} 已含訓練產物 {leftovers[:5]}。\n"
                f"       要重跑請先將該目錄改名，不要覆寫。")

    spec = importlib.util.spec_from_file_location("h64_capacity", TARGET)
    if spec is None or spec.loader is None:
        sys.exit(f"FATAL: 無法載入 {TARGET}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只定義，不執行 main()

    # --- 隔離：改寫模組層級全域變數 ---
    mod.RESULTS = out_results
    mod.MODELS = out_models

    argv = [
        str(TARGET),
        "--seeds", str(args.seed),
        "--max-epochs", str(FIXED["max_epochs"]),
        "--patience", str(FIXED["patience"]),
        "--batch-size", str(FIXED["batch_size"]),
        "--eval-batch-size", str(FIXED["eval_batch_size"]),
        "--lr", str(FIXED["lr"]),
        "--hidden-dim", str(FIXED["hidden_dim"]),
        "--torch-threads", str(FIXED["torch_threads"]),
        "--device", FIXED["device"],
        "--output-suffix", FIXED["output_suffix"],
        "--phase-label", FIXED["phase_label"],
    ]

    banner = {
        "seed": args.seed,
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

    # 僅真正執行時才落地啟動資訊（dry-run 寫檔會觸發上面的目錄防呆）
    (out_results / "runner_banner.json").write_text(
        json.dumps(banner, indent=2, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    sys.argv = argv
    status = "completed"
    try:
        mod.main()
    except BaseException as exc:          # noqa: BLE001 - 要記錄任何中止原因
        status = f"aborted: {type(exc).__name__}: {exc}"
        raise
    finally:
        (out_results / "runner_status.json").write_text(json.dumps({
            "seed": args.seed,
            "status": status,
            "wall_seconds": round(time.time() - t0, 1),
            "ended_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
