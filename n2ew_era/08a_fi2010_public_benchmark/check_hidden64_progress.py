#!/usr/bin/env python3
"""hidden64 seed1 / seed2 進度檢視器。隨時可執行，不影響訓練。

    python check_hidden64_progress.py

資料來源是 console log 的逐 epoch JSON 行（訓練途中就有），
而不是 training_curves CSV（那個要整個 run 結束才會寫出）。

seed0 對照（單獨執行、無並行）：epoch 1 = 507.6 s
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
SEED0_EPOCH1 = 507.6


def epochs_from_log(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not (line.startswith("{") and '"epoch"' in line):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in d and "train_loss" in d:
            out.append(d)
    return out


def main() -> None:
    print("=" * 76)
    print(f"  hidden64 進度  {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print("=" * 76)

    for seed in (1, 2):
        log = ROOT / f"results_seed{seed}_console.log"
        d = ROOT / f"results_seed{seed}"
        print(f"\n--- seed {seed} ---")

        if not log.exists():
            print("  尚未啟動")
            continue

        err = re.search(r"(FATAL|Traceback).*", log.read_text(encoding="utf-8", errors="replace"))
        if err:
            print(f"  [ALERT] log 中出現錯誤: {err.group(0)[:100]}")

        st = d / "runner_status.json"
        if st.exists():
            s = json.loads(st.read_text(encoding="utf-8"))
            print(f"  已結束: {s['status']}  (總計 {s['wall_seconds']/3600:.2f} h)")

        eps = epochs_from_log(log)
        if not eps:
            age = (datetime.now().timestamp() - log.stat().st_mtime) / 60
            print(f"  執行中，第一個 epoch 尚未完成（log 最後更新於 {age:.0f} 分鐘前）")
            if age > 25:
                print("  [WARN] 超過 25 分鐘無新輸出，建議檢查行程是否還活著")
            continue

        last = eps[-1]
        n = last["epoch"]
        per_ep = [e["epoch_wall_seconds"] for e in eps if "epoch_wall_seconds" in e]
        recent = sum(per_ep[-3:]) / len(per_ep[-3:]) if per_ep else 0.0
        best = max(eps, key=lambda e: e["val_macro_f1"])
        since_best = n - best["epoch"]
        left = max(0, 10 - since_best)

        print(f"  已完成 epoch      : {n}")
        print(f"  最近 epoch 耗時   : {recent:.0f} s   (seed0 單獨跑 epoch1 = {SEED0_EPOCH1:.0f} s"
              f" -> 並行慢 {recent/SEED0_EPOCH1:.2f}x)")
        print(f"  累計              : {last['elapsed_seconds']/3600:.2f} h")
        print(f"  最新 val_macro_f1 : {last['val_macro_f1']:.4f}   train_loss {last['train_loss']:.5f}")
        print(f"  最佳 val_macro_f1 : {best['val_macro_f1']:.4f}  (epoch {best['epoch']})")
        print(f"  距最佳            : {since_best}  (patience 剩 {left})")
        if left > 0:
            print(f"  最快還要          : >= {left} epochs, 約 {left*recent/3600:.1f} h")

        age = (datetime.now().timestamp() - log.stat().st_mtime) / 60
        if not st.exists() and age > (recent / 60) * 2 + 10:
            print(f"  [WARN] log 已 {age:.0f} 分鐘未更新，超過一個 epoch 的兩倍，可能已卡住")

    # 隔離檢查
    shared = ROOT / "results" / "training_curves_v3_hidden64_capacity.csv"
    if shared.exists():
        import csv
        with shared.open(newline="", encoding="utf-8") as fh:
            seeds = {r["seed"] for r in csv.DictReader(fh)}
        if seeds - {"0"}:
            print(f"\n[ALERT] 共用 results/ 出現非 seed0 資料 {sorted(seeds)} —— 隔離失效，請立即中止")
        else:
            print(f"\n[OK] 共用 results/ 仍只有 seed0，隔離正常")
    print()


if __name__ == "__main__":
    main()
