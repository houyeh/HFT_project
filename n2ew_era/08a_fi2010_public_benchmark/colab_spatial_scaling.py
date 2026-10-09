#!/usr/bin/env python3
"""
Colab 版：空間前端規模掃描（C = 32, 64），含 CPU/GPU 移植一致性閘。

為什麼改用 Colab
---------------
本機實測 C=32 的單 epoch 需 733 秒（C=8 為 58 秒，成本隨 C^2 增長），
推算 C=64 約 2,900 秒/epoch -> 3 seeds 需約 108 小時，CPU 不可行。
空間卷積是 GPU 的強項，且 Colab 不消耗 Delta 的 SU 額度。

Colab 的額外好處：檔案系統與專案 repo 完全分離，不可能污染既有結果。

執行順序
-------
  第 0 階段  環境與決定性設定檢查
  第 1 階段  移植一致性閘：C=8 seed0 跑 3 epoch，與本機 CPU 前 3 epoch 比對
            （不通過就停止，不要浪費時間跑完整掃描）
  第 2 階段  C=32 x seed{0,1,2}
  第 3 階段  C=64 x seed{0,1,2}   <- 參數配對點，約 23,252 params vs hidden64 23,876
  第 4 階段  彙整並輸出可貼進報告的 markdown

每個 run 結束立刻把結果複製到 Drive，Colab 斷線最多只損失當前那一個 run。

用法（Colab 儲存格）
------------------
  !python colab_spatial_scaling.py --stage check
  !python colab_spatial_scaling.py --stage c32
  !python colab_spatial_scaling.py --stage c64
  !python colab_spatial_scaling.py --stage report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "14_train_spatial_frontend_probe.py"
DRIVE = Path("/content/drive/MyDrive/fi2010_spatial_scaling")

# 本機 CPU 的 C=8 phc seed0 前三 epoch（移植一致性閘的參考值）
CPU_REF_C8_SEED0 = {
    1: {"train_loss": 1.0702502296431846, "val_macro_f1": 0.21862157438090812},
    2: {"train_loss": 1.02571047397467,   "val_macro_f1": 0.4469634910321025},
    3: {"train_loss": 0.9690387007444601, "val_macro_f1": 0.4962513817461465},
}
GATE_REL_PASS = 1e-3        # epoch 1 train_loss 相對差 < 1e-3 視為移植正確
GATE_REL_INVESTIGATE = 1e-1

# 既有基準（3 seeds，本機 CPU）
BASE = {"params": 2132, "macro": (63.97, 1.45), "weighted": (66.50, 1.15)}
C8 = {"params": 2420, "macro": (65.14, 1.10), "weighted": (67.58, 1.00)}
HIDDEN64 = {"params": 23876, "macro": (63.88, 0.55), "weighted": (65.74, 0.57)}


def expected_params(c: int) -> int:
    front = (1 * c * 2 + c) + (2 * c * c * 2 + c)
    return 2132 + front + max((c * 5 - 40) * 16, 0)


def build_cmd(channels: int, seed: int, max_epochs: int, suffix: str) -> list[str]:
    """固定配方；唯一變因是 front-channels。device/preflight/slow-abort 為環境調整，不影響數值。"""
    return [
        sys.executable, str(TARGET),
        "--model-kind", "phc",
        "--horizon", "50",
        "--seed", str(seed),
        "--front-channels", str(channels),
        "--hidden-dim", "16",
        "--kernel-size", "3",
        "--dropout", "0.1",
        "--loss-mode", "raw",
        "--max-epochs", str(max_epochs),
        "--patience", "10",
        "--min-delta", "1e-5",
        "--batch-size", "512",
        "--eval-batch-size", "1024",
        "--lr", "1e-3",
        "--lr-min", "1e-5",
        "--weight-decay", "0.0",
        "--grad-clip", "5.0",
        "--torch-threads", "2",
        "--device", "cuda",
        "--skip-preflight",                      # Colab 無 OneDrive/Anaconda，守門無意義
        "--max-expected-epoch-seconds", "3600",  # GPU 上不會觸發，設高避免誤殺
        "--output-suffix", suffix,
    ]


def restore_from_drive() -> None:
    """斷線重連後，把 Drive 的 metrics 還原回 results/，讓 14_ 的 metric_exists
    能跳過已完成的 (seed, front_channels) 組合。沒有備份就視為全新開始。"""
    res = ROOT / "results"
    res.mkdir(parents=True, exist_ok=True)
    restored = []
    for name in ("spatial_frontend_probe_metrics.csv",
                 "spatial_frontend_probe_training_curves.csv",
                 "spatial_frontend_probe_parameter_count.csv"):
        src = DRIVE / name
        if src.exists():
            shutil.copy2(src, res / name)
            restored.append(name)
    if restored:
        done = []
        m = res / "spatial_frontend_probe_metrics.csv"
        if m.exists():
            for r in csv.DictReader(m.open(encoding="utf-8")):
                if r.get("model_kind") == "phc":
                    done.append(f"C={r.get('front_channels')} seed={r.get('seed')}")
        print(f"[restore] 自 Drive 還原 {len(restored)} 個檔案；已完成的組合會被跳過：")
        print(f"[restore]   {done}")
    else:
        print("[restore] Drive 無既有備份，視為全新開始")


def stage_check() -> None:
    import torch
    print("=" * 72)
    print(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "(unset)"),
    }, indent=2))
    print("=" * 72)
    if not torch.cuda.is_available():
        sys.exit("FATAL: 沒有 GPU。Colab 選單 -> 執行階段 -> 變更執行階段類型 -> T4 GPU")

    print("\n[移植一致性閘] C=8 seed0 跑 3 epoch，與本機 CPU 比對 ...\n", flush=True)
    t0 = time.time()
    r = subprocess.run(build_cmd(8, 0, max_epochs=3, suffix="_spatial_c8_gatecheck"),
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:]); sys.exit("FATAL: 一致性閘的 run 失敗")

    got = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"epoch"' in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train_loss" in d and "epoch" in d:
                got[int(d["epoch"])] = d

    bar = "=" * 72
    print(bar)
    if 1 not in got:
        print("  VERDICT: INCOMPLETE —— 沒有 epoch 1，無法判定"); print(bar); return
    cpu_v = CPU_REF_C8_SEED0[1]["train_loss"]
    gpu_v = float(got[1]["train_loss"])
    rel = abs(gpu_v - cpu_v) / abs(cpu_v)
    verdict = ("PASS" if rel < GATE_REL_PASS else
               "INVESTIGATE" if rel < GATE_REL_INVESTIGATE else "FAIL")
    print(f"  PORT CHECK VERDICT: {verdict}")
    print(f"  epoch 1 train_loss   CPU={cpu_v!r}")
    print(f"                       GPU={gpu_v!r}")
    print(f"  relative |delta| = {rel:.3e}   (PASS < {GATE_REL_PASS:g})")
    for e in (2, 3):
        if e in got:
            c, g = CPU_REF_C8_SEED0[e]["train_loss"], float(got[e]["train_loss"])
            print(f"  epoch {e} train_loss   CPU={c:.10f}  GPU={g:.10f}  rel={abs(g-c)/abs(c):.3e}")
    print(f"  gate run wall = {time.time()-t0:.0f} s  "
          f"(epoch1 = {got[1].get('epoch_wall_seconds', '?')} s，本機 CPU 為 106.3 s)")
    print(bar)
    if verdict != "PASS":
        print("\n不要進入第 2 階段。把上面的輸出貼出來診斷。")


def stage_run(channels: int) -> None:
    DRIVE.mkdir(parents=True, exist_ok=True)
    restore_from_drive()
    suffix = f"_spatial_c{channels}_gpu"
    print(f"C={channels}  預估參數量 ~{expected_params(channels):,}", flush=True)
    for seed in (0, 1, 2):
        print(f"\n{'='*72}\n  C={channels}  seed={seed}  開始\n{'='*72}", flush=True)
        t0 = time.time()
        # 即時串流：capture_output=True 會把輸出吞到結束才印，
        # 導致單一 seed 的 1-2 小時完全看不到進度。改用 Popen 逐行轉發。
        proc = subprocess.Popen(
            build_cmd(channels, seed, max_epochs=60, suffix=suffix),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        captured = []
        for line in proc.stdout:
            captured.append(line)
            line = line.strip()
            if line.startswith("{") and '"epoch"' in line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    print(line[:200], flush=True); continue
                if "train_loss" in d:
                    print(f"    ep {d['epoch']:>3}  {d.get('epoch_wall_seconds', 0):>7.1f}s  "
                          f"train {d['train_loss']:.5f}  val_macroF1 {d['val_macro_f1']:.4f}  "
                          f"best {d.get('best_val_macro_f1_so_far', 0):.4f}@{d.get('best_epoch_so_far','?')}  "
                          f"patience {d.get('patience_left','?')}", flush=True)
                else:
                    print(f"    {line[:200]}", flush=True)
            elif line:
                print(f"    {line[:200]}", flush=True)
        proc.wait()
        r_stdout = "".join(captured)
        if proc.returncode != 0:
            print(f"[WARN] C={channels} seed={seed} 失敗 (rc={proc.returncode})，繼續下一個")
        print(f"  wall = {time.time()-t0:.0f} s", flush=True)
        # 每個 run 結束立刻備份到 Drive，斷線最多損失當前這一個
        for name in ("spatial_frontend_probe_metrics.csv",
                     "spatial_frontend_probe_training_curves.csv",
                     "spatial_frontend_probe_parameter_count.csv"):
            src = ROOT / "results" / name
            if src.exists():
                shutil.copy2(src, DRIVE / name)
        (DRIVE / f"log_c{channels}_seed{seed}.txt").write_text(
            r_stdout[-200000:], encoding="utf-8")
        print(f"  已備份至 {DRIVE}", flush=True)


def stage_report() -> None:
    restore_from_drive()
    m = ROOT / "results" / "spatial_frontend_probe_metrics.csv"
    rows = [r for r in csv.DictReader(m.open(encoding="utf-8")) if r.get("model_kind") == "phc"]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("front_channels", "?"), []).append(r)

    lines = ["### 表徵軸：空間前端規模掃描（PHC，hidden=16 固定）", "",
             "| front_channels | 參數量 | n | test macro-F1 % | test weighted-F1 % |",
             "|---:|---:|---:|---:|---:|",
             f"| 0（無前端） | {BASE['params']:,} | 3 | {BASE['macro'][0]:.2f} ± {BASE['macro'][1]:.2f} "
             f"| {BASE['weighted'][0]:.2f} ± {BASE['weighted'][1]:.2f} |"]
    for c in sorted(groups, key=lambda x: int(x)):
        g = groups[c]
        if len(g) < 2:
            lines.append(f"| {c} | — | {len(g)} | （樣本不足，略） | |"); continue
        ma = [float(x["test_macro_f1"]) * 100 for x in g]
        we = [float(x["test_weighted_f1"]) * 100 for x in g]
        pc = g[0].get("trainable_parameters", expected_params(int(c)))
        lines.append(f"| {c} | {int(float(pc)):,} | {len(g)} | {st.mean(ma):.2f} ± {st.stdev(ma):.2f} "
                     f"| {st.mean(we):.2f} ± {st.stdev(we):.2f} |")
    lines += ["", "### 參數配對比較（約 23k 參數，兩種花法）", "",
              "| 設計 | 參數量 | macro-F1 % | weighted-F1 % |", "|---|---:|---:|---:|",
              f"| 容量軸：hidden64 | {HIDDEN64['params']:,} "
              f"| {HIDDEN64['macro'][0]:.2f} ± {HIDDEN64['macro'][1]:.2f} "
              f"| {HIDDEN64['weighted'][0]:.2f} ± {HIDDEN64['weighted'][1]:.2f} |"]
    if "64" in groups and len(groups["64"]) >= 2:
        g = groups["64"]
        ma = [float(x["test_macro_f1"]) * 100 for x in g]
        we = [float(x["test_weighted_f1"]) * 100 for x in g]
        lines.append(f"| 表徵軸：spatial C=64 | ~{expected_params(64):,} "
                     f"| {st.mean(ma):.2f} ± {st.stdev(ma):.2f} | {st.mean(we):.2f} ± {st.stdev(we):.2f} |")
    else:
        lines.append("| 表徵軸：spatial C=64 | — | （尚未完成） | |")
    block = "\n".join(lines)
    DRIVE.mkdir(parents=True, exist_ok=True)
    (DRIVE / "REPORT_BLOCK_spatial.md").write_text(block, encoding="utf-8")
    print(block)
    print(f"\n已寫出 {DRIVE/'REPORT_BLOCK_spatial.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "c32", "c64", "report"])
    a = ap.parse_args()
    if a.stage == "check":
        stage_check()
    elif a.stage == "c32":
        stage_run(32)
    elif a.stage == "c64":
        stage_run(64)
    else:
        stage_report()
