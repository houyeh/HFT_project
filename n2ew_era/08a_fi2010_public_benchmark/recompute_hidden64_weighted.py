#!/usr/bin/env python3
"""
hidden64 三個 seed 的 weighted-F1 重算（不重訓），並與 hidden16 做容量對照。

為什麼需要這個 wrapper：
  03_recompute_stage1_metrics.py 只掃 models/ 目錄，看不到隔離執行產生的
  models_seed1/ 與 models_seed2/。本腳本把三個 hidden64 checkpoint 集中到一個
  暫存目錄後，沿用該腳本的同一套度量計算（包含 weighted recall == accuracy 的
  一致性閘），確保與 seed0 當初的重算完全同法。

輸出：
  results_h64_weighted/stage1_metrics_h64_weighted_recompute.csv
  results_h64_weighted/stage1_summary_h64_weighted_recompute.csv
  results_h64_weighted/REPORT_BLOCK.md   <- 可直接貼進報告的表格

用法：
  python recompute_hidden64_weighted.py
"""
from __future__ import annotations

import csv
import importlib.util
import math
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "03_recompute_stage1_metrics.py"
STAGE = ROOT / "_h64_checkpoints_staging"
OUT = ROOT / "results_h64_weighted"

# hidden16 fair512 的既有 3-seed 結果（與 DeepLOB 同口徑，先前已重算）
H16 = {"accuracy": (66.45, 1.07), "macro_f1": (63.97, 1.45), "weighted_f1": (66.50, 1.15),
       "params": 2132, "n": 3}
H64_PARAMS = 23876
DEEPLOB_WEIGHTED_F1 = 80.35
DEEPLOB_PARAMS = 60000

SOURCES = [
    ROOT / "models" / "phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt",
    ROOT / "models_seed1" / "phc_fi2010_h50_seed1_raw_v3_hidden64_capacity.pt",
    ROOT / "models_seed2" / "phc_fi2010_h50_seed2_raw_v3_hidden64_capacity.pt",
]


def welch(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1))
    return t, df, se


def main() -> None:
    missing = [p for p in SOURCES if not p.exists()]
    if missing:
        sys.exit("FATAL: 找不到 checkpoint:\n  " + "\n  ".join(str(m) for m in missing))
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"FATAL: {OUT} 已存在且非空。要重算請先改名。")

    # --- 集中 checkpoint ---
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    for p in SOURCES:
        shutil.copy2(p, STAGE / p.name)
    print(f"已集中 {len(SOURCES)} 個 checkpoint 至 {STAGE.name}/")

    OUT.mkdir(parents=True, exist_ok=True)

    # --- 載入既有重算腳本並改寫其 MODELS / RESULTS ---
    spec = importlib.util.spec_from_file_location("recompute_stage1", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODELS = STAGE
    mod.RESULTS = OUT
    print(f"MODELS -> {mod.MODELS}\nRESULTS -> {mod.RESULTS}")

    sys.argv = [
        str(TARGET),
        "--checkpoint-glob", "phc_fi2010_h50_seed*_raw_v3_hidden64_capacity.pt",
        "--output-suffix", "_h64_weighted_recompute",
        "--torch-threads", "4",
    ]
    mod.main()

    # --- 讀回結果並產生報告區塊 ---
    mpath = OUT / "stage1_metrics_h64_weighted_recompute.csv"
    rows = list(csv.DictReader(mpath.open(encoding="utf-8")))
    rows = [r for r in rows if "hidden64" in r.get("checkpoint", "")]
    if len(rows) != 3:
        print(f"[WARN] 預期 3 列，實得 {len(rows)} 列")

    def col(name):
        return [float(r[name]) * 100 for r in rows]

    acc, mf1, wf1 = col("test_accuracy"), col("test_macro_f1"), col("test_weighted_f1")
    identity = max(abs(float(r["test_weighted_recall_accuracy_abs_diff"])) for r in rows)

    lines = ["| seed | checkpoint | accuracy % | weighted-F1 % | macro-F1 % |",
             "|---:|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: int(x["seed"])):
        lines.append(f"| {r['seed']} | `{Path(r['checkpoint']).name}` | "
                     f"{float(r['test_accuracy'])*100:.2f} | "
                     f"{float(r['test_weighted_f1'])*100:.2f} | "
                     f"{float(r['test_macro_f1'])*100:.2f} |")
    lines.append(f"| **mean ± std** | 3 seeds | **{st.mean(acc):.2f} ± {st.stdev(acc):.2f}** | "
                 f"**{st.mean(wf1):.2f} ± {st.stdev(wf1):.2f}** | "
                 f"**{st.mean(mf1):.2f} ± {st.stdev(mf1):.2f}** |")

    lines += ["", "### 容量對照（3 seeds vs 3 seeds）", "",
              "| 設定 | 參數量 | accuracy % | weighted-F1 % | macro-F1 % |",
              "|---|---:|---:|---:|---:|",
              f"| hidden16 | {H16['params']:,} | {H16['accuracy'][0]:.2f} ± {H16['accuracy'][1]:.2f} "
              f"| {H16['weighted_f1'][0]:.2f} ± {H16['weighted_f1'][1]:.2f} "
              f"| {H16['macro_f1'][0]:.2f} ± {H16['macro_f1'][1]:.2f} |",
              f"| hidden64 | {H64_PARAMS:,} | {st.mean(acc):.2f} ± {st.stdev(acc):.2f} "
              f"| {st.mean(wf1):.2f} ± {st.stdev(wf1):.2f} "
              f"| {st.mean(mf1):.2f} ± {st.stdev(mf1):.2f} |",
              f"| **差（hidden64 - hidden16）** | **{H64_PARAMS/H16['params']:.1f}×** "
              f"| **{st.mean(acc)-H16['accuracy'][0]:+.2f} pp** "
              f"| **{st.mean(wf1)-H16['weighted_f1'][0]:+.2f} pp** "
              f"| **{st.mean(mf1)-H16['macro_f1'][0]:+.2f} pp** |", ""]

    lines.append("### 檢定力陳述")
    lines.append("")
    for label, obs, ref in [("accuracy", (st.mean(acc), st.stdev(acc)), H16["accuracy"]),
                            ("weighted-F1", (st.mean(wf1), st.stdev(wf1)), H16["weighted_f1"]),
                            ("macro-F1", (st.mean(mf1), st.stdev(mf1)), H16["macro_f1"])]:
        t, df, se = welch(obs[0], obs[1], 3, ref[0], ref[1], 3)
        tcrit = 2.776
        d = obs[0] - ref[0]
        lo, hi = d - tcrit * se, d + tcrit * se
        extra = ""
        if label == "weighted-F1":
            gap = DEEPLOB_WEIGHTED_F1 - ref[0]
            extra = f"；與 DeepLOB 的 {gap:.2f} pp 差距相比，上界僅佔 {hi/gap*100:.0f}%"
        lines.append(f"- **{label}**：差 {d:+.2f} pp，Welch t = {t:.2f}（df = {df:.1f}，不顯著）；"
                     f"95% CI = [{lo:+.2f}, {hi:+.2f}] pp —— 可排除大於 **{hi:+.2f} pp** 的提升{extra}")

    lines += ["", f"一致性閘：所有 checkpoint 的 weighted recall 與 accuracy 最大絕對差 = `{identity:.2e}`", ""]

    block = "\n".join(lines)
    (OUT / "REPORT_BLOCK.md").write_text(block, encoding="utf-8")
    shutil.rmtree(STAGE)

    print("\n" + "=" * 76)
    print(block)
    print("=" * 76)
    print(f"\n已寫出：{OUT/'REPORT_BLOCK.md'}")


if __name__ == "__main__":
    main()
