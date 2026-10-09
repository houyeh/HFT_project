#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修復 2026-10-03 隔離失敗所污染的 spatial front-end probe 產物。

## 污染原因

`14_train_spatial_frontend_probe.py` 於 import 時（第 37-42 行）即計算
METRICS_CSV / CURVES_CSV / PREFLIGHT_CSV / PARAMS_CSV / HISTORY_CSV / REPORT_MD
的絕對路徑。隔離包裝器覆寫 `mod.RESULTS` 無效，因此 C=32 的 CPU 試跑
（被 slow_epoch_abort 於第 1 epoch 砍掉）把結果寫進了共用檔案。

## 本腳本修復的三件事

1. `results/spatial_frontend_probe_metrics.csv`
   移除 front_channels != 8 的列（3 筆 C=32，best_epoch=1、slow_abort=True），
   並依本 repo 既有慣例搬進 history.csv（填入 archived_at / archive_reason）。

2. `results/spatial_frontend_probe_parameter_count.csv`
   被 C=64 的參數計數覆寫（23252/21224），C=8 的原值（2420/2632）遺失。
   自倖存的 metrics.csv 還原 C=8 列；C=64 的計數另存為獨立檔案（該數字本身有效
   且為第六章參數配對設計的依據，不刪除）。

3. `SPATIAL_CONV_FRONT_END_PROBE.md`
   「Parameter Counts」表被改寫為 C=32。還原為 C=8。
   其餘段落（Per-Seed Results、Decision Gate、Claim Boundary）經檢查未受污染。

## 不修復的項目（非本次污染）

- Baselines 表的 `n=1` 與 `test_weighted_F1 = nan`：為該腳本內建的 fallback
  常數，自始即為單 seed。正式報告的 +1.17 pp 使用的是 3v3 重算值，不是這個
  n=1 欄位（見下方驗證輸出）。此限制已寫入報告 §6.4。

## 使用

    python repair_spatial_probe_artifacts.py --dry-run   # 先看會動什麼
    python repair_spatial_probe_artifacts.py --apply
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

METRICS = RESULTS / "spatial_frontend_probe_metrics.csv"
HISTORY = RESULTS / "spatial_frontend_probe_metrics_history.csv"
CURVES = RESULTS / "spatial_frontend_probe_training_curves.csv"
PARAMS = RESULTS / "spatial_frontend_probe_parameter_count.csv"
PARAMS_C64 = RESULTS / "spatial_frontend_probe_parameter_count_c64_design.csv"
REPORT_MD = HERE / "SPATIAL_CONV_FRONT_END_PROBE.md"

VALID_FRONT_CHANNELS = "8"
ARCHIVE_REASON = (
    "discarded_c32_cpu_probe_slow_epoch_abort_epoch1_written_to_shared_csv_by_isolation_failure"
)

TPE = timezone(timedelta(hours=8))


def stamp() -> str:
    return datetime.now(TPE).strftime("%Y%m%dT%H%M%S%z")


def iso_now() -> str:
    return datetime.now(TPE).strftime("%Y-%m-%dT%H:%M:%S%z")


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        return list(rdr.fieldnames or []), list(rdr)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def backup(path: Path, tag: str, apply: bool) -> None:
    if not path.exists():
        return
    dst = path.with_name(f"{path.stem}.PRE_REPAIR_{tag}{path.suffix}")
    print(f"  [backup] {path.name} -> {dst.name}")
    if apply:
        shutil.copy2(path, dst)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)
    tag = stamp()

    print("=" * 78)
    print(f"spatial probe 產物修復　模式={'APPLY' if apply else 'DRY-RUN'}　{iso_now()}")
    print("=" * 78)

    for p in (METRICS, HISTORY, PARAMS, REPORT_MD):
        if not p.exists():
            print(f"[FATAL] 找不到 {p}")
            return 2

    # ---------------------------------------------------------------- 1. metrics
    print("\n[1] metrics.csv")
    m_fields, m_rows = read_csv(METRICS)
    keep = [r for r in m_rows if (r.get("front_channels") or "").strip() == VALID_FRONT_CHANNELS]
    drop = [r for r in m_rows if (r.get("front_channels") or "").strip() != VALID_FRONT_CHANNELS]

    print(f"  總列數 {len(m_rows)} -> 保留 {len(keep)}　移除 {len(drop)}")
    for r in drop:
        print(
            f"    [drop] {r.get('model_kind')} C={r.get('front_channels')} seed={r.get('seed')} "
            f"best_epoch={r.get('best_epoch')} slow_abort={r.get('slow_abort')} "
            f"test_macro_f1={r.get('test_macro_f1','')[:6]}"
        )

    if len(keep) != 6:
        print(f"  [FATAL] 期望保留 6 列（3 PHC + 3 TCN，C=8），實得 {len(keep)}。中止。")
        return 3
    kinds = sorted((r["model_kind"], r["seed"]) for r in keep)
    expect = sorted([("phc", s) for s in "012"] + [("tcn", s) for s in "012"])
    if kinds != expect:
        print(f"  [FATAL] 保留列的 (model, seed) 組合不符：{kinds}")
        return 3
    print("  [OK] 保留列為 phc/tcn x seed 0,1,2，C=8，全部 early_stop")

    if not drop:
        print("  [SKIP] 無需移除，metrics.csv 已乾淨")
    else:
        backup(METRICS, tag, apply)
        backup(HISTORY, tag, apply)
        h_fields, h_rows = read_csv(HISTORY)
        now = iso_now()
        for r in drop:
            r2 = dict(r)
            r2["archived_at"] = now
            r2["archive_reason"] = ARCHIVE_REASON
            h_rows.append(r2)
        merged = list(h_fields)
        for f in m_fields + ["archived_at", "archive_reason"]:
            if f not in merged:
                merged.append(f)
        print(f"  搬入 history.csv：{len(h_rows) - len(read_csv(HISTORY)[1])} 筆新增，"
              f"總計 {len(h_rows)} 筆")
        if apply:
            write_csv(HISTORY, merged, h_rows)
            write_csv(METRICS, m_fields, keep)
        print("  [OK] metrics.csv 僅留 6 筆 C=8；垃圾列已歸檔並標註原因")

    # ---------------------------------------------------------------- 2. curves
    print("\n[2] training_curves.csv")
    c_fields, c_rows = read_csv(CURVES)
    if "front_channels" in c_fields:
        c_keep = [r for r in c_rows if (r.get("front_channels") or "").strip() == VALID_FRONT_CHANNELS]
        c_drop = len(c_rows) - len(c_keep)
        print(f"  總列數 {len(c_rows)} -> 保留 {len(c_keep)}　移除 {c_drop}")
        if c_drop:
            backup(CURVES, tag, apply)
            if apply:
                write_csv(CURVES, c_fields, c_keep)
            print("  [OK] 已移除非 C=8 的 epoch 曲線")
        else:
            print("  [SKIP] 曲線檔無污染")
    else:
        print("  [SKIP] 無 front_channels 欄位")

    # ------------------------------------------------------------- 3. parameters
    print("\n[3] parameter_count.csv")
    p_fields, p_rows = read_csv(PARAMS)
    bad = [r for r in p_rows if (r.get("front_channels") or "").strip() != VALID_FRONT_CHANNELS]
    if not bad:
        print("  [SKIP] 已是 C=8，無需處理")
    else:
        for r in bad:
            print(f"    [found] C={r.get('front_channels')} "
                  f"{r.get('model_kind')} params={r.get('trainable_parameters')}")
        backup(PARAMS, tag, apply)
        # C=8 的真值自倖存的 metrics.csv 還原
        restored = []
        for kind in ("phc", "tcn"):
            src = next(r for r in keep if r["model_kind"] == kind)
            tp = int(src["trainable_parameters"])
            bp = int(src["baseline_trainable_parameters"])
            restored.append({
                "phase": src["phase"],
                "model_kind": kind,
                "front_channels": "8",
                "front_output_dim": src["front_output_dim"],
                "trainable_parameters": tp,
                "baseline_trainable_parameters": bp,
                "parameter_delta": tp - bp,
                "parameter_delta_pct": src["parameter_delta_pct"],
            })
            print(f"    [restore] {kind} C=8 params={tp} baseline={bp} delta=+{tp - bp}")
        print(f"  C=64 的設計用計數另存 -> {PARAMS_C64.name}（該數字有效，不刪除）")
        if apply:
            write_csv(PARAMS, p_fields, restored)
            write_csv(PARAMS_C64, p_fields, bad)
        print("  [OK] parameter_count.csv 還原為 C=8")

    # ---------------------------------------------------------------- 4. report
    print("\n[4] SPATIAL_CONV_FRONT_END_PROBE.md")
    md = REPORT_MD.read_text(encoding="utf-8")
    want_rows = {}
    for kind in ("phc", "tcn"):
        src = next(r for r in keep if r["model_kind"] == kind)
        tp, bp = int(src["trainable_parameters"]), int(src["baseline_trainable_parameters"])
        want_rows[kind] = f"| {kind} | 8 | {src['front_output_dim']} | {tp} | {bp} | +{tp - bp} |"

    changed = False
    new_md_lines = []
    for line in md.splitlines():
        m = re.match(r"^\|\s*(phc|tcn)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\+?\d+\s*\|\s*$", line)
        if m and m.group(2) != VALID_FRONT_CHANNELS:
            print(f"    [fix] {line.strip()}")
            print(f"       -> {want_rows[m.group(1)]}")
            new_md_lines.append(want_rows[m.group(1)])
            changed = True
        else:
            new_md_lines.append(line)
    if changed:
        backup(REPORT_MD, tag, apply)
        if apply:
            REPORT_MD.write_text("\n".join(new_md_lines) + "\n", encoding="utf-8")
        print("  [OK] Parameter Counts 表已還原為 C=8")
    else:
        print("  [SKIP] Parameter Counts 表已是 C=8")

    # ------------------------------------------------------------ 5. 驗證輸出
    print("\n" + "=" * 78)
    print("驗證：正式報告引用的 3v3 數字（不使用 CSV 內 n=1 的 *_delta_vs_baseline 欄位）")
    print("=" * 78)

    # 報告 §6.3.1 / §6.4 的 hidden16 三 seed 基線（已凍結，見 V3_FAIR512_REPORT.md）
    H16_MACRO = 0.6397
    H16_WEIGHTED = 0.6650

    for kind, label in (("phc", "PHC+spatial C=8"), ("tcn", "TCN+spatial C=8")):
        rows = sorted((r for r in keep if r["model_kind"] == kind), key=lambda r: r["seed"])
        macro = [float(r["test_macro_f1"]) for r in rows]
        weighted = [float(r["test_weighted_f1"]) for r in rows]
        print(f"\n{label}　n={len(rows)}　params={rows[0]['trainable_parameters']}")
        print(f"  per-seed test_macro_f1    : " + ", ".join(pct(v) for v in macro))
        print(f"  per-seed test_weighted_f1 : " + ", ".join(pct(v) for v in weighted))
        mm, sm = statistics.mean(macro), statistics.stdev(macro)
        mw, sw = statistics.mean(weighted), statistics.stdev(weighted)
        print(f"  macro-F1    = {pct(mm)} +/- {sm * 100:.2f}")
        print(f"  weighted-F1 = {pct(mw)} +/- {sw * 100:.2f}")
        if kind == "phc":
            print(f"  vs hidden16 3-seed 基線（macro {H16_MACRO * 100:.2f}%）"
                  f"：delta_macro = {(mm - H16_MACRO) * 100:+.2f} pp　"
                  f"<-- 報告 §6.4 的 +1.17 pp 應等於此值")
            print(f"  vs hidden16 3-seed 基線（weighted {H16_WEIGHTED * 100:.2f}%）"
                  f"：delta_weighted = {(mw - H16_WEIGHTED) * 100:+.2f} pp")

    print("\n[注意] CSV 的 `test_macro_f1_delta_vs_baseline` 欄位是對「hidden16 **seed0 單一** "
          "基線」算的（約 +2.83 pp 平均），**不可引用於報告**。報告一律使用上方的 3v3 值。")
    print("\n" + ("已套用。" if apply else "以上為 DRY-RUN，未改動任何檔案。加 --apply 執行。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
