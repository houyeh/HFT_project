from __future__ import annotations

"""
00 - 一鍵運行（Run All Ablations）
===============================

用途：
- 你只要「跑這一支」就會把你想做的 ablation 全部跑完：
  dataset -> train (30 epoch) -> evaluation (完整圖表/CSV) -> backtest (val sweep 選參數 -> test 固定一次)
- 最後自動產出 summary / report，並把所有 experiment 的圖表集中到 `_charts/` 方便檢視

輸出位置（results root）：
`new_era/15_models_playaround/015_models_playaround_result/`

你最常改的地方：
- `new_era/15_models_playaround/_common.py`（window/horizon/threshold、fees、max inventory、seed、device…）

建議用法：
- 直接按 VSCode 的 Run 按鈕（或 `python 00_run_all_one_click.py`）
- 若你只想跑第一批 label（batch1）：加 `--batches 1`
- 若你要跑 batch1+batch2：加 `--batches 1,2`
"""

import argparse
import csv
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from _common import _ensure_project_root_on_path, build_cfg, print_experiment_banner

_ensure_project_root_on_path()

from new_era.models_playaround_15.backtest_val_sweep import backtest_val_sweep_then_test  # noqa: E402
from new_era.models_playaround_15.charts_gallery import build_charts_gallery  # noqa: E402
from new_era.models_playaround_15.dataset_generation import generate_dataset_npz  # noqa: E402
from new_era.models_playaround_15.env_snapshot import write_environment_snapshot  # noqa: E402
from new_era.models_playaround_15.evaluate_tcn_full import evaluate_full  # noqa: E402
from new_era.models_playaround_15.experiment_matrix import iter_experiment_configs  # noqa: E402
from new_era.models_playaround_15.generate_experiment_report import main as generate_report_main  # noqa: E402
from new_era.models_playaround_15.paths import build_experiment_paths, experiment_id, get_results_root  # noqa: E402
from new_era.models_playaround_15.train_tcn import train_tcn  # noqa: E402

_VALID_STAGES: tuple[str, ...] = ("dataset", "train", "eval", "backtest")


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _dataset_npz_has_sequences(npz_path: Path) -> bool:
    if not npz_path.exists():
        return False
    try:
        import numpy as np

        data = np.load(npz_path, allow_pickle=False)
        return any(str(k).endswith("_X") for k in data.files)
    except Exception:
        return False


def _parse_stage_set(text: str) -> set[str]:
    stages: set[str] = set()
    for part in (text or "").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in _VALID_STAGES:
            raise ValueError(f"Unknown stage: {name!r}. Valid: {', '.join(_VALID_STAGES)}")
        stages.add(name)
    if not stages:
        stages = set(_VALID_STAGES)
    return stages


def _stage_complete(paths, stage: str) -> bool:
    if stage == "dataset":
        return _dataset_npz_has_sequences(paths.dataset_npz)
    if stage == "train":
        return paths.model_weights_pth.exists()
    if stage == "eval":
        return (paths.eval_dir / "evaluation_summary.json").exists()
    if stage == "backtest":
        return (paths.backtest_dir / "test_final_results.csv").exists()
    raise ValueError(f"Unknown stage: {stage!r}")


def _all_stages_complete(paths, stages: set[str]) -> bool:
    return all(_stage_complete(paths, s) for s in _VALID_STAGES if s in stages)


def _run_one(
    cfg,
    *,
    force: bool,
    stages: set[str],
    force_backtest: bool,
) -> dict[str, object]:
    paths = build_experiment_paths(cfg)
    paths.ensure_dirs()
    eid = experiment_id(cfg)

    if (not force) and _all_stages_complete(paths, stages) and (not (force_backtest and "backtest" in stages)):
        return {"experiment_id": eid, "status": "skipped(already_done)"}

    # 分段 resume：缺什麼補什麼（除非你自己刪檔）
    if "dataset" in stages and (force or (not _stage_complete(paths, "dataset"))):
        generate_dataset_npz(cfg, paths)
    if "train" in stages and (force or (not _stage_complete(paths, "train"))):
        train_tcn(cfg, paths)
    if "eval" in stages and (force or (not _stage_complete(paths, "eval"))):
        evaluate_full(cfg, paths)
    if "backtest" in stages and (force or force_backtest or (not _stage_complete(paths, "backtest"))):
        backtest_val_sweep_then_test(cfg, paths)

    return {"experiment_id": eid, "status": "ok"}


def _write_run_log_row(log_csv: Path, row: dict[str, object], *, write_header_if_needed: bool = True) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = log_csv.exists()
    with log_csv.open("a", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header_if_needed and not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=str, default="1", help="例如：1 或 1,2")
    parser.add_argument("--feature-counts", type=str, default="6,7,8,9")
    parser.add_argument(
        "--stages",
        type=str,
        default="dataset,train,eval,backtest",
        help="逗號分隔：dataset,train,eval,backtest（預設全跑）",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="強制重跑（不管檔案是否已存在）")
    parser.add_argument(
        "--only-backtest",
        action="store_true",
        help="只重跑 backtest（不重跑 dataset/train/eval）；預設會覆蓋 backtest_outputs（相當於 --stages backtest --force-backtest）",
    )
    parser.add_argument(
        "--force-backtest",
        action="store_true",
        help="即使 backtest_outputs/test_final_results.csv 已存在，也重跑 backtest（只影響 backtest stage）",
    )
    parser.add_argument("--no-gallery", action="store_true", help="不產生 _charts/ 統整圖表資料夾")
    args = parser.parse_args(list(argv) if argv is not None else None)

    results_root = get_results_root()
    results_root.mkdir(parents=True, exist_ok=True)
    run_log_csv = results_root / "run_all_log.csv"
    write_environment_snapshot(results_root)

    stages = _parse_stage_set(str(args.stages))
    force_backtest = bool(args.force_backtest)
    if bool(args.only_backtest):
        stages = {"backtest"}
        force_backtest = True

    base_cfg = build_cfg()
    base_cfg = replace(base_cfg, train=replace(base_cfg.train, epochs=int(args.epochs)))
    print_experiment_banner(base_cfg)

    batches = _parse_int_list(args.batches)
    if not batches:
        batches = [1]
    feature_counts = _parse_int_list(args.feature_counts)
    if not feature_counts:
        feature_counts = [6, 7, 8, 9]

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for batch in batches:
        for cfg in iter_experiment_configs(batch=int(batch), feature_counts=feature_counts, base=base_cfg):
            eid = experiment_id(cfg)
            total += 1
            t0 = time.time()
            row: dict[str, object] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment_id": eid,
                "batch": int(batch),
                "label_type": cfg.labels.label_type,
                "feature_count": cfg.features.feature_count,
                "epochs": cfg.train.epochs,
                "status": "",
                "seconds": "",
                "error": "",
            }

            print(f"\n=== [{total}] {eid} ===")
            try:
                out = _run_one(cfg, force=bool(args.force), stages=stages, force_backtest=force_backtest)
                status = str(out.get("status") or "ok")
                row["status"] = status
                if status.startswith("skipped"):
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                failed += 1
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                row["seconds"] = round(time.time() - t0, 3)
                _write_run_log_row(run_log_csv, row)

    # summary / report
    generate_report_main()

    if not args.no_gallery:
        index_md = build_charts_gallery(results_root, include_csv=True)
        print(f"[gallery] wrote: {index_md}")

    print("")
    print("=" * 80)
    print(f"Done. total={total} ok={ok} skipped={skipped} failed={failed}")
    print(f"Results: {results_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
