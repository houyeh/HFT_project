from __future__ import annotations

"""
00 - One-click run all ablations（單鍵跑完一批實驗）
==============================================

用途：
- 一次跑完你指定的 ablation matrix（label_type × feature_count）
- 每個 experiment 都會自動跑完：
  dataset -> train -> evaluate -> backtest(val sweep -> fixed test)
- 最後再產出 summary：experiment_summary.csv + experiment_report.md

你要改的地方（最常改）：
- `LABEL_TYPES_BATCH_1` / `LABEL_TYPES_BATCH_2`
- `FEATURE_COUNTS`
- `INCLUDE_BATCH_2`

輸出位置：
`new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/...`
"""

import argparse
import traceback
from dataclasses import replace

from _common import build_cfg
from new_era.models_playaround_15.backtest_val_sweep import backtest_val_sweep_then_test
from new_era.models_playaround_15.config import FeatureSet
from new_era.models_playaround_15.dataset_generation import generate_dataset_npz
from new_era.models_playaround_15.evaluate_tcn_full import evaluate_full
from new_era.models_playaround_15.generate_experiment_report import main as generate_summary_and_report
from new_era.models_playaround_15.paths import build_experiment_paths, experiment_id
from new_era.models_playaround_15.train_tcn import train_tcn

# =============================================================================
# 你通常只需要改這一區
# =============================================================================

FEATURE_COUNTS: tuple[int, ...] = (6, 7, 8, 9)

LABEL_TYPES_BATCH_1: tuple[str, ...] = (
    "mid_move_3class",
    "smoothed_mid_move_3class",
    "toxic_bid_binary",
    "toxic_ask_binary",
)

LABEL_TYPES_BATCH_2: tuple[str, ...] = (
    "adverse_bid_fill_binary",
    "adverse_ask_fill_binary",
    "active_buy_profitable_binary",
    "active_sell_profitable_binary",
)

INCLUDE_BATCH_2: bool = False


def _is_experiment_done(paths) -> bool:
    # 只要 test final 有產出，通常代表整條 pipeline 已跑完
    return (paths.backtest_dir / "test_final_results.csv").exists()


def run_one(cfg, *, skip_existing: bool) -> None:
    paths = build_experiment_paths(cfg)
    exp_id = experiment_id(cfg)
    print(f"\n=== {exp_id} ===")

    if skip_existing and _is_experiment_done(paths):
        print(f"[SKIP] already done: {paths.backtest_dir / 'test_final_results.csv'}")
        return

    try:
        generate_dataset_npz(cfg, paths)
        train_tcn(cfg, paths)
        evaluate_full(cfg, paths)
        backtest_val_sweep_then_test(cfg, paths)
    except Exception as exc:  # noqa: BLE001
        paths.ensure_dirs()
        err_path = paths.root_dir / "run_error.txt"
        err_path.write_text(
            f"experiment_id: {exp_id}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        print(f"[ERROR] {exc.__class__.__name__}: {exc}")
        print(f"[ERROR] wrote -> {err_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="若已完成就略過（避免重跑）")
    parser.add_argument("--include-batch2", action="store_true", help="把 batch2 也一起跑")
    args = parser.parse_args()

    base = build_cfg()
    label_types = list(LABEL_TYPES_BATCH_1)
    if INCLUDE_BATCH_2 or bool(args.include_batch2):
        label_types.extend(list(LABEL_TYPES_BATCH_2))

    planned = [(lt, fc) for fc in FEATURE_COUNTS for lt in label_types]
    print("[PLAN] experiments:", len(planned))
    for lt, fc in planned:
        print(f"  - label_type={lt} feature_count={fc}")

    for feature_count in FEATURE_COUNTS:
        for label_type in label_types:
            cfg = replace(
                base,
                features=FeatureSet(feature_count=int(feature_count)),
                labels=replace(base.labels, label_type=label_type),  # type: ignore[arg-type]
            )
            run_one(cfg, skip_existing=bool(args.skip_existing))

    generate_summary_and_report()
    print("[DONE] summary/report generated.")


if __name__ == "__main__":
    main()

