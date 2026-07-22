from __future__ import annotations

"""
03 - Evaluate（第三步）
====================

用途：
- 讀取 checkpoint 權重
- 對 val / test split 做 inference
- 產出完整評估圖表與 CSV（照你定義的規格）

輸出（eval_outputs/）：
- confusion matrix (count / percent) for val/test
- classification_report_val.csv / classification_report_test.csv
- prediction_distribution_val/test.png
- PR/ROC/calibration/threshold diagnostics（binary 與 3-class 都有對應版本）

你要改的地方：
- 主要還是在 `_common.py`（label_type/feature_count/horizon/threshold）
"""

from _common import build_cfg, build_paths, print_experiment_banner
from new_era.models_playaround_15.evaluate_tcn_full import evaluate_full


def main() -> None:
    cfg = build_cfg()
    print_experiment_banner(cfg)
    paths = build_paths(cfg)
    summary = evaluate_full(cfg, paths)
    print(f"[03] wrote -> {paths.eval_dir}")
    splits = summary.get("splits", {})
    if isinstance(splits, dict) and "val" in splits:
        print(f"[03] val acc={splits['val'].get('accuracy'):.4f} macro_f1={splits['val'].get('macro_f1'):.4f}")


if __name__ == "__main__":
    main()
