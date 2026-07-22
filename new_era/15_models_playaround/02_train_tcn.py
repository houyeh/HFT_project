from __future__ import annotations

"""
02 - Train TCN（第二步）
=====================

用途：
- 讀取 Step 01 產生的 `dataset_sequences.npz`
- 依照 feature_count 自動設定 `num_inputs`
- 訓練 TCN 並輸出 checkpoint + training history

輸出：
- `.../checkpoints/tcn_weights.pth`
- `.../checkpoints/checkpoint_metadata.json`
- `.../train_outputs/train_history.csv`
- `.../train_outputs/train_summary.json`

你要改的地方：
- `new_era/15_models_playaround/_common.py`（epochs/batch_size/lr/seed/device）
"""

from _common import build_cfg, build_paths, print_experiment_banner
from new_era.models_playaround_15.train_tcn import train_tcn


def main() -> None:
    cfg = build_cfg()
    print_experiment_banner(cfg)
    paths = build_paths(cfg)
    summary = train_tcn(cfg, paths)
    print(f"[02] best_epoch={summary['best_epoch']} best_val_acc={summary['best_val_acc']:.4f}")
    print(f"[02] weights -> {paths.model_weights_pth}")


if __name__ == "__main__":
    main()
