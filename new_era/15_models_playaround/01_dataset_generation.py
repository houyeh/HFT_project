from __future__ import annotations

"""
01 - Dataset Generation（第一步）
=============================

用途：
- 依照 `_common.py` 的設定（feature_count / label_type / horizon / threshold…）
  產生每個 session 的 sliding-window sequences（X）與 labels（y）

輸出（每個 experiment 一個資料夾）：
- `new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/dataset_sequences.npz`
- `new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/dataset_manifest.json`
- `new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/experiment_config.json`

你要改的地方：
- 請直接改 `new_era/15_models_playaround/_common.py`
"""

from _common import build_cfg, build_paths, print_experiment_banner
from new_era.models_playaround_15.dataset_generation import generate_dataset_npz


def main() -> None:
    cfg = build_cfg()
    print_experiment_banner(cfg)
    paths = build_paths(cfg)
    manifest = generate_dataset_npz(cfg, paths)
    print(f"[01] sessions={len(manifest['sessions'])} -> {paths.dataset_npz}")


if __name__ == "__main__":
    main()
