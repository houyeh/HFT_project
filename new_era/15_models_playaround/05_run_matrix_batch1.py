from __future__ import annotations

"""
05 - Run Matrix Batch 1（第五步）
===============================

第一批 label types：
- mid_move_3class
- smoothed_mid_move_3class
- toxic_bid_binary
- toxic_ask_binary

feature_count：
- 6/7/8/9

輸出：
每個實驗都會在 `new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/...`

你要改的地方：
- 若你想先縮小範圍（例如只跑 feature_count=8），可以改 `--feature-counts`
"""

import subprocess
import sys
from pathlib import Path

from _common import _ensure_project_root_on_path


def main() -> None:
    project_root = _ensure_project_root_on_path()
    script = project_root / "new_era" / "models_playaround_15" / "run_matrix.py"
    cmd = [sys.executable, str(script), "--batch", "1", "--feature-counts", "6,7,8,9"]
    print("[05] running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
