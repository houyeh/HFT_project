from __future__ import annotations

"""
06 - Run Matrix Batch 2（第六步）
===============================

第二批 label types：
- adverse_bid_fill_binary
- adverse_ask_fill_binary
- active_buy_profitable_binary
- active_sell_profitable_binary

feature_count：
- 6/7/8/9
"""

import subprocess
import sys
from pathlib import Path

from _common import _ensure_project_root_on_path


def main() -> None:
    project_root = _ensure_project_root_on_path()
    script = project_root / "new_era" / "models_playaround_15" / "run_matrix.py"
    cmd = [sys.executable, str(script), "--batch", "2", "--feature-counts", "6,7,8,9"]
    print("[06] running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

