from __future__ import annotations

"""
07 - Generate Summary + Report（第七步）
====================================

用途：
- 掃描 `new_era/15_models_playaround/015_models_playaround_result/*/`
- 彙整輸出：
  - `new_era/15_models_playaround/015_models_playaround_result/experiment_summary.csv`
  - `new_era/15_models_playaround/015_models_playaround_result/experiment_report.md`
"""

from _common import _ensure_project_root_on_path

_ensure_project_root_on_path()

from new_era.models_playaround_15.generate_experiment_report import (  # noqa: E402
    main as generate_report_main,
)


def main() -> None:
    generate_report_main()


if __name__ == "__main__":
    main()
