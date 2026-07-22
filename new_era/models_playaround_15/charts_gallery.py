from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_summary_rows(summary_csv: Path) -> dict[str, dict[str, Any]]:
    if not summary_csv.exists():
        return {}
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("experiment_id") or ""): row for row in reader if row.get("experiment_id")}


def _copy_pngs(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.glob("*.png")):
        shutil.copy2(path, dst_dir / path.name)
        count += 1
    return count


def _copy_csvs(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.glob("*.csv")):
        shutil.copy2(path, dst_dir / path.name)
        count += 1
    return count


def build_charts_gallery(
    results_root: Path,
    *,
    include_csv: bool = True,
    include_train_history: bool = True,
) -> Path:
    """
    Build a consolidated folder that collects all plots (and optionally CSVs) per experiment.

    Output:
      results_root/_charts/<experiment_id>/eval/*.png
      results_root/_charts/<experiment_id>/backtest/*.png
      results_root/_charts/<experiment_id>/eval/*.csv (optional)
      results_root/_charts/<experiment_id>/backtest/*.csv (optional)
      results_root/_charts/<experiment_id>/train/train_history.csv (optional)
      results_root/_charts/index.md
    """

    gallery_root = results_root / "_charts"
    gallery_root.mkdir(parents=True, exist_ok=True)

    summary_csv = results_root / "experiment_summary.csv"
    summary_rows = _read_summary_rows(summary_csv)

    experiments = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name,
    )

    index_lines: list[str] = []
    index_lines.append("# 015 Charts Gallery")
    index_lines.append("")
    index_lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    index_lines.append(f"- Experiments: {len(experiments)}")
    index_lines.append("")
    index_lines.append(
        "| experiment_id | label_type | feature_count | test_macro_f1 | pnl_improvement | links |"
    )
    index_lines.append("|---|---:|---:|---:|---:|---|")

    for exp_dir in experiments:
        exp_id = exp_dir.name
        dst_exp = gallery_root / exp_id
        dst_eval = dst_exp / "eval"
        dst_bt = dst_exp / "backtest"
        dst_train = dst_exp / "train"

        _copy_pngs(exp_dir / "eval_outputs", dst_eval)
        _copy_pngs(exp_dir / "backtest_outputs", dst_bt)
        if include_csv:
            _copy_csvs(exp_dir / "eval_outputs", dst_eval)
            _copy_csvs(exp_dir / "backtest_outputs", dst_bt)
        if include_train_history:
            src = exp_dir / "train_outputs" / "train_history.csv"
            if src.exists():
                dst_train.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_train / "train_history.csv")

        row = summary_rows.get(exp_id, {})
        label_type = str(row.get("label_type") or "")
        feature_count = str(row.get("feature_count") or "")
        test_macro_f1 = str(row.get("test_macro_f1") or "")
        pnl_improvement = str(row.get("pnl_improvement") or "")

        links = f"[eval](./{exp_id}/eval/) / [backtest](./{exp_id}/backtest/) / [raw](../{exp_id}/)"
        index_lines.append(
            f"| `{exp_id}` | {label_type} | {feature_count} | {test_macro_f1} | {pnl_improvement} | {links} |"
        )

    index_path = gallery_root / "index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    return index_path
