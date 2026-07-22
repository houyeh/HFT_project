from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from new_era.models_playaround_15.paths import get_results_root


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_one_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _exp_score(exp_dir: Path) -> tuple[int, float]:
    """
    Score an experiment directory for de-duplication.

    Higher is better:
    - prefer complete runs (has backtest/eval/train artifacts)
    - prefer short experiment ids (lt=...) over legacy label_type=... folders
    - break ties by most recent modified time
    """

    has_train = (exp_dir / "train_outputs" / "train_summary.json").exists()
    has_eval = (exp_dir / "eval_outputs" / "evaluation_summary.json").exists()
    has_bt = (exp_dir / "backtest_outputs" / "test_final_results.csv").exists()
    is_short = exp_dir.name.startswith("lt=")

    score = int(has_bt) * 8 + int(has_eval) * 4 + int(has_train) * 2 + int(is_short) * 1
    mtime = float(exp_dir.stat().st_mtime) if exp_dir.exists() else 0.0
    return score, mtime


def collect_experiment_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_root.exists():
        return rows

    # Only treat folders with an experiment_config.json as experiments.
    exp_dirs = [p for p in results_root.iterdir() if p.is_dir() and (p / "experiment_config.json").exists()]

    # Read raw rows (may include duplicates if legacy + short-id folders both exist).
    raw: list[tuple[Path, dict[str, Any]]] = []
    for exp_dir in sorted(exp_dirs):
        cfg = _read_json(exp_dir / "experiment_config.json") or {}
        train = _read_json(exp_dir / "train_outputs" / "train_summary.json") or {}
        eval_sum = _read_json(exp_dir / "eval_outputs" / "evaluation_summary.json") or {}
        selected = _read_json(exp_dir / "backtest_outputs" / "selected_strategy_config.json") or {}
        test_final = _read_csv_one_row(exp_dir / "backtest_outputs" / "test_final_results.csv") or {}

        val_metrics = (eval_sum.get("splits") or {}).get("val", {}) if isinstance(eval_sum.get("splits"), dict) else {}
        test_metrics = (eval_sum.get("splits") or {}).get("test", {}) if isinstance(eval_sum.get("splits"), dict) else {}

        label_type = (((cfg.get("labels") or {}).get("label_type")) if isinstance(cfg.get("labels"), dict) else None) or ""
        feature_count = (((cfg.get("features") or {}).get("feature_count")) if isinstance(cfg.get("features"), dict) else None)
        window_size = (((cfg.get("window") or {}).get("window_size")) if isinstance(cfg.get("window"), dict) else None)
        horizon = (((cfg.get("window") or {}).get("horizon")) if isinstance(cfg.get("window"), dict) else None)
        threshold_ticks = (((cfg.get("window") or {}).get("threshold_ticks")) if isinstance(cfg.get("window"), dict) else None)
        adverse_threshold_ticks = (((cfg.get("labels") or {}).get("adverse_threshold_ticks")) if isinstance(cfg.get("labels"), dict) else None)

        taker_fee_twd = (((cfg.get("labels") or {}).get("taker_fee_twd")) if isinstance(cfg.get("labels"), dict) else None)
        exit_cost_ticks = (((cfg.get("labels") or {}).get("exit_cost_ticks")) if isinstance(cfg.get("labels"), dict) else None)
        margin_ticks = (((cfg.get("labels") or {}).get("margin_ticks")) if isinstance(cfg.get("labels"), dict) else None)

        val_selected_threshold = ""
        if selected:
            pthr = selected.get("prob_threshold")
            ethr = selected.get("edge_threshold")
            if pthr is not None and ethr is not None:
                val_selected_threshold = f"pthr={pthr},ethr={ethr}"

        row = {
            "experiment_id": exp_dir.name,
            "label_type": label_type,
            "feature_count": feature_count,
            "window_size": window_size,
            "horizon": horizon,
            "threshold_ticks": threshold_ticks,
            "adverse_threshold_ticks": adverse_threshold_ticks,
            "_taker_fee_twd": taker_fee_twd,
            "_exit_cost_ticks": exit_cost_ticks,
            "_margin_ticks": margin_ticks,
            "num_train_samples": train.get("num_train_samples"),
            "num_val_samples": train.get("num_val_samples"),
            "num_test_samples": train.get("num_test_samples"),
            "class_distribution_train": json.dumps(train.get("class_distribution_train", {}), ensure_ascii=False),
            "class_distribution_val": json.dumps(train.get("class_distribution_val", {}), ensure_ascii=False),
            "class_distribution_test": json.dumps(train.get("class_distribution_test", {}), ensure_ascii=False),
            "best_epoch": train.get("best_epoch"),
            "best_val_loss": train.get("best_val_loss"),
            "best_val_acc": train.get("best_val_acc"),
            "best_val_balanced_acc": val_metrics.get("balanced_accuracy"),
            "best_val_macro_f1": val_metrics.get("macro_f1"),
            "test_loss": test_metrics.get("loss"),
            "test_acc": test_metrics.get("accuracy"),
            "test_balanced_acc": test_metrics.get("balanced_accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "val_selected_strategy": selected.get("strategy"),
            "val_selected_threshold": val_selected_threshold,
            "baseline_test_pnl": test_final.get("baseline_pnl"),
            "model_test_pnl": test_final.get("model_pnl"),
            "pnl_improvement": test_final.get("improvement_vs_baseline"),
            "baseline_test_pnl_flatten": test_final.get("baseline_pnl_flatten"),
            "model_test_pnl_flatten": test_final.get("model_pnl_flatten"),
            "pnl_improvement_flatten": test_final.get("improvement_vs_baseline_flatten"),
            "max_abs_inventory": test_final.get("max_abs_inventory"),
            "max_drawdown": test_final.get("max_drawdown"),
        }
        raw.append((exp_dir, row))

    # De-duplicate: keep the best row for each config key.
    best_by_key: dict[tuple[Any, ...], tuple[tuple[int, float], dict[str, Any]]] = {}
    for exp_dir, row in raw:
        label_type = row.get("label_type") or ""
        if not str(label_type):
            continue
        key = (
            label_type,
            row.get("feature_count"),
            row.get("window_size"),
            row.get("horizon"),
            row.get("threshold_ticks"),
            row.get("adverse_threshold_ticks"),
            row.get("_taker_fee_twd"),
            row.get("_exit_cost_ticks"),
            row.get("_margin_ticks"),
        )
        score = _exp_score(exp_dir)
        prev = best_by_key.get(key)
        if prev is None or score > prev[0]:
            best_by_key[key] = (score, row)

    rows = [entry[1] for entry in best_by_key.values()]
    rows.sort(key=lambda r: str(r.get("experiment_id") or ""))

    return rows


def write_experiment_summary(results_root: Path) -> Path:
    rows = collect_experiment_rows(results_root)
    out_path = results_root / "experiment_summary.csv"
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[[c for c in frame.columns if not str(c).startswith("_")]]
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def write_experiment_report(results_root: Path, summary_csv: Path) -> Path:
    df = pd.read_csv(summary_csv)
    out_path = results_root / "experiment_report.md"

    lines: list[str] = []
    lines.append("# Experiment Report (015_models_playaround)")
    lines.append("")
    lines.append(f"- Experiments: {len(df):,}")
    lines.append("")

    if df.empty:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines.append("## Class Distributions (by label_type)")
    lines.append("")
    for label_type in sorted(df["label_type"].dropna().unique().tolist()):
        sdf = df[df["label_type"] == label_type].copy()
        if sdf.empty:
            continue
        lines.append(f"### {label_type}")
        lines.append("")
        cols = [
            "feature_count",
            "class_distribution_train",
            "class_distribution_val",
            "class_distribution_test",
        ]
        lines.append(_table(sdf[cols].sort_values("feature_count")))
        lines.append("")

    lines.append("## Feature Count Comparison (Batch Summary)")
    lines.append("")
    agg = (
        df.groupby("feature_count", dropna=True)
        .agg(
            avg_test_macro_f1=("test_macro_f1", "mean"),
            avg_pnl_improvement=("pnl_improvement", "mean"),
            median_pnl_improvement=("pnl_improvement", "median"),
            experiments=("experiment_id", "count"),
        )
        .reset_index()
        .sort_values("feature_count")
    )
    lines.append(_table(agg))
    lines.append("")

    lines.append("## Best Candidates (Heuristic)")
    lines.append("")
    cand = df.dropna(subset=["pnl_improvement"]).copy()
    if cand.empty:
        lines.append("- No backtest results found yet.")
    else:
        best = cand.sort_values("pnl_improvement", ascending=False).iloc[0]
        lines.append(
            f"- Best PnL improvement: `{best['experiment_id']}` "
            f"(label={best['label_type']}, feat={best['feature_count']}, +{best['pnl_improvement']:.2f} TWD)"
        )
        if "pnl_improvement_flatten" in cand.columns and cand["pnl_improvement_flatten"].notna().any():
            cand_flat = cand.dropna(subset=["pnl_improvement_flatten"]).copy()
            best_flat = cand_flat.sort_values("pnl_improvement_flatten", ascending=False).iloc[0]
            lines.append(
                f"- Best PnL improvement (forced flatten): `{best_flat['experiment_id']}` "
                f"(label={best_flat['label_type']}, feat={best_flat['feature_count']}, "
                f"+{best_flat['pnl_improvement_flatten']:.2f} TWD)"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    results_root = get_results_root()
    summary_csv = write_experiment_summary(results_root)
    report_md = write_experiment_report(results_root, summary_csv)
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
