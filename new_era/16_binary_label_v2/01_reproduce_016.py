from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OLD_OUTPUTS = PROJECT_ROOT / "new_era" / "016_new_binary" / "outputs"
DEFAULT_OLD_FIX_OUTPUTS = PROJECT_ROOT / "new_era" / "016_new_binary" / "outputs_fix"
DEFAULT_OLD_SCRIPTS = PROJECT_ROOT / "new_era" / "016_new_binary" / "scripts"
DEFAULT_OUT_DIR = THIS_DIR / "outputs"

SIDES = ["bid", "ask"]


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required 016 artifact not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required 016 artifact not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_threshold_setup(text: str) -> tuple[float, float] | tuple[None, None]:
    match = re.search(r"bid=([0-9.]+),ask=([0-9.]+)", str(text))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _same_threshold_pair(left: tuple[float | None, float | None], right: tuple[float, float]) -> bool:
    if left[0] is None or left[1] is None:
        return False
    return bool(np.isclose(float(left[0]), right[0]) and np.isclose(float(left[1]), right[1]))


def build_label_distribution(old_outputs: Path, out_dir: Path) -> pd.DataFrame:
    src = old_outputs / "labels" / "binary_label_distribution.csv"
    df = _read_csv(src).copy()
    df.insert(0, "source_experiment", "016_new_binary")
    df.insert(1, "source_file", str(src))
    df.to_csv(out_dir / "label_distribution.csv", index=False)
    return df


def build_pr_auc_summary(old_outputs: Path, out_dir: Path) -> pd.DataFrame:
    training_summary_path = old_outputs / "models" / "binary_training_summary.csv"
    training_summary = _read_csv(training_summary_path)
    dataset_meta = _read_json(old_outputs / "datasets" / "primary_dataset_metadata.json")

    rows: list[dict[str, Any]] = []
    for side in SIDES:
        side_summary = training_summary[training_summary["side"] == side]
        if side_summary.empty:
            raise RuntimeError(f"Missing side={side} in {training_summary_path}")
        row = side_summary.iloc[0].to_dict()

        model_dir = Path(str(row["model_dir"]))
        model_cfg = _read_json(model_dir / "model_config.json")
        history = _read_csv(model_dir / "train_history.csv", required=False)
        threshold_diag = _read_csv(model_dir / "threshold_diagnostics_test.csv", required=False)

        max_history_val_pr_auc = float("nan")
        if not history.empty and "val_auc_pr" in history:
            max_history_val_pr_auc = float(history["val_auc_pr"].max())

        best_test_f1_threshold = float(row.get("best_threshold_by_f1", np.nan))
        best_test_f1 = float("nan")
        if not threshold_diag.empty and "f1" in threshold_diag:
            best_idx = threshold_diag["f1"].astype(float).idxmax()
            best_test_f1 = float(threshold_diag.loc[best_idx, "f1"])

        meta = dataset_meta.get(side, {})
        train_stats = meta.get("train_stats", {})
        val_stats = meta.get("val_stats", {})
        test_stats = meta.get("test_stats", {})

        rows.append(
            {
                "source_experiment": "016_new_binary",
                "side": side,
                "model_dir": str(model_dir),
                "best_epoch": int(row.get("best_epoch", -1)),
                "best_val_loss": float(row.get("best_val_loss", np.nan)),
                "val_pr_auc_best_checkpoint": float(row.get("val_auc_pr", np.nan)),
                "val_pr_auc_max_history": max_history_val_pr_auc,
                "test_pr_auc": float(row.get("test_auc_pr", np.nan)),
                "test_roc_auc": float(row.get("test_auc_roc", np.nan)),
                "train_positive_rate": float(model_cfg.get("positive_rate_train", np.nan)),
                "val_positive_rate": float(model_cfg.get("positive_rate_val", np.nan)),
                "test_positive_rate": float(model_cfg.get("positive_rate_test", np.nan)),
                "train_toxic_rate_from_dataset_meta": float(train_stats.get("toxic_rate", np.nan)),
                "val_toxic_rate_from_dataset_meta": float(val_stats.get("toxic_rate", np.nan)),
                "test_toxic_rate_from_dataset_meta": float(test_stats.get("toxic_rate", np.nan)),
                "best_threshold_by_f1": best_test_f1_threshold,
                "best_threshold_by_f1_source": "test",
                "best_test_f1": best_test_f1,
                "test_threshold_f1_tuning_violation": True,
                "notes": "Current 016 reports best_threshold_by_f1 from test threshold_diagnostics_test.csv.",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pr_auc_bid_ask.csv", index=False)
    return out


def build_pnl_summary(old_outputs: Path, old_fix_outputs: Path, old_scripts: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    comparison_path = old_outputs / "comparison" / "binary_vs_baselines.csv"
    backtest_path = old_outputs / "backtests" / "binary_backtest_summary.csv"

    comparison = _read_csv(comparison_path)
    backtest = _read_csv(backtest_path)
    new_test_sweep = backtest[backtest["dataset"] == "NEW_test"].copy()
    if new_test_sweep.empty:
        raise RuntimeError(f"No NEW_test rows found in {backtest_path}")

    new_test_sweep = new_test_sweep.sort_values("net_pnl", ascending=False).reset_index(drop=True)
    new_test_sweep["test_net_pnl_rank_desc"] = np.arange(1, len(new_test_sweep) + 1)
    best_test = new_test_sweep.iloc[0]
    best_test_pair = (float(best_test["threshold_bid"]), float(best_test["threshold_ask"]))

    val_selection_path = old_fix_outputs / "comparison" / "val_threshold_selection_binary.csv"
    val_selection = _read_csv(val_selection_path, required=False)
    has_val_only_fix = bool(not val_selection.empty and "selected" in val_selection and (val_selection["selected"] == 1).any())

    source_text = ""
    source_file = old_scripts / "backtest_binary_cancel.py"
    if source_file.exists():
        source_text = source_file.read_text(encoding="utf-8", errors="replace")

    source_mentions_test_selection = (
        "Choose best candidate on NEW test by net_pnl" in source_text
        or "sort_values(\"net_pnl\", ascending=False)" in source_text
    )

    rows: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        threshold_pair = _parse_threshold_setup(row.get("threshold_setup", ""))
        test_rank = np.nan
        if threshold_pair[0] is not None:
            rank_match = new_test_sweep[
                np.isclose(new_test_sweep["threshold_bid"].astype(float), float(threshold_pair[0]))
                & np.isclose(new_test_sweep["threshold_ask"].astype(float), float(threshold_pair[1]))
            ]
            if not rank_match.empty:
                test_rank = float(rank_match.iloc[0]["test_net_pnl_rank_desc"])

        selected_from_test = bool(
            row.get("dataset") == "NEW_test"
            and row.get("method") == "binary_best"
            and _same_threshold_pair(threshold_pair, best_test_pair)
        )

        rows.append(
            {
                **row.to_dict(),
                "threshold_bid": threshold_pair[0],
                "threshold_ask": threshold_pair[1],
                "test_net_pnl_rank_desc": test_rank,
                "selected_from_test_sweep": selected_from_test,
                "threshold_tuning_violation": selected_from_test,
                "selection_audit_note": (
                    "Current 016 binary_best equals the highest NEW_test net_pnl threshold pair."
                    if selected_from_test
                    else ""
                ),
            }
        )

    pnl_summary = pd.DataFrame(rows)
    pnl_summary.to_csv(out_dir / "pnl_summary.csv", index=False)

    audit = {
        "source_experiment": "016_new_binary",
        "old_outputs_dir": str(old_outputs),
        "old_fix_outputs_dir": str(old_fix_outputs),
        "required_no_test_tuning_rule": "threshold selection must use validation only",
        "current_016_binary_best_threshold_bid": best_test_pair[0],
        "current_016_binary_best_threshold_ask": best_test_pair[1],
        "current_016_binary_best_selected_from_new_test_sweep": bool(
            pnl_summary["selected_from_test_sweep"].fillna(False).any()
        ),
        "source_code_mentions_new_test_net_pnl_selection": bool(source_mentions_test_selection),
        "val_only_fix_artifact_available_in_outputs_fix": has_val_only_fix,
        "verdict": (
            "VIOLATION: current 016 binary_best was selected on NEW_test."
            if pnl_summary["selected_from_test_sweep"].fillna(False).any()
            else "No test-set threshold selection detected from reproduced artifacts."
        ),
    }
    (out_dir / "threshold_selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = [
        "# Current 016 Threshold Selection Audit",
        "",
        f"- Old outputs: `{old_outputs}`",
        f"- Current 016 selected threshold: bid={best_test_pair[0]:.2f}, ask={best_test_pair[1]:.2f}",
        f"- Selected from NEW_test sweep: {audit['current_016_binary_best_selected_from_new_test_sweep']}",
        f"- Source code contains NEW_test net-PnL selection pattern: {source_mentions_test_selection}",
        f"- Val-only fix artifact exists under outputs_fix: {has_val_only_fix}",
        "",
        f"Verdict: {audit['verdict']}",
    ]
    (out_dir / "threshold_selection_audit.md").write_text("\n".join(md), encoding="utf-8")
    return pnl_summary, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the current 016 binary baseline from existing 016 artifacts and "
            "audit whether thresholds were selected on test data."
        )
    )
    parser.add_argument("--old_outputs", type=Path, default=DEFAULT_OLD_OUTPUTS)
    parser.add_argument("--old_fix_outputs", type=Path, default=DEFAULT_OLD_FIX_OUTPUTS)
    parser.add_argument("--old_scripts", type=Path, default=DEFAULT_OLD_SCRIPTS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    build_label_distribution(args.old_outputs, out_dir)
    build_pr_auc_summary(args.old_outputs, out_dir)
    _, audit = build_pnl_summary(args.old_outputs, args.old_fix_outputs, args.old_scripts, out_dir)

    print("Reproduced current 016 baseline artifacts:")
    print(f"- {out_dir / 'label_distribution.csv'}")
    print(f"- {out_dir / 'pr_auc_bid_ask.csv'}")
    print(f"- {out_dir / 'pnl_summary.csv'}")
    print(f"Threshold audit verdict: {audit['verdict']}")


if __name__ == "__main__":
    main()
