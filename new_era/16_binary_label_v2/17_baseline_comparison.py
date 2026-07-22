from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round2_audit_utils import RESULTS, load_test_frame, simulate_cancel_policy


ROOT = Path(__file__).resolve().parent


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = load_test_frame()
    no_cancel, _ = simulate_cancel_policy(
        frame,
        np.zeros(len(frame), dtype=bool),
        np.zeros(len(frame), dtype=bool),
        method="no_cancel",
    )

    strict = pd.read_csv(results_dir / "strict_random_baseline_distribution.csv")
    v1 = strict[strict["method"] == "random_v1"].iloc[0]
    v3 = strict[strict["method"] == "random_v3"].iloc[0]
    random_95p_hurdle = float(v3["p95"])

    v2_summary = pd.read_csv(ROOT / "outputs" / "test_backtest_summary.csv").iloc[0]
    old_pnl = pd.read_csv(ROOT / "outputs" / "pnl_summary.csv")
    old_binary = old_pnl[(old_pnl["dataset"] == "NEW_test") & (old_pnl["method"] == "binary_best")].iloc[0]
    old_backtest = pd.read_csv(ROOT.parent / "016_new_binary" / "outputs" / "backtests" / "binary_backtest_summary.csv")
    old_backtest_row = old_backtest[
        (old_backtest["dataset"] == "NEW_test")
        & np.isclose(old_backtest["threshold_bid"].astype(float), 0.4)
        & np.isclose(old_backtest["threshold_ask"].astype(float), 0.2)
    ].iloc[0]

    smoothed_path = (
        ROOT.parent
        / "15_models_playaround"
        / "015_models_playaround_result"
        / "lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1"
        / "backtest_outputs"
        / "test_final_results.csv"
    )
    smoothed = pd.read_csv(smoothed_path).iloc[0]

    rows = []

    def add_row(
        *,
        method: str,
        test_pnl: float,
        trade_count: float,
        cancel_count: float,
        max_drawdown: float,
        hurdle: float,
        source: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "method": method,
                "test_pnl": float(test_pnl),
                "improvement_over_no_cancel": float(test_pnl - no_cancel["net_pnl"]),
                "random_95p_hurdle": float(hurdle) if np.isfinite(hurdle) else np.nan,
                "trade_count": float(trade_count),
                "cancel_count": float(cancel_count) if np.isfinite(cancel_count) else np.nan,
                "pnl_per_trade": float(test_pnl / max(float(trade_count), 1.0)),
                "max_drawdown": float(max_drawdown),
                "source": source,
                "notes": notes,
            }
        )

    add_row(
        method="no_cancel",
        test_pnl=no_cancel["net_pnl"],
        trade_count=no_cancel["trade_count"],
        cancel_count=0.0,
        max_drawdown=no_cancel["max_drawdown"],
        hurdle=random_95p_hurdle,
        source="recomputed_on_016_v2_test_predictions_with_no_pull_masks",
        notes="Reference policy; no model-driven cancels.",
    )
    add_row(
        method="015_smoothed",
        test_pnl=_safe_float(smoothed["model_pnl"]),
        trade_count=_safe_float(smoothed["trades"]),
        cancel_count=np.nan,
        max_drawdown=_safe_float(smoothed["max_drawdown"]),
        hurdle=random_95p_hurdle,
        source=str(smoothed_path),
        notes="Legacy 015 artifact does not expose quote-level cancel_count; field left NaN rather than guessed.",
    )
    old_shapes = json.loads((ROOT.parent / "016_new_binary" / "outputs" / "backtests" / "prediction_shapes.json").read_text())
    old_cancel_count = (float(old_backtest_row["signal_rate_bid"]) + float(old_backtest_row["signal_rate_ask"])) * float(
        old_shapes["NEW_test_events"]
    )
    add_row(
        method="016_old_binary",
        test_pnl=_safe_float(old_binary["net_pnl"]),
        trade_count=_safe_float(old_binary["trade_count"]),
        cancel_count=old_cancel_count,
        max_drawdown=_safe_float(old_binary["max_drawdown"]),
        hurdle=random_95p_hurdle,
        source="outputs/pnl_summary.csv and 016_new_binary backtest summary",
        notes="Known prior violation: binary_best was selected on NEW_test; included only as historical comparison.",
    )
    v2_cancel_count = (float(v2_summary["bid_signal_rate"]) + float(v2_summary["ask_signal_rate"])) * float(v2_summary["rows"])
    add_row(
        method="016_v2_binary",
        test_pnl=_safe_float(v2_summary["net_pnl"]),
        trade_count=_safe_float(v2_summary["trade_count"]),
        cancel_count=v2_cancel_count,
        max_drawdown=_safe_float(v2_summary["max_session_drawdown"]),
        hurdle=random_95p_hurdle,
        source="outputs/test_backtest_summary.csv",
        notes="Frozen validation-selected 10% thresholds; main audited candidate.",
    )
    add_row(
        method="matched_random_v1",
        test_pnl=_safe_float(v1["mean"]),
        trade_count=_safe_float(v1["mean_trade_count"]),
        cancel_count=_safe_float(v1["cancel_count"]),
        max_drawdown=_safe_float(v1["mean_max_drawdown"]),
        hurdle=_safe_float(v1["p95"]),
        source="results/strict_random_baseline_distribution.csv",
        notes="Count+side matched random baseline; reported as distribution mean.",
    )
    add_row(
        method="matched_random_v3",
        test_pnl=_safe_float(v3["mean"]),
        trade_count=_safe_float(v3["mean_trade_count"]),
        cancel_count=_safe_float(v3["cancel_count"]),
        max_drawdown=_safe_float(v3["mean_max_drawdown"]),
        hurdle=_safe_float(v3["p95"]),
        source="results/strict_random_baseline_distribution.csv",
        notes="Count+side+session+spread matched random baseline; main stricter hurdle.",
    )

    pd.DataFrame(rows).to_csv(results_dir / "baseline_comparison.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline comparison table.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"Baseline comparison written to {args.results_dir}")
