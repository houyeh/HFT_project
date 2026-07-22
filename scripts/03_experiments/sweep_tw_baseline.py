"""Batch launcher for Taiwan replay baseline training experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.baseline import run_signal_backtest
from src.backtest.event_driven import run_event_driven_backtest


BASE_ARGS = {
    "quote_stride": 10,
    "sample_stride": 10,
    "epochs": 4,
    "batch_size": 512,
}

CONFIGS = [
    {"name": "tx_mx_h5_t05_weighted", "symbols": "TXFJ5,MXFJ5", "horizon": 5, "threshold_ticks": 0.5, "weighted": True},
    {"name": "tx_mx_h5_t05_unweighted", "symbols": "TXFJ5,MXFJ5", "horizon": 5, "threshold_ticks": 0.5, "weighted": False},
    {"name": "tx_mx_h10_t10_weighted", "symbols": "TXFJ5,MXFJ5", "horizon": 10, "threshold_ticks": 1.0, "weighted": True},
    {"name": "tx_mx_h10_t10_unweighted", "symbols": "TXFJ5,MXFJ5", "horizon": 10, "threshold_ticks": 1.0, "weighted": False},
    {"name": "tx_mx_cd_h5_t05_weighted", "symbols": "TXFJ5,MXFJ5,CDFJ5", "horizon": 5, "threshold_ticks": 0.5, "weighted": True},
    {"name": "tx_mx_cd_h5_t05_unweighted", "symbols": "TXFJ5,MXFJ5,CDFJ5", "horizon": 5, "threshold_ticks": 0.5, "weighted": False},
    {"name": "tx_mx_cd_h10_t10_weighted", "symbols": "TXFJ5,MXFJ5,CDFJ5", "horizon": 10, "threshold_ticks": 1.0, "weighted": True},
    {"name": "tx_mx_cd_h10_t10_unweighted", "symbols": "TXFJ5,MXFJ5,CDFJ5", "horizon": 10, "threshold_ticks": 1.0, "weighted": False},
]


def latest_subdir(path: Path) -> Path:
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        raise FileNotFoundError(f"No run directory found under {path}")
    return max(children, key=lambda item: item.stat().st_mtime)


def run_config(config: dict, root_output_dir: Path) -> dict:
    target_root = root_output_dir / config["name"]
    cmd = [
        sys.executable,
        "scripts/02_train_eval/train_tw_baseline.py",
        "--symbols",
        config["symbols"],
        "--horizon",
        str(config["horizon"]),
        "--threshold-ticks",
        str(config["threshold_ticks"]),
        "--event-max-holding-steps",
        str(config["horizon"]),
        "--output-dir",
        str(target_root),
        "--quote-stride",
        str(BASE_ARGS["quote_stride"]),
        "--sample-stride",
        str(BASE_ARGS["sample_stride"]),
        "--epochs",
        str(BASE_ARGS["epochs"]),
        "--batch-size",
        str(BASE_ARGS["batch_size"]),
    ]
    if not config["weighted"]:
        cmd.append("--disable-class-weighting")

    subprocess.run(cmd, check=True)
    run_dir = latest_subdir(target_root)

    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    signal_05 = run_signal_backtest(predictions, spread_cost_multiplier=0.5)[1]
    signal_10 = run_signal_backtest(predictions, spread_cost_multiplier=1.0)[1]
    event_05 = run_event_driven_backtest(
        predictions,
        spread_scale=0.5,
        max_holding_steps=config["horizon"],
        close_on_stationary=True,
    )[1]
    event_10 = run_event_driven_backtest(
        predictions,
        spread_scale=1.0,
        max_holding_steps=config["horizon"],
        close_on_stationary=True,
    )[1]

    row = {
        "name": config["name"],
        "symbols": config["symbols"],
        "horizon": config["horizon"],
        "threshold_ticks": config["threshold_ticks"],
        "class_weighting": config["weighted"],
        "run_dir": str(run_dir),
        "test_accuracy": summary["test_accuracy"],
        "test_macro_f1": summary["test_macro_f1"],
        "signal_net_pnl_cost_0_5": signal_05["net_pnl_ticks"],
        "signal_net_pnl_cost_1_0": signal_10["net_pnl_ticks"],
        "event_net_pnl_cost_0_5": event_05["net_pnl_ticks"],
        "event_net_pnl_cost_1_0": event_10["net_pnl_ticks"],
        "event_trades_cost_1_0": event_10["trades"],
        "event_win_rate_cost_1_0": event_10["win_rate"],
    }
    return row


def main() -> None:
    root_output_dir = Path("artifacts/02_model_runs/tw_sweep")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in CONFIGS:
        print(f"Running {config['name']}...")
        rows.append(run_config(config, root_output_dir))

    results = pd.DataFrame(rows)
    active_results = results.loc[results["event_trades_cost_1_0"] >= 20].copy()
    ranking_source = active_results if not active_results.empty else results
    results = ranking_source.sort_values(
        ["event_net_pnl_cost_1_0", "test_macro_f1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(root_output_dir / "sweep_summary.csv", index=False)

    best = results.iloc[0].to_dict()
    with (root_output_dir / "best_run.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
