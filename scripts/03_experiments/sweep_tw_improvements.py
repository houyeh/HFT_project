"""Batch launcher for Taiwan replay improvement experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


BASE_ARGS = {
    "symbols": "TXFJ5,MXFJ5,CDFJ5",
    "horizon": 10,
    "threshold_ticks": 1.0,
    "quote_stride": 10,
    "sample_stride": 10,
    "epochs": 4,
    "batch_size": 512,
}

CONFIGS = [
    {
        "name": "baseline_market_no_gate",
        "threshold_mode": "static",
        "entry_mode": "market",
        "disable_probability_gating": True,
    },
    {
        "name": "market_gate_0336_m005",
        "threshold_mode": "static",
        "entry_mode": "market",
        "gating_prob_threshold": 0.336,
        "gating_margin_threshold": 0.005,
    },
    {
        "name": "hybrid_gate_0334_m005",
        "threshold_mode": "static",
        "entry_mode": "hybrid",
        "gating_prob_threshold": 0.334,
        "gating_margin_threshold": 0.005,
        "event_passive_confidence_threshold": 0.34,
        "event_max_passive_wait_steps": 4,
    },
    {
        "name": "passive_gate_0334_m005",
        "threshold_mode": "static",
        "entry_mode": "passive",
        "gating_prob_threshold": 0.334,
        "gating_margin_threshold": 0.005,
        "event_max_passive_wait_steps": 4,
    },
    {
        "name": "hybrid_dynamic_gate_0336_m005",
        "threshold_mode": "spread_volatility",
        "threshold_spread_multiplier": 0.10,
        "threshold_volatility_multiplier": 0.10,
        "entry_mode": "hybrid",
        "gating_prob_threshold": 0.336,
        "gating_margin_threshold": 0.005,
        "event_passive_confidence_threshold": 0.34,
        "event_max_passive_wait_steps": 4,
    },
    {
        "name": "passive_dynamic_gate_0336_m005",
        "threshold_mode": "spread_volatility",
        "threshold_spread_multiplier": 0.10,
        "threshold_volatility_multiplier": 0.10,
        "entry_mode": "passive",
        "gating_prob_threshold": 0.336,
        "gating_margin_threshold": 0.005,
        "event_max_passive_wait_steps": 4,
    },
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
        BASE_ARGS["symbols"],
        "--horizon",
        str(BASE_ARGS["horizon"]),
        "--threshold-ticks",
        str(BASE_ARGS["threshold_ticks"]),
        "--quote-stride",
        str(BASE_ARGS["quote_stride"]),
        "--sample-stride",
        str(BASE_ARGS["sample_stride"]),
        "--epochs",
        str(BASE_ARGS["epochs"]),
        "--batch-size",
        str(BASE_ARGS["batch_size"]),
        "--output-dir",
        str(target_root),
        "--event-max-holding-steps",
        str(BASE_ARGS["horizon"]),
        "--event-entry-mode",
        config["entry_mode"],
        "--threshold-mode",
        config.get("threshold_mode", "static"),
    ]

    optional_args = {
        "threshold_spread_multiplier": "--threshold-spread-multiplier",
        "threshold_volatility_multiplier": "--threshold-volatility-multiplier",
        "gating_prob_threshold": "--gating-prob-threshold",
        "gating_margin_threshold": "--gating-margin-threshold",
        "event_passive_confidence_threshold": "--event-passive-confidence-threshold",
        "event_max_passive_wait_steps": "--event-max-passive-wait-steps",
    }
    for key, flag in optional_args.items():
        if key in config:
            cmd.extend([flag, str(config[key])])
    if config.get("disable_probability_gating"):
        cmd.append("--disable-probability-gating")

    subprocess.run(cmd, check=True)
    run_dir = latest_subdir(target_root)

    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    with (run_dir / "event_backtest_summary.json").open("r", encoding="utf-8") as handle:
        event_summary = json.load(handle)

    row = {
        "name": config["name"],
        "run_dir": str(run_dir),
        "threshold_mode": config.get("threshold_mode", "static"),
        "entry_mode": config["entry_mode"],
        "probability_gating": not config.get("disable_probability_gating", False),
        "test_accuracy": summary["test_accuracy"],
        "test_macro_f1": summary["test_macro_f1"],
        "signal_backtest_net_pnl_ticks": summary["signal_backtest_net_pnl_ticks"],
        "event_backtest_net_pnl_ticks": summary["event_backtest_net_pnl_ticks"],
        "event_trades": event_summary["trades"],
        "event_win_rate": event_summary["win_rate"],
        "event_avg_pnl_ticks": event_summary["avg_pnl_ticks"],
    }
    return row


def main() -> None:
    root_output_dir = Path("artifacts/02_model_runs/tw_improvements")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in CONFIGS:
        print(f"Running {config['name']}...")
        rows.append(run_config(config, root_output_dir))

    results = pd.DataFrame(rows).sort_values(
        ["event_backtest_net_pnl_ticks", "test_macro_f1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(root_output_dir / "improvement_summary.csv", index=False)

    best = results.iloc[0].to_dict()
    with (root_output_dir / "best_improvement.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
