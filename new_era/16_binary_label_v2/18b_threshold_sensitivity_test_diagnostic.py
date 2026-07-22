from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round2_audit_utils import RESULTS, load_test_frame, simulate_cancel_policy


DIAGNOSTIC_NOTE = "DIAGNOSTIC ONLY — NOT USED FOR MODEL SELECTION"


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = results_dir / "threshold_sensitivity_val_thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Run 18a first; missing {thresholds_path}")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    frame = load_test_frame()

    rows = []
    for item in thresholds:
        target = float(item["target_cancel_rate"])
        bid_threshold = float(item["bid_threshold"])
        ask_threshold = float(item["ask_threshold"])
        pull_bid = frame["bid_prob_toxic"].to_numpy(dtype=float) >= bid_threshold
        pull_ask = frame["ask_prob_toxic"].to_numpy(dtype=float) >= ask_threshold
        summary, _ = simulate_cancel_policy(frame, pull_bid, pull_ask, method=f"test_diagnostic_target_{target:.3f}")
        rows.append(
            {
                "split": "test",
                "target_cancel_rate_selected_on_val": target,
                "bid_threshold_from_val": bid_threshold,
                "ask_threshold_from_val": ask_threshold,
                "bid_cancel_rate_on_test": float(np.mean(pull_bid)),
                "ask_cancel_rate_on_test": float(np.mean(pull_ask)),
                "any_side_cancel_rate_on_test": float(np.mean(pull_bid | pull_ask)),
                "cancel_count": int(summary["cancel_count"]),
                "trade_count": int(summary["trade_count"]),
                "net_pnl": float(summary["net_pnl"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "pnl_per_trade": float(summary["pnl_per_trade"]),
                "is_main_claim": False,
                "used_for_model_selection": False,
                "diagnostic_note": DIAGNOSTIC_NOTE,
            }
        )

    pd.DataFrame(rows).to_csv(results_dir / "threshold_sensitivity_test_diagnostic.csv", index=False)
    (results_dir / "threshold_sensitivity_test_diagnostic.md").write_text(
        f"# Test Threshold Sensitivity Diagnostic\n\n{DIAGNOSTIC_NOTE}.\n\n"
        "The rows apply validation-derived thresholds to test for sensitivity reporting only. "
        "They were not used to choose a model, feature set, cancel-rate target, or operating threshold.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-only test threshold sensitivity.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"Diagnostic test threshold sensitivity written to {args.results_dir}")
