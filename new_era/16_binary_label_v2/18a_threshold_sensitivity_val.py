from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round2_audit_utils import RESULTS, load_val_frame, simulate_cancel_policy, threshold_for_target_rate


TARGETS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


def _parse_targets(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run(results_dir: Path, targets: list[float]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = load_val_frame()
    rows = []
    threshold_payload = []
    for target in targets:
        bid_threshold = threshold_for_target_rate(frame["bid_prob_toxic"].to_numpy(dtype=float), target)
        ask_threshold = threshold_for_target_rate(frame["ask_prob_toxic"].to_numpy(dtype=float), target)
        pull_bid = frame["bid_prob_toxic"].to_numpy(dtype=float) >= bid_threshold
        pull_ask = frame["ask_prob_toxic"].to_numpy(dtype=float) >= ask_threshold
        summary, _ = simulate_cancel_policy(frame, pull_bid, pull_ask, method=f"val_target_{target:.3f}")
        is_selected = bool(np.isclose(target, 0.10))
        rows.append(
            {
                "split": "val",
                "target_cancel_rate": float(target),
                "bid_threshold": float(bid_threshold),
                "ask_threshold": float(ask_threshold),
                "bid_cancel_rate": float(np.mean(pull_bid)),
                "ask_cancel_rate": float(np.mean(pull_ask)),
                "any_side_cancel_rate": float(np.mean(pull_bid | pull_ask)),
                "cancel_count": int(summary["cancel_count"]),
                "trade_count": int(summary["trade_count"]),
                "net_pnl": float(summary["net_pnl"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "pnl_per_trade": float(summary["pnl_per_trade"]),
                "is_main_claim": is_selected,
                "used_for_model_selection": is_selected,
                "selection_basis": "validation_score_quantile",
            }
        )
        threshold_payload.append(
            {
                "target_cancel_rate": float(target),
                "bid_threshold": float(bid_threshold),
                "ask_threshold": float(ask_threshold),
                "selected_on": "val",
                "is_main_claim": is_selected,
            }
        )

    pd.DataFrame(rows).to_csv(results_dir / "threshold_sensitivity_val.csv", index=False)
    (results_dir / "threshold_sensitivity_val_thresholds.json").write_text(
        json.dumps({"thresholds": threshold_payload}, indent=2),
        encoding="utf-8",
    )
    (results_dir / "threshold_sensitivity_val.md").write_text(
        "# Validation Threshold Sensitivity\n\n"
        "All thresholds in this file are selected from validation scores only. "
        "The 10% row is the pre-approved main validation-selected operating point.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only threshold sensitivity.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    parser.add_argument("--targets", type=str, default=",".join(str(x) for x in TARGETS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir, _parse_targets(args.targets))
    print(f"Validation threshold sensitivity written to {args.results_dir}")
