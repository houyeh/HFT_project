"""CLI for comparing raw and gated execution behavior on saved predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.baseline import run_signal_backtest
from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating
from src.data.tw_replay import load_tw_replay_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gating and execution on saved predictions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--prob-thresholds", default="0.334,0.336,0.338")
    parser.add_argument("--margin-thresholds", default="0.0,0.005")
    parser.add_argument("--entry-modes", default="market,hybrid,passive")
    parser.add_argument("--passive-confidence-threshold", type=float, default=0.34)
    parser.add_argument("--max-holding-steps", type=int, default=10)
    parser.add_argument("--max-passive-wait-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    symbols = sorted(predictions["symbol"].unique().tolist())
    trade_tapes = load_tw_replay_trades(args.replay_path, symbols)

    prob_thresholds = [float(x) for x in args.prob_thresholds.split(",") if x.strip()]
    margin_thresholds = [float(x) for x in args.margin_thresholds.split(",") if x.strip()]
    entry_modes = [x.strip() for x in args.entry_modes.split(",") if x.strip()]

    rows = []
    for prob_threshold in prob_thresholds:
        for margin_threshold in margin_thresholds:
            gated = apply_probability_gating(
                predictions,
                probability_threshold=prob_threshold,
                margin_threshold=margin_threshold,
            )
            signal_summary = run_signal_backtest(
                gated,
                signal_column="gated_signal",
            )[1]
            for entry_mode in entry_modes:
                event_summary = run_event_driven_backtest(
                    gated,
                    signal_column="gated_signal",
                    trade_tapes=trade_tapes,
                    entry_mode=entry_mode,
                    passive_confidence_threshold=args.passive_confidence_threshold,
                    max_holding_steps=args.max_holding_steps,
                    max_passive_wait_steps=args.max_passive_wait_steps,
                )[1]
                rows.append(
                    {
                        "prob_threshold": prob_threshold,
                        "margin_threshold": margin_threshold,
                        "entry_mode": entry_mode,
                        "trade_rate": float(gated["trade_taken_by_gate"].mean()),
                        "signal_net_pnl_ticks": signal_summary["net_pnl_ticks"],
                        "event_net_pnl_ticks": event_summary["net_pnl_ticks"],
                        "event_trades": event_summary["trades"],
                        "event_win_rate": event_summary["win_rate"],
                        "event_avg_pnl_ticks": event_summary["avg_pnl_ticks"],
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        ["event_net_pnl_ticks", "event_avg_pnl_ticks"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(run_dir / "gating_execution_analysis.csv", index=False)

    best = results.iloc[0].to_dict()
    with (run_dir / "gating_execution_best.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
