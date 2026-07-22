"""CLI for checking how intraday time filters affect saved predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.event_driven import run_event_driven_backtest
from src.data.tw_replay import load_tw_replay_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze time-of-day filters on saved predictions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--start-times", default="08:45:00,09:00:00,09:15:00,09:30:00")
    parser.add_argument("--end-times", default="13:00:00,13:15:00,13:25:00,13:30:00")
    parser.add_argument("--holding-steps", type=int, default=8)
    parser.add_argument("--passive-wait", type=int, default=2)
    parser.add_argument("--cooldown-steps", type=int, default=10)
    parser.add_argument("--size-mode", default="fixed")
    parser.add_argument("--max-size", type=int, default=1)
    parser.add_argument("--confidence-floor", type=float, default=0.334)
    parser.add_argument("--confidence-ceiling", type=float, default=0.345)
    return parser.parse_args()


def parse_time_to_ns(text: str) -> int | None:
    if not text:
        return None
    hour, minute, second = [int(part) for part in text.split(":")]
    return (
        hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + second * 1_000_000_000
    )


def normalize_time_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    symbols = sorted(predictions["symbol"].unique().tolist())
    trade_tapes = load_tw_replay_trades(args.replay_path, symbols)

    start_times = normalize_time_list(args.start_times)
    end_times = normalize_time_list(args.end_times)

    rows = []
    for start_time in start_times:
        start_ns = parse_time_to_ns(start_time)
        for end_time in end_times:
            end_ns = parse_time_to_ns(end_time)
            if start_ns is not None and end_ns is not None and start_ns >= end_ns:
                continue

            summary = run_event_driven_backtest(
                predictions,
                signal_column="gated_signal",
                trade_tapes=trade_tapes,
                entry_mode="passive",
                max_holding_steps=args.holding_steps,
                max_passive_wait_steps=args.passive_wait,
                cooldown_steps=args.cooldown_steps,
                size_mode=args.size_mode,
                max_position_size=args.max_size,
                confidence_floor=args.confidence_floor,
                confidence_ceiling=args.confidence_ceiling,
                min_timestamp=start_ns,
                max_timestamp=end_ns,
            )[1]
            rows.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    **summary,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["net_pnl_ticks", "avg_pnl_ticks"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(run_dir / "time_filter_analysis.csv", index=False)

    best = results.iloc[0].to_dict()
    with (run_dir / "time_filter_best.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    _, _, best_by_symbol = run_event_driven_backtest(
        predictions,
        signal_column="gated_signal",
        trade_tapes=trade_tapes,
        entry_mode="passive",
        max_holding_steps=args.holding_steps,
        max_passive_wait_steps=args.passive_wait,
        cooldown_steps=args.cooldown_steps,
        size_mode=args.size_mode,
        max_position_size=args.max_size,
        confidence_floor=args.confidence_floor,
        confidence_ceiling=args.confidence_ceiling,
        min_timestamp=parse_time_to_ns(str(best["start_time"])),
        max_timestamp=parse_time_to_ns(str(best["end_time"])),
    )
    best_by_symbol.to_csv(run_dir / "time_filter_best_by_symbol.csv", index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
