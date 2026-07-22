"""CLI for inspecting how time and rule controls change strategy behavior."""

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
    parser = argparse.ArgumentParser(description="Analyze cooldown / sizing / holding controls.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--cooldowns", default="0,3,5,10")
    parser.add_argument("--holding-steps", default="5,8,10,15")
    parser.add_argument("--passive-waits", default="2,4")
    parser.add_argument("--size-modes", default="fixed,confidence")
    parser.add_argument("--max-sizes", default="1,2")
    parser.add_argument("--confidence-floor", type=float, default=0.334)
    parser.add_argument("--confidence-ceiling", type=float, default=0.345)
    parser.add_argument("--end-time", default="13:25:00")
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    symbols = sorted(predictions["symbol"].unique().tolist())
    trade_tapes = load_tw_replay_trades(args.replay_path, symbols)

    cooldowns = [int(x) for x in args.cooldowns.split(",") if x.strip()]
    holding_steps = [int(x) for x in args.holding_steps.split(",") if x.strip()]
    passive_waits = [int(x) for x in args.passive_waits.split(",") if x.strip()]
    size_modes = [x.strip() for x in args.size_modes.split(",") if x.strip()]
    max_sizes = [int(x) for x in args.max_sizes.split(",") if x.strip()]

    rows = []
    for cooldown in cooldowns:
        for hold in holding_steps:
            for wait in passive_waits:
                for size_mode in size_modes:
                    for max_size in max_sizes:
                        if size_mode == "fixed" and max_size != 1:
                            continue
                        summary = run_event_driven_backtest(
                            predictions,
                            signal_column="gated_signal",
                            trade_tapes=trade_tapes,
                            entry_mode="passive",
                            max_holding_steps=hold,
                            max_passive_wait_steps=wait,
                            cooldown_steps=cooldown,
                            size_mode=size_mode,
                            max_position_size=max_size,
                            confidence_floor=args.confidence_floor,
                            confidence_ceiling=args.confidence_ceiling,
                            max_timestamp=parse_time_to_ns(args.end_time),
                        )[1]
                        rows.append(
                            {
                                "cooldown": cooldown,
                                "holding_steps": hold,
                                "passive_wait": wait,
                                "size_mode": size_mode,
                                "max_size": max_size,
                                **summary,
                            }
                        )

    results = pd.DataFrame(rows).sort_values(
        ["net_pnl_ticks", "avg_pnl_ticks"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results.to_csv(run_dir / "strategy_control_analysis.csv", index=False)
    best = results.iloc[0].to_dict()
    with (run_dir / "strategy_control_best.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
