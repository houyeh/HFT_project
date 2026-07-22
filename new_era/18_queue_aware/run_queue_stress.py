from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from queue_stress_core import (
    ROOT,
    StressConfig,
    base_expected_values,
    ensure_random_masks,
    load_frames,
    policy_masks,
    random_v3_reference_pnl,
    simulate_policy,
)


def write_stress_definition(out_dir: Path) -> None:
    (out_dir / "stress_definition.md").write_text(
        "# Queue-Aware Stress Definition\n\n"
        "This stress engine evaluates how candidate cancel policies behave when the optimistic proxy execution assumptions are weakened.\n\n"
        "## Stress Dimensions\n\n"
        "1. `neutral_fill_haircut`: Uniformly keeps only a fixed fraction of realized maker fills. `1.0` means no queue haircut; `0.5` means each potential fill is kept only when its fixed random mask is below 0.5.\n\n"
        "2. `good_fill_haircut`: Preferentially removes favorable fills. A favorable bid fill is one whose future mid return is positive; a favorable ask fill is one whose future mid return is negative. This is intended for pessimistic appendix analysis.\n\n"
        "3. `toxic_multiplier`: Multiplies adverse fill damage. At `1.0`, no extra penalty is applied. Above `1.0`, adverse future-mid movement adds an extra penalty proportional to the side-specific adverse ticks.\n\n"
        "4. `cancel_delay_ticks`: Delays cancel effectiveness by event ticks. At delay `d`, a cancel signal at time `t` only protects quote events from `t+d` onward.\n\n"
        "## Fixed Random Masks\n\n"
        "The same random masks are used for every stress setting. This prevents noise from masquerading as stress sensitivity.\n",
        encoding="utf-8",
    )


def run(out_dir: Path, stress: StressConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_016, frame_015 = load_frames()
    actions, _ = ensure_random_masks(frame_016, out_dir)

    rows = []
    for name, (frame, pull_bid, pull_ask) in policy_masks(frame_016, frame_015).items():
        result = simulate_policy(frame, pull_bid, pull_ask, actions, stress)
        rows.append(
            {
                "policy": name,
                "neutral_fill_haircut": stress.neutral_fill_haircut,
                "good_fill_haircut": stress.good_fill_haircut,
                "toxic_multiplier": stress.toxic_multiplier,
                "cancel_delay_ticks": stress.cancel_delay_ticks,
                "net_pnl": result["net_pnl"],
                "trade_count": result["trade_count"],
                "max_drawdown": result["max_drawdown"],
            }
        )
    rows.append(
        {
            "policy": "random_v3_mean",
            "neutral_fill_haircut": stress.neutral_fill_haircut,
            "good_fill_haircut": stress.good_fill_haircut,
            "toxic_multiplier": stress.toxic_multiplier,
            "cancel_delay_ticks": stress.cancel_delay_ticks,
            "net_pnl": random_v3_reference_pnl(frame_016, actions, stress),
            "trade_count": float("nan"),
            "max_drawdown": float("nan"),
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "single_stress_run.csv", index=False)

    expected = base_expected_values()
    base = StressConfig()
    repro_rows = []
    for name, (frame, pull_bid, pull_ask) in policy_masks(frame_016, frame_015).items():
        result = simulate_policy(frame, pull_bid, pull_ask, actions, base)
        repro_rows.append(
            {
                "policy": name,
                "expected_base_pnl": expected[name],
                "stress_engine_base_pnl": result["net_pnl"],
                "abs_diff": abs(result["net_pnl"] - expected[name]),
                "pass": bool(abs(result["net_pnl"] - expected[name]) < 1e-9),
            }
        )
    random_base = random_v3_reference_pnl(frame_016, actions, base)
    repro_rows.append(
        {
            "policy": "random_v3_mean",
            "expected_base_pnl": expected["random_v3_mean"],
            "stress_engine_base_pnl": random_base,
            "abs_diff": abs(random_base - expected["random_v3_mean"]),
            "pass": bool(abs(random_base - expected["random_v3_mean"]) < 1e-9),
        }
    )
    pd.DataFrame(repro_rows).to_csv(out_dir / "stress_reproducibility_check.csv", index=False)
    write_stress_definition(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queue-aware stress backtest.")
    parser.add_argument("--out_dir", type=Path, default=ROOT)
    parser.add_argument("--neutral_fill_haircut", type=float, default=1.0)
    parser.add_argument("--good_fill_haircut", type=float, default=1.0)
    parser.add_argument("--toxic_multiplier", type=float, default=1.0)
    parser.add_argument("--cancel_delay_ticks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.out_dir,
        StressConfig(
            neutral_fill_haircut=args.neutral_fill_haircut,
            good_fill_haircut=args.good_fill_haircut,
            toxic_multiplier=args.toxic_multiplier,
            cancel_delay_ticks=args.cancel_delay_ticks,
        ),
    )
    print(f"Queue stress run written to {args.out_dir}")
