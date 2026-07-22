"""CLI for per-symbol regime-guard searches on saved prediction outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating
from src.data.tw_replay import load_tw_replay_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze symbol-aware gating, cutoffs, and regime switches.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--trade-symbols", default="")
    parser.add_argument("--base-prob-threshold", type=float, default=0.334)
    parser.add_argument("--prob-threshold-offsets", default="-0.002,0.0,0.002,0.004")
    parser.add_argument("--margin-thresholds", default="0.0,0.005,0.01")
    parser.add_argument("--end-times", default="11:45:00,12:00:00,12:15:00,12:30:00,12:45:00,13:00:00,13:25:00")
    parser.add_argument("--cooldowns", default="0,3,5,10")
    parser.add_argument("--holding-steps", type=int, default=10)
    parser.add_argument("--passive-wait", type=int, default=4)
    parser.add_argument("--vol-quantiles", default="0.80,0.85,0.90")
    parser.add_argument("--trade-quantiles", default="0.80,0.85,0.90")
    parser.add_argument("--cancel-quantiles", default="0.80,0.85,0.90")
    parser.add_argument("--liq-quantiles", default="0.10,0.20,0.25")
    return parser.parse_args()


def parse_time_to_ns(text: str) -> int:
    hour, minute, second = [int(part) for part in text.split(":")]
    return (
        hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + second * 1_000_000_000
    )


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_text_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_regime_profiles(
    vol_quantiles: list[float],
    trade_quantiles: list[float],
    cancel_quantiles: list[float],
    liq_quantiles: list[float],
) -> list[dict]:
    profiles: list[dict] = [{"name": "none"}]
    profiles.extend({"name": f"vol_guard_q{vol_q:.2f}", "vol_q": vol_q} for vol_q in vol_quantiles)
    profiles.extend({"name": f"trade_guard_q{trade_q:.2f}", "trade_q": trade_q} for trade_q in trade_quantiles)
    profiles.extend({"name": f"cancel_guard_q{cancel_q:.2f}", "cancel_q": cancel_q} for cancel_q in cancel_quantiles)
    profiles.extend({"name": f"liquidity_guard_q{liq_q:.2f}", "liq_q": liq_q} for liq_q in liq_quantiles)

    for vol_q in vol_quantiles:
        for liq_q in liq_quantiles:
            profiles.append(
                {
                    "name": f"vol_liq_guard_q{vol_q:.2f}_{liq_q:.2f}",
                    "vol_q": vol_q,
                    "liq_q": liq_q,
                }
            )
    for vol_q in vol_quantiles:
        for cancel_q in cancel_quantiles:
            profiles.append(
                {
                    "name": f"vol_cancel_guard_q{vol_q:.2f}_{cancel_q:.2f}",
                    "vol_q": vol_q,
                    "cancel_q": cancel_q,
                }
            )
    for vol_q in vol_quantiles:
        for trade_q in trade_quantiles:
            for liq_q in liq_quantiles:
                profiles.append(
                    {
                        "name": f"combined_guard_q{vol_q:.2f}_{trade_q:.2f}_{liq_q:.2f}",
                        "vol_q": vol_q,
                        "trade_q": trade_q,
                        "liq_q": liq_q,
                    }
                )
    return profiles


def quantile_thresholds(symbol_df: pd.DataFrame, profile: dict) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if "vol_q" in profile:
        thresholds["regime_volatility_thresholds"] = float(symbol_df["rolling_volatility"].quantile(profile["vol_q"]))
    if "trade_q" in profile:
        thresholds["regime_trade_intensity_thresholds"] = float(symbol_df["trade_intensity_short"].quantile(profile["trade_q"]))
    if "cancel_q" in profile:
        thresholds["regime_max_cancel_pressure"] = float(symbol_df["cancel_pressure_short"].abs().quantile(profile["cancel_q"]))
    if "liq_q" in profile:
        visible_queue = symbol_df["bid_vol_1"] + symbol_df["ask_vol_1"]
        thresholds["regime_min_visible_queue"] = float(visible_queue.quantile(profile["liq_q"]))
    return thresholds


def evaluate_symbol(
    symbol: str,
    symbol_predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    prob_thresholds: list[float],
    margin_thresholds: list[float],
    end_times: list[str],
    cooldowns: list[int],
    holding_steps: int,
    passive_wait: int,
    regime_profiles: list[dict],
) -> pd.DataFrame:
    rows = []
    for prob_threshold in prob_thresholds:
        for margin_threshold in margin_thresholds:
            for end_time in end_times:
                end_ns = parse_time_to_ns(end_time)
                for cooldown in cooldowns:
                    for profile in regime_profiles:
                        threshold_map = quantile_thresholds(symbol_predictions, profile)
                        gated = apply_probability_gating(
                            symbol_predictions,
                            probability_threshold=prob_threshold,
                            margin_threshold=margin_threshold,
                            symbol_probability_thresholds={symbol: prob_threshold},
                            symbol_margin_thresholds={symbol: margin_threshold},
                            symbol_end_times={symbol: end_ns},
                            regime_volatility_thresholds=(
                                {symbol: threshold_map["regime_volatility_thresholds"]}
                                if "regime_volatility_thresholds" in threshold_map
                                else None
                            ),
                            regime_trade_intensity_thresholds=(
                                {symbol: threshold_map["regime_trade_intensity_thresholds"]}
                                if "regime_trade_intensity_thresholds" in threshold_map
                                else None
                            ),
                            regime_min_visible_queue=(
                                {symbol: threshold_map["regime_min_visible_queue"]}
                                if "regime_min_visible_queue" in threshold_map
                                else None
                            ),
                            regime_max_cancel_pressure=(
                                {symbol: threshold_map["regime_max_cancel_pressure"]}
                                if "regime_max_cancel_pressure" in threshold_map
                                else None
                            ),
                        )
                        _, event_summary, _ = run_event_driven_backtest(
                            gated,
                            signal_column="gated_signal",
                            trade_tapes={symbol: trade_tapes.get(symbol, pd.DataFrame())},
                            entry_mode="passive",
                            max_holding_steps=holding_steps,
                            max_passive_wait_steps=passive_wait,
                            cooldown_steps=0,
                            symbol_cooldown_steps={symbol: cooldown},
                        )
                        rows.append(
                            {
                                "symbol": symbol,
                                "prob_threshold": prob_threshold,
                                "margin_threshold": margin_threshold,
                                "end_time": end_time,
                                "cooldown": cooldown,
                                "regime_profile": profile["name"],
                                "trade_rate": float(gated["trade_taken_by_gate"].mean()),
                                **event_summary,
                            }
                        )
    return pd.DataFrame(rows).sort_values(
        ["net_pnl_ticks", "avg_pnl_ticks", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    trade_symbol_list = (
        [symbol.strip() for symbol in args.trade_symbols.split(",") if symbol.strip()]
        if args.trade_symbols.strip()
        else sorted(predictions["symbol"].unique().tolist())
    )
    symbols = sorted(trade_symbol_list)
    trade_tapes = load_tw_replay_trades(args.replay_path, symbols)

    prob_thresholds = [args.base_prob_threshold + offset for offset in parse_float_list(args.prob_threshold_offsets)]
    margin_thresholds = parse_float_list(args.margin_thresholds)
    end_times = parse_text_list(args.end_times)
    cooldowns = parse_int_list(args.cooldowns)
    regime_profiles = build_regime_profiles(
        vol_quantiles=parse_float_list(args.vol_quantiles),
        trade_quantiles=parse_float_list(args.trade_quantiles),
        cancel_quantiles=parse_float_list(args.cancel_quantiles),
        liq_quantiles=parse_float_list(args.liq_quantiles),
    )

    per_symbol_frames = []
    best_rows = []
    for symbol in symbols:
        symbol_predictions = predictions.loc[predictions["symbol"] == symbol].copy().reset_index(drop=True)
        result = evaluate_symbol(
            symbol=symbol,
            symbol_predictions=symbol_predictions,
            trade_tapes=trade_tapes,
            prob_thresholds=prob_thresholds,
            margin_thresholds=margin_thresholds,
            end_times=end_times,
            cooldowns=cooldowns,
            holding_steps=args.holding_steps,
            passive_wait=args.passive_wait,
            regime_profiles=regime_profiles,
        )
        result.to_csv(run_dir / f"symbol_rule_analysis_{symbol}.csv", index=False)
        per_symbol_frames.append(result)
        best_rows.append(result.iloc[0].to_dict())

    best_by_symbol = pd.DataFrame(best_rows)
    best_by_symbol.to_csv(run_dir / "symbol_rule_best_by_symbol.csv", index=False)

    symbol_prob_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["prob_threshold"]))
    symbol_margin_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["margin_threshold"]))
    symbol_end_times = {
        symbol: parse_time_to_ns(end_time)
        for symbol, end_time in zip(best_by_symbol["symbol"], best_by_symbol["end_time"])
    }
    symbol_cooldowns = dict(zip(best_by_symbol["symbol"], best_by_symbol["cooldown"]))

    regime_volatility_thresholds: dict[str, float] = {}
    regime_trade_intensity_thresholds: dict[str, float] = {}
    regime_min_visible_queue: dict[str, float] = {}
    regime_max_cancel_pressure: dict[str, float] = {}

    for row in best_rows:
        symbol = row["symbol"]
        profile = next(profile for profile in regime_profiles if profile["name"] == row["regime_profile"])
        thresholds = quantile_thresholds(predictions.loc[predictions["symbol"] == symbol], profile)
        if "regime_volatility_thresholds" in thresholds:
            regime_volatility_thresholds[symbol] = thresholds["regime_volatility_thresholds"]
        if "regime_trade_intensity_thresholds" in thresholds:
            regime_trade_intensity_thresholds[symbol] = thresholds["regime_trade_intensity_thresholds"]
        if "regime_min_visible_queue" in thresholds:
            regime_min_visible_queue[symbol] = thresholds["regime_min_visible_queue"]
        if "regime_max_cancel_pressure" in thresholds:
            regime_max_cancel_pressure[symbol] = thresholds["regime_max_cancel_pressure"]

    combined_gated = apply_probability_gating(
        predictions,
        probability_threshold=args.base_prob_threshold,
        margin_threshold=min(margin_thresholds),
        symbol_probability_thresholds=symbol_prob_thresholds,
        symbol_margin_thresholds=symbol_margin_thresholds,
        symbol_end_times=symbol_end_times,
        regime_volatility_thresholds=regime_volatility_thresholds or None,
        regime_trade_intensity_thresholds=regime_trade_intensity_thresholds or None,
        regime_min_visible_queue=regime_min_visible_queue or None,
        regime_max_cancel_pressure=regime_max_cancel_pressure or None,
        allowed_symbols=set(symbols),
    )
    combined_gated.to_csv(run_dir / "symbol_rule_combined_predictions.csv", index=False)

    trade_log, combined_summary, combined_by_symbol = run_event_driven_backtest(
        combined_gated,
        signal_column="gated_signal",
        trade_tapes=trade_tapes,
        entry_mode="passive",
        max_holding_steps=args.holding_steps,
        max_passive_wait_steps=args.passive_wait,
        cooldown_steps=0,
        symbol_cooldown_steps=symbol_cooldowns,
    )
    trade_log.to_csv(run_dir / "symbol_rule_combined_trade_log.csv", index=False)
    combined_by_symbol.to_csv(run_dir / "symbol_rule_combined_by_symbol.csv", index=False)

    summary_payload = {
        "holding_steps": args.holding_steps,
        "passive_wait": args.passive_wait,
        "symbol_prob_thresholds": symbol_prob_thresholds,
        "symbol_margin_thresholds": symbol_margin_thresholds,
        "symbol_end_times": {symbol: str(end_time) for symbol, end_time in zip(best_by_symbol["symbol"], best_by_symbol["end_time"])},
        "symbol_cooldowns": symbol_cooldowns,
        "combined_summary": combined_summary,
    }
    with (run_dir / "symbol_rule_combined_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(best_by_symbol.to_string(index=False))
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    np.random.seed(42)
    main()
