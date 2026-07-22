"""Parameter-sweep helpers for the hybrid market-making backtest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.hybrid_market_maker import run_hybrid_market_maker_backtest
from src.data.tw_replay import load_tw_replay_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate a hybrid market-making baseline.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--trade-symbols", default="TXFJ5,MXFJ5")
    parser.add_argument("--score-thresholds", default="0.001,0.002")
    parser.add_argument("--one-sided-thresholds", default="0.004,0.006")
    parser.add_argument("--quote-off-thresholds", default="0.0,0.002,0.004")
    parser.add_argument("--toxicity-thresholds", default="0.03,0.06,99")
    parser.add_argument("--one-sided-toxicity-thresholds", default="0.02,0.04,99")
    parser.add_argument("--stationary-cutoffs", default="1.0,0.338,0.336")
    parser.add_argument("--inventory-penalties", default="0.0,0.5")
    parser.add_argument("--max-inventories", default="1,2")
    parser.add_argument("--end-times", default="12:00:00,12:30:00")
    parser.add_argument("--latency-steps", default="0,1,2,4")
    parser.add_argument("--quote-confirmation-steps", default="0,1")
    parser.add_argument("--alpha-smoothing-windows", default="1,2")
    parser.add_argument("--stop-loss-ticks", default="0,4,8")
    parser.add_argument("--take-profit-ticks", default="0,4,8,12")
    parser.add_argument("--alpha-decay-thresholds", default="0.0,0.002")
    parser.add_argument("--quote-step-ticks", type=float, default=1.0)
    parser.add_argument("--max-holding-steps", type=int, default=20)
    parser.add_argument("--adverse-score-threshold", type=float, default=0.010)
    parser.add_argument("--vol-quantiles", default="0.90")
    parser.add_argument("--trade-quantiles", default="0.90")
    parser.add_argument("--cancel-quantiles", default="0.90")
    parser.add_argument("--liq-quantiles", default="0.25")
    parser.add_argument("--alpha-profiles", default="base,flow_queue,full_exec")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_text_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_time_to_ns(text: str) -> int:
    hour, minute, second = [int(part) for part in text.split(":")]
    return (
        hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + second * 1_000_000_000
    )


def build_regime_profiles(
    vol_quantiles: list[float],
    trade_quantiles: list[float],
    cancel_quantiles: list[float],
    liq_quantiles: list[float],
) -> list[dict]:
    profiles: list[dict] = [{"name": "none"}]
    profiles.extend({"name": f"vol_guard_q{q:.2f}", "vol_q": q} for q in vol_quantiles)
    profiles.extend({"name": f"trade_guard_q{q:.2f}", "trade_q": q} for q in trade_quantiles)
    profiles.extend({"name": f"cancel_guard_q{q:.2f}", "cancel_q": q} for q in cancel_quantiles)
    profiles.extend({"name": f"liq_guard_q{q:.2f}", "liq_q": q} for q in liq_quantiles)
    for vol_q in vol_quantiles:
        for cancel_q in cancel_quantiles:
            profiles.append({"name": f"vol_cancel_q{vol_q:.2f}_{cancel_q:.2f}", "vol_q": vol_q, "cancel_q": cancel_q})
    return profiles


def alpha_profiles_from_names(names: list[str]) -> dict[str, dict[str, float]]:
    profiles = {
        "base": {"prob_edge": 1.0},
        "flow_queue": {
            "prob_edge": 1.0,
            "queue_change": 0.004,
            "aggressor_flow": 0.006,
        },
        "full_exec": {
            "prob_edge": 1.0,
            "queue_imbalance": 0.002,
            "queue_change": 0.004,
            "aggressor_flow": 0.008,
            "cancel_pressure": 0.004,
            "net_cancel_pressure": 0.002,
        },
        "flow_cancel": {
            "prob_edge": 1.0,
            "aggressor_flow": 0.010,
            "cancel_pressure": 0.005,
            "net_cancel_pressure": 0.003,
        },
        "toxic_flow": {
            "prob_edge": 1.0,
            "queue_imbalance": 0.002,
            "queue_change": 0.004,
            "aggressor_flow": 0.010,
            "cancel_pressure": 0.004,
            "net_cancel_pressure": 0.003,
            "bid_cancel_proxy": 0.003,
            "ask_cancel_proxy": 0.003,
            "toxicity_queue_imbalance": 0.020,
            "toxicity_queue_change": 0.015,
            "toxicity_aggressor_flow": 0.025,
            "toxicity_cancel_pressure": 0.020,
            "toxicity_bid_cancel_proxy": 0.010,
            "toxicity_ask_cancel_proxy": 0.010,
        },
        "toxic_book": {
            "prob_edge": 1.0,
            "queue_imbalance": 0.003,
            "queue_change": 0.006,
            "aggressor_flow": 0.008,
            "cancel_pressure": 0.004,
            "net_cancel_pressure": 0.004,
            "bid_cancel_proxy": 0.004,
            "ask_cancel_proxy": 0.004,
            "toxicity_queue_imbalance": 0.030,
            "toxicity_queue_change": 0.020,
            "toxicity_aggressor_flow": 0.020,
            "toxicity_cancel_pressure": 0.020,
            "toxicity_bid_cancel_proxy": 0.012,
            "toxicity_ask_cancel_proxy": 0.012,
        },
    }
    return {name: profiles[name] for name in names if name in profiles}


def quantile_thresholds(symbol_df: pd.DataFrame, profile: dict) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if "vol_q" in profile:
        thresholds["regime_volatility_thresholds"] = float(symbol_df["rolling_volatility"].quantile(profile["vol_q"]))
    if "trade_q" in profile:
        thresholds["regime_trade_intensity_thresholds"] = float(
            symbol_df["trade_intensity_short"].quantile(profile["trade_q"])
        )
    if "cancel_q" in profile:
        thresholds["regime_max_cancel_pressure"] = float(
            symbol_df["cancel_pressure_short"].abs().quantile(profile["cancel_q"])
        )
    if "liq_q" in profile:
        visible_queue = symbol_df["bid_vol_1"] + symbol_df["ask_vol_1"]
        thresholds["regime_min_visible_queue"] = float(visible_queue.quantile(profile["liq_q"]))
    return thresholds


def evaluate_symbol_configs(
    symbol: str,
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    score_thresholds: list[float],
    one_sided_thresholds: list[float],
    quote_off_thresholds: list[float],
    toxicity_thresholds: list[float],
    one_sided_toxicity_thresholds: list[float],
    stationary_cutoffs: list[float],
    inventory_penalties: list[float],
    max_inventories: list[int],
    end_times: list[str],
    quote_confirmation_steps_list: list[int],
    alpha_smoothing_windows: list[int],
    stop_loss_ticks_list: list[float],
    take_profit_ticks_list: list[float],
    alpha_decay_thresholds: list[float],
    max_holding_steps: int,
    adverse_score_threshold: float,
    quote_step_ticks: float,
    regime_profiles: list[dict],
    alpha_profiles: dict[str, dict[str, float]],
) -> pd.DataFrame:
    symbol_predictions = predictions.loc[predictions["symbol"] == symbol].copy().reset_index(drop=True)
    rows: list[dict] = []
    for score_threshold in score_thresholds:
        for one_sided_threshold in one_sided_thresholds:
            if one_sided_threshold < score_threshold:
                continue
            for quote_off_threshold in quote_off_thresholds:
                if quote_off_threshold > score_threshold:
                    continue
                for toxicity_threshold in toxicity_thresholds:
                    for one_sided_toxicity_threshold in one_sided_toxicity_thresholds:
                        if one_sided_toxicity_threshold > toxicity_threshold:
                            continue
                        for stationary_cutoff in stationary_cutoffs:
                            for alpha_name, alpha_coefficients in alpha_profiles.items():
                                for inventory_penalty in inventory_penalties:
                                    for max_inventory in max_inventories:
                                        for end_time in end_times:
                                            for quote_confirmation_steps in quote_confirmation_steps_list:
                                                for alpha_smoothing_window in alpha_smoothing_windows:
                                                    for stop_loss_ticks in stop_loss_ticks_list:
                                                        for take_profit_ticks in take_profit_ticks_list:
                                                            if (
                                                                stop_loss_ticks > 0
                                                                and take_profit_ticks > 0
                                                                and take_profit_ticks < stop_loss_ticks
                                                            ):
                                                                continue
                                                            for alpha_decay_threshold in alpha_decay_thresholds:
                                                                for profile in regime_profiles:
                                                                    thresholds = quantile_thresholds(symbol_predictions, profile)
                                                                    _, summary, _, _ = run_hybrid_market_maker_backtest(
                                                                        symbol_predictions,
                                                                        trade_tapes={symbol: trade_tapes.get(symbol, pd.DataFrame())},
                                                                        score_threshold=score_threshold,
                                                                        one_sided_threshold=one_sided_threshold,
                                                                        quote_off_score_threshold=quote_off_threshold,
                                                                        toxicity_threshold=toxicity_threshold,
                                                                        one_sided_toxicity_threshold=one_sided_toxicity_threshold,
                                                                        stationary_probability_cutoff=stationary_cutoff,
                                                                        alpha_coefficients=alpha_coefficients,
                                                                        inventory_penalty_ticks=inventory_penalty,
                                                                        quote_step_ticks=quote_step_ticks,
                                                                        max_inventory=max_inventory,
                                                                        max_holding_steps=max_holding_steps,
                                                                        adverse_score_threshold=adverse_score_threshold,
                                                                        quote_confirmation_steps=quote_confirmation_steps,
                                                                        alpha_smoothing_window=alpha_smoothing_window,
                                                                        inventory_stop_loss_ticks=(
                                                                            stop_loss_ticks if stop_loss_ticks > 0 else None
                                                                        ),
                                                                        inventory_take_profit_ticks=(
                                                                            take_profit_ticks if take_profit_ticks > 0 else None
                                                                        ),
                                                                        inventory_flatten_score_threshold=alpha_decay_threshold,
                                                                        symbol_end_times={symbol: parse_time_to_ns(end_time)},
                                                                        regime_volatility_thresholds=(
                                                                            {symbol: thresholds["regime_volatility_thresholds"]}
                                                                            if "regime_volatility_thresholds" in thresholds
                                                                            else None
                                                                        ),
                                                                        regime_trade_intensity_thresholds=(
                                                                            {symbol: thresholds["regime_trade_intensity_thresholds"]}
                                                                            if "regime_trade_intensity_thresholds" in thresholds
                                                                            else None
                                                                        ),
                                                                        regime_min_visible_queue=(
                                                                            {symbol: thresholds["regime_min_visible_queue"]}
                                                                            if "regime_min_visible_queue" in thresholds
                                                                            else None
                                                                        ),
                                                                        regime_max_cancel_pressure=(
                                                                            {symbol: thresholds["regime_max_cancel_pressure"]}
                                                                            if "regime_max_cancel_pressure" in thresholds
                                                                            else None
                                                                        ),
                                                                        allowed_symbols={symbol},
                                                                    )
                                                                    rows.append(
                                                                        {
                                                                            "symbol": symbol,
                                                                            "score_threshold": score_threshold,
                                                                            "one_sided_threshold": one_sided_threshold,
                                                                            "quote_off_threshold": quote_off_threshold,
                                                                            "toxicity_threshold": toxicity_threshold,
                                                                            "one_sided_toxicity_threshold": one_sided_toxicity_threshold,
                                                                            "stationary_cutoff": stationary_cutoff,
                                                                            "alpha_profile": alpha_name,
                                                                            "inventory_penalty_ticks": inventory_penalty,
                                                                            "max_inventory": max_inventory,
                                                                            "end_time": end_time,
                                                                            "quote_confirmation_steps": quote_confirmation_steps,
                                                                            "alpha_smoothing_window": alpha_smoothing_window,
                                                                            "stop_loss_ticks": stop_loss_ticks,
                                                                            "take_profit_ticks": take_profit_ticks,
                                                                            "alpha_decay_threshold": alpha_decay_threshold,
                                                                            "regime_profile": profile["name"],
                                                                            **summary,
                                                                        }
                                                                    )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["net_pnl_ticks", "avg_pnl_ticks", "win_rate", "fills"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return result


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir if not args.output_subdir else run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_symbols = [symbol.strip() for symbol in args.trade_symbols.split(",") if symbol.strip()]
    trade_tapes = load_tw_replay_trades(args.replay_path, trade_symbols)

    val_predictions = pd.read_csv(run_dir / "val_predictions.csv")
    test_predictions = pd.read_csv(run_dir / "test_predictions.csv")

    score_thresholds = parse_float_list(args.score_thresholds)
    one_sided_thresholds = parse_float_list(args.one_sided_thresholds)
    quote_off_thresholds = parse_float_list(args.quote_off_thresholds)
    toxicity_thresholds = parse_float_list(args.toxicity_thresholds)
    one_sided_toxicity_thresholds = parse_float_list(args.one_sided_toxicity_thresholds)
    stationary_cutoffs = parse_float_list(args.stationary_cutoffs)
    inventory_penalties = parse_float_list(args.inventory_penalties)
    max_inventories = parse_int_list(args.max_inventories)
    end_times = parse_text_list(args.end_times)
    latency_steps = parse_int_list(args.latency_steps)
    quote_confirmation_steps_list = parse_int_list(args.quote_confirmation_steps)
    alpha_smoothing_windows = parse_int_list(args.alpha_smoothing_windows)
    stop_loss_ticks_list = parse_float_list(args.stop_loss_ticks)
    take_profit_ticks_list = parse_float_list(args.take_profit_ticks)
    alpha_decay_thresholds = parse_float_list(args.alpha_decay_thresholds)
    alpha_profiles = alpha_profiles_from_names(parse_text_list(args.alpha_profiles))
    regime_profiles = build_regime_profiles(
        vol_quantiles=parse_float_list(args.vol_quantiles),
        trade_quantiles=parse_float_list(args.trade_quantiles),
        cancel_quantiles=parse_float_list(args.cancel_quantiles),
        liq_quantiles=parse_float_list(args.liq_quantiles),
    )

    best_rows: list[dict] = []
    for symbol in trade_symbols:
        result = evaluate_symbol_configs(
            symbol=symbol,
            predictions=val_predictions,
            trade_tapes=trade_tapes,
            score_thresholds=score_thresholds,
            one_sided_thresholds=one_sided_thresholds,
            quote_off_thresholds=quote_off_thresholds,
            toxicity_thresholds=toxicity_thresholds,
            one_sided_toxicity_thresholds=one_sided_toxicity_thresholds,
            stationary_cutoffs=stationary_cutoffs,
            inventory_penalties=inventory_penalties,
            max_inventories=max_inventories,
            end_times=end_times,
            quote_confirmation_steps_list=quote_confirmation_steps_list,
            alpha_smoothing_windows=alpha_smoothing_windows,
            stop_loss_ticks_list=stop_loss_ticks_list,
            take_profit_ticks_list=take_profit_ticks_list,
            alpha_decay_thresholds=alpha_decay_thresholds,
            max_holding_steps=args.max_holding_steps,
            adverse_score_threshold=args.adverse_score_threshold,
            quote_step_ticks=args.quote_step_ticks,
            regime_profiles=regime_profiles,
            alpha_profiles=alpha_profiles,
        )
        result.to_csv(output_dir / f"hybrid_mm_val_analysis_{symbol}.csv", index=False)
        best_rows.append(result.iloc[0].to_dict())

    best_by_symbol = pd.DataFrame(best_rows)
    best_by_symbol.to_csv(output_dir / "hybrid_mm_val_best_by_symbol.csv", index=False)

    symbol_score_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["score_threshold"]))
    symbol_one_sided_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["one_sided_threshold"]))
    symbol_inventory_penalties = dict(zip(best_by_symbol["symbol"], best_by_symbol["inventory_penalty_ticks"]))
    symbol_max_inventory = dict(zip(best_by_symbol["symbol"], best_by_symbol["max_inventory"]))
    symbol_end_times = {row["symbol"]: parse_time_to_ns(row["end_time"]) for _, row in best_by_symbol.iterrows()}
    symbol_quote_off_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["quote_off_threshold"]))
    symbol_toxicity_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["toxicity_threshold"]))
    symbol_one_sided_toxicity_thresholds = dict(
        zip(best_by_symbol["symbol"], best_by_symbol["one_sided_toxicity_threshold"])
    )
    symbol_stationary_cutoffs = dict(zip(best_by_symbol["symbol"], best_by_symbol["stationary_cutoff"]))
    symbol_alpha_coefficients = {
        row["symbol"]: alpha_profiles[row["alpha_profile"]]
        for _, row in best_by_symbol.iterrows()
    }
    symbol_quote_confirmation_steps = dict(zip(best_by_symbol["symbol"], best_by_symbol["quote_confirmation_steps"]))
    symbol_alpha_smoothing = dict(zip(best_by_symbol["symbol"], best_by_symbol["alpha_smoothing_window"]))
    symbol_stop_loss_ticks = dict(zip(best_by_symbol["symbol"], best_by_symbol["stop_loss_ticks"]))
    symbol_take_profit_ticks = dict(zip(best_by_symbol["symbol"], best_by_symbol["take_profit_ticks"]))
    symbol_alpha_decay_threshold = dict(zip(best_by_symbol["symbol"], best_by_symbol["alpha_decay_threshold"]))

    regime_volatility_thresholds: dict[str, float] = {}
    regime_trade_intensity_thresholds: dict[str, float] = {}
    regime_min_visible_queue: dict[str, float] = {}
    regime_max_cancel_pressure: dict[str, float] = {}
    for _, row in best_by_symbol.iterrows():
        profile = next(profile for profile in regime_profiles if profile["name"] == row["regime_profile"])
        symbol_df = val_predictions.loc[val_predictions["symbol"] == row["symbol"]]
        thresholds = quantile_thresholds(symbol_df, profile)
        if "regime_volatility_thresholds" in thresholds:
            regime_volatility_thresholds[row["symbol"]] = thresholds["regime_volatility_thresholds"]
        if "regime_trade_intensity_thresholds" in thresholds:
            regime_trade_intensity_thresholds[row["symbol"]] = thresholds["regime_trade_intensity_thresholds"]
        if "regime_min_visible_queue" in thresholds:
            regime_min_visible_queue[row["symbol"]] = thresholds["regime_min_visible_queue"]
        if "regime_max_cancel_pressure" in thresholds:
            regime_max_cancel_pressure[row["symbol"]] = thresholds["regime_max_cancel_pressure"]

    val_trade_log, val_summary, val_by_symbol, val_quote_log = run_hybrid_market_maker_backtest(
        val_predictions,
        trade_tapes=trade_tapes,
        score_threshold=min(score_thresholds),
        one_sided_threshold=max(one_sided_thresholds),
        quote_off_score_threshold=min(quote_off_thresholds),
        toxicity_threshold=max(toxicity_thresholds),
        one_sided_toxicity_threshold=max(one_sided_toxicity_thresholds),
        stationary_probability_cutoff=max(stationary_cutoffs),
        inventory_penalty_ticks=0.0,
        quote_step_ticks=args.quote_step_ticks,
        max_inventory=max(max_inventories),
        max_holding_steps=args.max_holding_steps,
        adverse_score_threshold=args.adverse_score_threshold,
        quote_confirmation_steps=0,
        alpha_smoothing_window=1,
        symbol_score_thresholds=symbol_score_thresholds,
        symbol_one_sided_thresholds=symbol_one_sided_thresholds,
        symbol_quote_off_thresholds=symbol_quote_off_thresholds,
        symbol_toxicity_thresholds=symbol_toxicity_thresholds,
        symbol_one_sided_toxicity_thresholds=symbol_one_sided_toxicity_thresholds,
        symbol_stationary_cutoffs=symbol_stationary_cutoffs,
        symbol_quote_confirmation_steps=symbol_quote_confirmation_steps,
        symbol_alpha_smoothing_windows=symbol_alpha_smoothing,
        symbol_inventory_penalties=symbol_inventory_penalties,
        symbol_max_inventory=symbol_max_inventory,
        symbol_end_times=symbol_end_times,
        symbol_alpha_coefficients=symbol_alpha_coefficients,
        symbol_stop_loss_ticks=symbol_stop_loss_ticks,
        symbol_take_profit_ticks=symbol_take_profit_ticks,
        symbol_alpha_decay_thresholds=symbol_alpha_decay_threshold,
        regime_volatility_thresholds=regime_volatility_thresholds or None,
        regime_trade_intensity_thresholds=regime_trade_intensity_thresholds or None,
        regime_min_visible_queue=regime_min_visible_queue or None,
        regime_max_cancel_pressure=regime_max_cancel_pressure or None,
        allowed_symbols=set(trade_symbols),
    )
    val_trade_log.to_csv(output_dir / "hybrid_mm_val_trade_log.csv", index=False)
    val_by_symbol.to_csv(output_dir / "hybrid_mm_val_by_symbol.csv", index=False)
    val_quote_log.to_csv(output_dir / "hybrid_mm_val_quote_log.csv", index=False)
    with (output_dir / "hybrid_mm_val_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(val_summary, handle, indent=2)

    latency_rows: list[dict] = []
    for latency in latency_steps:
        test_trade_log, test_summary, test_by_symbol, test_quote_log = run_hybrid_market_maker_backtest(
            test_predictions,
            trade_tapes=trade_tapes,
            score_threshold=min(score_thresholds),
            one_sided_threshold=max(one_sided_thresholds),
            quote_off_score_threshold=min(quote_off_thresholds),
            toxicity_threshold=max(toxicity_thresholds),
            one_sided_toxicity_threshold=max(one_sided_toxicity_thresholds),
            stationary_probability_cutoff=max(stationary_cutoffs),
            inventory_penalty_ticks=0.0,
            quote_step_ticks=args.quote_step_ticks,
            max_inventory=max(max_inventories),
            max_holding_steps=args.max_holding_steps,
            adverse_score_threshold=args.adverse_score_threshold,
            latency_steps=latency,
            quote_confirmation_steps=0,
            alpha_smoothing_window=1,
            symbol_score_thresholds=symbol_score_thresholds,
            symbol_one_sided_thresholds=symbol_one_sided_thresholds,
            symbol_quote_off_thresholds=symbol_quote_off_thresholds,
            symbol_toxicity_thresholds=symbol_toxicity_thresholds,
            symbol_one_sided_toxicity_thresholds=symbol_one_sided_toxicity_thresholds,
            symbol_stationary_cutoffs=symbol_stationary_cutoffs,
            symbol_quote_confirmation_steps=symbol_quote_confirmation_steps,
            symbol_alpha_smoothing_windows=symbol_alpha_smoothing,
            symbol_inventory_penalties=symbol_inventory_penalties,
            symbol_max_inventory=symbol_max_inventory,
            symbol_end_times=symbol_end_times,
            symbol_alpha_coefficients=symbol_alpha_coefficients,
            symbol_stop_loss_ticks=symbol_stop_loss_ticks,
            symbol_take_profit_ticks=symbol_take_profit_ticks,
            symbol_alpha_decay_thresholds=symbol_alpha_decay_threshold,
            regime_volatility_thresholds=regime_volatility_thresholds or None,
            regime_trade_intensity_thresholds=regime_trade_intensity_thresholds or None,
            regime_min_visible_queue=regime_min_visible_queue or None,
            regime_max_cancel_pressure=regime_max_cancel_pressure or None,
            allowed_symbols=set(trade_symbols),
        )
        prefix = f"hybrid_mm_test_latency_{latency}"
        test_trade_log.to_csv(output_dir / f"{prefix}_trade_log.csv", index=False)
        test_by_symbol.to_csv(output_dir / f"{prefix}_by_symbol.csv", index=False)
        test_quote_log.to_csv(output_dir / f"{prefix}_quote_log.csv", index=False)
        with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(test_summary, handle, indent=2)
        latency_rows.append({"latency_steps": latency, **test_summary})

    latency_df = pd.DataFrame(latency_rows).sort_values("latency_steps").reset_index(drop=True)
    latency_df.to_csv(output_dir / "hybrid_mm_test_latency_sweep.csv", index=False)
    best_latency_row = latency_df.sort_values(["net_pnl_ticks", "avg_pnl_ticks"], ascending=[False, False]).iloc[0].to_dict()

    summary = {
        "trade_symbols": trade_symbols,
        "best_by_symbol": best_by_symbol.to_dict(orient="records"),
        "best_latency_result": best_latency_row,
    }
    with (output_dir / "hybrid_mm_overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
