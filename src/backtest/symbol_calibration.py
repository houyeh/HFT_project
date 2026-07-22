"""Per-symbol search utilities for gating thresholds, time cutoffs, and regime guards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating


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


def _evaluate_symbol_rules(
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
                            allowed_symbols={symbol},
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


def fit_symbol_rule_configuration(
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    allowed_symbols: set[str],
    base_prob_threshold: float,
    prob_threshold_offsets: list[float],
    margin_thresholds: list[float],
    end_times: list[str],
    cooldowns: list[int],
    holding_steps: int,
    passive_wait: int,
    vol_quantiles: list[float],
    trade_quantiles: list[float],
    cancel_quantiles: list[float],
    liq_quantiles: list[float],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    regime_profiles = build_regime_profiles(vol_quantiles, trade_quantiles, cancel_quantiles, liq_quantiles)
    prob_thresholds = [base_prob_threshold + offset for offset in prob_threshold_offsets]
    symbols = sorted(symbol for symbol in predictions["symbol"].unique().tolist() if symbol in allowed_symbols)

    best_rows = []
    for symbol in symbols:
        symbol_predictions = predictions.loc[predictions["symbol"] == symbol].copy().reset_index(drop=True)
        result = _evaluate_symbol_rules(
            symbol=symbol,
            symbol_predictions=symbol_predictions,
            trade_tapes=trade_tapes,
            prob_thresholds=prob_thresholds,
            margin_thresholds=margin_thresholds,
            end_times=end_times,
            cooldowns=cooldowns,
            holding_steps=holding_steps,
            passive_wait=passive_wait,
            regime_profiles=regime_profiles,
        )
        best_rows.append(result.iloc[0].to_dict())

    best_by_symbol = pd.DataFrame(best_rows)
    symbol_prob_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["prob_threshold"]))
    symbol_margin_thresholds = dict(zip(best_by_symbol["symbol"], best_by_symbol["margin_threshold"]))
    symbol_end_times_ns = {
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

    config = {
        "symbol_prob_thresholds": symbol_prob_thresholds,
        "symbol_margin_thresholds": symbol_margin_thresholds,
        "symbol_end_times_text": dict(zip(best_by_symbol["symbol"], best_by_symbol["end_time"])),
        "symbol_end_times_ns": symbol_end_times_ns,
        "symbol_cooldowns": symbol_cooldowns,
        "regime_volatility_thresholds": regime_volatility_thresholds,
        "regime_trade_intensity_thresholds": regime_trade_intensity_thresholds,
        "regime_min_visible_queue": regime_min_visible_queue,
        "regime_max_cancel_pressure": regime_max_cancel_pressure,
        "allowed_symbols": sorted(allowed_symbols),
        "holding_steps": holding_steps,
        "passive_wait": passive_wait,
    }

    gated = apply_probability_gating(
        predictions,
        probability_threshold=base_prob_threshold,
        margin_threshold=min(margin_thresholds),
        symbol_probability_thresholds=symbol_prob_thresholds,
        symbol_margin_thresholds=symbol_margin_thresholds,
        symbol_end_times=symbol_end_times_ns,
        regime_volatility_thresholds=regime_volatility_thresholds or None,
        regime_trade_intensity_thresholds=regime_trade_intensity_thresholds or None,
        regime_min_visible_queue=regime_min_visible_queue or None,
        regime_max_cancel_pressure=regime_max_cancel_pressure or None,
        allowed_symbols=allowed_symbols,
    )
    trade_log, summary, by_symbol = run_event_driven_backtest(
        gated,
        signal_column="gated_signal",
        trade_tapes=trade_tapes,
        entry_mode="passive",
        max_holding_steps=holding_steps,
        max_passive_wait_steps=passive_wait,
        cooldown_steps=0,
        symbol_cooldown_steps=symbol_cooldowns,
    )
    return best_by_symbol, config, gated, trade_log, summary, by_symbol


def save_symbol_calibration_artifacts(
    output_dir: str | Path,
    prefix: str,
    best_by_symbol: pd.DataFrame,
    config: dict,
    gated_predictions: pd.DataFrame,
    trade_log: pd.DataFrame,
    summary: dict,
    by_symbol: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    best_by_symbol.to_csv(output_path / f"{prefix}_symbol_rule_best_by_symbol.csv", index=False)
    gated_predictions.to_csv(output_path / f"{prefix}_symbol_rule_predictions.csv", index=False)
    trade_log.to_csv(output_path / f"{prefix}_symbol_rule_trade_log.csv", index=False)
    by_symbol.to_csv(output_path / f"{prefix}_symbol_rule_by_symbol.csv", index=False)
    with (output_path / f"{prefix}_symbol_rule_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "summary": summary}, handle, indent=2)
