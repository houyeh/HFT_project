from __future__ import annotations

import pandas as pd


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
