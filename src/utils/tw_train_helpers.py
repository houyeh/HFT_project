from __future__ import annotations

import pandas as pd
import torch

from src.features.dimensionless_features import build_feature_dataframe
from src.features.execution_features import align_trade_features_to_quotes, compute_queue_imbalance


BASE_MODEL_FEATURE_COLS = [
    "normalized_spread",
    "micro_return",
    "obi",
    "weighted_mid_deviation",
    "deep_obi",
]
STRATEGY_AWARE_FEATURE_COLS = BASE_MODEL_FEATURE_COLS + [
    "rolling_volatility",
    "obi_change",
    "near_depth_obi",
    "book_slope_imbalance",
]
EXECUTION_AWARE_FEATURE_COLS = BASE_MODEL_FEATURE_COLS + [
    "queue_imbalance",
    "queue_imbalance_change",
    "aggressor_imbalance_interval",
    "rolling_signed_flow_ratio",
    "trade_intensity_short",
    "net_cancel_pressure",
    "cancel_pressure_short",
]
FULL_MICROSTRUCTURE_FEATURE_COLS = STRATEGY_AWARE_FEATURE_COLS + [
    "queue_imbalance",
    "queue_imbalance_change",
    "aggressor_imbalance_interval",
    "rolling_signed_flow_ratio",
    "trade_intensity_short",
    "net_cancel_pressure",
    "cancel_pressure_short",
]


def build_tw_feature_frame(
    lob_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    tick_size: float,
) -> pd.DataFrame:
    base = build_feature_dataframe(lob_df, symbol=symbol, tick_size=tick_size)
    base["weighted_mid_deviation"] = (base["weighted_mid_price"] - base["mid_price"]) / tick_size
    rolling_vol = (
        base["micro_return"]
        .rolling(window=50, min_periods=5)
        .std()
        .bfill()
        .fillna(0.0)
        .astype(float)
    )
    feature_df = pd.DataFrame(
        {
            "timestamp": base["timestamp"],
            "mid_price": base["mid_price"],
            "normalized_spread": base["normalized_spread"],
            "micro_return": base["micro_return"],
            "obi": base["obi"],
            "weighted_mid_deviation": base["weighted_mid_deviation"],
            "deep_obi": base["deep_obi"],
            "rolling_volatility": rolling_vol,
            "bid_px_1": lob_df["bid_px_1"].astype(float).values,
            "ask_px_1": lob_df["ask_px_1"].astype(float).values,
            "bid_vol_1": lob_df["bid_vol_1"].astype(float).values,
            "ask_vol_1": lob_df["ask_vol_1"].astype(float).values,
        }
    )
    feature_df["obi_change"] = feature_df["obi"].diff().fillna(0.0)
    near_bid = lob_df["bid_vol_1"].astype(float) + lob_df["bid_vol_2"].astype(float)
    near_ask = lob_df["ask_vol_1"].astype(float) + lob_df["ask_vol_2"].astype(float)
    near_total = (near_bid + near_ask).replace(0, 1.0)
    feature_df["near_depth_obi"] = ((near_bid - near_ask) / near_total).astype(float)

    bid_depth_span = (lob_df["bid_px_1"].astype(float) - lob_df["bid_px_5"].astype(float)) / tick_size
    ask_depth_span = (lob_df["ask_px_5"].astype(float) - lob_df["ask_px_1"].astype(float)) / tick_size
    bid_depth_vol = (
        lob_df[[f"bid_vol_{level}" for level in range(1, 6)]].astype(float).sum(axis=1).replace(0, 1.0)
    )
    ask_depth_vol = (
        lob_df[[f"ask_vol_{level}" for level in range(1, 6)]].astype(float).sum(axis=1).replace(0, 1.0)
    )
    feature_df["book_slope_imbalance"] = ((bid_depth_span / bid_depth_vol) - (ask_depth_span / ask_depth_vol)).astype(float)
    feature_df["queue_imbalance"] = compute_queue_imbalance(lob_df).astype(float)
    feature_df["queue_imbalance_change"] = feature_df["queue_imbalance"].diff().fillna(0.0)

    trade_features = align_trade_features_to_quotes(
        quotes=lob_df,
        trades=trade_df,
        tick_size=tick_size,
        rolling_window=20,
    )
    for column in trade_features.columns:
        feature_df[column] = trade_features[column].astype(float).values
    return feature_df


def compute_class_weights(dataset) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float32)
    for index in range(len(dataset)):
        label = int(dataset[index]["y"])
        counts[label] += 1

    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = counts.sum() / (counts.numel() * counts)
    return weights


def feature_columns_from_name(feature_set: str) -> list[str]:
    if feature_set == "base":
        return BASE_MODEL_FEATURE_COLS
    if feature_set == "strategy_aware":
        return STRATEGY_AWARE_FEATURE_COLS
    if feature_set == "execution_aware":
        return EXECUTION_AWARE_FEATURE_COLS
    if feature_set == "full_microstructure":
        return FULL_MICROSTRUCTURE_FEATURE_COLS
    raise ValueError(f"Unknown feature set: {feature_set}")


def parse_time_to_ns(text: str) -> int | None:
    if not text:
        return None
    hour, minute, second = [int(part) for part in text.split(":")]
    return (
        hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + second * 1_000_000_000
    )
