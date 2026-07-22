"""Lazy PyTorch datasets for chronologically split event-window training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from src.features.sliding_window import label_direction, label_direction_dynamic
from src.features.label_generation import LabelType, generate_labels


class WindowedLOBSequenceDataset(Dataset):
    """
    Lazy sliding-window dataset backed by a feature DataFrame.

    This avoids materializing the full 3D tensor in memory, which is important
    for event-level Taiwan replay data.
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        symbol: str,
        tick_size: float,
        window_size: int,
        horizon: int,
        threshold_ticks: float,
        sample_stride: int = 1,
        feature_cols: list[str] | None = None,
        threshold_mode: str = "static",
        threshold_spread_multiplier: float = 0.0,
        threshold_volatility_multiplier: float = 0.0,
        label_type: LabelType = "mid_move_3class",
        adverse_threshold_ticks: float = 1.0,
        taker_fee_twd: float = 50.0,
        point_value_twd_per_point: float = 200.0,
        exit_cost_ticks: float = 0.0,
        margin_ticks: float = 0.0,
        symbol_weight: float = 1.0,
        nonstationary_weight: float = 1.0,
        return_weight_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_cols is None:
            feature_cols = [col for col in feature_df.columns if col not in ("timestamp", "mid_price")]

        self.symbol = symbol
        self.tick_size = float(tick_size)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.threshold_ticks = float(threshold_ticks)
        self.sample_stride = int(sample_stride)
        self.feature_cols = feature_cols
        self.threshold_mode = threshold_mode
        self.threshold_spread_multiplier = float(threshold_spread_multiplier)
        self.threshold_volatility_multiplier = float(threshold_volatility_multiplier)
        self.label_type = label_type
        self.adverse_threshold_ticks = float(adverse_threshold_ticks)
        self.taker_fee_twd = float(taker_fee_twd)
        self.point_value_twd_per_point = float(point_value_twd_per_point)
        self.exit_cost_ticks = float(exit_cost_ticks)
        self.margin_ticks = float(margin_ticks)
        self.symbol_weight = float(symbol_weight)
        self.nonstationary_weight = float(nonstationary_weight)
        self.return_weight_alpha = float(return_weight_alpha)

        self.feature_matrix = feature_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.mid_prices = feature_df["mid_price"].to_numpy(dtype=np.float64, copy=True)
        self.spread_ticks = feature_df["normalized_spread"].to_numpy(dtype=np.float32, copy=True)
        self.timestamps = feature_df["timestamp"].to_numpy(dtype=np.int64, copy=True)
        self.rolling_volatility = feature_df.get(
            "rolling_volatility",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.bid_px_1 = feature_df.get(
            "bid_px_1",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.ask_px_1 = feature_df.get(
            "ask_px_1",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.bid_vol_1 = feature_df.get(
            "bid_vol_1",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.ask_vol_1 = feature_df.get(
            "ask_vol_1",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.queue_imbalance = feature_df.get(
            "queue_imbalance",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.queue_imbalance_change = feature_df.get(
            "queue_imbalance_change",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.trade_intensity_short = feature_df.get(
            "trade_intensity_short",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.aggressor_imbalance_interval = feature_df.get(
            "aggressor_imbalance_interval",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.bid_cancel_proxy = feature_df.get(
            "bid_cancel_proxy",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.ask_cancel_proxy = feature_df.get(
            "ask_cancel_proxy",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.net_cancel_pressure = feature_df.get(
            "net_cancel_pressure",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)
        self.cancel_pressure_short = feature_df.get(
            "cancel_pressure_short",
            pd.Series(np.zeros(len(feature_df), dtype=np.float32)),
        ).to_numpy(dtype=np.float32, copy=True)

        if self.label_type == "mid_move_3class":
            mid_prices_in_ticks = self.mid_prices / self.tick_size
            threshold_series = self._build_threshold_series()
            if self.threshold_mode == "static":
                self.labels = label_direction(
                    mid_prices=mid_prices_in_ticks,
                    horizon=self.horizon,
                    threshold=self.threshold_ticks,
                )
            else:
                self.labels = label_direction_dynamic(
                    mid_prices=mid_prices_in_ticks,
                    horizon=self.horizon,
                    thresholds=threshold_series,
                )
        else:
            self.labels = generate_labels(
                self.label_type,
                mid_prices=self.mid_prices,
                bid_px_1=self.bid_px_1,
                ask_px_1=self.ask_px_1,
                tick_size=self.tick_size,
                horizon=self.horizon,
                threshold_ticks=self.threshold_ticks,
                adverse_threshold_ticks=self.adverse_threshold_ticks,
                taker_fee_twd=self.taker_fee_twd,
                point_value_twd_per_point=self.point_value_twd_per_point,
                exit_cost_ticks=self.exit_cost_ticks,
                margin_ticks=self.margin_ticks,
            )

        raw_samples = len(self.labels) - self.window_size + 1
        if raw_samples <= 0:
            raise ValueError(
                f"{symbol}: insufficient rows for window_size={window_size} and horizon={horizon}."
            )

        self.sample_starts = np.arange(0, raw_samples, self.sample_stride, dtype=np.int64)

    def _build_threshold_series(self) -> np.ndarray:
        thresholds = np.full(len(self.mid_prices), self.threshold_ticks, dtype=np.float64)
        if self.threshold_mode in {"spread", "spread_volatility"}:
            thresholds += self.threshold_spread_multiplier * self.spread_ticks
        if self.threshold_mode in {"volatility", "spread_volatility"}:
            thresholds += self.threshold_volatility_multiplier * self.rolling_volatility
        return thresholds

    def __len__(self) -> int:
        return len(self.sample_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        start = int(self.sample_starts[index])
        end = start + self.window_size
        label_idx = end - 1
        future_idx = label_idx + self.horizon

        x = torch.from_numpy(self.feature_matrix[start:end].copy())
        y = torch.tensor(int(self.labels[label_idx]), dtype=torch.long)
        current_mid = torch.tensor(float(self.mid_prices[label_idx]), dtype=torch.float32)
        future_mid = torch.tensor(float(self.mid_prices[future_idx]), dtype=torch.float32)
        future_return_ticks = torch.tensor(
            float((self.mid_prices[future_idx] - self.mid_prices[label_idx]) / self.tick_size),
            dtype=torch.float32,
        )
        spread_ticks = torch.tensor(float(self.spread_ticks[label_idx]), dtype=torch.float32)
        rolling_volatility = torch.tensor(float(self.rolling_volatility[label_idx]), dtype=torch.float32)
        timestamp = torch.tensor(int(self.timestamps[label_idx]), dtype=torch.int64)
        tick_size = torch.tensor(self.tick_size, dtype=torch.float32)
        bid_px_1 = torch.tensor(float(self.bid_px_1[label_idx]), dtype=torch.float32)
        ask_px_1 = torch.tensor(float(self.ask_px_1[label_idx]), dtype=torch.float32)
        bid_vol_1 = torch.tensor(float(self.bid_vol_1[label_idx]), dtype=torch.float32)
        ask_vol_1 = torch.tensor(float(self.ask_vol_1[label_idx]), dtype=torch.float32)
        queue_imbalance = torch.tensor(float(self.queue_imbalance[label_idx]), dtype=torch.float32)
        queue_imbalance_change = torch.tensor(float(self.queue_imbalance_change[label_idx]), dtype=torch.float32)
        trade_intensity_short = torch.tensor(float(self.trade_intensity_short[label_idx]), dtype=torch.float32)
        aggressor_imbalance_interval = torch.tensor(
            float(self.aggressor_imbalance_interval[label_idx]),
            dtype=torch.float32,
        )
        bid_cancel_proxy = torch.tensor(float(self.bid_cancel_proxy[label_idx]), dtype=torch.float32)
        ask_cancel_proxy = torch.tensor(float(self.ask_cancel_proxy[label_idx]), dtype=torch.float32)
        net_cancel_pressure = torch.tensor(float(self.net_cancel_pressure[label_idx]), dtype=torch.float32)
        cancel_pressure_short = torch.tensor(float(self.cancel_pressure_short[label_idx]), dtype=torch.float32)
        sample_weight_value = self.symbol_weight
        if int(self.labels[label_idx]) != 1:
            sample_weight_value *= self.nonstationary_weight
        sample_weight_value *= 1.0 + self.return_weight_alpha * abs(float(future_return_ticks))
        sample_weight = torch.tensor(float(sample_weight_value), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "current_mid": current_mid,
            "future_mid": future_mid,
            "future_return_ticks": future_return_ticks,
            "spread_ticks": spread_ticks,
            "rolling_volatility": rolling_volatility,
            "timestamp": timestamp,
            "tick_size": tick_size,
            "bid_px_1": bid_px_1,
            "ask_px_1": ask_px_1,
            "bid_vol_1": bid_vol_1,
            "ask_vol_1": ask_vol_1,
            "queue_imbalance": queue_imbalance,
            "queue_imbalance_change": queue_imbalance_change,
            "trade_intensity_short": trade_intensity_short,
            "aggressor_imbalance_interval": aggressor_imbalance_interval,
            "bid_cancel_proxy": bid_cancel_proxy,
            "ask_cancel_proxy": ask_cancel_proxy,
            "net_cancel_pressure": net_cancel_pressure,
            "cancel_pressure_short": cancel_pressure_short,
            "sample_weight": sample_weight,
            "symbol": self.symbol,
        }


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    datasets_by_symbol: dict[str, WindowedLOBSequenceDataset]
    split_sizes: dict[str, dict[str, int]]


def chronological_split_indices(total_size: int, train_frac: float, val_frac: float) -> tuple[range, range, range]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    train_end = max(1, int(total_size * train_frac))
    val_end = max(train_end + 1, int(total_size * (train_frac + val_frac)))
    val_end = min(val_end, total_size - 1)

    return (
        range(0, train_end),
        range(train_end, val_end),
        range(val_end, total_size),
    )


def _concat_or_single(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_symbol_datasets(
    feature_frames: dict[str, pd.DataFrame],
    tick_sizes: dict[str, float],
    window_size: int,
    horizon: int,
    threshold_ticks: float,
    sample_stride: int,
    train_frac: float,
    val_frac: float,
    feature_cols: list[str] | None = None,
    threshold_mode: str = "static",
    threshold_spread_multiplier: float = 0.0,
    threshold_volatility_multiplier: float = 0.0,
    symbol_weights: dict[str, float] | None = None,
    nonstationary_weight: float = 1.0,
    return_weight_alpha: float = 0.0,
) -> DatasetSplits:
    train_sets: list[Dataset] = []
    val_sets: list[Dataset] = []
    test_sets: list[Dataset] = []
    raw_datasets: dict[str, WindowedLOBSequenceDataset] = {}
    split_sizes: dict[str, dict[str, int]] = {}

    for symbol, frame in feature_frames.items():
        dataset = WindowedLOBSequenceDataset(
            feature_df=frame,
            symbol=symbol,
            tick_size=tick_sizes[symbol],
            window_size=window_size,
            horizon=horizon,
            threshold_ticks=threshold_ticks,
            sample_stride=sample_stride,
            feature_cols=feature_cols,
            threshold_mode=threshold_mode,
            threshold_spread_multiplier=threshold_spread_multiplier,
            threshold_volatility_multiplier=threshold_volatility_multiplier,
            symbol_weight=1.0 if symbol_weights is None else float(symbol_weights.get(symbol, 1.0)),
            nonstationary_weight=nonstationary_weight,
            return_weight_alpha=return_weight_alpha,
        )
        raw_datasets[symbol] = dataset

        train_idx, val_idx, test_idx = chronological_split_indices(len(dataset), train_frac, val_frac)
        train_sets.append(Subset(dataset, list(train_idx)))
        val_sets.append(Subset(dataset, list(val_idx)))
        test_sets.append(Subset(dataset, list(test_idx)))
        split_sizes[symbol] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }

    return DatasetSplits(
        train=_concat_or_single(train_sets),
        val=_concat_or_single(val_sets),
        test=_concat_or_single(test_sets),
        datasets_by_symbol=raw_datasets,
        split_sizes=split_sizes,
    )
