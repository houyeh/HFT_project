from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from new_era.models_playaround_15.config import (
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSet,
    LabelConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    WindowConfig,
)
from new_era.models_playaround_15.paths import get_project_root


FEATURE_COUNTS: tuple[int, ...] = (6, 7, 8, 9)

LABEL_TYPES_BATCH_1: tuple[str, ...] = (
    "mid_move_3class",
    "smoothed_mid_move_3class",
    "toxic_bid_binary",
    "toxic_ask_binary",
)

LABEL_TYPES_BATCH_2: tuple[str, ...] = (
    "adverse_bid_fill_binary",
    "adverse_ask_fill_binary",
    "active_buy_profitable_binary",
    "active_sell_profitable_binary",
)


def default_base_config() -> ExperimentConfig:
    project_root = get_project_root()
    return ExperimentConfig(
        data=DataConfig(data_dir=project_root / "data" / "NEW", symbols=("TXFB6",), zscore_window=2000),
        window=WindowConfig(window_size=100, horizon=10, threshold_ticks=1.0, sample_stride_train=5),
        split=SplitConfig(train_session_frac=0.70, val_session_frac=0.15),
        features=FeatureSet(feature_count=8),
        labels=LabelConfig(label_type="mid_move_3class", adverse_threshold_ticks=1.0),
        model=ModelConfig(channel_list=(16, 16, 16), kernel_size=3, dropout=0.1),
        train=TrainingConfig(epochs=30, batch_size=1024, lr=1e-3, seed=42, device="auto"),
        backtest=BacktestConfig(
            symbol="TXFB6",
            point_value_twd_per_point=200.0,
            maker_fee_twd=15.0,
            taker_fee_twd=50.0,
            max_inventory=3,
            sample_stride=1,
            allow_active_alpha=False,
        ),
    )


def iter_experiment_configs(
    *,
    batch: int = 1,
    feature_counts: Iterable[int] = FEATURE_COUNTS,
    base: ExperimentConfig | None = None,
) -> Iterable[ExperimentConfig]:
    if base is None:
        base = default_base_config()

    if batch == 1:
        label_types: Iterable[str] = LABEL_TYPES_BATCH_1
    elif batch == 2:
        label_types = LABEL_TYPES_BATCH_2
    else:
        raise ValueError(f"batch must be 1 or 2; got {batch}")

    for feature_count in feature_counts:
        for label_type in label_types:
            yield replace(
                base,
                features=FeatureSet(feature_count=int(feature_count)),
                labels=replace(base.labels, label_type=label_type),  # type: ignore[arg-type]
            )
