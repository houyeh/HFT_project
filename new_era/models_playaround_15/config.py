from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from new_era.models_playaround_15.features import FeatureSet

LabelType = Literal[
    "mid_move_3class",
    "smoothed_mid_move_3class",
    "toxic_bid_binary",
    "toxic_ask_binary",
    "adverse_bid_fill_binary",
    "adverse_ask_fill_binary",
    "active_buy_profitable_binary",
    "active_sell_profitable_binary",
]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    symbols: tuple[str, ...] = ("TXFB6",)
    zscore_window: int = 2000


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 100
    horizon: int = 10
    threshold_ticks: float = 1.0
    sample_stride_train: int = 5


@dataclass(frozen=True)
class SplitConfig:
    train_session_frac: float = 0.70
    val_session_frac: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    channel_list: tuple[int, ...] = (16, 16, 16)
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "TXFB6"
    point_value_twd_per_point: float = 200.0
    maker_fee_twd: float = 15.0
    taker_fee_twd: float = 50.0
    max_inventory: int = 3
    sample_stride: int = 1
    allow_active_alpha: bool = False


@dataclass(frozen=True)
class LabelConfig:
    label_type: LabelType = "mid_move_3class"
    adverse_threshold_ticks: float = 1.0
    taker_fee_twd: float = 50.0
    point_value_twd_per_point: float = 200.0
    exit_cost_ticks: float = 0.0
    margin_ticks: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureSet = field(default_factory=FeatureSet)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def num_inputs(self) -> int:
        return len(self.features.selected_features)

    @property
    def num_classes(self) -> int:
        if self.labels.label_type in ("mid_move_3class", "smoothed_mid_move_3class"):
            return 3
        return 2

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)
