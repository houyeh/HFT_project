from __future__ import annotations

from dataclasses import dataclass


FEATURE_IMPORTANCE_RANKING: tuple[str, ...] = (
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
    "rolling_signed_flow_ratio",
    "micro_return",
    "net_cancel_pressure",
    "aggressor_imbalance_interval",
)


def select_top_features(feature_count: int) -> list[str]:
    if feature_count not in (6, 7, 8, 9):
        raise ValueError(f"feature_count must be one of (6, 7, 8, 9); got {feature_count}")
    return list(FEATURE_IMPORTANCE_RANKING[: int(feature_count)])


@dataclass(frozen=True)
class FeatureSet:
    feature_count: int = 8

    @property
    def selected_features(self) -> list[str]:
        return select_top_features(self.feature_count)

    @property
    def z_features(self) -> list[str]:
        return [f"{name}_z" for name in self.selected_features]

