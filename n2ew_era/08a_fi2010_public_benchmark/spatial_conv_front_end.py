from __future__ import annotations

import torch
from torch import nn

from fi2010_phc_lite import PHCLiteFI2010Classifier
from fi2010_tcn import TCNFI2010Classifier


class SpatialConvFrontEnd(nn.Module):
    """Lightweight level-aware FI-2010 input front end.

    FI-2010 rows have 40 features arranged as 10 price levels with four values
    per level: ask_price, ask_volume, bid_price, bid_volume. The first
    convolution is applied within each level to pair price/volume values. The
    second convolution then sees the two side-pair channels at every level and
    strides across adjacent price levels.
    """

    def __init__(self, channels: int = 8, levels: int = 10, values_per_level: int = 4) -> None:
        super().__init__()
        if levels % 2 != 0:
            raise ValueError("levels must be even for stride-2 adjacent-level convolution")
        if values_per_level != 4:
            raise ValueError("FI-2010 front end expects four values per level")
        self.channels = int(channels)
        self.levels = int(levels)
        self.values_per_level = int(values_per_level)
        self.input_dim = self.levels * self.values_per_level
        self.output_dim = self.channels * (self.levels // 2)

        self.pair_conv = nn.Conv1d(1, self.channels, kernel_size=2, stride=2)
        self.level_conv = nn.Conv1d(2 * self.channels, self.channels, kernel_size=2, stride=2)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dimension {self.input_dim}, got {x.shape[-1]}")
        prefix = x.shape[:-1]
        flat = x.reshape(-1, 1, self.input_dim)

        # (N, 1, 40) -> (N, C, 20), pairing price/volume in flattened level-major order.
        paired = self.activation(self.pair_conv(flat))

        # (N, C, levels, 2) -> (N, 2C, levels), so level_conv strides over levels.
        paired = paired.reshape(-1, self.channels, self.levels, 2)
        paired = paired.permute(0, 1, 3, 2).contiguous().reshape(-1, 2 * self.channels, self.levels)
        crossed = self.activation(self.level_conv(paired))
        return crossed.flatten(1).reshape(*prefix, self.output_dim)


class SpatialPHCLiteFI2010Classifier(nn.Module):
    """PHC-lite with a learned spatial convolutional FI-2010 front end."""

    def __init__(
        self,
        front_channels: int = 8,
        hidden_dim: int = 16,
        num_classes: int = 3,
        initial_decay: float = 0.90,
        initial_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.front_end = SpatialConvFrontEnd(channels=front_channels)
        self.core = PHCLiteFI2010Classifier(
            feature_dim=self.front_end.output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            initial_decay=initial_decay,
            initial_alpha=initial_alpha,
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.core.hidden_dim)

    @property
    def decay(self) -> torch.Tensor:
        return self.core.decay

    @property
    def alpha_scalar(self) -> torch.Tensor:
        return self.core.alpha_scalar

    def forward_step(self, x_t: torch.Tensor, state_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.forward_step(self.front_end(x_t), state_prev)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        return self.core(self.front_end(x_window))


class SpatialTCNFI2010Classifier(nn.Module):
    """TCN small with the same learned spatial convolutional FI-2010 front end."""

    def __init__(
        self,
        front_channels: int = 8,
        channels: int = 8,
        num_classes: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        input_output_bias: bool = False,
    ) -> None:
        super().__init__()
        self.front_end = SpatialConvFrontEnd(channels=front_channels)
        self.core = TCNFI2010Classifier(
            feature_dim=self.front_end.output_dim,
            channels=channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dropout=dropout,
            input_output_bias=input_output_bias,
        )

    def receptive_field(self) -> int:
        return int(self.core.receptive_field())

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        return self.core(self.front_end(x_window))


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
