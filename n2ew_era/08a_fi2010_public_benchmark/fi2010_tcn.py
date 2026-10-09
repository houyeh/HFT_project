from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TCNFI2010Block(nn.Module):
    """Two causal convs plus residual connection for FI-2010 windows."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv1 = nn.Conv1d(self.channels, self.channels, self.kernel_size, dilation=self.dilation)
        self.conv2 = nn.Conv1d(self.channels, self.channels, self.kernel_size, dilation=self.dilation)
        self.dropout = nn.Dropout(float(dropout))

    def causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.causal_conv(self.conv1, x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.causal_conv(self.conv2, out)
        out = F.relu(out)
        out = self.dropout(out)
        return F.relu(out + residual)


class TCNFI2010Classifier(nn.Module):
    """Feed-forward dilated causal-conv classifier for FI-2010 Setup 2.

    Input shape is ``(batch, time=100, feature=40)``. The 40 official FI-2010
    z-score LOB features are projected to 16 channels at every timestep to keep
    hidden width aligned with ``fi2010_phc_lite.py``. The model emits logits
    from the final timestep only, matching the benchmark decision target at the
    terminal event of each window.
    """

    DILATIONS = [1, 2, 4, 8, 16]

    def __init__(
        self,
        feature_dim: int = 40,
        channels: int = 16,
        num_classes: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        input_output_bias: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.kernel_size = int(kernel_size)
        self.dilations = list(self.DILATIONS)
        self.dropout_p = float(dropout)
        self.input_output_bias = bool(input_output_bias)

        self.input_proj = nn.Linear(self.feature_dim, self.channels, bias=self.input_output_bias)
        self.blocks = nn.ModuleList(
            [TCNFI2010Block(self.channels, self.kernel_size, d, self.dropout_p) for d in self.dilations]
        )
        self.readout = nn.Linear(self.channels, self.num_classes, bias=self.input_output_bias)
        rf = self.receptive_field()
        if rf < 100:
            raise ValueError(f"Receptive field {rf} does not cover FI-2010 WINDOW=100")

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        if x_window.ndim != 3:
            raise ValueError(f"Expected (batch, time, feature), got {tuple(x_window.shape)}")
        if x_window.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {x_window.shape[-1]}")
        x = self.input_proj(x_window)
        x = x.transpose(1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        terminal = x[:, :, -1]
        return self.readout(terminal)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
