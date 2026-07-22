from __future__ import annotations

import torch
import torch.nn as nn


class TCNPrePadBlock(nn.Module):
    """Two valid dilated convs with a cropped residual path.

    The caller prepends all causal padding before entering the model, so this
    block contains no ConstantPad1d and no F.pad. Each valid conv reduces time
    by ``(kernel_size - 1) * dilation``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.reduction = 2 * (self.kernel_size - 1) * self.dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        residual = x[:, :, -out.size(2) :].contiguous()
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNPrePadRegressor(nn.Module):
    """Five-block valid-convolution TCN for pre-padded causal inputs.

    Expected input shape is ``(B, F, 252)``. The final output time length is
    exactly 128 after reductions of 4 * sum([1, 2, 4, 8, 16]) = 124.
    """

    DILATIONS = [1, 2, 4, 8, 16]
    TOTAL_PAD = 124
    SEQ_LEN_OUT = 128
    SEQ_LEN_IN = SEQ_LEN_OUT + TOTAL_PAD

    def __init__(self, num_inputs: int, num_outputs: int, channels: int = 16, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = list(self.DILATIONS)
        layers: list[nn.Module] = []
        in_ch = self.num_inputs
        for dilation in self.dilations:
            layers.append(TCNPrePadBlock(in_ch, self.channels, self.kernel_size, dilation, dropout))
            in_ch = self.channels
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AvgPool1d(kernel_size=self.SEQ_LEN_OUT)
        self.fc = nn.Linear(self.channels, self.num_outputs)
        rf = self.receptive_field()
        if rf != 125:
            raise ValueError(f"TCNPrePadRegressor receptive field is {rf}, expected 125")

    def prepad(self, x_channel_first: torch.Tensor) -> torch.Tensor:
        """Prepend causal zeros to ``(B, F, 128)`` and return ``(B, F, 252)``."""
        zeros = torch.zeros(
            x_channel_first.size(0),
            x_channel_first.size(1),
            self.TOTAL_PAD,
            device=x_channel_first.device,
            dtype=x_channel_first.dtype,
        )
        return torch.cat([zeros, x_channel_first], dim=2)

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,F,T), got {tuple(x.shape)}")
        if x.size(2) != self.SEQ_LEN_IN:
            raise ValueError(f"Expected pre-padded time length {self.SEQ_LEN_IN}, got {x.size(2)}")
        x = self.tcn(x)
        if x.size(2) != self.SEQ_LEN_OUT:
            raise ValueError(f"Expected output time length {self.SEQ_LEN_OUT}, got {x.size(2)}")
        x = self.gap(x).squeeze(2)
        return self.fc(x)

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
