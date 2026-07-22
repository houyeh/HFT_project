from __future__ import annotations

import torch
from torch import nn


class PHCDiagonalSSM(nn.Module):
    """PHC-style encoder/readout with a diagonal per-channel state transition.

    This is intentionally weaker than PHCLiteSTP's dense recurrent core. The
    recurrence is O(hidden_dim) per step:

        state_t = decay * state_{t-1} + gain * encoded_t

    There is no hidden_dim x hidden_dim recurrent matmul, no dense synapse, and
    no cross-channel recurrence. Cross-channel mixing exists only in the input
    encoder and final readout, preserving the same I/O shape as the dense PHC
    baseline while isolating the state-transition structure.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        mtl_steps: int = 1,
        dropout: float = 0.0,
        initial_decay: float = 0.90,
        initial_gain: float = 0.50,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mtl_steps = int(mtl_steps)
        if self.mtl_steps != 1:
            raise ValueError("PHCDiagonalSSM keeps mtl_steps for protocol compatibility but only supports mtl_steps=1")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 2)
        initial_decay = min(max(float(initial_decay), 1e-4), 1.0 - 1e-4)
        initial_gain = min(max(float(initial_gain), 1e-4), 1.0 - 1e-4)
        self.diag_decay_param = nn.Parameter(torch.logit(torch.full((hidden_dim,), initial_decay, dtype=torch.float32)))
        self.diag_gain_param = nn.Parameter(torch.logit(torch.full((hidden_dim,), initial_gain, dtype=torch.float32)))

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        encoded_t = self.encoder(x_t)
        decay = torch.sigmoid(self.diag_decay_param).view(1, -1)
        gain = torch.sigmoid(self.diag_gain_param).view(1, -1)
        next_state = decay * state + gain * encoded_t
        return self.dropout(next_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        state = x.new_zeros((batch, self.hidden_dim))
        for t in range(seq_len):
            state = self.step(x[:, t, :], state)
        return self.readout(state)

    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
