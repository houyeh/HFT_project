from __future__ import annotations

import math

import torch
from torch import nn


class DaleSynapse(nn.Module):
    """Linear synapse layer with Dale's Law on presynaptic outgoing columns.

    For message = z @ W_eff.T + bias, presynaptic unit j corresponds to column j
    of W_eff. Dale's Law is enforced by assigning each column one fixed sign.
    """

    def __init__(self, hidden_dim: int, excitatory_ratio: float = 0.8) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        excitatory_count = int(round(hidden_dim * excitatory_ratio))
        excitatory_count = min(max(excitatory_count, 1), hidden_dim - 1)
        sign = torch.ones(hidden_dim, dtype=torch.float32)
        sign[excitatory_count:] = -1.0
        self.register_buffer("dale_sign", sign)
        self.raw_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))

    def effective_weight(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_weight) * self.dale_sign.view(1, -1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(z, self.effective_weight(), self.bias)


class PHCLiteDale(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        mtl_steps: int = 1,
        dropout: float = 0.0,
        initial_decay: float = 0.90,
        excitatory_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mtl_steps = int(mtl_steps)
        self.excitatory_ratio = float(excitatory_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)
        self.synapse = DaleSynapse(hidden_dim, excitatory_ratio=excitatory_ratio)
        self.loop_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 2)
        initial_decay = min(max(float(initial_decay), 1e-4), 1.0 - 1e-4)
        self.decay_param = nn.Parameter(torch.logit(torch.tensor(initial_decay, dtype=torch.float32)).repeat(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        state = x.new_zeros((batch, self.hidden_dim))
        decay = torch.sigmoid(self.decay_param).view(1, -1)
        flat = x.reshape(batch * seq_len, self.feature_dim)
        encoded = self.encoder(flat).reshape(batch, seq_len, self.hidden_dim)
        gates = torch.sigmoid(self.gate(encoded.reshape(batch * seq_len, self.hidden_dim))).reshape(batch, seq_len, self.hidden_dim)
        candidates = torch.tanh(self.candidate(encoded.reshape(batch * seq_len, self.hidden_dim))).reshape(batch, seq_len, self.hidden_dim)
        for t in range(seq_len):
            state_temporal = decay * state + gates[:, t, :] * candidates[:, t, :]
            z = state_temporal
            for _ in range(self.mtl_steps):
                message = torch.nn.functional.silu(self.synapse(z))
                message = self.dropout(message)
                z = self.loop_norm(z + self.alpha * message)
            state = z
        return self.readout(state)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
