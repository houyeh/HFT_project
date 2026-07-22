from __future__ import annotations

import torch
from torch import nn


class PHCLiteSTP(nn.Module):
    """PHC-lite with STP-like dynamic synaptic gain.

    This is a controlled ablation of PHC-lite v1. It is not full PHCSSM and
    not full biological STP because it does not implement biological
    short-term plasticity state variables. The only extension is a causal
    per-timestep gain applied to the synaptic message using current encoded
    input and previous recurrent state.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        mtl_steps: int = 1,
        dropout: float = 0.0,
        initial_decay: float = 0.90,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mtl_steps = int(mtl_steps)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)
        self.synapse = nn.Linear(hidden_dim, hidden_dim)
        self.gain_from_encoded = nn.Linear(hidden_dim, hidden_dim)
        self.gain_from_prev_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.loop_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 2)
        initial_decay = min(max(float(initial_decay), 1e-4), 1.0 - 1e-4)
        logit_decay = torch.logit(torch.tensor(initial_decay, dtype=torch.float32))
        self.decay_param = nn.Parameter(logit_decay.repeat(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        encoded_t = self.encoder(x_t)
        gate_t = torch.sigmoid(self.gate(encoded_t))
        candidate_t = torch.tanh(self.candidate(encoded_t))
        decay = torch.sigmoid(self.decay_param).view(1, -1)
        state_temporal = decay * state + gate_t * candidate_t
        z = state_temporal
        gain_t = torch.sigmoid(self.gain_from_encoded(encoded_t) + self.gain_from_prev_state(state))
        for _ in range(self.mtl_steps):
            message = torch.nn.functional.silu(self.synapse(z))
            message = self.dropout(gain_t * message)
            z = self.loop_norm(z + self.alpha * message)
        return z

    def step_with_gain(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_t = self.encoder(x_t)
        gate_t = torch.sigmoid(self.gate(encoded_t))
        candidate_t = torch.tanh(self.candidate(encoded_t))
        decay = torch.sigmoid(self.decay_param).view(1, -1)
        state_temporal = decay * state + gate_t * candidate_t
        z = state_temporal
        gain_t = torch.sigmoid(self.gain_from_encoded(encoded_t) + self.gain_from_prev_state(state))
        for _ in range(self.mtl_steps):
            message = torch.nn.functional.silu(self.synapse(z))
            message = self.dropout(gain_t * message)
            z = self.loop_norm(z + self.alpha * message)
        return z, gain_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        state = x.new_zeros((batch, self.hidden_dim))
        for t in range(seq_len):
            state = self.step(x[:, t, :], state)
        return self.readout(state)

    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit streaming alias for equivalence tests."""
        return self.forward(x)

    def forward_with_gains(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        state = x.new_zeros((batch, self.hidden_dim))
        gains = []
        for t in range(seq_len):
            state, gain_t = self.step_with_gain(x[:, t, :], state)
            gains.append(gain_t)
        return self.readout(state), torch.stack(gains, dim=1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
