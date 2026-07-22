from __future__ import annotations

import torch
from torch import nn


class PHCLite(nn.Module):
    """PHC-inspired lightweight microstructure sequence model.

    This is not full PHCSSM. It keeps only a streaming recurrent state and a
    small within-timestep multi-transmission loop for channel interaction.
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
        self.loop_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 2)
        initial_decay = min(max(float(initial_decay), 1e-4), 1.0 - 1e-4)
        logit_decay = torch.logit(torch.tensor(initial_decay, dtype=torch.float32))
        self.decay_param = nn.Parameter(logit_decay.repeat(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return bid/ask action-value scores for the final timestep.

        Args:
            x: tensor with shape [batch, seq_len, feature_dim].
        """
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
