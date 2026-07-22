from __future__ import annotations

import torch
from torch import nn


class PHCSTPStepHLS(nn.Module):
    """Single-step PHCLiteSTP inference graph for hls4ml experiments.

    Input shape: ``(B, feature_dim + hidden_dim)`` where the columns are
    ``cat([x_t, state_prev])``.

    Output shape: ``(B, 2 + hidden_dim)`` where the columns are
    ``cat([bid_ask_readout, state_next])``.

    This module intentionally exposes one recurrent step as a pure feed-forward
    graph. It contains no Python timestep loop and no tensor-value control flow.
    The recurrent constants are stored as buffers after weight migration.
    """

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_x = nn.Linear(self.feature_dim + self.hidden_dim, self.feature_dim, bias=False)
        self.input_state = nn.Linear(self.feature_dim + self.hidden_dim, self.hidden_dim, bias=False)
        self.enc_linear = nn.Linear(self.feature_dim, self.hidden_dim)
        self.enc_norm = nn.LayerNorm(self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.candidate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_encoded = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_prev_state = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.synapse = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.loop_norm = nn.LayerNorm(self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, 2)
        self.output_pack = nn.Linear(self.hidden_dim, 2 + self.hidden_dim)
        self.register_buffer("decay", torch.zeros(self.hidden_dim))
        self.register_buffer("alpha_scalar", torch.zeros(1))
        self._init_fixed_plumbing()

    def _init_fixed_plumbing(self) -> None:
        with torch.no_grad():
            self.input_x.weight.zero_()
            self.input_x.weight[:, : self.feature_dim].copy_(torch.eye(self.feature_dim))
            self.input_state.weight.zero_()
            self.input_state.weight[:, self.feature_dim :].copy_(torch.eye(self.hidden_dim))
            self.output_pack.weight.zero_()
            self.output_pack.bias.zero_()
            self.output_pack.weight[2:, :].copy_(torch.eye(self.hidden_dim))
        for parameter in self.input_x.parameters():
            parameter.requires_grad = False
        for parameter in self.input_state.parameters():
            parameter.requires_grad = False
        for parameter in self.output_pack.parameters():
            parameter.requires_grad = False

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        x_t = self.input_x(x_cat)
        state_prev = self.input_state(x_cat)

        encoded_pre = self.enc_norm(self.enc_linear(x_t))
        encoded = encoded_pre * torch.sigmoid(encoded_pre)
        gate_t = torch.sigmoid(self.gate(encoded))
        candidate_t = torch.tanh(self.candidate(encoded))
        state_temporal = self.decay * state_prev + gate_t * candidate_t

        gain_t = torch.sigmoid(self.gain_from_encoded(encoded) + self.gain_from_prev_state(state_prev))
        message_pre = self.synapse(state_temporal)
        message = message_pre * torch.sigmoid(message_pre)
        msg_gated = gain_t * message
        z = self.loop_norm(state_temporal + self.alpha_scalar * msg_gated)

        return self.output_pack(z)

    def split_output(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``(B, 2 + hidden_dim)`` into readout and next state."""
        return out[..., :2], out[..., 2:]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
