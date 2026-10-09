from __future__ import annotations

import torch
from torch import nn


class PHCLiteFI2010Classifier(nn.Module):
    """PHC-lite recurrent core adapted to FI-2010 classification.

    The recurrent equations intentionally match the Phase 06 PHC-lite step.
    Only the input projection width and final readout width differ:
    40 FI-2010 raw LOB features go in, and 3 class logits come out.

    v3 architecture fix: in the v2 implementation, `decay` and `alpha_scalar`
    were non-trainable buffers hardcoded to zero (see
    `results/decay_parameter_survey.md`), so the v2 checkpoints tested a
    reduced, non-recurrent classifier, not the real PHC-lite recursion. v3
    makes both trainable, mirroring the TXF Phase 05A convention exactly:
    `05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py`
    stores `decay_param` as an `nn.Parameter` initialized via
    `logit(initial_decay)` and uses `sigmoid(decay_param)` as the effective
    per-channel memory coefficient. `alpha_scalar` here plays the same role
    as that file's scalar `alpha` gate on the within-timestep message loop.
    """

    def __init__(
        self,
        feature_dim: int = 40,
        hidden_dim: int = 16,
        num_classes: int = 3,
        initial_decay: float = 0.90,
        initial_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.enc_linear = nn.Linear(self.feature_dim, self.hidden_dim)
        self.enc_norm = nn.LayerNorm(self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.candidate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_encoded = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gain_from_prev_state = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.synapse = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.loop_norm = nn.LayerNorm(self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, self.num_classes)

        # v3: decay/alpha are now trainable, initialized away from zero so the
        # model starts as a genuine leaky-integrator recurrence instead of the
        # v2 degenerate decay=0/alpha=0 point. Same logit-init trick as TXF's
        # phc_lite.py so the initial effective decay is `initial_decay`.
        initial_decay = min(max(float(initial_decay), 1e-4), 1.0 - 1e-4)
        logit_decay = torch.logit(torch.tensor(initial_decay, dtype=torch.float32))
        self.decay_param = nn.Parameter(logit_decay.repeat(self.hidden_dim))
        self.alpha_scalar = nn.Parameter(torch.tensor(float(initial_alpha), dtype=torch.float32))

    @property
    def decay(self) -> torch.Tensor:
        """Effective per-channel memory coefficient, sigmoid(decay_param) in (0, 1).

        Exposed as a property (not a plain attribute) so existing call sites
        that read `model.decay` (forward_step, and the v2-era
        `assert_fast_final_step_safe` guard in 02_train_phc_lite_stage1.py)
        keep working unchanged, while the underlying value is now learned.
        """
        return torch.sigmoid(self.decay_param)

    def forward_step(self, x_t: torch.Tensor, state_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        logits = self.readout(z)
        return logits, z

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        if x_window.ndim != 3:
            raise ValueError(f"Expected (batch, time, feature), got shape {tuple(x_window.shape)}")
        batch_size, seq_len, _ = x_window.shape
        state = x_window.new_zeros((batch_size, self.hidden_dim))
        logits = x_window.new_zeros((batch_size, self.num_classes))
        decay = self.decay.view(1, -1)

        flat = x_window.reshape(batch_size * seq_len, self.feature_dim)
        encoded_pre = self.enc_norm(self.enc_linear(flat))
        encoded_flat = encoded_pre * torch.sigmoid(encoded_pre)
        encoded = encoded_flat.reshape(batch_size, seq_len, self.hidden_dim)
        gates = torch.sigmoid(self.gate(encoded_flat)).reshape(batch_size, seq_len, self.hidden_dim)
        candidates = torch.tanh(self.candidate(encoded_flat)).reshape(batch_size, seq_len, self.hidden_dim)

        for t in range(seq_len):
            encoded_t = encoded[:, t, :]
            state_temporal = decay * state + gates[:, t, :] * candidates[:, t, :]
            gain_t = torch.sigmoid(self.gain_from_encoded(encoded_t) + self.gain_from_prev_state(state))
            message_pre = self.synapse(state_temporal)
            message = message_pre * torch.sigmoid(message_pre)
            msg_gated = gain_t * message
            state = self.loop_norm(state_temporal + self.alpha_scalar * msg_gated)
            logits = self.readout(state)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
