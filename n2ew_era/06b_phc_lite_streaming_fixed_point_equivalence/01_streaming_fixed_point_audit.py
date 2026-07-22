from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06b_phc_lite_streaming_fixed_point_equivalence"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402


Q_TOTAL_BITS = 24
Q_FRAC_BITS = 12
Q_SCALE = float(2**Q_FRAC_BITS)
Q_MIN = -(2 ** (Q_TOTAL_BITS - Q_FRAC_BITS - 1))
Q_MAX = (2 ** (Q_TOTAL_BITS - Q_FRAC_BITS - 1)) - (1.0 / Q_SCALE)


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def q(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x * Q_SCALE) / Q_SCALE, Q_MIN, Q_MAX)


def q_linear(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
    return q(F.linear(q(x), q(layer.weight), q(layer.bias) if layer.bias is not None else None))


def streaming_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    state = x.new_zeros((batch, model.hidden_dim))
    decay = torch.sigmoid(model.decay_param).view(1, -1)
    for t in range(seq_len):
        encoded = model.encoder(x[:, t, :])
        gate = torch.sigmoid(model.gate(encoded))
        candidate = torch.tanh(model.candidate(encoded))
        state_temporal = decay * state + gate * candidate
        z = state_temporal
        for _ in range(model.mtl_steps):
            message = torch.nn.functional.silu(model.synapse(z))
            message = model.dropout(message)
            z = model.loop_norm(z + model.alpha * message)
        state = z
    return model.readout(state)


def fixed_point_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    state = q(x.new_zeros((batch, model.hidden_dim)))
    decay = q(torch.sigmoid(q(model.decay_param)).view(1, -1))
    enc_linear = model.encoder[0]
    enc_norm = model.encoder[1]
    for t in range(seq_len):
        encoded = q_linear(x[:, t, :], enc_linear)
        encoded = q(enc_norm(encoded))
        encoded = q(torch.nn.functional.silu(encoded))
        gate = q(torch.sigmoid(q_linear(encoded, model.gate)))
        candidate = q(torch.tanh(q_linear(encoded, model.candidate)))
        state_temporal = q(q(decay * state) + q(gate * candidate))
        z = state_temporal
        for _ in range(model.mtl_steps):
            message = q(torch.nn.functional.silu(q_linear(z, model.synapse)))
            z = q(model.loop_norm(q(z + q(model.alpha) * message)))
        state = z
    return q_linear(state, model.readout)


def batched_scores(fn, model: torch.nn.Module, payload: dict[str, Any], target_scale: float, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    x = payload["x"]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(dtype=torch.float32)
            outs.append(fn(model, xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) * float(target_scale)


def selected_phc_model() -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    model.eval()
    return model, ckpt, policy


def metric_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, ckpt, policy = selected_phc_model()
    target_scale = float(ckpt["target_scale"])
    threshold = float(policy["threshold"])
    stream_rows = []
    fixed_rows = []
    causal_rows = []
    norms = p05a.compute_fill_norms_from_train_sequences()
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        fp32_scores = p05a.predict_scores(model, payload, target_scale=target_scale)
        stream_scores = batched_scores(streaming_forward, model, payload, target_scale=target_scale)
        fixed_scores = batched_scores(fixed_point_forward, model, payload, target_scale=target_scale)
        fp32_decision = p05a.policy_from_scores(fp32_scores, threshold)
        stream_decision = p05a.policy_from_scores(stream_scores, threshold)
        fixed_decision = p05a.policy_from_scores(fixed_scores, threshold)
        stream_rows.append(
            {
                "split": split,
                "row_count": len(fp32_decision),
                "max_abs_score_diff": float(np.max(np.abs(fp32_scores - stream_scores))),
                "mean_abs_score_diff": float(np.mean(np.abs(fp32_scores - stream_scores))),
                "action_agreement_rate": float((fp32_decision == stream_decision).mean()),
                "threshold": threshold,
                "new5_used": False,
            }
        )
        fixed_pnl = float(p05a.decision_pnl_from_payload(payload, fixed_decision).sum())
        fp32_pnl = float(p05a.decision_pnl_from_payload(payload, fp32_decision).sum())
        fixed_rows.append(
            {
                "split": split,
                "row_count": len(fp32_decision),
                "q_total_bits": Q_TOTAL_BITS,
                "q_frac_bits": Q_FRAC_BITS,
                "max_abs_score_diff": float(np.max(np.abs(fp32_scores - fixed_scores))),
                "mean_abs_score_diff": float(np.mean(np.abs(fp32_scores - fixed_scores))),
                "action_agreement_rate": float((fp32_decision == fixed_decision).mean()),
                "selected_action_agreement_rate": float((fp32_decision[fp32_decision != p05a.DECISION_NO_QUOTE] == fixed_decision[fp32_decision != p05a.DECISION_NO_QUOTE]).mean()),
                "fp32_total_excess_pnl_H20": fp32_pnl,
                "fixed_total_excess_pnl_H20": fixed_pnl,
                "pnl_delta_fixed_minus_fp32": fixed_pnl - fp32_pnl,
                "new5_used": False,
            }
        )
        for policy_id, decision, salt in [
            ("phc_lite_fp32", fp32_decision, "06b_phc_fp32"),
            ("phc_lite_fixed_point_emulated", fixed_decision, "06b_phc_fixed_point"),
        ]:
            probs = p05a.fill_probabilities(payload, decision, norms)
            fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, salt) < probs)
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
            causal_rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "sequential_excess_pnl": float(total),
                    "fill_count": int(fill_count),
                    "forced_no_quote_count": int(forced),
                    "max_abs_inventory": int(max_inv),
                    "max_drawdown": float(max_dd),
                    "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
                    "avg_trade_excess_pnl": total / max(fill_count, 1) if fill_count else 0.0,
                    "fill_uses_future_pnl": False,
                    "new5_used": False,
                }
            )
    stream = pd.DataFrame(stream_rows)
    fixed = pd.DataFrame(fixed_rows)
    causal = pd.DataFrame(causal_rows)
    stream.to_csv(RESULTS / "phc_streaming_equivalence.csv", index=False)
    fixed.to_csv(RESULTS / "phc_fixed_point_equivalence.csv", index=False)
    causal.to_csv(RESULTS / "phc_fixed_point_causal_fill_eval.csv", index=False)
    return stream, fixed, causal


def resource_estimate(model: torch.nn.Module) -> pd.DataFrame:
    feature_dim = int(model.feature_dim)
    hidden_dim = int(model.hidden_dim)
    macs = feature_dim * hidden_dim + 3 * hidden_dim * hidden_dim + hidden_dim * 2
    out = pd.DataFrame(
        [
            {
                "model": "PHC-lite v1",
                "q_total_bits": Q_TOTAL_BITS,
                "q_frac_bits": Q_FRAC_BITS,
                "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "state_elements": hidden_dim,
                "state_memory_bits": hidden_dim * Q_TOTAL_BITS,
                "estimated_mac_per_event": macs,
                "streaming_state_update": True,
                "window_buffer_required_for_exact_window_mode": True,
                "streaming_inference_plausible": True,
                "resource_estimate_type": "static_pre_hls_estimate_not_vendor_synthesis",
            }
        ]
    )
    out.to_csv(RESULTS / "phc_fixed_point_resource_estimate.csv", index=False)
    return out


def plots(stream: pd.DataFrame, fixed: pd.DataFrame, causal: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(stream["split"], stream["action_agreement_rate"])
    plt.ylim(0.98, 1.001)
    plt.title("PHC-lite Window vs Streaming Action Agreement")
    plt.tight_layout()
    plt.savefig(FIGURES / "phc_streaming_action_agreement.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(fixed["split"], fixed["action_agreement_rate"])
    plt.ylim(0.95, 1.001)
    plt.title("PHC-lite FP32 vs Fixed-point Action Agreement")
    plt.tight_layout()
    plt.savefig(FIGURES / "phc_fixed_point_action_agreement.png", dpi=160)
    plt.close()
    pivot = causal.pivot(index="split", columns="policy_id", values="sequential_excess_pnl")
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.title("PHC-lite Causal-fill PnL: FP32 vs Fixed-point")
    plt.tight_layout()
    plt.savefig(FIGURES / "phc_fixed_point_causal_fill_pnl.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()
    model, _, _ = selected_phc_model()
    stream, fixed, causal = metric_rows()
    estimate = resource_estimate(model)
    plots(stream, fixed, causal)
    stream_pass = bool((stream["action_agreement_rate"] > 0.99).all())
    fixed_pass = bool((fixed["action_agreement_rate"] > 0.98).all())
    causal_pivot = causal.pivot(index="split", columns="policy_id", values="sequential_excess_pnl")
    max_abs_causal_delta = float((causal_pivot["phc_lite_fixed_point_emulated"] - causal_pivot["phc_lite_fp32"]).abs().max())
    material_degradation = bool((causal_pivot["phc_lite_fixed_point_emulated"] < 0.95 * causal_pivot["phc_lite_fp32"]).any())
    status = "phc_streaming_fixed_point_ready" if stream_pass and fixed_pass and not material_degradation else "phc_streaming_fixed_point_needs_revision"
    write_markdown(
        ROOT / "PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md",
        [
            "# PHC-lite Streaming and Fixed-point Equivalence Report",
            "",
            f"Final status: `{status}`.",
            "",
            "This audit uses frozen PHC-lite v1 only. It does not retrain or retune the model. NEW5 is not used.",
            "",
            "## Streaming Equivalence",
            "",
            stream.to_markdown(index=False),
            "",
            "## Fixed-point Emulation",
            "",
            fixed.to_markdown(index=False),
            "",
            "## Causal-fill Sequential Check",
            "",
            causal.to_markdown(index=False),
            "",
            f"Max absolute causal-fill PnL delta: `{max_abs_causal_delta:,.2f}`.",
            f"Material degradation detected using 95% FP32-PnL tolerance: `{material_degradation}`.",
            "",
            "## Resource Estimate",
            "",
            estimate.to_markdown(index=False),
            "",
            "This is a pre-HLS fixed-point emulation and static resource estimate. It does not claim FPGA improves PnL, true exchange replay, or real-world profitability.",
        ],
    )
    print(json.dumps({"status": status, "stream_pass": stream_pass, "fixed_pass": fixed_pass, "material_degradation": material_degradation}, indent=2))


if __name__ == "__main__":
    main()
