from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
RESULTS = ROOT / "results"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2 = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain"

if str(PHASE05C2) not in sys.path:
    sys.path.insert(0, str(PHASE05C2))
if str(PHASE05A) not in sys.path:
    sys.path.insert(1, str(PHASE05A))
if str(ROOT) not in sys.path:
    sys.path.insert(2, str(ROOT))

import common_05a as p05a  # noqa: E402
import common_05c4 as p05c4  # noqa: E402

Q_TOTAL_BITS = 24
Q_FRAC_BITS = 12
Q_SCALE = float(2**Q_FRAC_BITS)
Q_MIN = -(2 ** (Q_TOTAL_BITS - Q_FRAC_BITS - 1))
Q_MAX = (2 ** (Q_TOTAL_BITS - Q_FRAC_BITS - 1)) - (1.0 / Q_SCALE)

POLICY_FP32 = "phc_stp_v2_cosine_lr_fp32"
POLICY_FIXED = "phc_stp_v2_cosine_lr_q24_12_emulated"
ACTION_AGREEMENT_PASS_THRESHOLD = 0.98
MATERIAL_DEGRADATION_RATIO = 0.95


def q(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x * Q_SCALE) / Q_SCALE, Q_MIN, Q_MAX)


def q_linear(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
    bias = q(layer.bias) if layer.bias is not None else None
    return q(F.linear(q(x), q(layer.weight), bias))


def fixed_point_forward_stp(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Q24.12 emulation of PHCLiteSTP, following the Phase 06B audit pattern.

    This is a strategy-level fixed-point emulation for the deployed 05C4 v2
    checkpoint. It is not HLS csim/cosim and does not alter the checkpoint.
    """

    batch, seq_len, _ = x.shape
    state = q(x.new_zeros((batch, model.hidden_dim)))
    decay = q(torch.sigmoid(q(model.decay_param)).view(1, -1))
    enc_linear = model.encoder[0]
    enc_norm = model.encoder[1]
    alpha = q(model.alpha)

    for t in range(seq_len):
        encoded = q_linear(x[:, t, :], enc_linear)
        encoded = q(enc_norm(encoded))
        encoded = q(torch.nn.functional.silu(encoded))

        gate = q(torch.sigmoid(q_linear(encoded, model.gate)))
        candidate = q(torch.tanh(q_linear(encoded, model.candidate)))
        state_temporal = q(q(decay * state) + q(gate * candidate))

        gain_encoded = q_linear(encoded, model.gain_from_encoded)
        gain_prev = q_linear(state, model.gain_from_prev_state)
        gain_t = q(torch.sigmoid(q(gain_encoded + gain_prev)))

        z = state_temporal
        for _ in range(model.mtl_steps):
            message = q(torch.nn.functional.silu(q_linear(z, model.synapse)))
            message = q(gain_t * message)
            z = q(model.loop_norm(q(z + q(alpha * message))))
        state = z

    return q_linear(state, model.readout)


def batched_scores(
    fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    payload: dict[str, Any],
    target_scale: float,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    x = payload["x"]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(dtype=torch.float32)
            out = fn(model, xb).detach().cpu().numpy().astype(np.float32)
            outs.append(out)
    return np.concatenate(outs, axis=0) * float(target_scale)


def causal_fill_row(split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray) -> dict[str, Any]:
    norms = p05a.compute_fill_norms_from_train_sequences()
    probs = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (
        p05a.stable_uniform(payload, decision, p05c4.FILL_RANDOM_SALT) < probs
    )
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(
        payload, decision, fills, pnl, inventory_cap=1
    )
    return {
        "split": split,
        "policy_id": policy_id,
        "horizon": p05a.HORIZON,
        "fill_model": "conservative_l2_stress",
        "inventory_cap": 1,
        "fee_multiplier": 1.0,
        "sequential_excess_pnl": float(total),
        "fill_count": int(fill_count),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "max_drawdown": float(max_dd),
        "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
        "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
        "avg_trade_excess_pnl": total / max(fill_count, 1) if fill_count else 0.0,
        "fill_uses_future_pnl": False,
        "phase": "05c4",
        "new5_used": False,
    }


def assert_selected_contract(selected: dict[str, Any], ckpt: dict[str, Any]) -> None:
    if selected.get("seed") != 2:
        raise RuntimeError(f"Expected selected seed=2, found {selected.get('seed')!r}")
    if selected.get("model") != p05c4.POLICY_ID:
        raise RuntimeError(f"Expected model {p05c4.POLICY_ID}, found {selected.get('model')!r}")
    if selected.get("phase") != "05c4" or ckpt.get("phase") != "05c4":
        raise RuntimeError("Refusing non-05C4 selected policy/checkpoint")
    for key in ["new3_used_for_selection", "new4_used_for_selection", "new5_used_for_selection"]:
        if selected.get(key) is not False:
            raise RuntimeError(f"Selection contract violation: {key}={selected.get(key)!r}")
    if ckpt.get("seed") != 2:
        raise RuntimeError(f"Expected checkpoint seed=2, found {ckpt.get('seed')!r}")


def comparison_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, ckpt, selected = p05c4.selected_model()
    assert_selected_contract(selected, ckpt)
    target_scale = float(ckpt["target_scale"])
    threshold = float(selected["threshold"])
    existing_oos = pd.read_csv(RESULTS / "phc_stp_v2_oos_eval.csv")
    existing_causal = pd.read_csv(RESULTS / "phc_stp_v2_causal_fill_eval.csv")

    equivalence_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []

    for split in p05c4.SPLITS:
        payload = p05a.load_sequences(split)
        fp32_decision, fp32_scores = p05c4.stp_decision_for_payload(payload, model, ckpt, threshold)
        fixed_scores = batched_scores(fixed_point_forward_stp, model, payload, target_scale=target_scale)
        fixed_decision = p05a.policy_from_scores(fixed_scores, threshold)

        fp32_metric = p05a.metric_from_decision(split, POLICY_FP32, payload, fp32_decision)
        fixed_metric = p05a.metric_from_decision(split, POLICY_FIXED, payload, fixed_decision)
        existing_metric = existing_oos[existing_oos["split"].eq(split)].iloc[0]
        oos_total_diff = float(fp32_metric["total_excess_pnl_H20"]) - float(existing_metric["total_excess_pnl_H20"])
        oos_win_diff = float(fp32_metric["win_rate"]) - float(existing_metric["win_rate"])
        if abs(oos_total_diff) > 1e-9 or abs(oos_win_diff) > 1e-15:
            raise RuntimeError(
                f"Float32 OOS sanity check failed for {split}: "
                f"total diff={oos_total_diff}, win_rate diff={oos_win_diff}"
            )

        fp32_causal = causal_fill_row(split, POLICY_FP32, payload, fp32_decision)
        fixed_causal = causal_fill_row(split, POLICY_FIXED, payload, fixed_decision)
        existing_causal_row = existing_causal[existing_causal["split"].eq(split)].iloc[0]
        causal_diff = float(fp32_causal["sequential_excess_pnl"]) - float(existing_causal_row["sequential_excess_pnl"])
        if abs(causal_diff) > 1e-9:
            raise RuntimeError(f"Float32 causal-fill sanity check failed for {split}: diff={causal_diff}")

        selected_mask = fp32_decision != p05a.DECISION_NO_QUOTE
        selected_action_agreement = (
            float((fp32_decision[selected_mask] == fixed_decision[selected_mask]).mean()) if selected_mask.any() else np.nan
        )
        fixed_less_than_material = bool(
            fixed_causal["sequential_excess_pnl"]
            < MATERIAL_DEGRADATION_RATIO * fp32_causal["sequential_excess_pnl"]
        )

        equivalence_rows.append(
            {
                "split": split,
                "model": "phc_stp_v2_cosine_lr",
                "seed": int(selected["seed"]),
                "checkpoint": str(p05c4.best_ckpt_path(int(selected["seed"]))),
                "threshold": threshold,
                "q_total_bits": Q_TOTAL_BITS,
                "q_frac_bits": Q_FRAC_BITS,
                "state_count": int(len(fp32_decision)),
                "max_abs_score_diff": float(np.max(np.abs(fp32_scores - fixed_scores))),
                "mean_abs_score_diff": float(np.mean(np.abs(fp32_scores - fixed_scores))),
                "action_agreement_rate": float((fp32_decision == fixed_decision).mean()),
                "selected_action_agreement_rate": selected_action_agreement,
                "decision_flip_count": int((fp32_decision != fixed_decision).sum()),
                "fp32_selected_action_count": int((fp32_decision != p05a.DECISION_NO_QUOTE).sum()),
                "fixed_selected_action_count": int((fixed_decision != p05a.DECISION_NO_QUOTE).sum()),
                "fp32_total_excess_pnl_H20": float(fp32_metric["total_excess_pnl_H20"]),
                "fixed_total_excess_pnl_H20": float(fixed_metric["total_excess_pnl_H20"]),
                "pnl_delta_fixed_minus_fp32": float(fixed_metric["total_excess_pnl_H20"])
                - float(fp32_metric["total_excess_pnl_H20"]),
                "action_agreement_pass_threshold": ACTION_AGREEMENT_PASS_THRESHOLD,
                "action_agreement_pass": bool(
                    float((fp32_decision == fixed_decision).mean()) >= ACTION_AGREEMENT_PASS_THRESHOLD
                ),
                "new5_used": False,
            }
        )
        causal_rows.extend([fp32_causal, fixed_causal])
        sanity_rows.extend(
            [
                {
                    "split": split,
                    "check": "fp32_action_oos_matches_existing_05c4",
                    "computed": float(fp32_metric["total_excess_pnl_H20"]),
                    "existing": float(existing_metric["total_excess_pnl_H20"]),
                    "delta": oos_total_diff,
                    "pass": True,
                },
                {
                    "split": split,
                    "check": "fp32_causal_fill_matches_existing_05c4",
                    "computed": float(fp32_causal["sequential_excess_pnl"]),
                    "existing": float(existing_causal_row["sequential_excess_pnl"]),
                    "delta": causal_diff,
                    "pass": True,
                },
                {
                    "split": split,
                    "check": "fixed_causal_fill_not_materially_degraded",
                    "computed": float(fixed_causal["sequential_excess_pnl"]),
                    "existing": float(fp32_causal["sequential_excess_pnl"]),
                    "delta": float(fixed_causal["sequential_excess_pnl"])
                    - float(fp32_causal["sequential_excess_pnl"]),
                    "pass": not fixed_less_than_material,
                },
            ]
        )

    equivalence = pd.DataFrame(equivalence_rows)
    causal = pd.DataFrame(causal_rows)
    sanity = pd.DataFrame(sanity_rows)
    return equivalence, causal, sanity


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(equivalence: pd.DataFrame, causal: pd.DataFrame, sanity: pd.DataFrame) -> str:
    causal_pivot = causal.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index()
    causal_pivot["causal_pnl_delta_fixed_minus_fp32"] = (
        causal_pivot[POLICY_FIXED] - causal_pivot[POLICY_FP32]
    )
    causal_pivot["causal_pnl_delta_pct_fixed_minus_fp32"] = (
        causal_pivot["causal_pnl_delta_fixed_minus_fp32"] / causal_pivot[POLICY_FP32].replace(0, np.nan)
    )
    causal_pivot["material_degradation"] = (
        causal_pivot[POLICY_FIXED] < MATERIAL_DEGRADATION_RATIO * causal_pivot[POLICY_FP32]
    )
    all_agreement_pass = bool(equivalence["action_agreement_pass"].astype(bool).all())
    material_degradation = bool(causal_pivot["material_degradation"].astype(bool).any())
    sanity_pass = bool(sanity["pass"].astype(bool).all())
    status = (
        "phc_stp_v2_fixed_point_equivalence_ready_to_replace_gap"
        if all_agreement_pass and not material_degradation and sanity_pass
        else "phc_stp_v2_fixed_point_equivalence_insufficient_to_replace_gap"
    )
    if status.endswith("ready_to_replace_gap"):
        replacement = (
            "This result can directly replace the AMD-track proposal Section 4.5 / "
            "Section 5 item 4 pending note for 05C4 v2 seed=2 fixed-point equivalence."
        )
    else:
        replacement = (
            "This result is still insufficient to replace the AMD-track proposal Section 4.5 / "
            "Section 5 item 4 pending note because action agreement, material degradation, "
            "or sanity checks did not all pass."
        )
    lines = [
        "# PHC STP v2 Q24.12 Fixed-point Equivalence Report",
        "",
        f"Final status: `{status}`.",
        "",
        replacement,
        "",
        "## Scope",
        "",
        "- Target model: `05C4 phc_stp_v2_cosine_lr`, selected seed `2`.",
        "- Checkpoint: `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt`.",
        "- Target: `excess_pnl_H20`.",
        "- Splits evaluated: `val`, `new3`, `new4`.",
        "- NEW5 is not read or used.",
        "- This is strategy-level `ap_fixed<24,12>` emulation against the float32 05C4 model. It is not HLS csim/cosim.",
        "- The Q24.12 method follows the Phase 06B fixed-point emulation pattern, but the results below are recomputed for the deployed 05C4 v2 seed=2 model. They are not copied from the PHC-lite v1 Phase 06B audit.",
        "",
        "## Action Equivalence",
        "",
        equivalence.to_markdown(index=False),
        "",
        "## Causal-fill PnL Equivalence",
        "",
        causal_pivot.to_markdown(index=False),
        "",
        "Material degradation is defined exactly as in the Phase 06B style check: fixed-point causal-fill PnL below 95% of the float32 causal-fill PnL.",
        "",
        "## Sanity Checks",
        "",
        sanity.to_markdown(index=False),
        "",
        "## Claim Boundary",
        "",
        "- No retraining was performed.",
        "- The 05C4 checkpoint and threshold were not modified.",
        "- `common_05a.py` and `common_05c4.py` were imported only; they were not edited.",
        "- This does not claim true exchange replay, real-world profitability, or hardware-caused PnL improvement.",
    ]
    out = ROOT / "PHC_STP_V2_FIXED_POINT_EQUIVALENCE_REPORT.md"
    write_markdown(out, lines)
    return status


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    equivalence, causal, sanity = comparison_rows()
    equivalence.to_csv(RESULTS / "phc_stp_v2_fixed_point_equivalence.csv", index=False)
    causal.to_csv(RESULTS / "phc_stp_v2_fixed_point_causal_fill_eval.csv", index=False)
    sanity.to_csv(RESULTS / "phc_stp_v2_fixed_point_sanity_checks.csv", index=False)
    status = report(equivalence, causal, sanity)
    payload = {
        "status": status,
        "min_action_agreement_rate": float(equivalence["action_agreement_rate"].min()),
        "max_abs_causal_pnl_delta": float(
            causal.pivot(index="split", columns="policy_id", values="sequential_excess_pnl")
            .assign(delta=lambda x: x[POLICY_FIXED] - x[POLICY_FP32])["delta"]
            .abs()
            .max()
        ),
        "new5_used": False,
    }
    print(json.dumps(payload, indent=2))
    print(equivalence[["split", "action_agreement_rate", "decision_flip_count", "pnl_delta_fixed_minus_fp32"]].to_string(index=False))


if __name__ == "__main__":
    main()
