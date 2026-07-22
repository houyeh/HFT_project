from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

import common_05d as c

PHASE05A = c.PHASE05A
if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402

POLICY_ID = "phc_lite_v1_streaming_dense"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def linear_np(x: np.ndarray, module: torch.nn.Linear) -> np.ndarray:
    weight = module.weight.detach().cpu().numpy().astype(np.float32)
    bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else 0.0
    return x @ weight.T + bias


def stream_frame(model: torch.nn.Module, frame: pd.DataFrame, scaler: dict[str, Any], target_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _ = p05a.normalized_features(frame, scaler)
    scores = np.zeros((len(x), 2), dtype=np.float32)
    states = np.zeros((len(x), int(model.hidden_dim)), dtype=np.float32)
    enc_linear = model.encoder[0]
    enc_norm = model.encoder[1]
    enc = linear_np(x.astype(np.float32), enc_linear)
    enc = layer_norm_np(
        enc,
        enc_norm.weight.detach().cpu().numpy().astype(np.float32),
        enc_norm.bias.detach().cpu().numpy().astype(np.float32),
        float(enc_norm.eps),
    ).astype(np.float32)
    enc = silu(enc).astype(np.float32)
    gates = sigmoid(linear_np(enc, model.gate)).astype(np.float32)
    candidates = np.tanh(linear_np(enc, model.candidate)).astype(np.float32)
    syn_w = model.synapse.weight.detach().cpu().numpy().astype(np.float32)
    syn_b = model.synapse.bias.detach().cpu().numpy().astype(np.float32)
    loop_w = model.loop_norm.weight.detach().cpu().numpy().astype(np.float32)
    loop_b = model.loop_norm.bias.detach().cpu().numpy().astype(np.float32)
    loop_eps = float(model.loop_norm.eps)
    read_w = model.readout.weight.detach().cpu().numpy().astype(np.float32)
    read_b = model.readout.bias.detach().cpu().numpy().astype(np.float32)
    decay = sigmoid(model.decay_param.detach().cpu().numpy().astype(np.float32))
    alpha = float(model.alpha.detach().cpu().numpy())
    state = np.zeros(int(model.hidden_dim), dtype=np.float32)
    for i in range(len(x)):
        z = decay * state + gates[i] * candidates[i]
        for _ in range(int(model.mtl_steps)):
            message = silu(z @ syn_w.T + syn_b).astype(np.float32)
            z = layer_norm_np((z + alpha * message).astype(np.float32), loop_w, loop_b, loop_eps).astype(np.float32)
        state = z
        scores[i] = (state @ read_w.T + read_b).astype(np.float32) * float(target_scale)
        states[i] = state
    decision = p05a.policy_from_scores(scores, 0.0)
    return scores, decision, states


def dense_payload_from_frame(frame: pd.DataFrame, scaler: dict[str, Any]) -> dict[str, Any]:
    _, raw = p05a.normalized_features(frame, scaler)
    sessions = frame["session_id"].astype(str).to_numpy()
    labels, codes = np.unique(sessions, return_inverse=True)
    return {
        "y_bid": torch.from_numpy(frame["bid_excess_pnl_H20"].to_numpy(dtype=np.float32)),
        "y_ask": torch.from_numpy(frame["ask_excess_pnl_H20"].to_numpy(dtype=np.float32)),
        "target_raw": torch.from_numpy(raw.astype(np.float32)),
        "session_code": torch.from_numpy(codes.astype(np.int32)),
        "session_labels": labels.tolist(),
        "event_index": torch.from_numpy(pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)),
        "timestamp_us": torch.from_numpy(pd.to_numeric(frame["timestamp_us"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)),
    }


def split_event_frames(split: str) -> list[pd.DataFrame]:
    frames = []
    for path in p05a.part_paths(split):
        frame = p05a.read_event_frame(path).sort_values(["session_id", "event_index"]).reset_index(drop=True)
        frames.append(frame)
    return frames


def streaming_scores_for_split(split: str, model: torch.nn.Module, scaler: dict[str, Any], target_scale: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    payloads = []
    score_chunks = []
    decision_chunks = []
    for frame in split_event_frames(split):
        # Reset state at every parquet/session part; Phase 04B writes one session per part in practice.
        scores, decision, _ = stream_frame(model, frame, scaler, target_scale)
        payloads.append(dense_payload_from_frame(frame, scaler))
        score_chunks.append(scores)
        decision_chunks.append(decision)
        print(f"streamed split={split} frame_rows={len(frame)} selected={int((decision != p05a.DECISION_NO_QUOTE).sum())}", flush=True)
    merged = merge_payloads(payloads)
    return merged, np.concatenate(score_chunks, axis=0), np.concatenate(decision_chunks)


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    labels: list[str] = []
    label_map: dict[str, int] = {}
    codes = []
    for payload in payloads:
        local_codes = payload["session_code"].numpy()
        local_labels = payload["session_labels"]
        mapped = np.empty_like(local_codes, dtype=np.int32)
        for code, label in enumerate(local_labels):
            if label not in label_map:
                label_map[label] = len(labels)
                labels.append(label)
            mapped[local_codes == code] = label_map[label]
        codes.append(mapped)
    return {
        "y_bid": torch.cat([p["y_bid"] for p in payloads]),
        "y_ask": torch.cat([p["y_ask"] for p in payloads]),
        "target_raw": torch.cat([p["target_raw"] for p in payloads]),
        "session_code": torch.from_numpy(np.concatenate(codes).astype(np.int32)),
        "session_labels": labels,
        "event_index": torch.cat([p["event_index"] for p in payloads]),
        "timestamp_us": torch.cat([p["timestamp_us"] for p in payloads]),
    }


def terminal_equivalence(split: str, dense_scores: np.ndarray, dense_payload: dict[str, Any]) -> pd.DataFrame:
    window_payload = p05a.load_sequences(split)
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    window_scores = p05a.predict_scores(model, window_payload, target_scale=float(ckpt["target_scale"]))
    dense_key = pd.DataFrame(
        {
            "session_id": c.session_ids(dense_payload),
            "event_index": dense_payload["event_index"].numpy().astype(np.int64),
            "dense_row": np.arange(len(dense_scores), dtype=np.int64),
        }
    )
    window_key = pd.DataFrame(
        {
            "session_id": c.session_ids(window_payload),
            "event_index": window_payload["event_index"].numpy().astype(np.int64),
            "window_row": np.arange(len(window_scores), dtype=np.int64),
        }
    )
    joined = window_key.merge(dense_key, on=["session_id", "event_index"], how="inner")
    w = window_scores[joined["window_row"].to_numpy()]
    d = dense_scores[joined["dense_row"].to_numpy()]
    window_decision = p05a.policy_from_scores(w, 0.0)
    dense_decision = p05a.policy_from_scores(d, 0.0)
    return pd.DataFrame(
        [
            {
                "split": split,
                "matched_terminal_count": int(len(joined)),
                "window_terminal_count": int(len(window_scores)),
                "dense_state_count": int(len(dense_scores)),
                "score_bid_mae": float(np.mean(np.abs(w[:, 0] - d[:, 0]))) if len(joined) else np.nan,
                "score_ask_mae": float(np.mean(np.abs(w[:, 1] - d[:, 1]))) if len(joined) else np.nan,
                "score_bid_max_abs_error": float(np.max(np.abs(w[:, 0] - d[:, 0]))) if len(joined) else np.nan,
                "score_ask_max_abs_error": float(np.max(np.abs(w[:, 1] - d[:, 1]))) if len(joined) else np.nan,
                "action_agreement": float((window_decision == dense_decision).mean()) if len(joined) else np.nan,
                "selected_action_agreement": float((window_decision[window_decision != p05a.DECISION_NO_QUOTE] == dense_decision[window_decision != p05a.DECISION_NO_QUOTE]).mean()) if (window_decision != p05a.DECISION_NO_QUOTE).any() else np.nan,
                "equivalence_exact_expected": False,
                "reason": "window inference resets state at each 128-row window; streaming dense carries state from session start",
            }
        ]
    )


def evaluate_dense() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    scaler = p05a.load_scaler()
    target_scale = float(ckpt["target_scale"])
    eval_rows = []
    causal_rows = []
    equivalence_rows = []
    for split in c.SPLITS:
        payload, scores, decision = streaming_scores_for_split(split, model, scaler, target_scale)
        metric = p05a.metric_from_decision(split, POLICY_ID, payload, decision)
        oracle = c.oracle_capture_for_decision(payload, decision)
        metric.update({"oracle_capture_ratio": oracle["capture_ratio"], "matched_count_oracle_pnl": oracle["matched_count_oracle_pnl"]})
        eval_rows.append(metric)
        causal_rows.append(c.causal_fill_for_decision(split, POLICY_ID, payload, decision, "05d_streaming_dense"))
        equivalence_rows.append(terminal_equivalence(split, scores, payload))
    eval_df = pd.DataFrame(eval_rows)
    causal_df = pd.DataFrame(causal_rows)
    equiv_df = pd.concat(equivalence_rows, ignore_index=True)
    eval_df.to_csv(c.RESULTS / "phc_streaming_dense_eval.csv", index=False)
    causal_df.to_csv(c.RESULTS / "phc_streaming_dense_causal_fill.csv", index=False)
    equiv_df.to_csv(c.RESULTS / "phc_streaming_equivalence.csv", index=False)
    return eval_df, causal_df, equiv_df


def write_report(eval_df: pd.DataFrame, causal_df: pd.DataFrame, equiv_df: pd.DataFrame) -> None:
    lines = [
        "# PHC Streaming Dense Inference Report",
        "",
        "This audit runs the frozen PHC-lite v1 as a session-reset streaming state machine. It does not retrain the model and does not use NEW5.",
        "",
        "## Terminal Equivalence",
        equiv_df.to_markdown(index=False),
        "",
        "Exact equality is not expected because the original PHC-window inference resets hidden state inside each 128-row window, while dense streaming carries state from the session start. The key diagnostic is action agreement at matching terminal events.",
        "",
        "## Dense Streaming Action-Level Evaluation",
        eval_df[["split", "state_count", "selected_action_count", "selection_rate", "total_excess_pnl_H20", "oracle_capture_ratio", "max_drawdown"]].to_markdown(index=False),
        "",
        "## Dense Streaming Causal-Fill Evaluation",
        causal_df[["split", "sequential_excess_pnl", "fill_count", "max_abs_inventory", "max_drawdown"]].to_markdown(index=False),
        "",
        "## Interpretation",
        "Dense streaming is HLS-relevant because each event updates one recurrent state and emits bid/ask scores. This remains proxy excess markout and causal-fill simulation, not true exchange replay.",
    ]
    c.write_markdown(c.ROOT / "PHC_STREAMING_DENSE_INFERENCE_REPORT.md", lines)


def main() -> None:
    c.ensure_dirs()
    eval_df, causal_df, equiv_df = evaluate_dense()
    write_report(eval_df, causal_df, equiv_df)
    print("wrote PHC streaming dense inference outputs")


if __name__ == "__main__":
    main()
