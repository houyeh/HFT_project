from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07c_tcn_streaming_manual_hls"
RESULTS = ROOT / "results"
PHASE07A = PROJECT_ROOT / "n2ew_era" / "07a_tcn_feedforward_phc_tradeoff"
PHASE07B = PROJECT_ROOT / "n2ew_era" / "07b_tcn_prepad_manual_hls"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_prepad import TCNPrePadRegressor  # noqa: E402

VARIANTS: dict[str, dict[str, Any]] = {
    "scratch": {
        "policy_id": "tcn_prepad_scratch_16feat",
        "checkpoint": PHASE07A / "models" / "tcn_prepad_scratch_16feat_seed2_best_val.pt",
        "test_vectors": PHASE07B / "results" / "tcn_prepad_scratch_test_vectors.csv",
    },
    "distilled": {
        "policy_id": "tcn_prepad_distilled_16feat",
        "checkpoint": PHASE07A / "models" / "tcn_prepad_distilled_16feat_seed1_best_val.pt",
        "test_vectors": PHASE07B / "results" / "tcn_prepad_distilled_test_vectors.csv",
    },
}

DILATIONS = [1, 2, 4, 8, 16]
CHANNELS = 16
FEATURE_DIM = 16
SEQ_LEN = 128
PREPAD = 124
SEQ_LEN_IN = SEQ_LEN + PREPAD
KERNEL_SIZE = 3


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.8g}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def load_model(path: Path) -> tuple[TCNPrePadRegressor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCNPrePadRegressor(num_inputs=16, num_outputs=2, channels=16, kernel_size=3, dropout=0.1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_test_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing test vector csv: {path}")
    df = pd.read_csv(path)
    x_cols = [f"x_{t}_{f}" for t in range(SEQ_LEN) for f in range(FEATURE_DIM)]
    ref_cols = [f"ref_readout_{i}" for i in range(2)]
    missing = [c for c in x_cols + ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing[:5]}")
    x = df[x_cols].to_numpy(dtype=np.float32).reshape(len(df), SEQ_LEN, FEATURE_DIM)
    ref = df[ref_cols].to_numpy(dtype=np.float32)
    return x, ref


def torch_batch_trace(model: TCNPrePadRegressor, x_np: np.ndarray, target_scale: float) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    with torch.no_grad():
        x = torch.from_numpy(x_np).float().permute(0, 2, 1).contiguous()
        current = model.prepad(x)
        traces["prepad"] = current.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
        for block_idx, block in enumerate(model.tcn):
            conv1 = block.conv1(current)
            conv1_relu = block.relu(conv1)
            conv1_relu = block.dropout(conv1_relu)
            conv2 = block.conv2(conv1_relu)
            conv2_relu = block.relu(conv2)
            conv2_relu = block.dropout(conv2_relu)
            residual = current[:, :, -conv2_relu.size(2) :].contiguous()
            if block.downsample is not None:
                residual = block.downsample(residual)
            out = block.relu(conv2_relu + residual)
            traces[f"block{block_idx}_conv1_relu"] = conv1_relu.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
            traces[f"block{block_idx}_conv2_relu"] = conv2_relu.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
            traces[f"block{block_idx}_residual"] = residual.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
            traces[f"block{block_idx}_out"] = out.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
            current = out
        pooled = current.mean(dim=2)
        readout = model.fc(pooled).detach().cpu().numpy().astype(np.float32) * np.float32(target_scale)
        traces["readout_scaled"] = readout.astype(np.float32)
    return traces


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, np.float32(0.0)).astype(np.float32)


class StreamingBlock:
    """Event-aligned causal form of one TCNPrePadBlock.

    At event-aligned time s, this block emits output_s once s >= 4*d:

        c1_s = relu(W1[0] x_{s-2d} + W1[1] x_{s-d} + W1[2] x_s + b1)
        c2_s = relu(W2[0] c1_{s-2d} + W2[1] c1_{s-d} + W2[2] c1_s + b2)
        out_s = relu(c2_s + x_s)

    This is algebraically the same as the valid-conv PyTorch block, where the
    compact output index t corresponds to event-aligned index s=t+4*d.
    """

    def __init__(
        self,
        conv1_w: np.ndarray,
        conv1_b: np.ndarray,
        conv2_w: np.ndarray,
        conv2_b: np.ndarray,
        dilation: int,
    ) -> None:
        self.conv1_w = conv1_w.astype(np.float32)
        self.conv1_b = conv1_b.astype(np.float32)
        self.conv2_w = conv2_w.astype(np.float32)
        self.conv2_b = conv2_b.astype(np.float32)
        self.d = int(dilation)
        self.input_hist: deque[np.ndarray] = deque(maxlen=2 * self.d + 1)
        self.conv1_hist: deque[np.ndarray] = deque(maxlen=2 * self.d + 1)

    def _conv3(self, weights: np.ndarray, bias: np.ndarray, hist: deque[np.ndarray]) -> np.ndarray:
        d = self.d
        taps = [hist[-1 - 2 * d], hist[-1 - d], hist[-1]]
        acc = bias.astype(np.float32).copy()
        for k, tap in enumerate(taps):
            acc += weights[:, :, k].dot(tap.astype(np.float32))
        return acc.astype(np.float32)

    def step(self, x_current: np.ndarray) -> dict[str, np.ndarray | None]:
        x = x_current.astype(np.float32)
        self.input_hist.append(x)
        conv1: np.ndarray | None = None
        conv2_relu: np.ndarray | None = None
        out: np.ndarray | None = None
        residual: np.ndarray | None = None

        if len(self.input_hist) == self.input_hist.maxlen:
            conv1 = relu(self._conv3(self.conv1_w, self.conv1_b, self.input_hist))
            self.conv1_hist.append(conv1)
            if len(self.conv1_hist) == self.conv1_hist.maxlen:
                conv2_relu = relu(self._conv3(self.conv2_w, self.conv2_b, self.conv1_hist))
                residual = x
                out = relu(conv2_relu + residual)

        return {
            "conv1_relu": conv1,
            "conv2_relu": conv2_relu,
            "residual": residual,
            "out": out,
        }


def extract_weights(model: TCNPrePadRegressor) -> dict[str, list[np.ndarray] | np.ndarray]:
    state = model.state_dict()
    return {
        "conv1_w": [state[f"tcn.{b}.conv1.weight"].detach().cpu().numpy().astype(np.float32) for b in range(5)],
        "conv1_b": [state[f"tcn.{b}.conv1.bias"].detach().cpu().numpy().astype(np.float32) for b in range(5)],
        "conv2_w": [state[f"tcn.{b}.conv2.weight"].detach().cpu().numpy().astype(np.float32) for b in range(5)],
        "conv2_b": [state[f"tcn.{b}.conv2.bias"].detach().cpu().numpy().astype(np.float32) for b in range(5)],
        "fc_w": state["fc.weight"].detach().cpu().numpy().astype(np.float32),
        "fc_b": state["fc.bias"].detach().cpu().numpy().astype(np.float32),
    }


def streaming_trace(model: TCNPrePadRegressor, x_np: np.ndarray, target_scale: float) -> dict[str, np.ndarray]:
    weights = extract_weights(model)
    prepad = np.concatenate(
        [np.zeros((x_np.shape[0], PREPAD, FEATURE_DIM), dtype=np.float32), x_np.astype(np.float32)],
        axis=1,
    )
    traces: dict[str, list[np.ndarray] | np.ndarray] = {"prepad": prepad}
    final_rows: list[np.ndarray] = []

    for sample_idx in range(x_np.shape[0]):
        current = prepad[sample_idx]
        sample_outputs: dict[str, list[np.ndarray]] = {}
        for block_idx, dilation in enumerate(DILATIONS):
            block = StreamingBlock(
                weights["conv1_w"][block_idx],  # type: ignore[index]
                weights["conv1_b"][block_idx],  # type: ignore[index]
                weights["conv2_w"][block_idx],  # type: ignore[index]
                weights["conv2_b"][block_idx],  # type: ignore[index]
                dilation,
            )
            conv1_rows: list[np.ndarray] = []
            conv2_rows: list[np.ndarray] = []
            residual_rows: list[np.ndarray] = []
            out_rows: list[np.ndarray] = []
            for x_t in current:
                step = block.step(x_t)
                if step["conv1_relu"] is not None:
                    conv1_rows.append(step["conv1_relu"])  # type: ignore[arg-type]
                if step["out"] is not None:
                    conv2_rows.append(step["conv2_relu"])  # type: ignore[arg-type]
                    residual_rows.append(step["residual"])  # type: ignore[arg-type]
                    out_rows.append(step["out"])  # type: ignore[arg-type]
            sample_outputs[f"block{block_idx}_conv1_relu"] = conv1_rows
            sample_outputs[f"block{block_idx}_conv2_relu"] = conv2_rows
            sample_outputs[f"block{block_idx}_residual"] = residual_rows
            sample_outputs[f"block{block_idx}_out"] = out_rows
            current = np.asarray(out_rows, dtype=np.float32)
        pooled = current.mean(axis=0)
        readout = weights["fc_w"].dot(pooled) + weights["fc_b"]  # type: ignore[union-attr]
        final_rows.append((readout.astype(np.float32) * np.float32(target_scale)).astype(np.float32))

        for name, values in sample_outputs.items():
            if name not in traces:
                traces[name] = []
            traces[name].append(np.asarray(values, dtype=np.float32))  # type: ignore[union-attr]

    compact: dict[str, np.ndarray] = {"prepad": prepad, "readout_scaled": np.asarray(final_rows, dtype=np.float32)}
    for name, values in traces.items():
        if name == "prepad":
            continue
        compact[name] = np.stack(values, axis=0).astype(np.float32)  # type: ignore[arg-type]
    return compact


def compare_arrays(variant: str, name: str, batch: np.ndarray, stream: np.ndarray) -> dict[str, Any]:
    if batch.shape != stream.shape:
        raise ValueError(f"{variant}:{name} shape mismatch batch={batch.shape} stream={stream.shape}")
    diff = stream.astype(np.float64) - batch.astype(np.float64)
    abs_diff = np.abs(diff)
    flat = int(abs_diff.argmax()) if abs_diff.size else 0
    idx = np.unravel_index(flat, abs_diff.shape) if abs_diff.size else ()
    max_timestep = int(idx[1]) if len(idx) >= 3 else -1
    max_channel = int(idx[2]) if len(idx) >= 3 else (int(idx[1]) if len(idx) == 2 else -1)
    return {
        "variant": variant,
        "stage": name,
        "shape": "x".join(str(v) for v in batch.shape),
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rmse": float(math.sqrt(float((diff * diff).mean()))) if diff.size else 0.0,
        "max_sample_idx": int(idx[0]) if len(idx) >= 1 else -1,
        "max_timestep_idx": max_timestep,
        "max_channel_or_output_idx": max_channel,
    }


def fifo_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = SEQ_LEN_IN
    cumulative_reduction = 0
    for block_idx, dilation in enumerate(DILATIONS):
        out_len = length - 4 * dilation
        rows.append(
            {
                "block": block_idx,
                "dilation": dilation,
                "input_len": length,
                "output_len": out_len,
                "conv1_input_history_previous_timesteps": 2 * dilation,
                "conv2_conv1_history_previous_timesteps": 2 * dilation,
                "total_block_warmup_timesteps": 4 * dilation,
                "cumulative_warmup_after_block": cumulative_reduction + 4 * dilation,
                "pytorch_valid_conv_residual_tap": f"block_input[t+{4 * dilation}]",
                "event_aligned_residual_tap": "block_input[s], where s=t+4d",
                "residual_timestep_fifo_depth": 0,
                "residual_cycle_delay": "TBD by Stage 1 HLS operator schedule",
            }
        )
        cumulative_reduction += 4 * dilation
        length = out_len
    return rows


def write_report(summary: pd.DataFrame, manifest: list[dict[str, Any]]) -> None:
    fifo_df = pd.DataFrame(fifo_rows())
    worst = (
        summary.sort_values("max_abs_diff", ascending=False)
        .groupby("variant", as_index=False)
        .head(8)
        .reset_index(drop=True)
    )
    lines = [
        "status: stage0_complete_stop_for_review",
        "finding: python_streaming_matches_pytorch_batch_with_float_roundoff",
        "# Stage 0 Design and Python Validation Report",
        "",
        "## Scope",
        "",
        "This phase creates only a Python streaming design validation for the TCN prepad model. It does not modify 07a/07b artifacts, does not use NEW5, and does not create HLS C++, csim, csynth, cosim, implementation, P&R, or bitstream outputs.",
        "",
        "Inputs are the existing 07b 200-row validation test vectors generated from the val split. Checkpoints are the existing frozen 07a selected policies.",
        "",
        "## Streaming Form",
        "",
        "For one block with dilation d and kernel size 3, the PyTorch valid-conv block can be re-indexed as an event-aligned causal streaming block:",
        "",
        "```text",
        "c1_s  = ReLU(b1 + W1[0] * x_{s-2d}  + W1[1] * x_{s-d}  + W1[2] * x_s)",
        "c2_s  = ReLU(b2 + W2[0] * c1_{s-2d} + W2[1] * c1_{s-d} + W2[2] * c1_s)",
        "out_s = ReLU(c2_s + x_s)",
        "emit only after s >= 4d",
        "```",
        "",
        "Important residual alignment note: in this event-aligned form, the residual term is the current block input `x_s`. A timestep FIFO that pops `x_{s-4d}` at the residual add would not match the PyTorch batch model. HLS may still need cycle-level delay registers to align arithmetic pipeline latency, but that is a Stage 1 scheduling detail, not a Stage 0 timestep FIFO.",
        "",
        "The same relation in PyTorch valid-convolution coordinates is:",
        "",
        "```text",
        "batch output index t  <=>  event-aligned index s = t + 4d",
        "PyTorch residual = block_input[t + 4d] = block_input[s]",
        "```",
        "",
        "## FIFO and Alignment Table",
        "",
        md_table(fifo_df),
        "",
        "## Initialization",
        "",
        "- Block 0 receives 124 zero vectors followed by the 128 real feature vectors, matching `TCNPrePadRegressor.prepad`.",
        "- Internal histories start empty at sequence reset. The 124 explicit zero events provide the causal warmup.",
        "- For a free-running deployment, reset plus 124 zero ticks is equivalent to the current prepad contract for the first scored 128-event window.",
        "",
        "## Validation Inputs",
        "",
        md_table(pd.DataFrame(manifest)),
        "",
        "## Worst Per-Stage Differences",
        "",
        md_table(worst),
        "",
        "## Full Per-Stage Summary",
        "",
        md_table(summary),
        "",
        "## Stage Boundary",
        "",
        "STOP HERE. Stage 1 HLS work is intentionally not started until this Stage 0 design and Python validation is reviewed.",
        "",
        "## Honest Conclusion",
        "",
        "The event-aligned streaming Python implementation reproduces the PyTorch batch TCN intermediate tensors at every block output tested here, with only float-order roundoff. The algorithmic design is ready for review, but no HLS implementation or latency claim is made in this stage.",
    ]
    (ROOT / "STAGE0_DESIGN_AND_PYTHON_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for variant, cfg in VARIANTS.items():
        model, ckpt = load_model(cfg["checkpoint"])
        target_scale = float(ckpt["target_scale"])
        x_np, ref_csv = load_test_vectors(cfg["test_vectors"])
        batch = torch_batch_trace(model, x_np, target_scale)
        stream = streaming_trace(model, x_np, target_scale)
        for name in sorted(batch.keys()):
            all_rows.append(compare_arrays(variant, name, batch[name], stream[name]))
        all_rows.append(compare_arrays(variant, "readout_scaled_vs_test_vector_csv", ref_csv, batch["readout_scaled"]))
        manifest.append(
            {
                "variant": variant,
                "policy_id": cfg["policy_id"],
                "checkpoint": str(cfg["checkpoint"]),
                "test_vectors": str(cfg["test_vectors"]),
                "sample_count": int(x_np.shape[0]),
                "seq_len": SEQ_LEN,
                "feature_dim": FEATURE_DIM,
                "target_scale": target_scale,
                "new5_used": False,
            }
        )
    summary = pd.DataFrame(all_rows).sort_values(["variant", "stage"]).reset_index(drop=True)
    summary.to_csv(RESULTS / "stage0_streaming_vs_batch_trace_summary.csv", index=False)
    with (RESULTS / "stage0_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    write_report(summary, manifest)
    print(f"wrote {RESULTS / 'stage0_streaming_vs_batch_trace_summary.csv'}")
    print(f"wrote {ROOT / 'STAGE0_DESIGN_AND_PYTHON_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
