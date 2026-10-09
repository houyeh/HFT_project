from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STEP_MODEL_PATH = PROJECT_ROOT / "n2ew_era" / "06e_phc_stp_step_hls4ml" / "models" / "phc_stp_step_hls_seed2.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phc_stp_step_hls import PHCSTPStepHLS  # noqa: E402


FEATURE_DIM = 16
HIDDEN_DIM = 16
FPGA_LATENCY_US = 25.733
DEFAULT_REPLAY_CSV = RESULTS / "demo_replay_input_features.csv"
DEFAULT_MANIFEST = RESULTS / "replay_vector_manifest.json"
DEFAULT_METADATA = RESULTS / "demo_replay_metadata.json"
LIBTORCH_NUM_THREADS = 8
UNOPTIMIZED_EAGER_BASELINE = {
    "label": "previous_python_eager_thread1",
    "mean_us": 191.1911865234375,
    "median_us": 110.3,
    "p99_us": 1662.4750000000022,
    "jitter_p99_minus_median_us": 1552.1750000000022,
    "min_us": 76.7,
    "max_us": 42285.4,
    "torch_threads": 1,
    "warmup_iters": 100,
    "model_execution_mode": "eager",
    "timed_region": "eager model forward plus Python split_output",
    "source_file": "results/cpu_live_replay_latency_summary.json before 2026-08-08 optimization patch",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_step_model() -> tuple[PHCSTPStepHLS, dict[str, Any]]:
    if not STEP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing migrated 05C4 seed2 step checkpoint: {STEP_MODEL_PATH}")
    payload = torch_load(STEP_MODEL_PATH)
    model = PHCSTPStepHLS(feature_dim=int(payload["feature_dim"]), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def build_timed_model(model: torch.nn.Module, x_cat_example: torch.Tensor, mode: str) -> torch.nn.Module:
    if mode == "eager":
        return model
    if mode != "trace":
        raise ValueError(f"Unsupported model mode: {mode}")
    with torch.no_grad():
        traced = torch.jit.trace(model, x_cat_example, strict=True)
        traced = torch.jit.optimize_for_inference(traced)
        traced.eval()
    return traced


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarize(latencies_us: np.ndarray, fpga_latency_us: float) -> dict[str, Any]:
    valid = latencies_us[np.isfinite(latencies_us)]
    if len(valid) == 0:
        raise RuntimeError("No valid latency measurements.")
    median = percentile(valid, 50)
    p99 = percentile(valid, 99)
    return {
        "count": int(len(valid)),
        "mean_us": float(valid.mean()),
        "median_us": median,
        "p50_us": median,
        "p90_us": percentile(valid, 90),
        "p95_us": percentile(valid, 95),
        "p99_us": p99,
        "min_us": float(valid.min()),
        "max_us": float(valid.max()),
        "std_us": float(valid.std(ddof=0)),
        "jitter_p99_minus_median_us": float(p99 - median),
        "fpga_latency_us": float(fpga_latency_us),
        "deadline_miss_rate_vs_fpga": float((valid > fpga_latency_us).mean()),
        "deadline_miss_count_vs_fpga": int((valid > fpga_latency_us).sum()),
    }


def load_replay_inputs(replay_csv: Path, manifest_path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict[str, Any]]:
    if not replay_csv.exists():
        raise FileNotFoundError(f"Missing replay input CSV: {replay_csv}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing replay manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("new5_used"):
        raise RuntimeError("Hard rule violation: replay manifest is marked new5_used=true.")
    expected_checkpoint = "n2ew_era\\06e_phc_stp_step_hls4ml\\models\\phc_stp_step_hls_seed2.pt"
    if str(manifest.get("step_model_checkpoint")) != expected_checkpoint:
        raise RuntimeError(
            "Replay manifest does not point at the expected 05C4 seed2 migrated step checkpoint: "
            f"{manifest.get('step_model_checkpoint')}"
        )
    with replay_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Replay CSV has no header: {replay_csv}")
        rows = list(reader)
    x_cols = [f"x_t_{i}" for i in range(FEATURE_DIM)]
    missing = [c for c in x_cols if c not in reader.fieldnames]
    if missing:
        raise RuntimeError(f"Replay CSV is missing feature columns: {missing}")
    if len(rows) != int(manifest.get("row_count", len(rows))):
        raise RuntimeError(f"Replay row count mismatch: csv={len(rows)} manifest={manifest.get('row_count')}")
    if "source_group" in reader.fieldnames and any(row.get("source_group", "").upper() == "NEW5" for row in rows):
        raise RuntimeError("Hard rule violation: replay CSV contains NEW5 rows.")
    x_np = np.asarray([[float(row[c]) for c in x_cols] for row in rows], dtype=np.float32)
    return rows, x_np, manifest


def render_terminal_plot(latencies: list[float], fpga_latency_us: float, total_events: int) -> None:
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) == 0:
        return
    stats = summarize(arr, fpga_latency_us)
    recent = arr[-64:]
    scale_max = max(float(np.percentile(recent, 95)), fpga_latency_us * 4.0, 1.0)
    chars = []
    for value in recent:
        if value <= fpga_latency_us:
            chars.append(".")
        elif value <= fpga_latency_us * 2.0:
            chars.append("+")
        elif value <= scale_max:
            chars.append("*")
        else:
            chars.append("!")
    hist_upper = max(scale_max, float(arr.max()), fpga_latency_us * 8.0) + 1e-9
    bins = np.asarray([0.0, fpga_latency_us, fpga_latency_us * 2, fpga_latency_us * 4, fpga_latency_us * 8, hist_upper])
    hist, edges = np.histogram(arr, bins=bins)
    max_count = max(int(hist.max()), 1)
    hist_lines = []
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * max(1, int(round(36 * count / max_count))) if count else ""
        hist_lines.append(f"{left:8.2f}-{right:8.2f} us | {bar} {int(count)}")

    print("\033[2J\033[H", end="")
    print("PHC CPU Live Replay Latency Demo")
    print(f"events {len(arr)}/{total_events} | FPGA fixed reference {fpga_latency_us:.3f} us")
    print(f"mean={stats['mean_us']:.2f}us median={stats['median_us']:.2f}us p99={stats['p99_us']:.2f}us "
          f"jitter={stats['jitter_p99_minus_median_us']:.2f}us miss={stats['deadline_miss_rate_vs_fpga']:.2%}")
    print()
    print("recent latency marks: . <= FPGA, + <= 2x FPGA, * slower, ! tail")
    print("".join(chars))
    print()
    print("running histogram")
    print("\n".join(hist_lines))


def setup_live_plot(fpga_latency_us: float, max_events: int):
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_scatter, ax_hist) = plt.subplots(2, 1, figsize=(11, 7))
    fig.canvas.manager.set_window_title("PHC CPU Live Replay Latency Demo")
    ax_scatter.set_title("CPU PyTorch per-event latency vs fixed FPGA latency")
    ax_scatter.set_xlabel("Replay event index")
    ax_scatter.set_ylabel("Latency (us)")
    ax_scatter.axhline(fpga_latency_us, color="#2ca02c", linewidth=1.5, label=f"FPGA fixed {fpga_latency_us:.3f} us")
    scatter = ax_scatter.scatter([], [], s=8, color="#d62728", alpha=0.75, label="CPU PyTorch")
    ax_scatter.set_xlim(0, max(max_events - 1, 1))
    ax_scatter.set_ylim(0, max(100.0, fpga_latency_us * 5.0))
    ax_scatter.grid(alpha=0.25)
    ax_scatter.legend(loc="upper right")

    ax_hist.set_title("Running histogram")
    ax_hist.set_xlabel("Latency (us)")
    ax_hist.set_ylabel("Count")
    text = fig.text(0.02, 0.02, "warming up...", fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show(block=False)
    return plt, fig, ax_scatter, ax_hist, scatter, text


def refresh_plot(
    plt,
    fig,
    ax_scatter,
    ax_hist,
    scatter,
    text,
    latencies: list[float],
    fpga_latency_us: float,
) -> None:
    arr = np.asarray(latencies, dtype=np.float64)
    idx = np.arange(len(arr), dtype=np.float64)
    scatter.set_offsets(np.column_stack([idx, arr]) if len(arr) else np.empty((0, 2)))
    ymax = max(float(np.nanmax(arr)) if len(arr) else 0.0, fpga_latency_us * 4.0, 50.0)
    ax_scatter.set_ylim(0, ymax * 1.12)
    ax_hist.cla()
    ax_hist.hist(arr, bins=40, color="#4c78a8", alpha=0.85)
    ax_hist.axvline(fpga_latency_us, color="#2ca02c", linewidth=1.5, label=f"FPGA {fpga_latency_us:.3f} us")
    ax_hist.set_title("Running histogram")
    ax_hist.set_xlabel("Latency (us)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.2)
    ax_hist.legend(loc="upper right")
    stats = summarize(arr, fpga_latency_us)
    text.set_text(
        "events={count}  mean={mean_us:.2f}us  median={median_us:.2f}us  "
        "p99={p99_us:.2f}us  jitter={jitter_p99_minus_median_us:.2f}us  "
        "miss>{fpga_latency_us:.3f}us={deadline_miss_rate_vs_fpga:.2%}".format(**stats)
    )
    fig.canvas.draw_idle()
    plt.pause(0.001)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    replay_rows, x_np, manifest = load_replay_inputs(args.replay_csv, args.manifest)
    if args.max_events is not None:
        x_np = x_np[: args.max_events]
        replay_rows = replay_rows[: args.max_events]
    if len(x_np) == 0:
        raise RuntimeError("No replay events selected.")

    torch.set_num_threads(int(args.torch_threads))
    try:
        torch.set_num_interop_threads(int(args.torch_threads))
        interop_threads = int(torch.get_num_interop_threads())
    except RuntimeError:
        interop_threads = int(torch.get_num_interop_threads())
    model, payload = load_step_model()
    if payload.get("new5_used"):
        raise RuntimeError("Hard rule violation: step model payload is marked new5_used=true.")
    if int(payload.get("source_seed", -1)) != 2 or str(payload.get("source_phase")) != "05c4":
        raise RuntimeError(f"Unexpected step model lineage: source_phase={payload.get('source_phase')} source_seed={payload.get('source_seed')}")
    model.eval()

    x_tensor = torch.from_numpy(x_np).to(dtype=torch.float32)
    x_cat = torch.zeros(1, FEATURE_DIM + HIDDEN_DIM, dtype=torch.float32)
    timed_model = build_timed_model(model, x_cat, args.model_mode)

    with torch.no_grad():
        warm_state = torch.zeros(1, HIDDEN_DIM, dtype=torch.float32)
        warm_x = x_tensor[:1]
        for _ in range(int(args.warmup_iters)):
            x_cat[:, :FEATURE_DIM].copy_(warm_x)
            x_cat[:, FEATURE_DIM:].copy_(warm_state)
            warm_out = timed_model(x_cat)
            warm_state = warm_out[..., 2:]
        state = torch.zeros(1, HIDDEN_DIM, dtype=torch.float32)

    plot_mode = "none" if args.no_plot else args.plot_mode
    plot_handles = None
    if plot_mode == "matplotlib":
        try:
            plot_handles = setup_live_plot(args.fpga_latency_us, len(x_np))
        except ImportError as exc:
            print(f"WARNING: matplotlib is unavailable ({exc}); falling back to terminal live plot.")
            plot_mode = "terminal"

    latencies: list[float] = []
    readout_rows: list[tuple[float, float]] = []
    wall_started = time.perf_counter()
    last_stdout_update = 0.0

    with torch.no_grad():
        for i in range(len(x_np)):
            x_cat[:, :FEATURE_DIM].copy_(x_tensor[i : i + 1])
            x_cat[:, FEATURE_DIM:].copy_(state)
            t0 = time.perf_counter_ns()
            out = timed_model(x_cat)
            t1 = time.perf_counter_ns()
            readout = out[..., :2]
            state = out[..., 2:]
            latency_us = (t1 - t0) / 1000.0
            latencies.append(float(latency_us))
            readout_rows.append((float(readout[0, 0].item()), float(readout[0, 1].item())))
            now = time.perf_counter()
            if plot_mode == "terminal" and ((i + 1) % int(args.plot_every) == 0 or i == len(x_np) - 1):
                render_terminal_plot(latencies, args.fpga_latency_us, len(x_np))
            elif now - last_stdout_update > 0.5 or i == len(x_np) - 1:
                stats = summarize(np.asarray(latencies, dtype=np.float64), args.fpga_latency_us)
                print(
                    "\r"
                    f"event={i + 1}/{len(x_np)} "
                    f"mean={stats['mean_us']:.2f}us median={stats['median_us']:.2f}us "
                    f"p99={stats['p99_us']:.2f}us miss>{args.fpga_latency_us:.3f}us={stats['deadline_miss_rate_vs_fpga']:.2%}",
                    end="",
                    flush=True,
                )
                last_stdout_update = now
            if plot_handles is not None and ((i + 1) % int(args.plot_every) == 0 or i == len(x_np) - 1):
                refresh_plot(*plot_handles, latencies, args.fpga_latency_us)
    print()

    arr = np.asarray(latencies, dtype=np.float64)
    stats = summarize(arr, args.fpga_latency_us)
    elapsed = time.perf_counter() - wall_started
    latency_csv = RESULTS / "cpu_live_replay_latency.csv"
    with latency_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "row_id",
            "source_replay_event_id",
            "event_index",
            "timestamp_us",
            "cpu_latency_us",
            "fpga_reference_latency_us",
            "deadline_miss_vs_fpga",
            "readout_0",
            "readout_1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, latency_us in enumerate(arr):
            row = replay_rows[i] if i < len(replay_rows) else {}
            writer.writerow(
                {
                    "row_id": i,
                    "source_replay_event_id": row.get("replay_event_id", i),
                    "event_index": row.get("event_index", -1),
                    "timestamp_us": row.get("timestamp_us", ""),
                    "cpu_latency_us": f"{float(latency_us):.9g}",
                    "fpga_reference_latency_us": f"{float(args.fpga_latency_us):.9g}",
                    "deadline_miss_vs_fpga": int(float(latency_us) > float(args.fpga_latency_us)),
                    "readout_0": f"{readout_rows[i][0]:.9g}",
                    "readout_1": f"{readout_rows[i][1]:.9g}",
                }
            )

    metadata = read_json(DEFAULT_METADATA) if DEFAULT_METADATA.exists() else {}
    summary = {
        "status": "cpu_live_replay_demo_completed",
        "model": "PHC-lite STP step model (PyTorch float32)",
        "step_model_checkpoint": str(STEP_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "source_checkpoint": payload.get("source_checkpoint"),
        "source_phase": payload.get("source_phase"),
        "source_seed": payload.get("source_seed"),
        "replay_csv": display_path(args.replay_csv),
        "manifest": display_path(args.manifest),
        "selected_window": metadata.get("selected_window", {}),
        "new5_used": False,
        "device": "CPU",
        "torch_threads": int(args.torch_threads),
        "torch_interop_threads": int(interop_threads),
        "warmup_iters": int(args.warmup_iters),
        "model_execution_mode": str(args.model_mode),
        "timed_region": "TorchScript/eager forward(x_cat) only; x_cat copy and output slicing are outside the measured region.",
        "comparison_reference": "cpp_libtorch_benchmark.cpp uses torch::set_num_threads(8), TorchScript module.forward, warmup=100.",
        "event_count": int(len(arr)),
        "wall_elapsed_sec": float(elapsed),
        **stats,
    }
    summary_json = RESULTS / "cpu_live_replay_latency_summary.json"
    write_json(summary_json, summary)

    parity_json = RESULTS / "cpu_trace_constant_forward_parity.json"
    root_cause_json = RESULTS / "cpu_latency_root_cause_audit_20260808.json"
    parity_summary = read_json(parity_json) if parity_json.exists() else None
    root_cause = read_json(root_cause_json) if root_cause_json.exists() else None
    if parity_summary is None:
        parity_table_row = "| Python trace parity check | not run | 8 | fixed zero input, forward only | n/a | n/a | n/a | n/a | n/a | n/a |"
        parity_note = (
            "The Python trace parity check was not available when this report was generated. "
            "Run `04_cpu_trace_parity_benchmark.py` before claiming methodology parity with the C++ baseline."
        )
    else:
        parity_table_row = (
            f"| Python trace parity check | trace | {parity_summary.get('torch_threads', 'n/a')} | fixed zero input, forward only | "
            f"{float(parity_summary['mean_us']):.3f} | {float(parity_summary['median_us']):.3f} | "
            f"{float(parity_summary['p99_us']):.3f} | "
            f"{float(parity_summary['p99_us']) - float(parity_summary['median_us']):.3f} | "
            f"{float(parity_summary['min_us']):.3f} | {float(parity_summary['max_us']):.3f} |"
        )
        parity_note = (
            "The fixed-zero-input Python trace parity check aligns the input and iteration protocol more closely with "
            "`cpp_libtorch_benchmark.cpp`; it does not show a stable Python-over-C++ advantage because the mean/p99 remain tail-sensitive."
        )
    root_cause_lines = [
        "## Root Cause Determination",
        "",
    ]
    if root_cause is None:
        root_cause_lines.extend(
            [
                "Root-cause audit file was not available when this report was generated. Run `05_cpu_latency_root_cause_audit.py` before making a final C++/Python parity claim.",
                "",
            ]
        )
    else:
        eq = root_cause.get("checkpoint_equivalence", {})
        cpp = root_cause.get("cpp_baseline", {})
        py = root_cause.get("python_parity", {})
        snapshots = root_cause.get("snapshots", {})
        before_load = snapshots.get("before_cpp", {}).get("cpu_load_counter", {}).get("parsed", {}).get("CookedValue")
        after_load = snapshots.get("after_python", {}).get("cpu_load_counter", {}).get("parsed", {}).get("CookedValue")
        root_cause_lines.extend(
            [
                f"- TorchScript-vs-eager checkpoint equivalence: `{eq.get('status', 'unknown')}`, max_abs_diff `{float(eq.get('max_abs_diff', float('nan'))):.6g}`.",
                f"- Controlled C++ retest: mean `{float(cpp.get('mean_us', float('nan'))):.3f}us`, median `{float(cpp.get('median_us', float('nan'))):.3f}us`, p99 `{float(cpp.get('p99_us', float('nan'))):.3f}us`.",
                f"- Controlled Python trace retest: mean `{float(py.get('mean_us', float('nan'))):.3f}us`, median `{float(py.get('median_us', float('nan'))):.3f}us`, p99 `{float(py.get('p99_us', float('nan'))):.3f}us`.",
                f"- CPU load during the audit was not single-digit idle: before C++ `{float(before_load):.3f}%`, after Python `{float(after_load):.3f}%`." if before_load is not None and after_load is not None else "- CPU load snapshot was unavailable in the root-cause audit.",
                "- The older C++ baseline files do not contain CPU-load snapshots and were produced on different dates, so historical load causality cannot be proven retroactively.",
                "",
                "Final determination: model/checkpoint mismatch is excluded. Background host load is the most plausible explanation for the C++/Python spread and p99 instability, but it remains partly inferential because the older runs did not record load. For demo/QA, quote the controlled C++ retest as the current optimized CPU single-stream reference and describe Python live replay as comparable, not faster.",
                "",
            ]
        )

    report = [
        "# CPU Live Replay Latency Demo",
        "",
        "status: `cpu_live_replay_demo_ready`",
        "",
        "## Scope",
        "",
        "- Laptop-side software demo only; no RTL, bitstream, threshold, or checkpoint was modified.",
        "- The script uses the existing 4096-event replay burst prepared for the FPGA demo.",
        "- The CPU model is the PyTorch float32 step model migrated from 05C4 seed2, matching the HLS deployment lineage.",
        "- NEW5 is not read; the script consumes only the prepared 06S2 replay CSV and manifest.",
        "",
        "## Inputs",
        "",
        f"- Replay CSV: `{display_path(args.replay_csv)}`",
        f"- Manifest: `{display_path(args.manifest)}`",
        f"- Step checkpoint: `{STEP_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- FPGA reference latency: `{args.fpga_latency_us:.3f} us`",
        f"- CPU execution mode: `{summary['model_execution_mode']}`",
        f"- Torch threads: `{summary['torch_threads']}`; interop threads: `{summary['torch_interop_threads']}`",
        f"- Warmup iterations before timing: `{summary['warmup_iters']}`",
        f"- Timed region: `{summary['timed_region']}`",
        "",
        "## One-Click Launch",
        "",
        "From `cmd.exe` or a double-clickable terminal:",
        "",
        "```cmd",
        "cd /d C:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\n2ew_era\\06s2_pynq_z2_wrapper_bringup",
        "scripts\\run_cpu_live_replay_demo.cmd",
        "```",
        "",
        "Headless smoke test, useful if the plotting window gets stuck:",
        "",
        "```cmd",
        "scripts\\run_cpu_live_replay_demo.cmd --no-plot --no-hold",
        "```",
        "",
        "## Expected Demo Screen",
        "",
        "- Default terminal mode: a live recent-latency strip plus a running histogram. `.` means CPU <= FPGA latency, `+` means <= 2x FPGA, `*` and `!` mark slower tail events.",
        "- Optional matplotlib mode: use `scripts\\run_cpu_live_replay_demo.cmd --plot-mode matplotlib` for a red CPU scatter plot and running histogram with a green FPGA reference line.",
        "- Running stats show `mean`, `median`, `p99`, `jitter = p99 - median`, and deadline-miss rate vs the FPGA reference.",
        "- In matplotlib mode, the final plot stays open after the 4096-event replay. Close the window to exit.",
        "",
        "## Final Summary",
        "",
        "| run | mode | threads | timed region | mean_us | median_us | p99_us | jitter_us | min_us | max_us |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        f"| previous unoptimized Python | {UNOPTIMIZED_EAGER_BASELINE['model_execution_mode']} | {UNOPTIMIZED_EAGER_BASELINE['torch_threads']} | {UNOPTIMIZED_EAGER_BASELINE['timed_region']} | {UNOPTIMIZED_EAGER_BASELINE['mean_us']:.3f} | {UNOPTIMIZED_EAGER_BASELINE['median_us']:.3f} | {UNOPTIMIZED_EAGER_BASELINE['p99_us']:.3f} | {UNOPTIMIZED_EAGER_BASELINE['jitter_p99_minus_median_us']:.3f} | {UNOPTIMIZED_EAGER_BASELINE['min_us']:.3f} | {UNOPTIMIZED_EAGER_BASELINE['max_us']:.3f} |",
        f"| optimized Python demo | {summary['model_execution_mode']} | {summary['torch_threads']} | forward only | {summary['mean_us']:.3f} | {summary['median_us']:.3f} | {summary['p99_us']:.3f} | {summary['jitter_p99_minus_median_us']:.3f} | {summary['min_us']:.3f} | {summary['max_us']:.3f} |",
        f"| known LibTorch C++ baseline | TorchScript C++ | 8 | module.forward only | 77.450 | 51-57.800 | not rerun here | not rerun here | not rerun here | not rerun here |",
        parity_table_row,
        "",
        f"Optimized Python deadline miss rate vs `{args.fpga_latency_us:.3f} us`: `{summary['deadline_miss_rate_vs_fpga']:.4%}`.",
        "",
        "The optimized Python demo now uses the same broad CPU opponent class as the report baseline: warmup, 8 Torch threads, and a traced TorchScript-style forward path. It remains a Python-hosted live visualizer, not the C++ LibTorch executable itself.",
        (
            "The optimized Python mean is still above 120 us, so this should be described as the best Python-side live-demo result on this environment; the already reported C++ LibTorch baseline remains the stronger optimized CPU reference."
            if summary["mean_us"] > 120.0
            else "The optimized Python mean is close to the already reported optimized CPU reference, while preserving live replay visualization."
        ),
        "",
        "## Methodology Parity Check vs LibTorch C++ Baseline",
        "",
        "Known C++ baseline source: `n2ew_era/06f_phc_lite_manual_hls/cpp_libtorch_benchmark.cpp`, with measured provenance in `n2ew_era/06f_phc_lite_manual_hls/results/_tmp_libtorch_probe.json` (`mean=77.4484us`, `median=57.8us`, `p99=361.1us`) and a later rerun in `cpu_baseline_timing_libtorch_raw_rerun2.json` (`mean=64.0703us`, `median=43.8us`, `p99=434.1us`).",
        "",
        "| item | C++ LibTorch baseline | Python live replay demo | Python trace parity check |",
        "|---|---|---|---|",
        "| Input | fixed `torch::zeros({1,32})` | 4096 real replay events plus recurrent state | fixed `torch.zeros(1,32)` |",
        "| Timed region | `module.forward(inputs)` only | `timed_model(x_cat)` only; tensor copy and slicing excluded | `timed_model(x_cat)` only |",
        "| Warmup | 100 | 100 | 100 |",
        "| Iterations | 10000 | 4096 replay events | 10000 |",
        "| Threads | `torch::set_num_threads(8)`, interop 8 | `torch.set_num_threads(8)`, interop 8 | same Python setting |",
        "| Runtime | C++ LibTorch executable | Python host with TorchScript trace | Python host with TorchScript trace |",
        "",
        parity_note,
        "",
        "Conclusion: do not say in the demo that the Python live visualizer is faster than the C++ LibTorch baseline. The honest line is: `the Python live demo is in the same order of magnitude as the optimized C++ CPU baseline, but it remains a Python-hosted visualization path; the FPGA reference is deterministic and already routed.`",
        "",
        *root_cause_lines,
        "## Demo Interpretation",
        "",
        "Use this as a latency/jitter visualization, not a PnL claim. The intended visual message is that the CPU per-event latency moves around while the FPGA reference is a fixed routed latency line.",
        "",
        "The CPU latency is measured live on the laptop for each replay event. The FPGA line is the routed hardware reference latency and is not live-synchronized to the board in this software-only demo.",
        "",
        "If the plot window stops responding, close it and rerun the launcher. Use the headless command above to confirm the model/data path is healthy before the live presentation.",
        "",
        "## Outputs",
        "",
        f"- `{display_path(latency_csv)}`",
        f"- `{display_path(summary_json)}`",
        "- `figures/cpu_live_replay_latency.png` when plotting is enabled.",
    ]
    (ROOT / "CPU_LIVE_REPLAY_DEMO.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if plot_handles is not None:
        plt, fig, *_ = plot_handles
        fig_path = FIGURES / "cpu_live_replay_latency.png"
        fig.savefig(fig_path, dpi=160)
        if not args.no_hold:
            print("Close the matplotlib window to exit.")
            plt.ioff()
            plt.show()
        else:
            plt.close(fig)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live CPU PyTorch replay latency demo for the PHC FPGA presentation.")
    parser.add_argument("--replay-csv", type=Path, default=DEFAULT_REPLAY_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=32)
    parser.add_argument("--plot-mode", choices=["terminal", "matplotlib", "none"], default="terminal")
    parser.add_argument("--torch-threads", type=int, default=LIBTORCH_NUM_THREADS)
    parser.add_argument("--model-mode", choices=["trace", "eager"], default="trace")
    parser.add_argument("--fpga-latency-us", type=float, default=FPGA_LATENCY_US)
    parser.add_argument("--no-plot", action="store_true", help="Alias for --plot-mode none; only write CSV/JSON/MD outputs.")
    parser.add_argument("--no-hold", action="store_true", help="Do not keep the matplotlib window open after completion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_every <= 0:
        raise ValueError("--plot-every must be positive.")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive.")
    summary = run_demo(args)
    print(
        "FINAL "
        f"mean_us={summary['mean_us']:.3f} "
        f"median_us={summary['median_us']:.3f} "
        f"p99_us={summary['p99_us']:.3f} "
        f"jitter_us={summary['jitter_p99_minus_median_us']:.3f} "
        f"miss_vs_fpga={summary['deadline_miss_rate_vs_fpga']:.4%}"
    )


if __name__ == "__main__":
    main()
