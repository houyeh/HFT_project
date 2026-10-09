from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
BRAM = ROOT / "bram"
FIGURES = ROOT / "figures"

DEFAULT_SOURCE_CSV = RESULTS / "demo_replay_seed2_90_hls_csim_sequential_out.csv"
DEFAULT_REPLAY_CSV = RESULTS / "demo_replay_input_features.csv"
DEFAULT_BRAM_MEM = BRAM / "demo_replay_x_readout_q24_12.mem"
DEFAULT_BRAM_MANIFEST = RESULTS / "bram_expected_from_hls_csim_manifest.json"
DEFAULT_METADATA = RESULTS / "demo_replay_metadata.json"
DEFAULT_OUTPUT_CSV = RESULTS / "fpga_known_result_replay_viewer_rows.csv"
DEFAULT_SUMMARY_JSON = RESULTS / "fpga_known_result_replay_viewer_summary.json"
DEFAULT_SCREENSHOT = FIGURES / "fpga_known_result_replay_viewer.png"

FEATURE_DIM = 16
READOUT_DIM = 2
WORD_HEX_DIGITS = 6
WORD_BITS = 24
SIGN_BIT = 1 << (WORD_BITS - 1)
WORD_MASK = (1 << WORD_BITS) - 1
Q_SCALE = 4096.0
FPGA_LATENCY_US = 25.733

DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2
DECISION_TO_NAME = {
    DECISION_NO_QUOTE: "no_quote",
    DECISION_BID: "bid",
    DECISION_ASK: "ask",
}
DECISION_TO_Y = {
    "ask": -1,
    "no_quote": 0,
    "bid": 1,
}
DECISION_COLORS = {
    "ask": "#c44e52",
    "no_quote": "#7f7f7f",
    "bid": "#2ca02c",
}


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signed24(raw: int) -> int:
    raw &= WORD_MASK
    if raw & SIGN_BIT:
        return raw - (1 << WORD_BITS)
    return raw


def decode_x_readout_mem(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    with path.open("r", encoding="utf-8") as f:
        for row_id, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            expected_digits = (FEATURE_DIM + READOUT_DIM) * WORD_HEX_DIGITS
            if len(line) != expected_digits:
                raise RuntimeError(
                    f"BRAM line {row_id} has {len(line)} hex digits; expected {expected_digits}: {path}"
                )
            words = [line[i : i + WORD_HEX_DIGITS] for i in range(0, len(line), WORD_HEX_DIGITS)]
            bid_lsb = signed24(int(words[-2], 16))
            ask_lsb = signed24(int(words[-1], 16))
            rows.append(
                {
                    "row_id": row_id,
                    "bram_bid_lsb": bid_lsb,
                    "bram_ask_lsb": ask_lsb,
                    "bram_bid_score": bid_lsb / Q_SCALE,
                    "bram_ask_score": ask_lsb / Q_SCALE,
                }
            )
    if not rows:
        raise RuntimeError(f"No BRAM rows decoded from {path}")
    return pd.DataFrame(rows)


def policy_from_lsb(bid_lsb: np.ndarray, ask_lsb: np.ndarray, threshold_lsb: int = 0) -> np.ndarray:
    decision = np.full(len(bid_lsb), DECISION_NO_QUOTE, dtype=np.int64)
    bid_mask = (bid_lsb >= ask_lsb) & (bid_lsb > threshold_lsb)
    ask_mask = (ask_lsb > bid_lsb) & (ask_lsb > threshold_lsb)
    decision[bid_mask] = DECISION_BID
    decision[ask_mask] = DECISION_ASK
    return decision


def decision_name(decision: int) -> str:
    return DECISION_TO_NAME[int(decision)]


def quantized_lsb_from_score(values: pd.Series) -> np.ndarray:
    # The source is HLS DUT output already printed on the Q24.12 grid; round avoids CSV decimal noise.
    return np.rint(values.to_numpy(dtype=np.float64) * Q_SCALE).astype(np.int64)


def verify_manifest(source_csv: Path, bram_manifest: Path) -> dict[str, Any]:
    manifest = read_json(bram_manifest)
    expected_status = "updated_from_hls_csim"
    if manifest.get("status") != expected_status:
        raise RuntimeError(f"Unexpected BRAM manifest status: {manifest.get('status')} != {expected_status}")
    manifest_source = str(manifest.get("csim_output", ""))
    normalized_manifest_source = manifest_source.replace("/", "\\")
    normalized_source = display_path(source_csv).replace("/", "\\")
    if normalized_manifest_source.lower() != normalized_source.lower():
        raise RuntimeError(
            "Selected source CSV does not match the BRAM update manifest. "
            f"manifest={manifest_source} selected={display_path(source_csv)}"
        )
    return manifest


def load_known_results(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in [args.source_csv, args.replay_csv, args.bram_mem, args.bram_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = verify_manifest(args.source_csv, args.bram_manifest)
    source = pd.read_csv(args.source_csv)
    replay = pd.read_csv(args.replay_csv)
    bram = decode_x_readout_mem(args.bram_mem)

    required_source_cols = {"row_id", "dut_readout_0", "dut_readout_1"}
    missing_source = required_source_cols - set(source.columns)
    if missing_source:
        raise RuntimeError(f"Source CSV missing columns: {sorted(missing_source)}")

    required_replay_cols = {
        "replay_event_id",
        "source_split",
        "source_group",
        "session_id",
        "symbol",
        "event_index",
        "timestamp_us",
        "timestamp_rel_us",
    }
    missing_replay = required_replay_cols - set(replay.columns)
    if missing_replay:
        raise RuntimeError(f"Replay CSV missing columns: {sorted(missing_replay)}")

    if len(source) != len(replay) or len(source) != len(bram):
        raise RuntimeError(f"Row count mismatch: source={len(source)} replay={len(replay)} bram={len(bram)}")
    if int(manifest.get("row_count", len(bram))) != len(bram):
        raise RuntimeError(f"BRAM manifest row_count={manifest.get('row_count')} but decoded rows={len(bram)}")
    if "source_group" in replay.columns and replay["source_group"].astype(str).str.upper().eq("NEW5").any():
        raise RuntimeError("Hard rule violation: replay CSV contains NEW5 rows.")

    source = source.sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(source["row_id"].to_numpy(dtype=np.int64), np.arange(len(source), dtype=np.int64)):
        raise RuntimeError("Source CSV row_id is not contiguous 0..N-1.")
    if not np.array_equal(bram["row_id"].to_numpy(dtype=np.int64), np.arange(len(bram), dtype=np.int64)):
        raise RuntimeError("Decoded BRAM row_id is not contiguous 0..N-1.")

    source_bid_lsb = quantized_lsb_from_score(source["dut_readout_0"])
    source_ask_lsb = quantized_lsb_from_score(source["dut_readout_1"])
    residual_bid_lsb = np.abs(source["dut_readout_0"].to_numpy(dtype=np.float64) * Q_SCALE - source_bid_lsb)
    residual_ask_lsb = np.abs(source["dut_readout_1"].to_numpy(dtype=np.float64) * Q_SCALE - source_ask_lsb)
    max_grid_residual_lsb = float(max(residual_bid_lsb.max(), residual_ask_lsb.max()))
    if max_grid_residual_lsb > 1e-3:
        raise RuntimeError(f"Source readout is not on Q24.12 grid; max residual {max_grid_residual_lsb:.6f} LSB")

    bram_bid_lsb = bram["bram_bid_lsb"].to_numpy(dtype=np.int64)
    bram_ask_lsb = bram["bram_ask_lsb"].to_numpy(dtype=np.int64)
    mismatch = (source_bid_lsb != bram_bid_lsb) | (source_ask_lsb != bram_ask_lsb)
    if bool(mismatch.any()):
        first = int(np.flatnonzero(mismatch)[0])
        raise RuntimeError(
            "BRAM readout does not match the selected HLS CSim source. "
            f"first_mismatch_row={first} source=({source_bid_lsb[first]}, {source_ask_lsb[first]}) "
            f"bram=({bram_bid_lsb[first]}, {bram_ask_lsb[first]})"
        )

    source_decision = policy_from_lsb(source_bid_lsb, source_ask_lsb)
    bram_decision = policy_from_lsb(bram_bid_lsb, bram_ask_lsb)
    decision_match = source_decision == bram_decision
    if not bool(decision_match.all()):
        first = int(np.flatnonzero(~decision_match)[0])
        raise RuntimeError(f"Source-vs-BRAM decision mismatch at row {first}")

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(bram), dtype=np.int64),
            "replay_event_id": replay["replay_event_id"].to_numpy(dtype=np.int64),
            "source_split": replay["source_split"].astype(str),
            "source_group": replay["source_group"].astype(str),
            "session_id": replay["session_id"].astype(str),
            "symbol": replay["symbol"].astype(str),
            "event_index": replay["event_index"].to_numpy(dtype=np.int64),
            "timestamp_us": replay["timestamp_us"].to_numpy(dtype=np.float64),
            "timestamp_rel_us": replay["timestamp_rel_us"].to_numpy(dtype=np.float64),
            "fpga_playback_elapsed_us": np.arange(len(bram), dtype=np.float64) * float(args.fpga_latency_us),
            "bid_lsb": bram_bid_lsb,
            "ask_lsb": bram_ask_lsb,
            "bid_score": bram_bid_lsb / Q_SCALE,
            "ask_score": bram_ask_lsb / Q_SCALE,
            "decision_code": bram_decision,
            "decision": [decision_name(v) for v in bram_decision],
            "source_decision": [decision_name(v) for v in source_decision],
            "match_known_source": decision_match.astype(np.int64),
        }
    )
    out["cumulative_bid"] = (out["decision"] == "bid").cumsum()
    out["cumulative_ask"] = (out["decision"] == "ask").cumsum()
    out["cumulative_no_quote"] = (out["decision"] == "no_quote").cumsum()
    out["cumulative_match_rate"] = out["match_known_source"].expanding().mean()

    summary = build_summary(out, args, manifest, max_grid_residual_lsb)
    return out, summary


def build_summary(df: pd.DataFrame, args: argparse.Namespace, manifest: dict[str, Any], residual_lsb: float) -> dict[str, Any]:
    counts = df["decision"].value_counts().reindex(["bid", "ask", "no_quote"], fill_value=0)
    total = int(len(df))
    return {
        "status": "ready",
        "scope": "software visualization only; no RTL/bitstream/threshold/checkpoint changes; no FPGA readback",
        "source_csv": display_path(args.source_csv),
        "bram_mem": display_path(args.bram_mem),
        "bram_manifest": display_path(args.bram_manifest),
        "replay_csv": display_path(args.replay_csv),
        "metadata_json": display_path(args.metadata),
        "row_count": total,
        "fpga_latency_us": float(args.fpga_latency_us),
        "fpga_playback_total_us": float((total - 1) * args.fpga_latency_us if total else 0.0),
        "fpga_playback_total_ms": float(((total - 1) * args.fpga_latency_us) / 1000.0 if total else 0.0),
        "source_vs_bram_readout_lsb_match_count": total,
        "source_vs_bram_readout_lsb_mismatch_count": 0,
        "source_vs_bram_decision_match_rate": float(df["match_known_source"].mean()),
        "max_source_grid_residual_lsb": residual_lsb,
        "decision_counts": {k: int(v) for k, v in counts.items()},
        "decision_rates": {k: float(v / total) if total else float("nan") for k, v in counts.items()},
        "timestamp_first_us": float(df["timestamp_us"].iloc[0]),
        "timestamp_last_us": float(df["timestamp_us"].iloc[-1]),
        "timestamp_rel_last_us": float(df["timestamp_rel_us"].iloc[-1]),
        "session_id": str(df["session_id"].iloc[0]),
        "symbol": str(df["symbol"].iloc[0]),
        "source_group": str(df["source_group"].iloc[0]),
        "bram_manifest_source": manifest,
    }


def import_pyplot(headless: bool):
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_dashboard(fig, axes: dict[str, Any], df: pd.DataFrame, frame_idx: int, tail_events: int, summary: dict[str, Any]) -> None:
    frame_idx = max(0, min(int(frame_idx), len(df) - 1))
    start = max(0, frame_idx - tail_events + 1)
    window = df.iloc[start : frame_idx + 1]
    current = df.iloc[frame_idx]

    ax_seq = axes["sequence"]
    ax_score = axes["scores"]
    ax_bar = axes["bars"]
    ax_text = axes["text"]
    for ax in [ax_seq, ax_score, ax_bar, ax_text]:
        ax.clear()

    x_ms = window["fpga_playback_elapsed_us"].to_numpy(dtype=np.float64) / 1000.0
    y = window["decision"].map(DECISION_TO_Y).to_numpy(dtype=np.int64)
    colors = window["decision"].map(DECISION_COLORS).to_list()

    ax_seq.scatter(x_ms, y, c=colors, s=18, alpha=0.9, linewidths=0)
    ax_seq.axvline(float(current["fpga_playback_elapsed_us"]) / 1000.0, color="#222222", linewidth=1.0, alpha=0.45)
    ax_seq.set_title("FPGA known decision replay")
    ax_seq.set_ylabel("decision")
    ax_seq.set_yticks([-1, 0, 1])
    ax_seq.set_yticklabels(["ask", "no_quote", "bid"])
    ax_seq.grid(alpha=0.22)
    ax_seq.set_xlim(max(0.0, float(x_ms.min()) - 0.05), float(x_ms.max()) + 0.05 if len(x_ms) else 1.0)

    ax_score.plot(x_ms, window["bid_score"], color=DECISION_COLORS["bid"], linewidth=1.7, label="bid score")
    ax_score.plot(x_ms, window["ask_score"], color=DECISION_COLORS["ask"], linewidth=1.7, label="ask score")
    ax_score.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.65)
    ax_score.axvline(float(current["fpga_playback_elapsed_us"]) / 1000.0, color="#222222", linewidth=1.0, alpha=0.45)
    ax_score.set_title("Q24.12 readout scores from board BRAM source")
    ax_score.set_xlabel("FPGA elapsed time model (ms)")
    ax_score.set_ylabel("score")
    ax_score.grid(alpha=0.22)
    ax_score.legend(loc="upper right")

    counts = pd.Series(
        {
            "bid": int(current["cumulative_bid"]),
            "ask": int(current["cumulative_ask"]),
            "no_quote": int(current["cumulative_no_quote"]),
        }
    )
    ax_bar.bar(counts.index, counts.values, color=[DECISION_COLORS[k] for k in counts.index])
    ax_bar.set_title("Cumulative decision distribution")
    ax_bar.set_ylabel("count")
    ax_bar.grid(axis="y", alpha=0.22)

    total_seen = frame_idx + 1
    current_text = (
        f"row {int(current['row_id']) + 1}/{len(df)} | decision {current['decision']}\n"
        f"session {current['session_id']}\n"
        f"event_index {int(current['event_index'])}\n"
        f"timestamp_us {float(current['timestamp_us']):.1f}\n"
        f"timestamp_rel_us {float(current['timestamp_rel_us']):.1f}\n"
        f"FPGA elapsed {float(current['fpga_playback_elapsed_us']):.3f} us\n\n"
        f"bid {float(current['bid_score']): .6f} ({int(current['bid_lsb'])} LSB)\n"
        f"ask {float(current['ask_score']): .6f} ({int(current['ask_lsb'])} LSB)\n\n"
        f"known-source match {float(current['cumulative_match_rate']) * 100.0:.3f}%\n"
        f"seen {total_seen} events\n"
        f"fixed latency {summary['fpga_latency_us']:.3f} us/event"
    )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        current_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.0,
        color="#222222",
    )
    fig.suptitle(
        f"PHC FPGA Known-Result Replay Viewer | {summary['symbol']} {summary['source_group']} | "
        f"BRAM/source decision match {summary['source_vs_bram_decision_match_rate'] * 100.0:.3f}%",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Synchronized visualization from precomputed FPGA replay artifacts; no live FPGA readback.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))


def create_figure(plt):
    fig = plt.figure(figsize=(13.5, 8.7), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.55, 1.35], width_ratios=[1.9, 1.1])
    axes = {
        "sequence": fig.add_subplot(gs[0, :]),
        "scores": fig.add_subplot(gs[1, :]),
        "bars": fig.add_subplot(gs[2, 0]),
        "text": fig.add_subplot(gs[2, 1]),
    }
    return fig, axes


def save_outputs(df: pd.DataFrame, summary: dict[str, Any], output_csv: Path, summary_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_screenshot(df: pd.DataFrame, summary: dict[str, Any], args: argparse.Namespace) -> None:
    plt = import_pyplot(headless=True)
    fig, axes = create_figure(plt)
    draw_dashboard(fig, axes, df, args.screenshot_frame, args.tail_events, summary)
    args.save_screenshot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save_screenshot, dpi=160)
    plt.close(fig)


def run_interactive(df: pd.DataFrame, summary: dict[str, Any], args: argparse.Namespace) -> None:
    plt = import_pyplot(headless=False)
    plt.ion()
    fig, axes = create_figure(plt)
    try:
        fig.canvas.manager.set_window_title("PHC FPGA Known-Result Replay Viewer")
    except Exception:
        pass

    max_events = len(df) if args.max_events is None else min(args.max_events, len(df))
    event_sleep_s = max(0.0, (args.fpga_latency_us * args.playback_slowdown) / 1_000_000.0)
    last_frame = -1
    for frame_idx in range(max_events):
        should_draw = frame_idx == 0 or frame_idx == max_events - 1 or frame_idx % args.plot_every == 0
        if should_draw:
            draw_dashboard(fig, axes, df, frame_idx, args.tail_events, summary)
            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_frame = frame_idx
        if event_sleep_s > 0:
            time.sleep(event_sleep_s)

    if last_frame != max_events - 1:
        draw_dashboard(fig, axes, df, max_events - 1, args.tail_events, summary)
        fig.canvas.draw_idle()
        plt.pause(0.001)
    if not args.no_hold:
        print("Close the FPGA known-result replay window to exit.")
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Laptop-side synchronized visualization of the precomputed FPGA demo readout stream. "
            "This script reads existing HLS CSim/BRAM replay artifacts only; it does not talk to the FPGA."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--replay-csv", type=Path, default=DEFAULT_REPLAY_CSV)
    parser.add_argument("--bram-mem", type=Path, default=DEFAULT_BRAM_MEM)
    parser.add_argument("--bram-manifest", type=Path, default=DEFAULT_BRAM_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--save-screenshot", type=Path, default=None)
    parser.add_argument("--screenshot-frame", type=int, default=2048)
    parser.add_argument("--fpga-latency-us", type=float, default=FPGA_LATENCY_US)
    parser.add_argument("--tail-events", type=int, default=240)
    parser.add_argument("--plot-every", type=int, default=4)
    parser.add_argument("--playback-slowdown", type=float, default=250.0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fpga_latency_us <= 0:
        raise ValueError("--fpga-latency-us must be positive.")
    if args.tail_events <= 0:
        raise ValueError("--tail-events must be positive.")
    if args.plot_every <= 0:
        raise ValueError("--plot-every must be positive.")
    if args.playback_slowdown < 0:
        raise ValueError("--playback-slowdown must be non-negative.")

    df, summary = load_known_results(args)
    save_outputs(df, summary, args.output_csv, args.summary_json)

    screenshot_path = args.save_screenshot
    if screenshot_path is None and args.no_show:
        screenshot_path = DEFAULT_SCREENSHOT
        args.save_screenshot = screenshot_path
    if screenshot_path is not None:
        args.save_screenshot = screenshot_path
        save_screenshot(df, summary, args)
    if not args.no_show:
        run_interactive(df, summary, args)

    print(
        "FINAL "
        f"rows={summary['row_count']} "
        f"source_vs_bram_readout_mismatch={summary['source_vs_bram_readout_lsb_mismatch_count']} "
        f"decision_match_rate={summary['source_vs_bram_decision_match_rate']:.6f} "
        f"bid={summary['decision_counts']['bid']} "
        f"ask={summary['decision_counts']['ask']} "
        f"no_quote={summary['decision_counts']['no_quote']} "
        f"output_csv={display_path(args.output_csv)} "
        f"summary_json={display_path(args.summary_json)}"
    )
    if args.save_screenshot is not None:
        print(f"SCREENSHOT {display_path(args.save_screenshot)}")


if __name__ == "__main__":
    main()
