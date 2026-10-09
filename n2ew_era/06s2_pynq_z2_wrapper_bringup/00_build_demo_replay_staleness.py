from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s2_pynq_z2_wrapper_bringup"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ACTION_ROOT = (
    PROJECT_ROOT
    / "n2ew_era"
    / "04b_txf_only_final_development_rerun"
    / "data"
    / "txf_only_action_universe.parquet"
)
SCALER_PATH = (
    PROJECT_ROOT
    / "n2ew_era"
    / "05a_phc_lite_microstructure_sequence_feasibility"
    / "data"
    / "phc_lite_feature_scaler.json"
)
CPU_N4_CSV = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results" / "cpu_concurrent_benchmark_N4.csv"
CPU_N4_JSON = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results" / "cpu_concurrent_benchmark_N4.json"
SPLITS = ["val", "new3", "new4"]
ACTION_BID = 0
DEMO_EVENT_COUNT = 4096
BURST_SEARCH_WINDOW_US = 1_000_000.0
CALM_SEARCH_WINDOW_US = 1_000_000.0
FPGA_LATENCY_US = 25.733
FPGA_LATENCY_SOURCE = (
    "06S seed2 / 06L unroll16 90 MHz routed timing-met wrapper target; "
    "2316 cycles at 90 MHz = 25.733 us"
)
CPU_SAMPLE_SEED = 20260806
SCOREBOARD_STALE_METRIC = "stale_top_change_events"
SCOREBOARD_LABEL = "top-of-book change stale events"
SCOREBOARD_DEFINITION_NOTE = (
    "scoreboard uses stale_top_change_events: future events after decision event t where best_bid or best_ask "
    "actually changes before the latency arrival point. This matches the 06T event-shift definition."
)
PHC_FEATURES = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
    "log1p_bid_sz_1",
    "log1p_ask_sz_1",
    "log1p_top_depth",
    "clipped_inter_arrival_ms",
    "zscore_volatility_past_50_by_train_stats",
]
READ_COLUMNS = [
    "split",
    "source_group",
    "calendar_session_id",
    "session_id",
    "symbol",
    "event_index",
    "timestamp_us",
    "action_slot",
    "best_bid",
    "best_ask",
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "imbalance_change_past",
    "spread_change_past",
]


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def assert_no_new5(path_or_text: Any) -> None:
    if "NEW5" in str(path_or_text).upper():
        raise RuntimeError(f"NEW5 is forbidden for this demo replay phase: {path_or_text}")


def action_paths() -> list[Path]:
    paths: list[Path] = []
    for split in SPLITS:
        split_dir = ACTION_ROOT / f"split={split}"
        paths.extend(sorted(split_dir.glob("*.parquet")))
    for path in paths:
        assert_no_new5(path)
    return paths


def load_event_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=READ_COLUMNS)
    if raw["source_group"].astype(str).str.upper().eq("NEW5").any():
        raise RuntimeError(f"NEW5 row encountered in {path}")
    frame = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].copy()
    frame = frame.sort_values(["session_id", "timestamp_us", "event_index"], kind="mergesort").reset_index(drop=True)
    return frame


def rolling_count_peak(ts_us: np.ndarray, window_us: float) -> tuple[int, int, float]:
    starts = np.searchsorted(ts_us, ts_us - window_us, side="left")
    counts = np.arange(len(ts_us), dtype=np.int64) - starts + 1
    peak_idx = int(np.argmax(counts))
    peak_count = int(counts[peak_idx])
    rate_per_sec = float(peak_count / max(window_us / 1_000_000.0, 1e-9))
    return peak_idx, peak_count, rate_per_sec


def segment_bounds(center_idx: int, n: int, count: int) -> tuple[int, int]:
    start = max(0, center_idx - count // 2)
    end = min(n, start + count)
    start = max(0, end - count)
    return start, end


def candidate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in action_paths():
        frame = load_event_frame(path)
        if len(frame) < DEMO_EVENT_COUNT:
            continue
        ts = pd.to_numeric(frame["timestamp_us"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.diff(ts) >= 0):
            raise RuntimeError(f"Timestamps are not monotonic after sorting: {path}")
        peak_idx, peak_count, peak_rate = rolling_count_peak(ts, BURST_SEARCH_WINDOW_US)
        start, end = segment_bounds(peak_idx, len(frame), DEMO_EVENT_COUNT)
        duration_us = float(ts[end - 1] - ts[start]) if end > start else 0.0
        segment_rate = float((end - start) / max(duration_us / 1_000_000.0, 1e-9))
        calm_idx, calm_count, calm_peak_rate = calm_window_index(ts)
        calm_start, calm_end = segment_bounds(calm_idx, len(frame), min(DEMO_EVENT_COUNT, len(frame)))
        rows.append(
            {
                "path": str(path),
                "split": str(frame["split"].iloc[0]),
                "source_group": str(frame["source_group"].iloc[0]),
                "session_id": str(frame["session_id"].iloc[0]),
                "symbol": str(frame["symbol"].iloc[0]),
                "event_rows": int(len(frame)),
                "peak_idx": peak_idx,
                "peak_count_1s": peak_count,
                "peak_event_rate_per_sec": peak_rate,
                "segment_start": start,
                "segment_end": end,
                "segment_duration_us": duration_us,
                "segment_event_rate_per_sec": segment_rate,
                "median_event_rate_past_50": float(pd.to_numeric(frame["event_rate_past_50"], errors="coerce").median()),
                "p99_event_rate_past_50": float(pd.to_numeric(frame["event_rate_past_50"], errors="coerce").quantile(0.99)),
                "calm_idx": calm_idx,
                "calm_count_1s": calm_count,
                "calm_event_rate_per_sec": calm_peak_rate,
                "calm_start": calm_start,
                "calm_end": calm_end,
            }
        )
    return rows


def calm_window_index(ts_us: np.ndarray) -> tuple[int, int, float]:
    starts = np.searchsorted(ts_us, ts_us - CALM_SEARCH_WINDOW_US, side="left")
    counts = np.arange(len(ts_us), dtype=np.int64) - starts + 1
    margin = max(DEMO_EVENT_COUNT // 2, 1)
    valid = np.arange(len(ts_us))
    valid = valid[(valid >= margin) & (valid < len(ts_us) - margin)]
    if len(valid) == 0:
        idx = int(np.argmin(counts))
    else:
        idx = int(valid[np.argmin(counts[valid])])
    count = int(counts[idx])
    return idx, count, float(count / max(CALM_SEARCH_WINDOW_US / 1_000_000.0, 1e-9))


def feature_frame(frame: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    base_cols = [
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "volatility_past_50",
        "event_rate_past_50",
        "inter_arrival_ms",
        "imbalance_change_past",
        "spread_change_past",
    ]
    for col in base_cols:
        out[col] = (
            pd.to_numeric(frame[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    out["log1p_bid_sz_1"] = np.log1p(np.maximum(out["bid_sz_1"], 0.0)).astype(np.float32)
    out["log1p_ask_sz_1"] = np.log1p(np.maximum(out["ask_sz_1"], 0.0)).astype(np.float32)
    out["log1p_top_depth"] = np.log1p(np.maximum(out["top_depth"], 0.0)).astype(np.float32)
    out["clipped_inter_arrival_ms"] = np.clip(out["inter_arrival_ms"], 0.0, 5000.0).astype(np.float32)
    out["zscore_volatility_past_50_by_train_stats"] = (
        (out["volatility_past_50"] - float(scaler["volatility_mean_train"]))
        / max(float(scaler["volatility_std_train"]), 1e-6)
    ).astype(np.float32)
    raw_features = pd.DataFrame(out, index=frame.index)[PHC_FEATURES]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    arr = raw_features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norm = (arr - mean) / np.maximum(std, 1e-6)
    return pd.DataFrame(norm.astype(np.float32), columns=[f"x_t_{i}" for i in range(len(PHC_FEATURES))])


def load_cpu_latencies() -> tuple[np.ndarray, dict[str, Any]]:
    df = pd.read_csv(CPU_N4_CSV)
    if "latency_us" not in df.columns:
        raise RuntimeError(f"Missing latency_us column in {CPU_N4_CSV}")
    lat = pd.to_numeric(df["latency_us"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(lat) == 0:
        raise RuntimeError(f"No latency samples in {CPU_N4_CSV}")
    meta = read_json(CPU_N4_JSON)
    return lat, meta


def staleness_counts(ts_us: np.ndarray, best_bid: np.ndarray, best_ask: np.ndarray, latency_us: np.ndarray) -> dict[str, np.ndarray]:
    row_index = np.arange(len(ts_us), dtype=np.int64)
    target = ts_us + latency_us
    idx_right = np.searchsorted(ts_us, target, side="right") - 1
    idx_right = np.maximum(idx_right, row_index)
    all_events = idx_right - row_index
    strict_left = np.searchsorted(ts_us, ts_us, side="right")
    strict_right = np.searchsorted(ts_us, target, side="right")
    strict_time_events = np.maximum(strict_right - strict_left, 0)
    top_change = np.zeros(len(ts_us), dtype=bool)
    if len(ts_us) > 1:
        top_change[1:] = (best_bid[1:] != best_bid[:-1]) | (best_ask[1:] != best_ask[:-1])
    change_positions = np.flatnonzero(top_change)
    change_start = np.searchsorted(change_positions, row_index, side="right")
    change_end = np.searchsorted(change_positions, idx_right, side="right")
    top_changes = change_end - change_start
    change_ts = ts_us[change_positions] if len(change_positions) else np.array([], dtype=np.float64)
    strict_change_start = np.searchsorted(change_ts, ts_us, side="right")
    strict_change_end = np.searchsorted(change_ts, target, side="right")
    strict_time_top_changes = np.maximum(strict_change_end - strict_change_start, 0)
    return {
        "target_index": idx_right.astype(np.int64),
        "stale_all_events": all_events.astype(np.int64),
        "stale_top_change_events": top_changes.astype(np.int64),
        "stale_strict_time_events": strict_time_events.astype(np.int64),
        "stale_strict_time_top_change_events": strict_time_top_changes.astype(np.int64),
    }


def build_timeseries(frame: pd.DataFrame, cpu_source: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    ts = pd.to_numeric(frame["timestamp_us"], errors="coerce").to_numpy(dtype=np.float64)
    best_bid = pd.to_numeric(frame["best_bid"], errors="coerce").to_numpy(dtype=np.float64)
    best_ask = pd.to_numeric(frame["best_ask"], errors="coerce").to_numpy(dtype=np.float64)
    cpu_latency = rng.choice(cpu_source, size=len(frame), replace=True)
    fpga_latency = np.full(len(frame), FPGA_LATENCY_US, dtype=np.float64)
    cpu = staleness_counts(ts, best_bid, best_ask, cpu_latency)
    fpga = staleness_counts(ts, best_bid, best_ask, fpga_latency)
    out = pd.DataFrame(
        {
            "replay_event_id": np.arange(len(frame), dtype=np.int64),
            "source_split": frame["split"].astype(str).to_numpy(),
            "source_group": frame["source_group"].astype(str).to_numpy(),
            "session_id": frame["session_id"].astype(str).to_numpy(),
            "symbol": frame["symbol"].astype(str).to_numpy(),
            "event_index": pd.to_numeric(frame["event_index"], errors="coerce").to_numpy(dtype=np.int64),
            "timestamp_us": ts,
            "timestamp_rel_us": ts - ts[0],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "event_rate_past_50": pd.to_numeric(frame["event_rate_past_50"], errors="coerce").to_numpy(dtype=np.float64),
            "fpga_latency_us": fpga_latency,
            "cpu_latency_us": cpu_latency,
            "fpga_target_event_offset": fpga["stale_all_events"],
            "cpu_target_event_offset": cpu["stale_all_events"],
            "fpga_stale_all_events": fpga["stale_all_events"],
            "cpu_stale_all_events": cpu["stale_all_events"],
            "fpga_stale_top_change_events": fpga["stale_top_change_events"],
            "cpu_stale_top_change_events": cpu["stale_top_change_events"],
            "fpga_stale_strict_time_events": fpga["stale_strict_time_events"],
            "cpu_stale_strict_time_events": cpu["stale_strict_time_events"],
            "fpga_stale_strict_time_top_change_events": fpga["stale_strict_time_top_change_events"],
            "cpu_stale_strict_time_top_change_events": cpu["stale_strict_time_top_change_events"],
            "fpga_cumulative_stale_all_events": np.cumsum(fpga["stale_all_events"], dtype=np.int64),
            "cpu_cumulative_stale_all_events": np.cumsum(cpu["stale_all_events"], dtype=np.int64),
            "fpga_cumulative_stale_top_change_events": np.cumsum(fpga["stale_top_change_events"], dtype=np.int64),
            "cpu_cumulative_stale_top_change_events": np.cumsum(cpu["stale_top_change_events"], dtype=np.int64),
            "fpga_cumulative_stale_strict_time_events": np.cumsum(fpga["stale_strict_time_events"], dtype=np.int64),
            "cpu_cumulative_stale_strict_time_events": np.cumsum(cpu["stale_strict_time_events"], dtype=np.int64),
            "fpga_cumulative_stale_strict_time_top_change_events": np.cumsum(
                fpga["stale_strict_time_top_change_events"], dtype=np.int64
            ),
            "cpu_cumulative_stale_strict_time_top_change_events": np.cumsum(
                cpu["stale_strict_time_top_change_events"], dtype=np.int64
            ),
        }
    )
    return out


def summarize_timeseries(name: str, ts: pd.DataFrame) -> dict[str, Any]:
    scoreboard_cpu_col = f"cpu_{SCOREBOARD_STALE_METRIC}"
    scoreboard_fpga_col = f"fpga_{SCOREBOARD_STALE_METRIC}"
    return {
        "window_name": name,
        "scoreboard_metric": SCOREBOARD_STALE_METRIC,
        "scoreboard_definition": SCOREBOARD_DEFINITION_NOTE,
        "event_count": int(len(ts)),
        "duration_us": float(ts["timestamp_rel_us"].iloc[-1]) if len(ts) else 0.0,
        "duration_sec": float(ts["timestamp_rel_us"].iloc[-1] / 1_000_000.0) if len(ts) else 0.0,
        "event_rate_per_sec": float(len(ts) / max(ts["timestamp_rel_us"].iloc[-1] / 1_000_000.0, 1e-9)) if len(ts) else 0.0,
        "cpu_latency_mean_us": float(ts["cpu_latency_us"].mean()),
        "cpu_latency_p50_us": float(ts["cpu_latency_us"].quantile(0.5)),
        "cpu_latency_p99_us": float(ts["cpu_latency_us"].quantile(0.99)),
        "cpu_latency_max_us": float(ts["cpu_latency_us"].max()),
        "fpga_total_stale_all_events": int(ts["fpga_stale_all_events"].sum()),
        "cpu_total_stale_all_events": int(ts["cpu_stale_all_events"].sum()),
        "fpga_total_stale_top_change_events": int(ts["fpga_stale_top_change_events"].sum()),
        "cpu_total_stale_top_change_events": int(ts["cpu_stale_top_change_events"].sum()),
        "fpga_total_stale_strict_time_events": int(ts["fpga_stale_strict_time_events"].sum()),
        "cpu_total_stale_strict_time_events": int(ts["cpu_stale_strict_time_events"].sum()),
        "fpga_total_stale_strict_time_top_change_events": int(ts["fpga_stale_strict_time_top_change_events"].sum()),
        "cpu_total_stale_strict_time_top_change_events": int(ts["cpu_stale_strict_time_top_change_events"].sum()),
        "fpga_nonzero_stale_all_rate": float((ts["fpga_stale_all_events"] > 0).mean()),
        "cpu_nonzero_stale_all_rate": float((ts["cpu_stale_all_events"] > 0).mean()),
        "fpga_nonzero_stale_top_change_rate": float((ts["fpga_stale_top_change_events"] > 0).mean()),
        "cpu_nonzero_stale_top_change_rate": float((ts["cpu_stale_top_change_events"] > 0).mean()),
        "fpga_nonzero_stale_strict_time_rate": float((ts["fpga_stale_strict_time_events"] > 0).mean()),
        "cpu_nonzero_stale_strict_time_rate": float((ts["cpu_stale_strict_time_events"] > 0).mean()),
        "fpga_scoreboard_stale_events": int(ts[scoreboard_fpga_col].sum()),
        "cpu_scoreboard_stale_events": int(ts[scoreboard_cpu_col].sum()),
        "cpu_minus_fpga_scoreboard_stale_events": int(ts[scoreboard_cpu_col].sum() - ts[scoreboard_fpga_col].sum()),
        "fpga_nonzero_scoreboard_stale_rate": float((ts[scoreboard_fpga_col] > 0).mean()),
        "cpu_nonzero_scoreboard_stale_rate": float((ts[scoreboard_cpu_col] > 0).mean()),
        "cpu_minus_fpga_stale_all_events": int(ts["cpu_stale_all_events"].sum() - ts["fpga_stale_all_events"].sum()),
        "cpu_minus_fpga_stale_top_change_events": int(
            ts["cpu_stale_top_change_events"].sum() - ts["fpga_stale_top_change_events"].sum()
        ),
        "cpu_minus_fpga_stale_strict_time_events": int(
            ts["cpu_stale_strict_time_events"].sum() - ts["fpga_stale_strict_time_events"].sum()
        ),
        "cpu_minus_fpga_stale_strict_time_top_change_events": int(
            ts["cpu_stale_strict_time_top_change_events"].sum() - ts["fpga_stale_strict_time_top_change_events"].sum()
        ),
    }


def plot_scoreboard(ts: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x_ms = ts["timestamp_rel_us"].to_numpy(dtype=np.float64) / 1000.0
    axes[0].plot(x_ms, ts["event_rate_past_50"], color="#4c78a8", linewidth=0.9)
    axes[0].set_ylabel("event_rate_past_50")
    axes[0].set_title("Selected burst replay window")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x_ms, ts["cpu_cumulative_stale_top_change_events"], label="CPU N=4 empirical latency", color="#e45756")
    axes[1].plot(
        x_ms,
        ts["fpga_cumulative_stale_top_change_events"],
        label=f"FPGA fixed {FPGA_LATENCY_US:.3f}us",
        color="#54a24b",
    )
    axes[1].set_ylabel("cumulative stale top-of-book changes")
    axes[1].set_xlabel("replay time (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(selected: dict[str, Any], burst_summary: dict[str, Any], calm_summary: dict[str, Any], cpu_meta: dict[str, Any]) -> None:
    lines = [
        "# Demo Replay Staleness Scoreboard",
        "",
        "status: `part_a_replay_scoreboard_ready_for_selected_06s2_board_bitstream`",
        "",
        "## Scope",
        "",
        "- This is a pure software replay/staleness data-prep artifact for the demo scoreboard.",
        "- No model was trained, no checkpoint was changed, and no threshold was selected.",
        "- NEW5 was not touched; the selected replay comes from the existing 04B action universe splits `val/new3/new4` only.",
        "- CPU latency is sampled from the existing measured N=4 empirical distribution, not re-measured per event during this replay.",
        "",
        "## Selected Burst Window",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    for key in [
        "split",
        "source_group",
        "session_id",
        "symbol",
        "peak_count_1s",
        "peak_event_rate_per_sec",
        "segment_event_rate_per_sec",
        "segment_duration_us",
        "median_event_rate_past_50",
        "p99_event_rate_past_50",
    ]:
        lines.append(f"| `{key}` | `{selected[key]}` |")
    lines += [
        "",
        f"Source parquet: `{Path(selected['path']).relative_to(PROJECT_ROOT)}`",
        "",
        "The selected window maximizes the 1-second rolling event count among `val/new3/new4` sessions. "
        "This is a reproducible burst criterion, not a hand-picked dramatic interval.",
        "",
        "## Latency Sources",
        "",
        f"- FPGA latency: fixed `{FPGA_LATENCY_US}` us.",
        f"- FPGA latency source: {FPGA_LATENCY_SOURCE}.",
        "- 06U V1 `22.36 us` remains the lower-latency candidate, but the repo-local 06U RTL export is not self-contained for this wrapper bring-up. The selected board bitstream therefore uses the complete 06S seed2 / 06L unroll16 90 MHz RTL.",
        f"- CPU latency: empirical raw samples from `{CPU_N4_CSV.relative_to(PROJECT_ROOT)}`.",
        f"- CPU benchmark metadata: concurrency `{cpu_meta.get('concurrency')}`, samples per round `{cpu_meta.get('samples_per_round')}`, deadline miss rate round1 `{cpu_meta.get('round_summaries', [{}])[0].get('deadline_miss_rate')}`.",
        "",
        "## Scoreboard Definition",
        "",
        "- For event `i`, arrival time is `timestamp_us[i] + latency_us`.",
        "- Formal scoreboard metric: `stale_top_change_events`.",
        "- `stale_top_change_events` counts future events after decision event `t` where `best_bid` or `best_ask` actually changes before the latency arrival point. This matches the 06T event-shift definition: the kth future top-of-book price change after `t`.",
        "- `stale_strict_time_top_change_events` is also exported as a timestamp-conservative top-change audit variant.",
        "- `stale_strict_time_events` and `stale_all_events` are auxiliary audit counters only, not the formal scoreboard definition. `stale_strict_time_events` counts all real L2 events with `timestamp > timestamp[i]` and `timestamp <= arrival_time`; `stale_all_events` additionally counts later rows with the same timestamp by event-index order.",
        "",
        "## Burst vs Calm Sanity Check",
        "",
        "| window | events | duration_sec | event_rate/s | CPU scoreboard top-change stale | FPGA scoreboard top-change stale | CPU-FPGA scoreboard delta | CPU nonzero scoreboard rate | FPGA nonzero scoreboard rate | CPU strict-time top-change audit | FPGA strict-time top-change audit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in [burst_summary, calm_summary]:
        lines.append(
            f"| {s['window_name']} | {s['event_count']} | {s['duration_sec']:.6f} | {s['event_rate_per_sec']:.2f} | "
            f"{s['cpu_scoreboard_stale_events']} | {s['fpga_scoreboard_stale_events']} | {s['cpu_minus_fpga_scoreboard_stale_events']} | "
            f"{s['cpu_nonzero_scoreboard_stale_rate']:.4%} | {s['fpga_nonzero_scoreboard_stale_rate']:.4%} | "
            f"{s['cpu_total_stale_strict_time_top_change_events']} | {s['fpga_total_stale_strict_time_top_change_events']} |"
        )
    lines += [
        "",
        "Interpretation for the demo:",
        "",
        "- The official scoreboard should be described as top-of-book-change staleness, not strict timestamp all-event staleness.",
        "- If the burst CPU top-change counter finishes above FPGA, the replay supports the intended latency-backlog story under the 06T-compatible definition.",
        "- If both official counters are close, say that honestly and use the auxiliary strict-time/all-event counters only as audit context.",
        "- The calm window is included as a control; both counters should be materially flatter than the burst window.",
        "",
        "## Outputs",
        "",
        "- `results/demo_replay_staleness_timeseries.csv`",
        "- `results/demo_replay_staleness_timeseries.json`",
        "- `results/demo_replay_input_features.csv`",
        "- `results/demo_replay_window_selection.csv`",
        "- `results/demo_replay_summary.csv`",
        "- `figures/demo_replay_staleness_scoreboard.png`",
    ]
    (ROOT / "DEMO_REPLAY_STALENESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    scaler = read_json(SCALER_PATH)
    candidates = candidate_rows()
    if not candidates:
        raise RuntimeError("No replay candidates found.")
    pd.DataFrame(candidates).to_csv(RESULTS / "demo_replay_window_selection.csv", index=False)
    selected = max(candidates, key=lambda row: row["peak_event_rate_per_sec"])
    full = load_event_frame(Path(selected["path"]))
    burst = full.iloc[int(selected["segment_start"]) : int(selected["segment_end"])].reset_index(drop=True)
    calm = full.iloc[int(selected["calm_start"]) : int(selected["calm_end"])].reset_index(drop=True)
    cpu_latencies, cpu_meta = load_cpu_latencies()
    rng = np.random.default_rng(CPU_SAMPLE_SEED)
    burst_ts = build_timeseries(burst, cpu_latencies, rng)
    calm_ts = build_timeseries(calm, cpu_latencies, rng)
    features = feature_frame(burst, scaler)
    feature_out = pd.concat(
        [
            burst_ts[
                [
                    "replay_event_id",
                    "source_split",
                    "source_group",
                    "session_id",
                    "symbol",
                    "event_index",
                    "timestamp_us",
                    "timestamp_rel_us",
                ]
            ],
            features,
        ],
        axis=1,
    )
    burst_summary = summarize_timeseries("burst", burst_ts)
    calm_summary = summarize_timeseries("calm_control", calm_ts)
    burst_ts.to_csv(RESULTS / "demo_replay_staleness_timeseries.csv", index=False)
    burst_ts.to_json(RESULTS / "demo_replay_staleness_timeseries.json", orient="records", indent=2)
    calm_ts.to_csv(RESULTS / "demo_replay_calm_staleness_timeseries.csv", index=False)
    feature_out.to_csv(RESULTS / "demo_replay_input_features.csv", index=False)
    pd.DataFrame([burst_summary, calm_summary]).to_csv(RESULTS / "demo_replay_summary.csv", index=False)
    write_json(RESULTS / "demo_replay_metadata.json", {"selected_window": selected, "cpu_sample_seed": CPU_SAMPLE_SEED})
    plot_scoreboard(burst_ts, FIGURES / "demo_replay_staleness_scoreboard.png")
    write_report(selected, burst_summary, calm_summary, cpu_meta)
    print(json.dumps({"selected": selected, "burst_summary": burst_summary, "calm_summary": calm_summary}, indent=2))


if __name__ == "__main__":
    main()
