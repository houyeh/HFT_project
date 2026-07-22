from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PHASE06F_RESULTS = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_phase02f import (  # noqa: E402
    RESULTS,
    SELECTED_POLICY_ID,
    iter_excess_event_frames,
    latency_events_to_skip_by_timestamp,
    policy_decision,
    policy_set,
    stable_uniform,
    write_markdown,
)

SPLIT = "val"
HORIZON = 20
GAP_TOP_N = 20
ABNORMAL_SHIFT_US = 10_000.0
CPU_RAW_CSV = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch_raw.csv"


def load_cpu_latency_samples() -> np.ndarray:
    raw = pd.read_csv(CPU_RAW_CSV)
    samples = pd.to_numeric(raw["latency_us"], errors="coerce").dropna().to_numpy(dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0.0)]
    if len(samples) == 0:
        raise RuntimeError(f"No usable latency samples in {CPU_RAW_CSV}")
    return samples


def empirical_latency_for_frame(frame: pd.DataFrame, decision: np.ndarray, samples: np.ndarray, salt_suffix: str) -> np.ndarray:
    u = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=f"02f|val_gap_diag|{salt_suffix}")
    idx = np.minimum((u * len(samples)).astype(np.int64), len(samples) - 1)
    return samples[idx]


def collect_top_gaps(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame_idx, frame in enumerate(frames):
        for session_id, group in frame.groupby(frame["session_id"].astype(str), sort=False):
            g = group.reset_index(drop=False).rename(columns={"index": "frame_row_index"})
            ts = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=float)
            event_index = pd.to_numeric(g["event_index"], errors="coerce").to_numpy(dtype=float)
            symbol = g["symbol"].astype(str).to_numpy() if "symbol" in g.columns else np.array([""] * len(g))
            if len(g) < 2:
                continue
            dt_ns = np.diff(ts)
            order = np.argsort(dt_ns)[::-1][:GAP_TOP_N]
            event_diff = np.diff(event_index)
            for local_i in order:
                rows.append(
                    {
                        "frame_part": int(frame_idx),
                        "session_id": str(session_id),
                        "local_gap_rank_input": int(local_i),
                        "gap_seconds": float(dt_ns[local_i] / 1_000_000_000.0),
                        "gap_us": float(dt_ns[local_i] / 1_000.0),
                        "before_local_row": int(local_i),
                        "after_local_row": int(local_i + 1),
                        "before_frame_row": int(g["frame_row_index"].iloc[local_i]),
                        "after_frame_row": int(g["frame_row_index"].iloc[local_i + 1]),
                        "before_event_index": int(event_index[local_i]) if np.isfinite(event_index[local_i]) else None,
                        "after_event_index": int(event_index[local_i + 1]) if np.isfinite(event_index[local_i + 1]) else None,
                        "event_index_delta": float(event_diff[local_i]) if np.isfinite(event_diff[local_i]) else np.nan,
                        "before_timestamp_ns": float(ts[local_i]),
                        "after_timestamp_ns": float(ts[local_i + 1]),
                        "before_symbol": str(symbol[local_i]),
                        "after_symbol": str(symbol[local_i + 1]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("gap_seconds", ascending=False).head(GAP_TOP_N).reset_index(drop=True)
    out.insert(0, "gap_rank", np.arange(1, len(out) + 1))
    return out


def collect_abnormal_shift_stats(frames: list[pd.DataFrame], cpu_samples: np.ndarray) -> pd.DataFrame:
    rows = []
    policy = policy_set()[SELECTED_POLICY_ID]
    for scenario in ["cpu_empirical", "fpga_fixed_83p16us"]:
        selected_total = 0
        valid_total = 0
        overflow_total = 0
        abnormal_total = 0
        max_shift_us = 0.0
        p99_chunks = []
        for frame_idx, frame in enumerate(frames):
            decision = policy_decision(policy, frame).astype(np.int8)
            selected = decision != 0
            if scenario == "cpu_empirical":
                latency = empirical_latency_for_frame(frame, decision, cpu_samples, salt_suffix=f"{SPLIT}|part{frame_idx}")
            else:
                latency = 83.16
            source_idx = latency_events_to_skip_by_timestamp(frame, latency)
            row_idx = np.arange(len(frame), dtype=np.int64)
            valid = selected & (source_idx >= 0)
            overflow = selected & (source_idx < 0)
            ts = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=float)
            shift_us = np.array([], dtype=float)
            if valid.any():
                shift_us = (ts[source_idx[valid]] - ts[row_idx[valid]]) / 1_000.0
                p99_chunks.append(shift_us)
            selected_total += int(selected.sum())
            valid_total += int(valid.sum())
            overflow_total += int(overflow.sum())
            abnormal_total += int((shift_us > ABNORMAL_SHIFT_US).sum())
            if len(shift_us):
                max_shift_us = max(max_shift_us, float(np.max(shift_us)))
        all_shift = np.concatenate(p99_chunks) if p99_chunks else np.array([], dtype=float)
        rows.append(
            {
                "split": SPLIT,
                "scenario": scenario,
                "policy_id": SELECTED_POLICY_ID,
                "selected_action_count": int(selected_total),
                "timestamp_valid_selected_count": int(valid_total),
                "timestamp_overflow_selected_count": int(overflow_total),
                "abnormal_shift_threshold_us": float(ABNORMAL_SHIFT_US),
                "abnormal_shift_count": int(abnormal_total),
                "abnormal_shift_rate_selected": float(abnormal_total / selected_total) if selected_total else 0.0,
                "timestamp_shift_us_p50": float(np.quantile(all_shift, 0.50)) if len(all_shift) else 0.0,
                "timestamp_shift_us_p99": float(np.quantile(all_shift, 0.99)) if len(all_shift) else 0.0,
                "timestamp_shift_us_max": float(max_shift_us),
            }
        )
    return pd.DataFrame(rows)


def infer_gap_interpretation(top_gaps: pd.DataFrame, stats: pd.DataFrame) -> list[str]:
    if top_gaps.empty:
        return ["No timestamp gaps were found."]
    max_gap = float(top_gaps["gap_seconds"].max())
    max_event_delta = float(pd.to_numeric(top_gaps["event_index_delta"], errors="coerce").abs().max())
    same_symbol_rate = float((top_gaps["before_symbol"].astype(str) == top_gaps["after_symbol"].astype(str)).mean()) if "before_symbol" in top_gaps else 0.0
    lines = []
    if max_gap > 3600.0:
        lines.append(
            f"The largest gap is {max_gap:.2f} seconds, which is hours long. This is far outside a microstructure latency scale and should not be interpreted as latency adverse selection."
        )
    elif max_gap > 10.0:
        lines.append(
            f"The largest gap is {max_gap:.2f} seconds, which is much larger than the CPU/FPGA latency inputs and likely reflects missing updates or session segmentation rather than ordinary microstructure."
        )
    else:
        lines.append(
            f"The largest gap is {max_gap:.6f} seconds. From timestamp size alone this could still be a quiet period, but it remains much larger than the configured latency threshold."
        )
    if max_event_delta <= 1.0:
        lines.append(
            "The largest gaps occur between consecutive event_index values, which suggests the discontinuity is in timestamp spacing rather than event_index jumps."
        )
    else:
        lines.append(
            f"Some large gaps also have event_index_delta up to {max_event_delta:g}, so event numbering may also contain discontinuities."
        )
    if same_symbol_rate >= 0.95:
        lines.append(
            "Symbols before and after the top gaps are almost always identical, so the gap does not look like an obvious symbol switch from the available columns."
        )
    else:
        lines.append(
            "Some top gaps cross symbol values, which may indicate mixed streams or segmentation issues."
        )
    lines.append(
        "From the available columns alone, this cannot be cleanly classified as a true trading halt versus a session-splitting artifact. The next audit should inspect the Phase 01 session construction/raw timestamp parser."
    )
    if not stats.empty:
        for _, row in stats.iterrows():
            lines.append(
                f"{row['scenario']}: {int(row['abnormal_shift_count']):,} selected decisions exceed {ABNORMAL_SHIFT_US / 1000:.1f}ms timestamp shift ({row['abnormal_shift_rate_selected']:.4%} of selected actions)."
            )
    return lines


def main() -> None:
    frames = list(iter_excess_event_frames(SPLIT, horizons=[HORIZON]))
    cpu_samples = load_cpu_latency_samples()
    top_gaps = collect_top_gaps(frames)
    stats = collect_abnormal_shift_stats(frames, cpu_samples)
    top_gaps.to_csv(RESULTS / "val_timestamp_top_gaps.csv", index=False)
    stats.to_csv(RESULTS / "val_timestamp_abnormal_shift_stats.csv", index=False)

    lines = [
        "# Val Timestamp Gap Diagnostic",
        "",
        "Scope: val split only. This diagnostic does not change latency-aware fill logic, does not run PnL, and does not touch NEW5.",
        "",
        f"Abnormal shift threshold: `{ABNORMAL_SHIFT_US:.0f}us` (10ms).",
        "",
        "## Top 20 Timestamp Gaps",
        "",
        top_gaps.to_markdown(index=False) if not top_gaps.empty else "No gaps found.",
        "",
        "## Selected Decision Impact",
        "",
        stats.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {line}" for line in infer_gap_interpretation(top_gaps, stats))
    write_markdown(ROOT / "VAL_TIMESTAMP_GAP_DIAGNOSTIC.md", lines)
    print(top_gaps.head(20).to_string(index=False))
    print(stats.to_string(index=False))
    print(f"wrote {ROOT / 'VAL_TIMESTAMP_GAP_DIAGNOSTIC.md'}")


if __name__ == "__main__":
    main()
