from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

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
    latency_events_to_skip,
    latency_events_to_skip_by_timestamp,
    policy_decision,
    policy_set,
    stable_uniform,
    write_markdown,
)

SPLIT = "val"
HORIZON = 20
FPGA_LATENCY_US = 83.16
CPU_RAW_CSV = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch_raw.csv"


def load_cpu_latency_samples() -> np.ndarray:
    raw = pd.read_csv(CPU_RAW_CSV)
    samples = pd.to_numeric(raw["latency_us"], errors="coerce").dropna().to_numpy(dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0.0)]
    if len(samples) == 0:
        raise RuntimeError(f"No usable latency samples in {CPU_RAW_CSV}")
    return samples


def empirical_latency_for_frame(frame: pd.DataFrame, decision: np.ndarray, samples: np.ndarray, salt_suffix: str) -> np.ndarray:
    u = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=f"02f|latency_shift_diag|{salt_suffix}")
    idx = np.minimum((u * len(samples)).astype(np.int64), len(samples) - 1)
    return samples[idx]


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def summarize_scenario(scenario: str, latencies_by_part: list[np.ndarray] | None = None) -> dict[str, Any]:
    policy = policy_set()[SELECTED_POLICY_ID]
    old_skips_selected: list[np.ndarray] = []
    new_skips_selected: list[np.ndarray] = []
    timestamp_shift_us_selected: list[np.ndarray] = []
    selected_count = 0
    valid_count = 0
    overflow_count = 0
    part_count = 0

    for part_idx, frame in enumerate(iter_excess_event_frames(SPLIT, horizons=[HORIZON])):
        decision = policy_decision(policy, frame).astype(np.int8)
        selected = decision != 0
        selected_count += int(selected.sum())
        if scenario == "cpu_empirical":
            if latencies_by_part is None:
                raise ValueError("latencies_by_part required for cpu_empirical")
            latency = latencies_by_part[part_idx]
        elif scenario == "fpga_fixed_83p16us":
            latency = FPGA_LATENCY_US
        else:
            raise ValueError(scenario)

        old_skip = latency_events_to_skip(frame, latency)
        source_idx = latency_events_to_skip_by_timestamp(frame, latency)
        row_idx = np.arange(len(frame), dtype=np.int64)
        valid = selected & (source_idx >= 0)
        overflow = selected & (source_idx < 0)
        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=float)
        old_skips_selected.append(old_skip[selected].astype(float))
        new_skips_selected.append((source_idx[valid] - row_idx[valid]).astype(float))
        timestamp_shift_us_selected.append(((timestamps[source_idx[valid]] - timestamps[valid]) / 1_000.0).astype(float))
        valid_count += int(valid.sum())
        overflow_count += int(overflow.sum())
        part_count += 1

    old_arr = np.concatenate(old_skips_selected) if old_skips_selected else np.array([], dtype=float)
    new_arr = np.concatenate(new_skips_selected) if new_skips_selected else np.array([], dtype=float)
    ts_arr = np.concatenate(timestamp_shift_us_selected) if timestamp_shift_us_selected else np.array([], dtype=float)
    old_q = quantiles(old_arr)
    new_q = quantiles(new_arr)
    ts_q = quantiles(ts_arr)
    return {
        "split": SPLIT,
        "scenario": scenario,
        "policy_id": SELECTED_POLICY_ID,
        "part_count": int(part_count),
        "selected_action_count": int(selected_count),
        "timestamp_valid_selected_count": int(valid_count),
        "timestamp_overflow_selected_count": int(overflow_count),
        "timestamp_overflow_rate_selected": float(overflow_count / selected_count) if selected_count else 0.0,
        "old_event_rate_skip_mean": old_q["mean"],
        "old_event_rate_skip_p50": old_q["p50"],
        "old_event_rate_skip_p99": old_q["p99"],
        "old_event_rate_skip_max": old_q["max"],
        "new_timestamp_event_shift_mean": new_q["mean"],
        "new_timestamp_event_shift_p50": new_q["p50"],
        "new_timestamp_event_shift_p99": new_q["p99"],
        "new_timestamp_event_shift_max": new_q["max"],
        "timestamp_shift_us_mean": ts_q["mean"],
        "timestamp_shift_us_p50": ts_q["p50"],
        "timestamp_shift_us_p99": ts_q["p99"],
        "timestamp_shift_us_max": ts_q["max"],
        "new_nonzero_shift_rate_selected": float((new_arr > 0).mean()) if len(new_arr) else 0.0,
        "old_nonzero_shift_rate_selected": float((old_arr > 0).mean()) if len(old_arr) else 0.0,
    }


def main() -> None:
    cpu_samples = load_cpu_latency_samples()
    cpu_latencies_by_part: list[np.ndarray] = []
    policy = policy_set()[SELECTED_POLICY_ID]
    for part_idx, frame in enumerate(iter_excess_event_frames(SPLIT, horizons=[HORIZON])):
        decision = policy_decision(policy, frame).astype(np.int8)
        cpu_latencies_by_part.append(empirical_latency_for_frame(frame, decision, cpu_samples, salt_suffix=f"{SPLIT}|part{part_idx}"))

    rows = [
        summarize_scenario("cpu_empirical", latencies_by_part=cpu_latencies_by_part),
        summarize_scenario("fpga_fixed_83p16us"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "latency_shift_diagnostic_val.csv", index=False)
    lines = [
        "# Val Latency Shift Diagnostic",
        "",
        "Scope: val split only, selected policy decisions only. This compares the legacy event-rate approximation against the timestamp searchsorted lookup. No matched-random, no PnL rerun, and no NEW5 access.",
        "",
        "Interpretation: the legacy event-rate approximation rounds microsecond-scale latency to zero events in this split. The timestamp method uses the last observed L2 event at or before `timestamp + latency_us * 1000`, so the book remains unchanged when no update occurs during the latency window.",
        "",
        "Caveat: val timestamps are monotonic non-decreasing, but many consecutive rows share the same timestamp. Event-shift counts can be nonzero even when timestamp shift is 0us because multiple event rows can share one timestamp.",
        "",
        out.to_markdown(index=False),
    ]
    write_markdown(ROOT / "LATENCY_SHIFT_DIAGNOSTIC_VAL.md", lines)
    print(out.to_string(index=False))
    print(json.dumps({"output_csv": str(RESULTS / "latency_shift_diagnostic_val.csv")}, indent=2))


if __name__ == "__main__":
    main()
