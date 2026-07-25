from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = ROOT / "n2ew_era" / "06p_event_interarrival_latency_budget"
RESULTS_DIR = PHASE_DIR / "results"
FIGURES_DIR = PHASE_DIR / "figures"
SOURCE_DIR = (
    ROOT
    / "n2ew_era"
    / "04b_txf_only_final_development_rerun"
    / "data"
    / "txf_only_action_universe.parquet"
)

SCRIPT_REL = "n2ew_era/06p_event_interarrival_latency_budget/00_event_interarrival_latency_budget.py"
REPORT_PATH = PHASE_DIR / "AUDIT_event_interarrival_latency_budget.md"
SUMMARY_CSV = RESULTS_DIR / "event_interarrival_latency_budget_summary.csv"
SESSION_CSV = RESULTS_DIR / "event_interarrival_session_summary.csv"
RESOLUTION_CSV = RESULTS_DIR / "timestamp_resolution_audit.csv"
CDF_PATH = FIGURES_DIR / "event_interarrival_cdf.png"

SPLITS = ["train", "val", "new3", "new4"]
SPLIT_TO_SOURCE = {
    "train": "NEW2",
    "val": "NEW1",
    "new3": "NEW3",
    "new4": "NEW4",
}
FORBIDDEN_SOURCE_GROUPS = {"NEW5"}
FORBIDDEN_SPLITS = {"new5", "NEW5"}

LATENCY_FAST_US = 23.16
LATENCY_BASELINE_US = 83.16
ONE_MS_US = 1_000.0
TEN_MS_US = 10_000.0
DAY_US = 86_400_000_000
PEAK_WINDOW_US = 1_000_000  # 1 second windows
PEAK_WINDOW_FRACTION = 0.10

STATE_COLUMNS = [
    "split",
    "source_group",
    "calendar_session_id",
    "session_id",
    "product_id",
    "event_index",
    "timestamp_us",
    "action_slot",
    "best_bid",
    "best_ask",
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def unwrap_session_timestamp_us(ts: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert time-of-day microseconds into session-clock microseconds.

    Raw `timestamp_us` is time of day. Night sessions can cross midnight, and a few
    raw chunks can arrive in non-chronological event_index order. We anchor on the
    first observed timestamp: if the session begins after noon, timestamps more
    than six hours behind the first timestamp are treated as after-midnight events.
    The caller then sorts by this unwrapped timestamp, with event_index as tie-break.
    """
    ts = np.rint(ts.astype(np.float64)).astype(np.int64)
    if ts.size == 0:
        return ts, 0
    first = int(ts[0])
    shifted = np.zeros(ts.size, dtype=bool)
    if first >= 12 * 60 * 60 * 1_000_000:
        shifted = ts < (first - 6 * 60 * 60 * 1_000_000)
    return ts + shifted.astype(np.int64) * DAY_US, int(shifted.sum())


def quote_eligible_mask(frame: pd.DataFrame) -> np.ndarray:
    required = ["best_bid", "best_ask", "bid_sz_1", "ask_sz_1", "spread_ticks"]
    work = frame[required]
    finite = np.isfinite(work.to_numpy(dtype=np.float64)).all(axis=1)
    return (
        finite
        & frame["best_bid"].to_numpy(dtype=np.float64).__lt__(frame["best_ask"].to_numpy(dtype=np.float64))
        & (frame["bid_sz_1"].to_numpy(dtype=np.float64) > 0)
        & (frame["ask_sz_1"].to_numpy(dtype=np.float64) > 0)
        & (frame["spread_ticks"].to_numpy(dtype=np.float64) > 0)
    )


def peak_interval_mask(unwrapped_ts: np.ndarray) -> np.ndarray:
    """Return interval mask where the later event is in a top-10% 1s window."""
    if unwrapped_ts.size < 2:
        return np.zeros(0, dtype=bool)
    windows = np.floor_divide(unwrapped_ts, PEAK_WINDOW_US)
    unique, counts = np.unique(windows, return_counts=True)
    if unique.size == 0:
        return np.zeros(unwrapped_ts.size - 1, dtype=bool)
    top_n = max(1, int(math.ceil(unique.size * PEAK_WINDOW_FRACTION)))
    order = np.argsort(counts)[::-1]
    peak_windows = unique[order[:top_n]]
    return np.isin(windows[1:], peak_windows)


def describe_diffs(diffs: np.ndarray, split: str, scope: str, source_group: str) -> dict[str, object]:
    diffs = np.asarray(diffs, dtype=np.float64)
    total = int(diffs.size)
    if total == 0:
        return {
            "split": split,
            "source_group": source_group,
            "scope": scope,
            "interval_count": 0,
            "same_timestamp_count": 0,
            "same_timestamp_rate": np.nan,
            "min_nonzero_dt_us": np.nan,
            "p50_dt_us": np.nan,
            "p90_dt_us": np.nan,
            "p99_dt_us": np.nan,
            "p999_dt_us": np.nan,
            "p_lt_23p16us_observed_resolution_limited": np.nan,
            "p_lt_83p16us_observed_resolution_limited": np.nan,
            "p_lt_1ms_observed": np.nan,
            "p_lt_10ms_observed": np.nan,
        }
    nonzero = diffs[diffs > 0]
    return {
        "split": split,
        "source_group": source_group,
        "scope": scope,
        "interval_count": total,
        "same_timestamp_count": int((diffs == 0).sum()),
        "same_timestamp_rate": float((diffs == 0).mean()),
        "min_nonzero_dt_us": float(nonzero.min()) if nonzero.size else np.nan,
        "p50_dt_us": float(np.quantile(diffs, 0.50)),
        "p90_dt_us": float(np.quantile(diffs, 0.90)),
        "p99_dt_us": float(np.quantile(diffs, 0.99)),
        "p999_dt_us": float(np.quantile(diffs, 0.999)),
        # These are timestamp-observed shares only. The report treats sub-ms thresholds as not measurable
        # when the verified timestamp granularity is 1 ms.
        "p_lt_23p16us_observed_resolution_limited": float((diffs < LATENCY_FAST_US).mean()),
        "p_lt_83p16us_observed_resolution_limited": float((diffs < LATENCY_BASELINE_US).mean()),
        "p_lt_1ms_observed": float((diffs < ONE_MS_US).mean()),
        "p_lt_10ms_observed": float((diffs < TEN_MS_US).mean()),
    }


def read_state_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=STATE_COLUMNS)
    if frame["split"].astype(str).str.lower().isin(FORBIDDEN_SPLITS).any():
        raise RuntimeError(f"Forbidden NEW5-like split found in {path}")
    if frame["source_group"].astype(str).isin(FORBIDDEN_SOURCE_GROUPS).any():
        raise RuntimeError(f"Forbidden NEW5 source_group found in {path}")
    # One row per L2 state. 04B emits bid/ask/no_quote; action_slot 0 is aligned with the state.
    frame = frame[frame["action_slot"].astype(int).eq(0)].copy()
    frame = frame.sort_values(["event_index"], kind="mergesort").drop_duplicates("event_index", keep="first")
    return frame.reset_index(drop=True)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    aggregate: dict[str, dict[str, list[np.ndarray]]] = {}
    session_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    cdf_arrays: dict[str, list[np.ndarray]] = {
        "all_event_flow": [],
        "quote_eligible_state_subset": [],
        "peak10_all_event_flow": [],
        "peak10_quote_eligible_state_subset": [],
    }

    for split in SPLITS:
        split_dir = SOURCE_DIR / f"split={split}"
        files = sorted(split_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found for split={split}: {split_dir}")
        aggregate.setdefault(split, {})
        aggregate.setdefault("all_new1_new4", {})

        for path in files:
            if "NEW5" in path.name.upper() or "NEW5" in str(path).upper():
                raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")

            frame = read_state_frame(path)
            if frame.empty:
                continue
            session_id = str(frame["session_id"].iloc[0])
            source_group = str(frame["source_group"].iloc[0])
            product_id = str(frame["product_id"].iloc[0])
            timestamps_raw_event_index_order = frame["timestamp_us"].to_numpy(dtype=np.float64)
            event_index_order_diffs = np.diff(np.rint(timestamps_raw_event_index_order).astype(np.int64))
            event_index_negative_timestamp_count = int((event_index_order_diffs < 0).sum())

            timestamps_unwrapped_event_index_order, midnight_shifted_rows = unwrap_session_timestamp_us(
                timestamps_raw_event_index_order
            )
            frame = frame.assign(timestamp_unwrapped_us=timestamps_unwrapped_event_index_order)
            frame = frame.sort_values(["timestamp_unwrapped_us", "event_index"], kind="mergesort").reset_index(drop=True)
            timestamps = frame["timestamp_unwrapped_us"].to_numpy(dtype=np.int64)
            diffs = np.diff(timestamps).astype(np.float64)
            if np.any(diffs < 0):
                raise RuntimeError(f"Negative diffs remain after unwrap in {path}")

            quote_mask = quote_eligible_mask(frame)
            quote_interval = quote_mask[1:] & quote_mask[:-1]
            peak_mask = peak_interval_mask(timestamps)

            scoped = {
                "all_event_flow": diffs,
                "quote_eligible_state_subset": diffs[quote_interval],
                "peak10_all_event_flow": diffs[peak_mask],
                "peak10_quote_eligible_state_subset": diffs[peak_mask & quote_interval],
            }

            for scope, arr in scoped.items():
                aggregate[split].setdefault(scope, []).append(arr)
                aggregate["all_new1_new4"].setdefault(scope, []).append(arr)
                cdf_arrays[scope].append(arr)

            nonzero = diffs[diffs > 0]
            session_rows.append(
                {
                    "split": split,
                    "source_group": source_group,
                    "session_id": session_id,
                    "product_id": product_id,
                    "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "state_count": int(len(frame)),
                    "interval_count": int(len(diffs)),
                    "quote_eligible_state_count": int(quote_mask.sum()),
                    "quote_eligible_state_rate": float(quote_mask.mean()),
                    "midnight_shifted_rows": int(midnight_shifted_rows),
                    "event_index_order_negative_timestamp_count": event_index_negative_timestamp_count,
                    "same_timestamp_count": int((diffs == 0).sum()),
                    "same_timestamp_rate": float((diffs == 0).mean()) if diffs.size else np.nan,
                    "min_nonzero_dt_us": float(nonzero.min()) if nonzero.size else np.nan,
                    "p50_dt_us": float(np.quantile(diffs, 0.50)) if diffs.size else np.nan,
                    "p99_dt_us": float(np.quantile(diffs, 0.99)) if diffs.size else np.nan,
                    "max_dt_us": float(diffs.max()) if diffs.size else np.nan,
                }
            )
            resolution_rows.append(
                {
                    "split": split,
                    "source_group": source_group,
                    "session_id": session_id,
                    "timestamp_column": "timestamp_us",
                    "timestamp_unit": "microseconds_from_info_time_time_of_day",
                    "parser_source": "Phase 04A.1 interprets info_time as hour:minute:second:millisecond:flag",
                    "min_nonzero_adjacent_diff_us": float(nonzero.min()) if nonzero.size else np.nan,
                    "same_timestamp_rate": float((diffs == 0).mean()) if diffs.size else np.nan,
                    "midnight_unwrap_applied": bool(midnight_shifted_rows),
                    "midnight_shifted_row_count": int(midnight_shifted_rows),
                    "event_index_order_negative_timestamp_count": event_index_negative_timestamp_count,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for split, scopes in aggregate.items():
        source_group = "NEW1-NEW4" if split == "all_new1_new4" else SPLIT_TO_SOURCE[split]
        for scope, parts in scopes.items():
            arr = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
            summary_rows.append(describe_diffs(arr, split, scope, source_group))

    return (
        pd.DataFrame(summary_rows).sort_values(["split", "scope"]).reset_index(drop=True),
        pd.DataFrame(session_rows).sort_values(["split", "session_id"]).reset_index(drop=True),
        pd.DataFrame(resolution_rows).sort_values(["split", "session_id"]).reset_index(drop=True),
        cdf_arrays,
    )


def fmt_us(value: float) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}s"
    if value >= 1_000:
        return f"{value / 1_000:.3f}ms"
    return f"{value:.3f}us"


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.6f}%"


def make_cdf_figure(cdf_arrays: dict[str, list[np.ndarray]]) -> None:
    plt.figure(figsize=(10, 6))
    styles = {
        "all_event_flow": ("All event flow", "#1f77b4"),
        "quote_eligible_state_subset": ("Quote-eligible subset", "#ff7f0e"),
        "peak10_all_event_flow": ("Peak 10% windows, all", "#2ca02c"),
        "peak10_quote_eligible_state_subset": ("Peak 10% windows, quote-eligible", "#d62728"),
    }
    for scope, (label, color) in styles.items():
        parts = cdf_arrays.get(scope, [])
        if not parts:
            continue
        diffs = np.concatenate(parts)
        if diffs.size == 0:
            continue
        zero_mass = float((diffs == 0).mean())
        positive = np.sort(diffs[diffs > 0])
        if positive.size == 0:
            continue
        y = (np.arange(1, positive.size + 1) + int((diffs == 0).sum())) / diffs.size
        plt.step(positive, y, where="post", label=f"{label} (same-ts={zero_mass:.2%})", color=color)

    plt.axvline(LATENCY_FAST_US, color="black", linestyle="--", linewidth=1.2, label="23.16us reference")
    plt.axvline(LATENCY_BASELINE_US, color="black", linestyle=":", linewidth=1.4, label="83.16us reference")
    plt.axvline(ONE_MS_US, color="gray", linestyle="-.", linewidth=1.0, label="1ms timestamp tick")
    plt.xscale("log")
    plt.xlabel("Adjacent event inter-arrival time, delta_t (us, log scale)")
    plt.ylabel("Cumulative probability")
    plt.title("TXF-only NEW1-NEW4 L2 Event Inter-arrival CDF")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CDF_PATH, dpi=180)
    plt.close()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False)


def write_report(summary: pd.DataFrame, session_summary: pd.DataFrame, resolution: pd.DataFrame) -> None:
    min_nonzero = float(resolution["min_nonzero_adjacent_diff_us"].min())
    resolution_limited = min_nonzero >= ONE_MS_US
    status = "resolution_limited_microsecond_probabilities_not_measurable" if resolution_limited else "microsecond_latency_probabilities_measurable"

    aggregate = summary[summary["split"].eq("all_new1_new4")].copy()
    agg_all = aggregate[aggregate["scope"].eq("all_event_flow")].iloc[0]
    agg_peak = aggregate[aggregate["scope"].eq("peak10_all_event_flow")].iloc[0]
    display = aggregate.copy()
    for col in ["same_timestamp_rate", "p_lt_1ms_observed", "p_lt_10ms_observed"]:
        display[col] = display[col].map(pct)
    for col in ["min_nonzero_dt_us", "p50_dt_us", "p90_dt_us", "p99_dt_us", "p999_dt_us"]:
        display[col] = display[col].map(fmt_us)
    display["p_lt_23p16us"] = "N/A below 1ms timestamp resolution"
    display["p_lt_83p16us"] = "N/A below 1ms timestamp resolution"

    split_display = summary[summary["split"].isin(SPLITS) & summary["scope"].eq("all_event_flow")].copy()
    for col in ["same_timestamp_rate", "p_lt_1ms_observed", "p_lt_10ms_observed"]:
        split_display[col] = split_display[col].map(pct)
    for col in ["min_nonzero_dt_us", "p50_dt_us", "p90_dt_us", "p99_dt_us", "p999_dt_us"]:
        split_display[col] = split_display[col].map(fmt_us)

    quote_rates = session_summary.groupby("split", as_index=False).agg(
        state_count=("state_count", "sum"),
        quote_eligible_state_count=("quote_eligible_state_count", "sum"),
    )
    quote_rates["quote_eligible_state_rate"] = quote_rates["quote_eligible_state_count"] / quote_rates["state_count"]
    quote_rates["quote_eligible_state_rate"] = quote_rates["quote_eligible_state_rate"].map(pct)

    lines = [
        "# AUDIT: Event Inter-arrival Latency Budget",
        "",
        f"Status: `{status}`.",
        "",
        "## Timestamp Resolution Statement",
        "",
        f"The verified minimum non-zero adjacent timestamp difference across NEW1-NEW4 TXF-only sessions is `{fmt_us(min_nonzero)}`.",
        "",
        "The source timestamp column is `timestamp_us`, derived from `info_time` parsed as `hour:minute:second:millisecond:flag`; Phase 04A.1 found the fifth segment always zero and interpreted the fourth segment as milliseconds. Night-session time-of-day values are unwrapped with a session-start anchor, then events are ordered by unwrapped timestamp with `event_index` as a tie-break. This avoids treating raw chunk-order overlaps as real negative time intervals.",
        "",
        "Because the effective timestamp granularity is 1ms, the requested probabilities `P(delta_t < 23.16us)` and `P(delta_t < 83.16us)` are **not directly measurable** from this dataset. Any sub-millisecond event ordering inside a single timestamp batch is collapsed into `delta_t = 0`, and any event that crosses a millisecond boundary may have true spacing below 1ms but observed spacing of 1ms. The report therefore presents `P(delta_t < 1ms timestamp tick)` as the measurable resolution-limited substitute, which is the same-timestamp adjacent-event share.",
        "",
        "## Data Scope",
        "",
        "- Source dataset: `n2ew_era/04b_txf_only_final_development_rerun/data/txf_only_action_universe.parquet/`.",
        "- Splits used: `train` = NEW2, `val` = NEW1, `new3` = NEW3, `new4` = NEW4.",
        "- NEW5: not read, not referenced for selection, and not used.",
        "- Event stream: one row per `(session_id, event_index)`, taken from `action_slot == 0` to remove bid/ask/no_quote action-row duplication.",
        "- Quote-eligible subset: finite top-of-book bid/ask, positive top sizes, positive spread. Since Phase 04B already builds a quote-eligible TXF action universe, this subset is effectively the same state universe; the exact rates are shown below.",
        "- Peak slice: per session, 1-second windows are ranked by event count; intervals whose later event falls in the top 10% windows are retained.",
        f"- Calculation script: `{SCRIPT_REL}`.",
        "",
        "## Aggregate NEW1-NEW4 Results",
        "",
        markdown_table(
            display,
            [
                "scope",
                "interval_count",
                "same_timestamp_rate",
                "min_nonzero_dt_us",
                "p50_dt_us",
                "p90_dt_us",
                "p99_dt_us",
                "p999_dt_us",
                "p_lt_23p16us",
                "p_lt_83p16us",
                "p_lt_1ms_observed",
                "p_lt_10ms_observed",
            ],
        ),
        "",
        "## Split-Level All-Event Results",
        "",
        markdown_table(
            split_display,
            [
                "split",
                "source_group",
                "interval_count",
                "same_timestamp_rate",
                "min_nonzero_dt_us",
                "p50_dt_us",
                "p90_dt_us",
                "p99_dt_us",
                "p999_dt_us",
                "p_lt_1ms_observed",
                "p_lt_10ms_observed",
            ],
        ),
        "",
        "## Quote-Eligible Coverage",
        "",
        markdown_table(quote_rates, ["split", "state_count", "quote_eligible_state_count", "quote_eligible_state_rate"]),
        "",
        "## Interpretation",
        "",
        "- The dataset can support millisecond-level latency-budget claims, not direct 23.16us or 83.16us opportunity-loss probabilities.",
        "- The measurable sub-1ms proxy is the same-timestamp adjacent-event rate. This is not a true microsecond probability; it is a resolution-limited batch indicator.",
        f"- In this audit the same-timestamp adjacent-event share is `{pct(agg_all['same_timestamp_rate'])}` overall and `{pct(agg_peak['same_timestamp_rate'])}` in peak 10% event-rate windows. Because same-timestamp batches dominate, this dataset cannot establish that 83.16us or 23.16us misses only a small number of true microstructure opportunities.",
        f"- Among timestamp-resolved nonzero moves, the minimum observed adjacent difference is `{fmt_us(agg_all['min_nonzero_dt_us'])}`; observed p90 is `{fmt_us(agg_all['p90_dt_us'])}` overall and `{fmt_us(agg_peak['p90_dt_us'])}` in peak windows.",
        "- Safe claim: under the available millisecond timestamp feed, sub-millisecond latency loss is resolution-limited and unquantified. Do not present the 23.16us/83.16us thresholds as directly measured opportunity-loss probabilities.",
        "- A fresh data feed with exchange or receive timestamps finer than 83us would be required to make the stronger claim `P(true delta_t < 83.16us)` directly.",
        "",
        "## Output Files",
        "",
        f"- Summary CSV: `results/{SUMMARY_CSV.name}`",
        f"- Session CSV: `results/{SESSION_CSV.name}`",
        f"- Timestamp resolution CSV: `results/{RESOLUTION_CSV.name}`",
        f"- CDF figure: `figures/{CDF_PATH.name}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    summary, session_summary, resolution, cdf_arrays = collect()
    summary.to_csv(SUMMARY_CSV, index=False)
    session_summary.to_csv(SESSION_CSV, index=False)
    resolution.to_csv(RESOLUTION_CSV, index=False)
    make_cdf_figure(cdf_arrays)
    write_report(summary, session_summary, resolution)
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {SESSION_CSV}")
    print(f"wrote {RESOLUTION_CSV}")
    print(f"wrote {CDF_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(summary[summary['split'].eq('all_new1_new4')].to_string(index=False))


if __name__ == "__main__":
    main()

