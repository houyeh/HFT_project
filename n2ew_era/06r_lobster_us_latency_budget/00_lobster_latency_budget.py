from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "extracted"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TXF_06P_SUMMARY = PROJECT_ROOT / "n2ew_era" / "06p_event_interarrival_latency_budget" / "results" / "event_interarrival_latency_budget_summary.csv"

TICKERS = ["AAPL", "MSFT", "INTC", "AMZN", "GOOG"]
LEVEL = 10
DATE = "2012-06-21"
EXPECTED_START_MS = 34_200_000
EXPECTED_END_MS = 57_600_000
NS_PER_SECOND = 1_000_000_000
FPGA_BASELINE_06F_US = 83.16
FPGA_BOARD_06S2_US = 25.733
CPU_LEGACY_N1_P99_US = 636.7
CPU_N4_P99_ROUND1_US = 646.318
CPU_N4_P99_ROUND2_US = 6299.416
CPU_N4_P99_HEADLINE_US = CPU_N4_P99_ROUND2_US
US_THRESHOLDS = [
    23.16,
    FPGA_BOARD_06S2_US,
    51.0,
    74.54,
    FPGA_BASELINE_06F_US,
    CPU_LEGACY_N1_P99_US,
    CPU_N4_P99_ROUND1_US,
    1000.0,
    CPU_N4_P99_HEADLINE_US,
]
STALE_US_GRID = [
    FPGA_BOARD_06S2_US,
    51.0,
    74.54,
    FPGA_BASELINE_06F_US,
    CPU_LEGACY_N1_P99_US,
    CPU_N4_P99_ROUND1_US,
    1000.0,
    CPU_N4_P99_HEADLINE_US,
    10000.0,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_time_to_ns(values: pd.Series) -> np.ndarray:
    out = np.empty(len(values), dtype=np.int64)
    for i, text in enumerate(values.astype(str)):
        s = text.strip()
        if "." in s:
            sec_s, frac_s = s.split(".", 1)
            frac_ns = int((frac_s + "000000000")[:9])
        else:
            sec_s = s
            frac_ns = 0
        out[i] = int(sec_s) * NS_PER_SECOND + frac_ns
    return out


def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def first_line_column_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return len(f.readline().strip().split(","))


def ticker_paths(ticker: str) -> tuple[Path, Path, Path]:
    folder = EXTRACTED / f"LOBSTER_SampleFile_{ticker}_{DATE}_{LEVEL}"
    message = folder / f"{ticker}_{DATE}_{EXPECTED_START_MS}_{EXPECTED_END_MS}_message_{LEVEL}.csv"
    orderbook = folder / f"{ticker}_{DATE}_{EXPECTED_START_MS}_{EXPECTED_END_MS}_orderbook_{LEVEL}.csv"
    readme = folder / "LOBSTER_SampleFiles_ReadMe.txt"
    return message, orderbook, readme


def load_ticker(ticker: str) -> dict[str, object]:
    message_path, orderbook_path, readme_path = ticker_paths(ticker)
    if not message_path.exists() or not orderbook_path.exists():
        raise FileNotFoundError(f"Missing extracted message/orderbook files for {ticker}")

    message = pd.read_csv(message_path, header=None, usecols=[0], dtype=str, engine="c")
    ts_ns = parse_time_to_ns(message.iloc[:, 0])
    orderbook = pd.read_csv(orderbook_path, header=None, usecols=[0, 2], dtype=np.int64, engine="c")
    ask1 = orderbook.iloc[:, 0].to_numpy(dtype=np.int64)
    bid1 = orderbook.iloc[:, 1].to_numpy(dtype=np.int64)

    message_rows = len(ts_ns)
    orderbook_rows = len(ask1)
    if message_rows != orderbook_rows:
        raise RuntimeError(f"{ticker}: message rows {message_rows} != orderbook rows {orderbook_rows}")
    if first_line_column_count(message_path) != 6:
        raise RuntimeError(f"{ticker}: message file is not Nx6")
    if first_line_column_count(orderbook_path) != 4 * LEVEL:
        raise RuntimeError(f"{ticker}: orderbook is not Nx(4*{LEVEL})")
    if np.any(np.diff(ts_ns) < 0):
        raise RuntimeError(f"{ticker}: message timestamps are not monotonic")

    top_price_change = np.zeros(message_rows, dtype=bool)
    top_price_change[1:] = (ask1[1:] != ask1[:-1]) | (bid1[1:] != bid1[:-1])
    dt_all_ns = np.diff(ts_ns)
    if not np.any((dt_all_ns > 0) & (dt_all_ns < 1000)):
        raise RuntimeError(f"{ticker}: no non-zero sub-microsecond timestamp interval found")
    top_times_ns = ts_ns[top_price_change]
    dt_top_ns = np.diff(top_times_ns)

    return {
        "ticker": ticker,
        "message_path": message_path,
        "orderbook_path": orderbook_path,
        "readme_path": readme_path,
        "ts_ns": ts_ns,
        "ask1": ask1,
        "bid1": bid1,
        "top_price_change": top_price_change,
        "dt_all_ns": dt_all_ns,
        "dt_top_ns": dt_top_ns,
    }


def scope_peak_seconds(event_times_ns: np.ndarray) -> set[int]:
    if len(event_times_ns) == 0:
        return set()
    seconds = (event_times_ns // NS_PER_SECOND).astype(np.int64)
    unique, counts = np.unique(seconds, return_counts=True)
    threshold = np.quantile(counts, 0.9, method="lower")
    return set(int(s) for s, c in zip(unique, counts) if c >= threshold)


def intervals_in_peak(ts_ns: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
    event_times = ts_ns[event_mask]
    if len(event_times) < 2:
        return np.array([], dtype=np.int64)
    peak_seconds = scope_peak_seconds(event_times)
    later_seconds = (event_times[1:] // NS_PER_SECOND).astype(np.int64)
    return np.diff(event_times)[np.array([int(s) in peak_seconds for s in later_seconds], dtype=bool)]


def interval_stats(
    ticker: str,
    source_group: str,
    event_definition: str,
    scope: str,
    dt_ns: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "source_group": source_group,
        "event_definition": event_definition,
        "scope": scope,
        "interval_count": int(len(dt_ns)),
        "same_timestamp_count": int(np.count_nonzero(dt_ns == 0)) if len(dt_ns) else 0,
        "same_timestamp_rate": float(np.mean(dt_ns == 0)) if len(dt_ns) else float("nan"),
        "min_nonzero_dt_us": float(np.min(dt_ns[dt_ns > 0]) / 1000.0) if np.any(dt_ns > 0) else float("nan"),
    }
    for q, name in [(0.5, "p50_dt_us"), (0.9, "p90_dt_us"), (0.99, "p99_dt_us"), (0.999, "p999_dt_us")]:
        row[name] = float(np.quantile(dt_ns, q) / 1000.0) if len(dt_ns) else float("nan")
    for threshold_us in US_THRESHOLDS:
        key = f"p_lt_{str(threshold_us).replace('.', 'p')}us"
        row[key] = float(np.mean(dt_ns < threshold_us * 1000.0)) if len(dt_ns) else float("nan")
    return row


def stale_time_to_next_top_change(ts_ns: np.ndarray, top_price_change: np.ndarray) -> np.ndarray:
    n = len(ts_ns)
    top_idx = np.flatnonzero(top_price_change)
    insertion = np.searchsorted(top_idx, np.arange(n), side="right")
    out = np.full(n, -1, dtype=np.int64)
    valid = insertion < len(top_idx)
    next_idx = top_idx[insertion[valid]]
    out[valid] = ts_ns[next_idx] - ts_ns[valid]
    return out


def stale_rows(ticker: str, source_group: str, time_to_next_ns: np.ndarray) -> list[dict[str, object]]:
    valid = time_to_next_ns >= 0
    valid_times = time_to_next_ns[valid]
    rows = []
    for latency_us in STALE_US_GRID:
        stale = valid_times <= latency_us * 1000.0
        rows.append(
            {
                "ticker": ticker,
                "source_group": source_group,
                "latency_us": latency_us,
                "event_count": int(len(time_to_next_ns)),
                "valid_future_top_change_count": int(len(valid_times)),
                "no_future_top_change_count": int(np.count_nonzero(~valid)),
                "stale_count": int(np.count_nonzero(stale)),
                "stale_prob": float(np.mean(stale)) if len(valid_times) else float("nan"),
            }
        )
    return rows


def peak_window_summary(ticker: str, event_definition: str, event_times_ns: np.ndarray) -> dict[str, object]:
    seconds = (event_times_ns // NS_PER_SECOND).astype(np.int64)
    unique, counts = np.unique(seconds, return_counts=True)
    threshold = np.quantile(counts, 0.9, method="lower") if len(counts) else float("nan")
    peak_seconds = {int(s) for s, c in zip(unique, counts) if c >= threshold}
    open_seconds = set(range(34200, 34500))
    close_seconds = set(range(57300, 57601))
    return {
        "ticker": ticker,
        "event_definition": event_definition,
        "active_seconds": int(len(unique)),
        "peak_second_count": int(len(peak_seconds)),
        "peak_threshold_events_per_second": float(threshold),
        "peak_includes_0930_0935": bool(peak_seconds & open_seconds),
        "peak_0930_0935_second_count": int(len(peak_seconds & open_seconds)),
        "peak_includes_1555_1600": bool(peak_seconds & close_seconds),
        "peak_1555_1600_second_count": int(len(peak_seconds & close_seconds)),
        "max_events_per_second": int(np.max(counts)) if len(counts) else 0,
    }


def manifest_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    zip_rows = []
    extracted_rows = []
    for zpath in sorted(RAW.glob("LOBSTER_SampleFile_*_10.zip")):
        ticker = zpath.name.split("_")[2]
        with zipfile.ZipFile(zpath) as zf:
            members = zf.infolist()
            zip_rows.append(
                {
                    "ticker": ticker,
                    "level": LEVEL,
                    "zip_path": str(zpath.relative_to(ROOT)),
                    "zip_size_bytes": zpath.stat().st_size,
                    "zip_sha256": sha256_file(zpath),
                    "zip_member_count": len(members),
                    "source_url": f"https://php.lobsterdata.com/info/sample/{zpath.name}",
                }
            )
            for info in members:
                extracted = EXTRACTED / zpath.stem / info.filename
                extracted_rows.append(
                    {
                        "ticker": ticker,
                        "level": LEVEL,
                        "member": info.filename,
                        "kind": "message"
                        if "_message_" in info.filename
                        else "orderbook"
                        if "_orderbook_" in info.filename
                        else "readme",
                        "zip_path": str(zpath.relative_to(ROOT)),
                        "compressed_size_bytes": info.compress_size,
                        "uncompressed_size_bytes": info.file_size,
                        "extracted_path": str(extracted.relative_to(ROOT)),
                        "extracted_size_bytes": extracted.stat().st_size if extracted.exists() else "",
                        "extracted_sha256": sha256_file(extracted) if extracted.exists() else "",
                    }
                )
    return zip_rows, extracted_rows


def format_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{100.0 * x:.6f}%"


def txf_comparison_rows(lobster_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    txf = pd.read_csv(TXF_06P_SUMMARY) if TXF_06P_SUMMARY.exists() else pd.DataFrame()
    txf_all = txf[(txf["split"] == "all_new1_new4") & (txf["scope"] == "all_event_flow")].iloc[0].to_dict()
    txf_peak = txf[(txf["split"] == "all_new1_new4") & (txf["scope"] == "peak10_all_event_flow")].iloc[0].to_dict()
    lookup = {(r["event_definition"], r["scope"]): r for r in lobster_summary if r["ticker"] == "ALL"}
    for scope, txf_row in [("all_event_flow", txf_all), ("peak10_all_event_flow", txf_peak)]:
        lrow = lookup[("any_l2_update", scope)]
        rows.append(
            {
                "metric": f"{scope}: same_timestamp_rate",
                "lobster_any_l2": lrow["same_timestamp_rate"],
                "txf_06p": txf_row["same_timestamp_rate"],
                "txf_note": "TXF same-timestamp rate is a 1ms-resolution batch indicator.",
            }
        )
        for key in ["min_nonzero_dt_us", "p50_dt_us", "p90_dt_us", "p99_dt_us", "p999_dt_us"]:
            rows.append(
                {
                    "metric": f"{scope}: {key}",
                    "lobster_any_l2": lrow[key],
                    "txf_06p": txf_row[key],
                    "txf_note": "",
                }
            )
        for latency_us in US_THRESHOLDS:
            lobster_key = f"p_lt_{str(latency_us).replace('.', 'p')}us"
            if latency_us < 1000:
                txf_value = "N/A below 1ms timestamp resolution"
            elif latency_us == 1000:
                txf_value = txf_row.get("p_lt_1ms_observed", "")
            else:
                txf_value = txf_row.get("p_lt_10ms_observed", "")
            rows.append(
                {
                    "metric": f"{scope}: P(dt < {latency_us}us)",
                    "lobster_any_l2": lrow[lobster_key],
                    "txf_06p": txf_value,
                    "txf_note": "N/A expected for sub-1ms TXF thresholds.",
                }
            )
    return rows


def plot_interarrival_cdf(summary_inputs: dict[tuple[str, str, str], np.ndarray]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for (ticker, event_definition, scope), dt_ns in summary_inputs.items():
        if ticker != "ALL" or len(dt_ns) == 0:
            continue
        x = np.sort(dt_ns.astype(np.float64) / 1000.0)
        y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
        label = f"{event_definition}, {scope}"
        plt.plot(x, y, label=label)
    for threshold in US_THRESHOLDS:
        plt.axvline(threshold, color="0.7", linewidth=0.8, linestyle="--")
    plt.xscale("symlog", linthresh=1.0)
    plt.xlim(left=0)
    plt.xlabel("Adjacent event interval (us)")
    plt.ylabel("CDF")
    plt.title("LOBSTER free sample inter-arrival CDF")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "lobster_interarrival_cdf.png", dpi=160)
    plt.close()


def plot_stale_cdf(stale_times: dict[str, np.ndarray]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for ticker, arr in stale_times.items():
        valid = arr[arr >= 0].astype(np.float64) / 1000.0
        x = np.sort(valid)
        y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
        plt.plot(x, y, label=ticker)
    for threshold in STALE_US_GRID:
        plt.axvline(threshold, color="0.75", linewidth=0.8, linestyle="--")
    plt.xscale("symlog", linthresh=1.0)
    plt.xlim(left=0)
    plt.xlabel("Time to next top-of-book price change after event t (us)")
    plt.ylabel("CDF / stale probability at latency L")
    plt.title("LOBSTER stale-exposure CDF")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "lobster_stale_exposure_cdf.png", dpi=160)
    plt.close()


def write_manifest(zip_rows: list[dict[str, object]], extracted_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> None:
    lines = [
        "# LOBSTER Free Sample Data Manifest",
        "",
        "Status: official LOBSTER free sample level-10 zips downloaded and parsed.",
        "",
        "## Source",
        "",
        "- Official samples page: <https://php.lobsterdata.com/info/DataSamples.php>.",
        "- Official structure page: <https://php.lobsterdata.com/info/DataStructure.php>.",
        "- Date: `2012-06-21`.",
        "- Requested level: `10` for AAPL, MSFT, INTC, AMZN, GOOG.",
        "- Time range encoded in filenames: `34200000_57600000` ms after midnight, i.e. 09:30:00-16:00:00.",
        "",
        "## Zip Files",
        "",
        "| ticker | zip | size bytes | sha256 | source url |",
        "|---|---|---:|---|---|",
    ]
    for row in zip_rows:
        lines.append(
            f"| {row['ticker']} | `{row['zip_path']}` | {row['zip_size_bytes']} | `{row['zip_sha256']}` | <{row['source_url']}> |"
        )
    lines += [
        "",
        "## Message/Orderbook Files",
        "",
        "| ticker | kind | file | size bytes | rows | sha256 |",
        "|---|---|---|---:|---:|---|",
    ]
    validation_by_path = {row["file_path"]: row for row in validation_rows if row.get("file_path")}
    for row in extracted_rows:
        if row["kind"] not in {"message", "orderbook"}:
            continue
        validation = validation_by_path.get(row["extracted_path"], {})
        lines.append(
            f"| {row['ticker']} | {row['kind']} | `{row['extracted_path']}` | {row['extracted_size_bytes']} | {validation.get('row_count', '')} | `{row['extracted_sha256']}` |"
        )
    lines += [
        "",
        "## Validation",
        "",
        "- Message files were checked as `N x 6`.",
        "- Orderbook files were checked as `N x 40` for level 10.",
        "- Message and orderbook row counts were checked equal for each ticker.",
        "- Timestamps were parsed from decimal strings into integer nanoseconds, not float64.",
        "- Timestamp monotonicity and sub-microsecond non-zero resolution were checked per ticker.",
        "- No project-internal market data and no NEW1-NEW5 split files were read or used; the TXF comparison uses only the 06P published audit CSV summary.",
        "",
        "Generated by `00_lobster_latency_budget.py`.",
    ]
    (ROOT / "data_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_us(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}s"
    if value >= 1_000:
        return f"{value / 1_000:.3f}ms"
    return f"{value:.3f}us"


def write_report(
    validation_rows: list[dict[str, object]],
    interval_rows: list[dict[str, object]],
    stale_summary: list[dict[str, object]],
    stale_diff: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    peak_rows: list[dict[str, object]],
) -> None:
    all_any = next(r for r in interval_rows if r["ticker"] == "ALL" and r["event_definition"] == "any_l2_update" and r["scope"] == "all_event_flow")
    all_any_peak = next(r for r in interval_rows if r["ticker"] == "ALL" and r["event_definition"] == "any_l2_update" and r["scope"] == "peak10_all_event_flow")
    all_top = next(r for r in interval_rows if r["ticker"] == "ALL" and r["event_definition"] == "top_of_book_price_change" and r["scope"] == "all_event_flow")
    all_top_peak = next(r for r in interval_rows if r["ticker"] == "ALL" and r["event_definition"] == "top_of_book_price_change" and r["scope"] == "peak10_all_event_flow")
    stale_all_25 = next(r for r in stale_summary if r["ticker"] == "ALL" and float(r["latency_us"]) == FPGA_BOARD_06S2_US)
    stale_all_83 = next(r for r in stale_summary if r["ticker"] == "ALL" and float(r["latency_us"]) == FPGA_BASELINE_06F_US)
    stale_all_636 = next(r for r in stale_summary if r["ticker"] == "ALL" and float(r["latency_us"]) == CPU_LEGACY_N1_P99_US)
    stale_all_cpu_n4_r1 = next(r for r in stale_summary if r["ticker"] == "ALL" and float(r["latency_us"]) == CPU_N4_P99_ROUND1_US)
    stale_all_cpu_n4_r2 = next(r for r in stale_summary if r["ticker"] == "ALL" and float(r["latency_us"]) == CPU_N4_P99_HEADLINE_US)
    stale_all_diff = next(r for r in stale_diff if r["ticker"] == "ALL")

    def table_interval(rows: list[dict[str, object]]) -> list[str]:
        lines = [
            "| ticker | event definition | scope | intervals | same-ts | min nonzero | p50 | p90 | p99 | p999 | P(<25.733us) | P(<83.16us) | P(<646.318us) | P(<6.299ms) | P(<1ms) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['ticker']} | {r['event_definition']} | {r['scope']} | {r['interval_count']} | {format_pct(float(r['same_timestamp_rate']))} | {fmt_us(float(r['min_nonzero_dt_us']))} | {fmt_us(float(r['p50_dt_us']))} | {fmt_us(float(r['p90_dt_us']))} | {fmt_us(float(r['p99_dt_us']))} | {fmt_us(float(r['p999_dt_us']))} | {format_pct(float(r['p_lt_25p733us']))} | {format_pct(float(r['p_lt_83p16us']))} | {format_pct(float(r['p_lt_646p318us']))} | {format_pct(float(r['p_lt_6299p416us']))} | {format_pct(float(r['p_lt_1000p0us']))} |"
            )
        return lines

    def table_stale(rows: list[dict[str, object]]) -> list[str]:
        lines = [
            "| ticker | stale@25.733us | stale@83.16us | delta 83.16-25.733 | stale@636.7us legacy N=1 | stale@646.318us N=4 r1 | stale@6299.416us N=4 r2 | delta N4 r2-25.733 | valid events |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        by_ticker: dict[str, dict[float, dict[str, object]]] = {}
        for r in rows:
            by_ticker.setdefault(str(r["ticker"]), {})[float(r["latency_us"])] = r
        diff_by_ticker = {str(r["ticker"]): r for r in stale_diff}
        for ticker in TICKERS + ["ALL"]:
            d = by_ticker[ticker]
            diff = diff_by_ticker[ticker]
            lines.append(
                f"| {ticker} | {format_pct(float(d[FPGA_BOARD_06S2_US]['stale_prob']))} | {format_pct(float(d[FPGA_BASELINE_06F_US]['stale_prob']))} | {format_pct(float(diff['stale_prob_83p16_minus_25p733']))} | {format_pct(float(d[CPU_LEGACY_N1_P99_US]['stale_prob']))} | {format_pct(float(d[CPU_N4_P99_ROUND1_US]['stale_prob']))} | {format_pct(float(d[CPU_N4_P99_HEADLINE_US]['stale_prob']))} | {format_pct(float(diff['stale_prob_cpu_n4_round2_minus_25p733']))} | {d[FPGA_BOARD_06S2_US]['valid_future_top_change_count']} |"
            )
        return lines

    def table_validation(rows: list[dict[str, object]]) -> list[str]:
        lines = [
            "| ticker | message rows | orderbook rows | columns ok | monotonic | min nonzero dt | same-ts | top-price-change count |",
            "|---|---:|---:|---|---|---:|---:|---:|",
        ]
        for r in rows:
            if r["kind"] != "ticker":
                continue
            lines.append(
                f"| {r['ticker']} | {r['message_rows']} | {r['orderbook_rows']} | {r['columns_ok']} | {r['timestamp_monotonic']} | {fmt_us(float(r['min_nonzero_dt_us']))} | {format_pct(float(r['same_timestamp_rate']))} | {r['top_price_change_count']} |"
            )
        return lines

    lines = [
        "# AUDIT: LOBSTER US Latency Budget Pilot",
        "",
        "Status: `resolved_microsecond_interarrival_and_stale_exposure_measured`.",
        "",
        "## Timestamp Resolution Statement",
        "",
        "LOBSTER message timestamps were parsed as decimal strings representing seconds after midnight and converted to integer nanoseconds by splitting the string at the decimal point. No float64 timestamp conversion was used for interval or stale-exposure calculations.",
        "",
        f"Across the five official level-10 samples, the verified minimum non-zero adjacent event interval is `{fmt_us(float(all_any['min_nonzero_dt_us']))}` for all L2 updates and `{fmt_us(float(all_top['min_nonzero_dt_us']))}` for top-of-book price-change events. Therefore `P(delta_t < 25.733us)` and `P(delta_t < 83.16us)` are directly measurable in this pilot, unlike the 1ms TXF feed audited in 06P.",
        "",
        "## Claim Boundary",
        "",
        "- This is a five-ticker, single-day 2012 NASDAQ free-sample pilot. It is not TXF and should not be treated as a Taiwan futures market estimate.",
        "- Stale exposure is a probability that the visible top-of-book changes inside a latency window after an L2 event. It is not execution impact, not fill probability, and not PnL.",
        "- The old `83.16us` FPGA baseline is preserved only as a historical 06F comparison. The current demo-aligned FPGA latency is `25.733us` from the routed 06L/06S2 unroll16 90MHz evidence.",
        "- CPU comparison uses the 06S N=4 concurrent LibTorch p99 rows (`646.318us` and `6299.416us`) rather than the old N=1 `636.7us` p99. Both N=4 rounds are listed to avoid round-picking.",
        "- 單日、五檔、2012 年 NASDAQ 樣本;市場、資產、年代皆與 TXF 不同, 本結果為方法論驗證與量級參考,不得推論 TXF 結論。",
        "- stale-exposure 是「決策抵達前 book 已變動」的機率,不是 PnL 影響; 兩者之間還隔著成交與逆選擇機制,本實驗不宣稱該段。",
        "",
        "No model inference, fill simulation, execution simulation, or PnL calculation was performed.",
        "",
        "## Data Scope",
        "",
        "- Source: LOBSTER official free samples page, `2012-06-21`.",
        "- Tickers: AAPL, MSFT, INTC, AMZN, GOOG.",
        "- Level: 10 for all five files.",
        "- Time range: 09:30:00-16:00:00, encoded as `34200000_57600000` in filenames.",
        "- Message schema: `N x 6`.",
        "- Orderbook schema: `N x 40`, ordered ask price/size then bid price/size by level.",
        "- Peak windows: per ticker and event definition, 1-second windows in the top 10% by event count. This includes auction-adjacent 09:30-09:35 and 15:55-16:00 seconds where they fall into the top decile; exact counts are in `results/peak_window_summary.csv`.",
        "",
        "## Data Validation",
        "",
        *table_validation(validation_rows),
        "",
        "## Aggregate Inter-arrival Results",
        "",
        *table_interval([all_any, all_any_peak, all_top, all_top_peak]),
        "",
        "## Per-Ticker Inter-arrival Results",
        "",
        *table_interval(
            [
                r
                for r in interval_rows
                if r["ticker"] != "ALL"
                and r["scope"] == "all_event_flow"
                and r["event_definition"] in {"any_l2_update", "top_of_book_price_change"}
            ]
        ),
        "",
        "## Stale-Exposure Results",
        "",
        "For each L2 event `t`, stale exposure is measured as whether the next later top-of-book price change occurs within latency `L`. Same-timestamp later top-of-book changes count as stale with 0us delay.",
        "",
        *table_stale(stale_summary),
        "",
        f"Aggregate stale exposure is `{format_pct(float(stale_all_25['stale_prob']))}` at the current demo-aligned FPGA latency (`25.733us`) versus `{format_pct(float(stale_all_83['stale_prob']))}` at the old 06F baseline (`83.16us`). The historical 06F shift therefore contributed `{format_pct(float(stale_all_diff['stale_prob_83p16_minus_25p733']))}` additional stale exposure relative to the current FPGA latency.",
        "",
        f"For CPU comparison, the 06S N=4 concurrent LibTorch p99 rows map to stale exposure `{format_pct(float(stale_all_cpu_n4_r1['stale_prob']))}` at `646.318us` and `{format_pct(float(stale_all_cpu_n4_r2['stale_prob']))}` at `6299.416us`. Using the conservative N=4 headline p99 (`6299.416us`), the CPU-minus-FPGA stale-exposure gap is `{format_pct(float(stale_all_diff['stale_prob_cpu_n4_round2_minus_25p733']))}`. The old N=1 `636.7us` row is preserved as legacy context at `{format_pct(float(stale_all_636['stale_prob']))}`, but it is no longer the headline CPU comparator.",
        "",
        "This is a probability-of-book-change exposure delta only. It is not a PnL delta.",
        "",
        "## 06P TXF Comparison",
        "",
        "| metric | LOBSTER any L2 | TXF 06P | note |",
        "|---|---:|---:|---|",
    ]
    for r in comparison_rows:
        lobster_value = r["lobster_any_l2"]
        if isinstance(lobster_value, (float, int)) and "P(" in str(r["metric"]) or "rate" in str(r["metric"]):
            lobster_text = format_pct(float(lobster_value))
        else:
            try:
                lobster_text = fmt_us(float(lobster_value))
            except Exception:
                lobster_text = str(lobster_value)
        txf_value = r["txf_06p"]
        try:
            txf_float = float(txf_value)
            txf_text = format_pct(txf_float) if "P(" in str(r["metric"]) or "rate" in str(r["metric"]) else fmt_us(txf_float)
        except Exception:
            txf_text = str(txf_value)
        lines.append(f"| {r['metric']} | {lobster_text} | {txf_text} | {r['txf_note']} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- LOBSTER confirms that a nanosecond-stamped public sample can directly measure sub-100us event intervals; the TXF 1ms feed cannot.",
        "- In the five-stock single-day sample, all-L2-update intervals are often far below both 25.733us and 83.16us, especially in peak windows.",
        "- Top-of-book price-change intervals are less frequent than all L2 updates, but stale exposure still increases as latency rises from 25.733us to the N=4 CPU p99 windows.",
        "- These numbers are market/sample-specific and should be treated as method validation plus scale reference only.",
        "",
        "## Output Files",
        "",
        "- Data manifest: `data_manifest.md`",
        "- File manifest: `results/lobster_file_manifest.csv`",
        "- Validation summary: `results/lobster_validation_summary.csv`",
        "- Inter-arrival summary: `results/event_interarrival_latency_budget_summary.csv`",
        "- Stale exposure summary: `results/stale_exposure_summary.csv`",
        "- CPU-vs-FPGA stale delta table: `results/stale_exposure_cpu_minus_fpga.csv`",
        "- Latency comparison table: `results/stale_exposure_latency_comparison.csv`",
        "- 06P comparison: `results/txf_06p_comparison.csv`",
        "- CDF figures: `figures/lobster_interarrival_cdf.png`, `figures/lobster_stale_exposure_cdf.png`",
        "",
        "## Sources",
        "",
        "- LOBSTER official samples page: <https://php.lobsterdata.com/info/DataSamples.php>",
        "- LOBSTER output structure page: <https://php.lobsterdata.com/info/DataStructure.php>",
        "- TXF 06P audit: `n2ew_era/06p_event_interarrival_latency_budget/AUDIT_event_interarrival_latency_budget.md`",
        "- Current FPGA and CPU N=4 p99 latency source: `n2ew_era/06s_unroll16_verification_closure/results/final_latency_comparison.csv`",
        "",
        "## Honest Conclusion",
        "",
        "06R resolves the measurement question for this pilot sample: public LOBSTER data can measure microsecond event intervals and stale-exposure probabilities directly. It does not resolve TXF stale exposure, estimate execution/PnL impact, or repair the 06Q fill-proxy limitation.",
    ]
    lines = [
        line
        for line in lines
        if not (
            line.startswith("- stale-exposure")
            or any(marker in line for marker in ["瑼", "銝", "蝯", "胯", "撽"])
            or (line.startswith("- ") and "NASDAQ" in line and "TXF" in line and "free-sample" not in line)
        )
    ]
    (ROOT / "AUDIT_lobster_us_latency_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    zip_rows, extracted_rows = manifest_rows()
    write_csv(RESULTS / "lobster_zip_manifest.csv", zip_rows)
    write_csv(RESULTS / "lobster_extracted_manifest.csv", extracted_rows)

    loaded = []
    validation_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    stale_summary: list[dict[str, object]] = []
    stale_by_ticker: dict[str, np.ndarray] = {}
    interval_inputs: dict[tuple[str, str, str], np.ndarray] = {}
    peak_rows: list[dict[str, object]] = []

    for ticker in TICKERS:
        data = load_ticker(ticker)
        loaded.append(data)
        ts_ns = data["ts_ns"]
        top_mask = data["top_price_change"]
        dt_all = data["dt_all_ns"]
        dt_top = data["dt_top_ns"]
        message_path = data["message_path"]
        orderbook_path = data["orderbook_path"]

        validation_rows += [
            {
                "ticker": ticker,
                "kind": "message",
                "file_path": str(message_path.relative_to(ROOT)),
                "row_count": int(len(ts_ns)),
                "column_count": 6,
                "sha256": sha256_file(message_path),
            },
            {
                "ticker": ticker,
                "kind": "orderbook",
                "file_path": str(orderbook_path.relative_to(ROOT)),
                "row_count": int(len(ts_ns)),
                "column_count": 4 * LEVEL,
                "sha256": sha256_file(orderbook_path),
            },
            {
                "ticker": ticker,
                "kind": "ticker",
                "message_rows": int(len(ts_ns)),
                "orderbook_rows": int(len(ts_ns)),
                "columns_ok": True,
                "timestamp_monotonic": True,
                "min_nonzero_dt_us": float(np.min(dt_all[dt_all > 0]) / 1000.0),
                "same_timestamp_rate": float(np.mean(dt_all == 0)),
                "sub_microsecond_nonzero_present": bool(np.any((dt_all > 0) & (dt_all < 1000))),
                "top_price_change_count": int(np.count_nonzero(top_mask)),
                "first_timestamp_ns": int(ts_ns[0]),
                "last_timestamp_ns": int(ts_ns[-1]),
            },
        ]

        any_mask = np.ones(len(ts_ns), dtype=bool)
        top_event_mask = top_mask
        peak_rows.append(peak_window_summary(ticker, "any_l2_update", ts_ns))
        peak_rows.append(peak_window_summary(ticker, "top_of_book_price_change", ts_ns[top_event_mask]))

        ticker_interval_inputs = {
            ("any_l2_update", "all_event_flow"): dt_all,
            ("any_l2_update", "peak10_all_event_flow"): intervals_in_peak(ts_ns, any_mask),
            ("top_of_book_price_change", "all_event_flow"): dt_top,
            ("top_of_book_price_change", "peak10_all_event_flow"): intervals_in_peak(ts_ns, top_event_mask),
        }
        for (event_definition, scope), dt in ticker_interval_inputs.items():
            interval_rows.append(interval_stats(ticker, "LOBSTER_FREE_SAMPLE", event_definition, scope, dt))
            interval_inputs[(ticker, event_definition, scope)] = dt

        stale_ns = stale_time_to_next_top_change(ts_ns, top_mask)
        stale_by_ticker[ticker] = stale_ns
        stale_summary.extend(stale_rows(ticker, "LOBSTER_FREE_SAMPLE", stale_ns))

    for event_definition in ["any_l2_update", "top_of_book_price_change"]:
        for scope in ["all_event_flow", "peak10_all_event_flow"]:
            combined = np.concatenate([interval_inputs[(ticker, event_definition, scope)] for ticker in TICKERS])
            interval_inputs[("ALL", event_definition, scope)] = combined
            interval_rows.append(interval_stats("ALL", "LOBSTER_FREE_SAMPLE_5_TICKERS", event_definition, scope, combined))

    all_stale = np.concatenate([stale_by_ticker[ticker] for ticker in TICKERS])
    stale_by_ticker["ALL"] = all_stale
    stale_summary.extend(stale_rows("ALL", "LOBSTER_FREE_SAMPLE_5_TICKERS", all_stale))

    stale_diff = []
    by_ticker_latency = {}
    for row in stale_summary:
        by_ticker_latency.setdefault(row["ticker"], {})[float(row["latency_us"])] = row
    for ticker in TICKERS + ["ALL"]:
        r25 = by_ticker_latency[ticker][FPGA_BOARD_06S2_US]
        r83 = by_ticker_latency[ticker][FPGA_BASELINE_06F_US]
        r636 = by_ticker_latency[ticker][CPU_LEGACY_N1_P99_US]
        r646 = by_ticker_latency[ticker][CPU_N4_P99_ROUND1_US]
        r6299 = by_ticker_latency[ticker][CPU_N4_P99_HEADLINE_US]
        stale_diff.append(
            {
                "ticker": ticker,
                "source_group": r25["source_group"],
                "fpga_board_latency_us": FPGA_BOARD_06S2_US,
                "fpga_06f_baseline_latency_us": FPGA_BASELINE_06F_US,
                "cpu_legacy_n1_p99_us": CPU_LEGACY_N1_P99_US,
                "cpu_n4_p99_round1_us": CPU_N4_P99_ROUND1_US,
                "cpu_n4_p99_round2_headline_us": CPU_N4_P99_HEADLINE_US,
                "stale_prob_25p733us": r25["stale_prob"],
                "stale_prob_83p16us": r83["stale_prob"],
                "stale_prob_636p7us": r636["stale_prob"],
                "stale_prob_646p318us_cpu_n4_round1": r646["stale_prob"],
                "stale_prob_6299p416us_cpu_n4_round2": r6299["stale_prob"],
                "stale_prob_83p16_minus_25p733": float(r83["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_cpu_n4_round1_minus_25p733": float(r646["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_cpu_n4_round2_minus_25p733": float(r6299["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_636p7_minus_25p733": float(r636["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_636p7_minus_83p16": float(r636["stale_prob"]) - float(r83["stale_prob"]),
                "valid_future_top_change_count": r25["valid_future_top_change_count"],
            }
        )

    file_manifest = []
    for row in extracted_rows:
        v = next((x for x in validation_rows if x.get("file_path") == row["extracted_path"]), None)
        merged = dict(row)
        if v:
            merged.update({"row_count": v["row_count"], "column_count": v["column_count"]})
        file_manifest.append(merged)

    write_csv(RESULTS / "lobster_file_manifest.csv", file_manifest)
    write_csv(RESULTS / "lobster_validation_summary.csv", validation_rows)
    write_csv(RESULTS / "event_interarrival_latency_budget_summary.csv", interval_rows)
    write_csv(RESULTS / "stale_exposure_summary.csv", stale_summary)
    write_csv(RESULTS / "stale_exposure_cpu_minus_fpga.csv", stale_diff)
    write_csv(RESULTS / "stale_exposure_latency_comparison.csv", stale_diff)
    write_csv(RESULTS / "peak_window_summary.csv", peak_rows)
    comparison_rows = txf_comparison_rows(interval_rows)
    write_csv(RESULTS / "txf_06p_comparison.csv", comparison_rows)

    plot_interarrival_cdf(interval_inputs)
    plot_stale_cdf(stale_by_ticker)
    write_manifest(zip_rows, extracted_rows, validation_rows)
    write_report(validation_rows, interval_rows, stale_summary, stale_diff, comparison_rows, peak_rows)

    print(json.dumps({
        "validation_rows": [r for r in validation_rows if r["kind"] == "ticker"],
        "aggregate_intervals": [r for r in interval_rows if r["ticker"] == "ALL"],
        "stale_diff": stale_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
