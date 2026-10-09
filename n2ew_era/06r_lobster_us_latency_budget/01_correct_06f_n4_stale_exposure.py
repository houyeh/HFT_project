from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ORIGINAL_SCRIPT = ROOT / "00_lobster_latency_budget.py"

FPGA_BOARD_06S2_US = 25.733
FPGA_BASELINE_06F_US = 83.16
CPU_N4_06F_ROUND1_P99_US = 262.6
CPU_LEGACY_N1_P99_US = 636.7
CPU_N4_06S_STRESS_ROUND1_US = 646.318
CPU_N4_06S_STRESS_ROUND2_US = 6299.416

CORRECTED_STALE_GRID_US = [
    FPGA_BOARD_06S2_US,
    FPGA_BASELINE_06F_US,
    CPU_N4_06F_ROUND1_P99_US,
    CPU_LEGACY_N1_P99_US,
    CPU_N4_06S_STRESS_ROUND1_US,
    CPU_N4_06S_STRESS_ROUND2_US,
]


def load_original_module():
    spec = importlib.util.spec_from_file_location("lobster_latency_budget_06r", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import original 06R script: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p06r = load_original_module()
    p06r.STALE_US_GRID = CORRECTED_STALE_GRID_US

    stale_by_ticker: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for ticker in p06r.TICKERS:
        payload = p06r.load_ticker(ticker)
        ts_ns = payload["ts_ns"]
        top_mask = payload["top_price_change"]
        stale_ns = p06r.stale_time_to_next_top_change(ts_ns, top_mask)
        stale_by_ticker[ticker] = stale_ns
        summary_rows.extend(p06r.stale_rows(ticker, "LOBSTER_FREE_SAMPLE", stale_ns))

    all_stale = np.concatenate([stale_by_ticker[ticker] for ticker in p06r.TICKERS])
    summary_rows.extend(p06r.stale_rows("ALL", "LOBSTER_FREE_SAMPLE_5_TICKERS", all_stale))

    by_ticker_latency: dict[str, dict[float, dict[str, object]]] = {}
    for row in summary_rows:
        by_ticker_latency.setdefault(str(row["ticker"]), {})[float(row["latency_us"])] = row

    diff_rows: list[dict[str, object]] = []
    for ticker in [*p06r.TICKERS, "ALL"]:
        by_latency = by_ticker_latency[ticker]
        r25 = by_latency[FPGA_BOARD_06S2_US]
        r83 = by_latency[FPGA_BASELINE_06F_US]
        r262 = by_latency[CPU_N4_06F_ROUND1_P99_US]
        r636 = by_latency[CPU_LEGACY_N1_P99_US]
        r646 = by_latency[CPU_N4_06S_STRESS_ROUND1_US]
        r6299 = by_latency[CPU_N4_06S_STRESS_ROUND2_US]
        diff_rows.append(
            {
                "ticker": ticker,
                "source_group": r25["source_group"],
                "fpga_board_latency_us": FPGA_BOARD_06S2_US,
                "fpga_06f_baseline_latency_us": FPGA_BASELINE_06F_US,
                "cpu_n4_06f_round1_p99_us": CPU_N4_06F_ROUND1_P99_US,
                "cpu_legacy_n1_p99_us": CPU_LEGACY_N1_P99_US,
                "cpu_n4_06s_stress_round1_us": CPU_N4_06S_STRESS_ROUND1_US,
                "cpu_n4_06s_stress_round2_us": CPU_N4_06S_STRESS_ROUND2_US,
                "stale_prob_25p733us": r25["stale_prob"],
                "stale_prob_83p16us": r83["stale_prob"],
                "stale_prob_262p6us_cpu_n4_06f_round1": r262["stale_prob"],
                "stale_prob_636p7us_legacy_n1": r636["stale_prob"],
                "stale_prob_646p318us_06s_stress_round1": r646["stale_prob"],
                "stale_prob_6299p416us_06s_stress_round2": r6299["stale_prob"],
                "stale_prob_83p16_minus_25p733": float(r83["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_cpu_n4_06f_round1_minus_25p733": float(r262["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_06s_stress_round1_minus_25p733": float(r646["stale_prob"]) - float(r25["stale_prob"]),
                "stale_prob_06s_stress_round2_minus_25p733": float(r6299["stale_prob"]) - float(r25["stale_prob"]),
                "valid_future_top_change_count": r25["valid_future_top_change_count"],
            }
        )

    p06r.write_csv(RESULTS / "stale_exposure_summary_06f_corrected.csv", summary_rows)
    p06r.write_csv(RESULTS / "stale_exposure_cpu_minus_fpga_06f_corrected.csv", diff_rows)

    all_row = next(row for row in diff_rows if row["ticker"] == "ALL")
    print(json.dumps(all_row, indent=2))


if __name__ == "__main__":
    main()
