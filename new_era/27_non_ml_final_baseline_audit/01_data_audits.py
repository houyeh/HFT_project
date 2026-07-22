from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from common_027 import (
    DATA_HIST,
    DATA_NEW4,
    MAKER_FEE,
    POINT_VALUE,
    PROJECT_ROOT,
    RESULTS,
    ensure_dirs,
    first_csv_header,
    load_historical_label_columns,
    raw_csv_inventory,
    write_markdown,
)
from new_era018_helpers import detect_symbol_from_raw_dir, infer_tick_size_simple
from new_era.models_playaround_15.data_loading import load_new_tw_data


def _capability_status(header: list[str]) -> list[dict[str, str]]:
    normalized = [h.lower() for h in header]

    def has_any(patterns: list[str]) -> bool:
        return any(any(re.search(p, h) for h in normalized) for p in patterns)

    return [
        {
            "capability": "timestamp/session/symbol",
            "status": "available",
            "evidence": "Raw loader parses row[0] timestamp, parent folder session_id, and row[3] symbol.",
            "queue_sim_implication": "Chronological replay and session-reset audits are possible.",
        },
        {
            "capability": "L1-L5 bid/ask price and depth",
            "status": "available",
            "evidence": "Raw loader maps row[12:31] to bid/ask price-volume levels.",
            "queue_sim_implication": "Spread and depth-aware quote eligibility can be simulated as a proxy.",
        },
        {
            "capability": "last trade price and quantity",
            "status": "available",
            "evidence": "Raw loader maps row[6] trade_price and row[7] trade_volume when volume is positive.",
            "queue_sim_implication": "Trade-hit proxy can be constructed, but not aggressor-side truth.",
        },
        {
            "capability": "order_id",
            "status": "unavailable" if not has_any(["order.?id", "ordno", "order_no"]) else "header_match_found",
            "evidence": "No reliable order_id field is consumed by the canonical loader.",
            "queue_sim_implication": "Cannot reconstruct exact queue position or order-level fills.",
        },
        {
            "capability": "aggressor_side",
            "status": "unavailable" if not has_any(["aggress", "initiator", "buyer", "seller"]) else "header_match_found",
            "evidence": "No aggressor_side field is consumed by the canonical loader.",
            "queue_sim_implication": "Fill toxicity remains a proxy, not true exchange-side fill toxicity.",
        },
        {
            "capability": "exchange queue position",
            "status": "unavailable",
            "evidence": "No order queue IDs, add/cancel messages, or order priority fields are present in the loader output.",
            "queue_sim_implication": "Queue-aware stress tests are scenario/proxy analyses, not production fill replay.",
        },
    ]


def write_raw_data_capability_report() -> None:
    hist_file, hist_header = first_csv_header(DATA_HIST)
    new4_file, new4_header = first_csv_header(DATA_NEW4)
    inv = pd.concat([raw_csv_inventory(DATA_HIST), raw_csv_inventory(DATA_NEW4)], ignore_index=True)
    inv.to_csv(RESULTS / "raw_csv_inventory.csv", index=False)
    capability = pd.DataFrame(_capability_status(hist_header or new4_header))
    capability.to_csv(RESULTS / "raw_data_capability_table.csv", index=False)

    lines = [
        "# Raw Data Capability Report",
        "",
        "This audit defines what can and cannot be simulated from the available order-book snapshots.",
        "",
        f"- Historical sample header file: `{hist_file.relative_to(PROJECT_ROOT) if hist_file else 'missing'}`.",
        f"- NEW4 sample header file: `{new4_file.relative_to(PROJECT_ROOT) if new4_file else 'missing'}`.",
        "- Canonical loader: `new_era/models_playaround_15/data_loading.py::load_new_tw_data`.",
        "",
        "## Capability Table",
        "",
    ]
    for row in capability.to_dict("records"):
        lines.append(f"- `{row['capability']}`: `{row['status']}`. {row['queue_sim_implication']}")
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "The dataset supports spread/depth/event-rate proxy controls. It does not support exact exchange queue reconstruction, true order-level fill priority, or aggressor-side-labeled toxicity. All queue/fill PnL remains a simulator-derived proxy.",
        ]
    )
    write_markdown(RESULTS / "raw_data_capability_report.md", lines)


def write_data_compatibility_audit() -> None:
    label_cols = ["dataset", "symbol", "session_id", "split", "tick_size", "spread_ticks"]
    hist = load_historical_label_columns(label_cols)
    symbol = detect_symbol_from_raw_dir(DATA_NEW4, preferred="TXFA6")
    lob_frames, _ = load_new_tw_data(DATA_NEW4, [symbol])
    new4 = lob_frames[symbol].copy()
    new4_tick = infer_tick_size_simple(new4)

    rows = [
        {
            "check": "point_value",
            "historical_setting": POINT_VALUE,
            "new4_setting": POINT_VALUE,
            "status": "match",
            "notes": "Both evaluations use common_027.POINT_VALUE imported from 026 baseline utilities.",
        },
        {
            "check": "maker_fee",
            "historical_setting": MAKER_FEE,
            "new4_setting": MAKER_FEE,
            "status": "match",
            "notes": "Both evaluations use common_027.MAKER_FEE imported from 026 baseline utilities.",
        },
        {
            "check": "tick_size",
            "historical_setting": ",".join(str(x) for x in sorted(hist["tick_size"].dropna().unique())[:10]),
            "new4_setting": new4_tick,
            "status": "match" if float(new4_tick) in set(float(x) for x in hist["tick_size"].dropna().unique()) else "review",
            "notes": "NEW4 tick size is inferred from observed valid L1 prices.",
        },
        {
            "check": "symbol",
            "historical_setting": ",".join(str(x) for x in sorted(hist["symbol"].dropna().unique())[:10]),
            "new4_setting": symbol,
            "status": "match" if symbol in set(hist["symbol"].astype(str)) else "review",
            "notes": "NEW4 detected symbol is used only for external OOS evaluation.",
        },
        {
            "check": "loader",
            "historical_setting": "load_new_tw_data / binary_labels_v2 derived from same raw format",
            "new4_setting": "load_new_tw_data",
            "status": "match",
            "notes": "NEW4 is read with the same canonical TW loader.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "data_compatibility_audit.csv", index=False)

    split_counts = hist.groupby(["split"], observed=False).agg(rows=("split", "count"), sessions=("session_id", "nunique")).reset_index()
    split_counts.to_csv(RESULTS / "historical_split_inventory.csv", index=False)

    new4_sessions = new4.groupby("session_id", observed=False).agg(rows=("session_id", "count")).reset_index()
    new4_sessions.to_csv(RESULTS / "new4_session_inventory.csv", index=False)

    lines = [
        "# Data Compatibility Audit",
        "",
        f"- NEW4 source: `{DATA_NEW4.relative_to(PROJECT_ROOT)}`.",
        f"- Detected NEW4 symbol: `{symbol}`.",
        f"- Inferred NEW4 tick size: `{new4_tick}`.",
        f"- Point value: `{POINT_VALUE}`.",
        f"- Maker fee: `{MAKER_FEE}`.",
        "",
        "## Audit Results",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['check']}`: `{row['status']}`. Historical=`{row['historical_setting']}`, NEW4=`{row['new4_setting']}`.")
    lines.extend(
        [
            "",
            "## Use In Reports",
            "",
            "NEW4 is treated as external out-of-sample evidence only. No policy threshold, side rule, stress setting, fee, multiplier, or tick-size assumption is tuned on NEW4.",
        ]
    )
    write_markdown(RESULTS / "data_compatibility_audit.md", lines)


def main() -> None:
    ensure_dirs()
    write_raw_data_capability_report()
    write_data_compatibility_audit()


if __name__ == "__main__":
    main()
    print(f"Wrote data audits to {RESULTS}")

