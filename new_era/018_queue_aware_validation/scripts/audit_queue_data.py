from __future__ import annotations

import csv
from pathlib import Path

from common import AUDIT_DIR, DATA_NEW, DATA_NEW3, SYMBOL_NEW, SYMBOL_NEW3, ensure_dirs, get_logger


def _scan_dataset(data_dir: Path, sample_files: int = 8, sample_rows_per_file: int = 4000) -> dict:
    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir)))
    if not csv_paths:
        return {
            "lob_depth_available": False,
            "best_level_vol_available": False,
            "trade_tape_available": False,
            "aggressor_side_available": False,
            "row_count_scanned": 0,
            "sample_header": [],
        }

    lob_depth_ok = False
    best_vol_ok = False
    trade_ok = False
    aggressor_side = False
    total_scanned = 0
    sample_header: list[str] = []

    for csv_path in csv_paths[:sample_files]:
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not sample_header:
                sample_header = [h.strip() for h in header]
            lower_header = {h.strip().lower() for h in header}
            if {"aggressor_side", "side", "trade_side", "is_buyer_maker", "buy_sell_flag"} & lower_header:
                aggressor_side = True

            for r_idx, row in enumerate(reader):
                if r_idx >= sample_rows_per_file:
                    break
                total_scanned += 1
                if len(row) < 34:
                    continue
                try:
                    # same indices used by load_new_tw_data
                    bid1 = float(row[21])
                    ask1 = float(row[22])
                    bidv1 = float(row[20])
                    askv1 = float(row[23])
                    bid2 = float(row[19]); ask2 = float(row[24]); bidv2 = float(row[18]); askv2 = float(row[25])
                    bid3 = float(row[17]); ask3 = float(row[26]); bidv3 = float(row[16]); askv3 = float(row[27])
                    bid4 = float(row[15]); ask4 = float(row[28]); bidv4 = float(row[14]); askv4 = float(row[29])
                    bid5 = float(row[13]); ask5 = float(row[30]); bidv5 = float(row[12]); askv5 = float(row[31])
                    _ = [bid1, ask1, bidv1, askv1, bid2, ask2, bidv2, askv2, bid3, ask3, bidv3, askv3, bid4, ask4, bidv4, askv4, bid5, ask5, bidv5, askv5]
                    lob_depth_ok = True
                    best_vol_ok = True
                except ValueError:
                    pass
                try:
                    trade_price = float(row[6])
                    trade_volume = float(row[7])
                    if trade_price > 0 and trade_volume > 0:
                        trade_ok = True
                except ValueError:
                    pass

    return {
        "lob_depth_available": lob_depth_ok,
        "best_level_vol_available": best_vol_ok,
        "trade_tape_available": trade_ok,
        "aggressor_side_available": aggressor_side,
        "row_count_scanned": total_scanned,
        "sample_header": sample_header,
    }


def main() -> None:
    ensure_dirs()
    logger = get_logger("audit_queue_data")

    new_scan = _scan_dataset(DATA_NEW)
    new3_scan = _scan_dataset(DATA_NEW3)

    lob_depth_yes = bool(new_scan["lob_depth_available"] and new3_scan["lob_depth_available"])
    best_level_vol_yes = bool(new_scan["best_level_vol_available"] and new3_scan["best_level_vol_available"])
    trade_tape_yes = bool(new_scan["trade_tape_available"] and new3_scan["trade_tape_available"])
    aggressor_available = bool(new_scan["aggressor_side_available"] or new3_scan["aggressor_side_available"])

    lines: list[str] = []
    lines.append("# Queue Data Audit")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- NEW source: `{DATA_NEW}` (requested symbol `{SYMBOL_NEW}`)")
    lines.append(f"- NEW3 source: `{DATA_NEW3}` (requested symbol `{SYMBOL_NEW3}`)")
    lines.append(f"- NEW sampled rows scanned: {new_scan['row_count_scanned']:,}")
    lines.append(f"- NEW3 sampled rows scanned: {new3_scan['row_count_scanned']:,}")
    lines.append("")
    lines.append("## Field Availability")
    lines.append(f"- LOB depth available? {'YES' if lob_depth_yes else 'NO'}")
    lines.append(f"- Best level volume available? {'YES' if best_level_vol_yes else 'NO'}")
    lines.append(f"- Trade tape available? {'YES' if trade_tape_yes else 'NO'}")
    lines.append(f"- Aggressor side available? {'YES' if aggressor_available else 'NO (likely)'}")
    lines.append("")
    lines.append("## Queue Reconstruction Feasibility")
    lines.append("- Can true exchange-level queue be reconstructed? NO (expected).")
    lines.append(
        "- True queue replay is not possible. This task uses conservative queue-aware sensitivity proxies."
    )
    lines.append("")
    lines.append("## Proxy Assumptions Used in 018")
    lines.append("- Fill probability haircut scenarios with fixed random seeds.")
    lines.append("- Profitable-fill haircut stronger than toxic-fill haircut (adverse selection stress).")
    lines.append("- Best-level queue-ahead approximation using `bid_vol_1` / `ask_vol_1`.")
    lines.append(
        "- Trade-tape volume in future event windows is used as a conservative proxy for queue depletion when aggressor side is unavailable."
    )
    lines.append("- Cancel-delay sensitivity combined with conservative fill assumptions.")
    lines.append("")
    lines.append("## Caveat")
    lines.append(
        "- Without aggressor_side and full per-order queue events, outputs are sensitivity tests, not exchange-faithful queue replay."
    )
    lines.append("")
    lines.append("## Raw Header Sample (NEW)")
    lines.append(f"- {new_scan['sample_header'][:20]}")
    lines.append("## Raw Header Sample (NEW3)")
    lines.append(f"- {new3_scan['sample_header'][:20]}")

    out_path = AUDIT_DIR / "queue_data_audit.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote queue data audit to %s", out_path)


if __name__ == "__main__":
    main()
