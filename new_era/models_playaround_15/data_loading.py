from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

SESSION_NS_SPACING = 36 * 60 * 60 * 1_000_000_000
_LOAD_CACHE: dict[tuple[str, tuple[str, ...]], tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = {}


def parse_new_tw_timestamp(raw_str: str) -> int:
    parts = raw_str.split(":")
    # Common formats observed in the raw data:
    # - HH:MM:SS:ms
    # - HH:MM:SS:ms:sub_ms (often 0; treat as microseconds)
    if len(parts) == 4:
        hour, minute, second, millis = parts
        micros = "0"
    elif len(parts) == 5:
        hour, minute, second, millis, micros = parts
    else:
        raise ValueError(f"Unexpected timestamp format: {raw_str!r}")

    return (
        int(hour) * 3_600_000_000_000
        + int(minute) * 60_000_000_000
        + int(second) * 1_000_000_000
        + int(millis) * 1_000_000
        + int(micros) * 1_000
    )


def load_new_tw_data(data_dir: Path, target_symbols: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cache_key = (str(data_dir.resolve()), tuple(sorted(target_symbols)))
    cached = _LOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lob_records: dict[str, list[dict]] = defaultdict(list)
    trade_records: dict[str, list[dict]] = defaultdict(list)

    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda path: str(path.relative_to(data_dir)))
    session_order_map: dict[str, int] = {}

    for csv_path in csv_paths:
        session_id = str(csv_path.parent.relative_to(data_dir)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem

        session_order = session_order_map.setdefault(session_id, len(session_order_map))
        session_offset = session_order * SESSION_NS_SPACING

        with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)

            for row in reader:
                if len(row) < 34:
                    continue

                symbol = row[3].strip()
                if symbol not in target_symbols:
                    continue

                try:
                    timestamp = session_offset + parse_new_tw_timestamp(row[0].strip())
                except ValueError:
                    continue

                try:
                    last_trade_qty = int(float(row[7]))
                    last_trade_price = float(row[6])
                except ValueError:
                    last_trade_qty = 0
                    last_trade_price = 0.0

                if last_trade_qty > 0:
                    trade_records[symbol].append(
                        {
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "trade_price": last_trade_price,
                            "trade_volume": last_trade_qty,
                        }
                    )

                try:
                    if float(row[21]) > 0 and float(row[22]) > 0:
                        lob_records[symbol].append(
                            {
                                "timestamp": timestamp,
                                "session_id": session_id,
                                "bid_px_1": float(row[21]),
                                "bid_vol_1": int(float(row[20])),
                                "bid_px_2": float(row[19]),
                                "bid_vol_2": int(float(row[18])),
                                "bid_px_3": float(row[17]),
                                "bid_vol_3": int(float(row[16])),
                                "bid_px_4": float(row[15]),
                                "bid_vol_4": int(float(row[14])),
                                "bid_px_5": float(row[13]),
                                "bid_vol_5": int(float(row[12])),
                                "ask_px_1": float(row[22]),
                                "ask_vol_1": int(float(row[23])),
                                "ask_px_2": float(row[24]),
                                "ask_vol_2": int(float(row[25])),
                                "ask_px_3": float(row[26]),
                                "ask_vol_3": int(float(row[27])),
                                "ask_px_4": float(row[28]),
                                "ask_vol_4": int(float(row[29])),
                                "ask_px_5": float(row[30]),
                                "ask_vol_5": int(float(row[31])),
                            }
                        )
                except ValueError:
                    continue

    lob_frames = {
        symbol: pd.DataFrame(lob_records[symbol]).sort_values(["session_id", "timestamp"]).reset_index(drop=True)
        for symbol in target_symbols
        if lob_records[symbol]
    }
    trade_frames = {
        symbol: pd.DataFrame(trade_records[symbol]).sort_values(["session_id", "timestamp"]).reset_index(drop=True)
        for symbol in target_symbols
        if trade_records[symbol]
    }

    _LOAD_CACHE[cache_key] = (lob_frames, trade_frames)
    return lob_frames, trade_frames
