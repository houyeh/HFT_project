"""Unit tests for the CME Logic20 adapter."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.cme_logic20 import (
    infer_session_date_from_path,
    parse_cme_time_to_ns,
    parse_logic20_header,
    _build_symbol_frames,
)


def test_parse_cme_time_to_ns_converts_millisecond_timestamp():
    assert parse_cme_time_to_ns("04:00:00:007") == 14_400_007_000_000


def test_infer_session_date_from_path_uses_folder_prefix():
    path = "data/raw/CME/ES_MES/20260319_ES_MES/Logic20_SetProduct/sample.csv"
    assert infer_session_date_from_path(path) == date(2026, 3, 19)


def test_parse_logic20_header_extracts_products():
    header = [
        "info_time",
        "SeqNum",
        "Prod",
        "ESM6BQ1",
        "ESM6BP1",
        "ESM6BQ0",
        "ESM6BP0",
        "ESM6",
        "ESM6AP0",
        "ESM6AQ0",
        "ESM6AP1",
        "ESM6AQ1",
        "MESM6BQ1",
        "MESM6BP1",
        "MESM6BQ0",
        "MESM6BP0",
        "MESM6",
        "MESM6AP0",
        "MESM6AQ0",
        "MESM6AP1",
        "MESM6AQ1",
        "ESM6LP",
        "ESM6LQ",
        "MESM6LP",
        "MESM6LQ",
    ]
    products = parse_logic20_header(header)
    assert set(products) == {"ESM6", "MESM6"}
    assert products["ESM6"]["BP0"] == "ESM6BP0"
    assert products["MESM6"]["AQ1"] == "MESM6AQ1"


def test_build_symbol_frames_expands_to_five_levels_and_extracts_trades():
    raw = pd.DataFrame(
        {
            "info_time": ["04:00:00:007", "04:00:00:007"],
            "SeqNum": [1, 2],
            "ESM6BP0": [6682.50, 6682.50],
            "ESM6BQ0": [16, 15],
            "ESM6BP1": [6682.25, 6682.25],
            "ESM6BQ1": [8, 7],
            "ESM6AP0": [6682.75, 6682.75],
            "ESM6AQ0": [3, 4],
            "ESM6AP1": [6683.00, 6683.00],
            "ESM6AQ1": [6, 7],
            "ESM6LP": [0.0, 6682.75],
            "ESM6LQ": [0.0, 2.0],
        }
    )
    book, trades, tick_size = _build_symbol_frames(
        raw=raw,
        symbol="ESM6",
        field_map={
            "BP0": "ESM6BP0",
            "BQ0": "ESM6BQ0",
            "BP1": "ESM6BP1",
            "BQ1": "ESM6BQ1",
            "AP0": "ESM6AP0",
            "AQ0": "ESM6AQ0",
            "AP1": "ESM6AP1",
            "AQ1": "ESM6AQ1",
            "LP": "ESM6LP",
            "LQ": "ESM6LQ",
        },
        session_day=date(2026, 3, 19),
        target_depth=5,
        dedupe_quotes=False,
    )

    assert tick_size == 0.25
    assert "bid_px_5" in book.columns
    assert "ask_px_5" in book.columns
    assert len(book) == 2
    assert len(trades) == 1
    assert trades.iloc[0]["trade_price"] == 6682.75
