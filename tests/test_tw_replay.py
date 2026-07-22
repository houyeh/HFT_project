"""Tests for Taiwan replay parsing and dataset construction."""

import pandas as pd

from src.data.tw_replay import infer_tick_size_from_lob, load_tw_replay_as_lob, parse_tw_timestamp_to_ns
from src.training.datasets import WindowedLOBSequenceDataset


def test_parse_tw_timestamp_to_ns_orders_within_day():
    t1 = parse_tw_timestamp_to_ns("[08:45:00:0116.0000]")
    t2 = parse_tw_timestamp_to_ns("[08:45:00:0117.0000]")
    assert t2 > t1


def test_load_tw_replay_as_lob_normalizes_book(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "\n".join(
            [
                "TXFJ5,[08:45:00:0116.0000],,,26842.00(1),26845.00(1),0.00(0),26848.00(5),26849.00(6),--,26853.00(7),26854.00(12),0.00(0),0.00(0),0.00(0)",
                "TXFJ5,[08:45:00:0117.0000],26853.00,1,,,,,,--,,,,,",
                "TXFJ5,[08:45:00:0118.0000],,,0.00(0),26847.00(1),26848.00(2),0.00(0),26850.00(2),--,26853.00(4),0.00(0),26855.00(2),0.00(0),0.00(0)",
            ]
        ),
        encoding="utf-8",
    )

    data = load_tw_replay_as_lob(csv_path, symbols=["TXFJ5"], depth=5, quote_stride=1, dedupe_quotes=False)
    frame = data["TXFJ5"]

    assert len(frame) == 2
    assert frame.loc[0, "bid_px_1"] == 26849.00
    assert frame.loc[0, "ask_px_1"] == 26853.00
    assert frame.loc[1, "bid_px_1"] == 26850.00


def test_infer_tick_size_and_window_dataset():
    frame = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "mid_price": [100.0, 101.0, 100.0, 101.0, 102.0, 103.0],
            "normalized_spread": [1, 1, 1, 1, 1, 1],
            "micro_return": [0, 1, -1, 1, 1, 1],
            "obi": [0.1, 0.2, 0.1, 0.0, -0.1, 0.3],
            "weighted_mid_deviation": [0.0, 0.1, -0.1, 0.2, 0.1, 0.0],
            "deep_obi": [0.05, 0.04, 0.03, 0.02, 0.01, 0.00],
        }
    )

    lob = pd.DataFrame(
        {
            "bid_px_1": [99.0, 100.0],
            "ask_px_1": [100.0, 101.0],
            "bid_px_2": [98.0, 99.0],
            "ask_px_2": [101.0, 102.0],
            "bid_px_3": [0.0, 0.0],
            "ask_px_3": [0.0, 0.0],
            "bid_px_4": [0.0, 0.0],
            "ask_px_4": [0.0, 0.0],
            "bid_px_5": [0.0, 0.0],
            "ask_px_5": [0.0, 0.0],
        }
    )
    assert infer_tick_size_from_lob(lob) == 1.0

    dataset = WindowedLOBSequenceDataset(
        feature_df=frame,
        symbol="TXFJ5",
        tick_size=1.0,
        window_size=3,
        horizon=1,
        threshold_ticks=0.5,
        sample_stride=1,
    )
    sample = dataset[0]

    assert sample["x"].shape == (3, 5)
    assert int(sample["y"]) in {0, 1, 2}
    assert sample["symbol"] == "TXFJ5"
    assert float(sample["sample_weight"]) >= 1.0


def test_window_dataset_can_upweight_nonstationary_samples():
    frame = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "mid_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "normalized_spread": [1, 1, 1, 1, 1, 1],
            "micro_return": [0, 1, 1, 1, 1, 1],
            "obi": [0.1, 0.2, 0.3, 0.1, 0.1, 0.2],
            "weighted_mid_deviation": [0.0, 0.1, 0.2, 0.1, 0.1, 0.0],
            "deep_obi": [0.05, 0.04, 0.04, 0.03, 0.02, 0.01],
        }
    )

    dataset = WindowedLOBSequenceDataset(
        feature_df=frame,
        symbol="TXFJ5",
        tick_size=1.0,
        window_size=3,
        horizon=1,
        threshold_ticks=0.5,
        sample_stride=1,
        symbol_weight=0.5,
        nonstationary_weight=2.0,
        return_weight_alpha=0.1,
    )
    sample = dataset[0]

    assert float(sample["sample_weight"]) > 1.0
