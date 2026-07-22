from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_phase02f import (  # noqa: E402
    FillNorms,
    fill_probabilities,
    latency_aware_fill_probabilities,
    latency_events_to_skip,
    latency_events_to_skip_by_timestamp,
)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2"],
            "event_index": [0, 1, 2, 0],
            "timestamp": [0.0, 1_000.0, 2_000.0, 0.0],
            "top_depth": [10.0, 20.0, 30.0, 100.0],
            "bid_sz_1": [5.0, 6.0, 7.0, 50.0],
            "ask_sz_1": [8.0, 9.0, 10.0, 80.0],
            "event_rate_past_50": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )


def assert_close_array(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"{name} mismatch: actual={actual} expected={expected}")


def test_existing_fill_probabilities_regression() -> None:
    frame = toy_frame()
    decision = np.array([1, 2, 0, 1], dtype=np.int8)
    norms = FillNorms(top_depth_median=10.0, bid_sz_1_median=5.0, ask_sz_1_median=8.0)
    depth = fill_probabilities(frame, decision, "depth_haircut", norms)
    queue = fill_probabilities(frame, decision, "queue_side_haircut", norms)
    stress = fill_probabilities(frame, decision, "conservative_l2_stress", norms)
    assert_close_array(depth, np.array([0.5, 1.0 / 3.0, 0.0, 1.0 / 11.0]), "depth_haircut")
    assert_close_array(queue, np.array([0.5, 1.0 / (1.0 + 9.0 / 8.0), 0.0, 1.0 / 11.0]), "queue_side_haircut")
    expected_stress = np.array(
        [
            0.5 * 0.5 * 0.5,
            0.5 * (1.0 / 3.0) * (1.0 / (1.0 + 9.0 / 8.0)),
            0.0,
            max(0.01, 0.5 * (1.0 / 11.0) * (1.0 / 11.0)),
        ]
    )
    assert_close_array(stress, expected_stress, "conservative_l2_stress")


def test_latency_events_to_skip_units() -> None:
    frame = toy_frame()
    skips = latency_events_to_skip(frame, np.array([0.0, 0.49, 0.50, 1.49]))
    assert_close_array(skips, np.array([0, 0, 1, 1]), "events_to_skip_rounding")


def test_latency_timestamp_lookup_irregular_searchsorted() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s1", "s2", "s2"],
            "event_index": [0, 1, 2, 3, 0, 1],
            "timestamp": [0.0, 1_000.0, 1_350.0, 10_000.0, 50.0, 500.0],
            "top_depth": [1, 1, 1, 1, 1, 1],
            "bid_sz_1": [1, 1, 1, 1, 1, 1],
            "ask_sz_1": [1, 1, 1, 1, 1, 1],
            "event_rate_past_50": [100.0] * 6,
        }
    )
    latency_us = np.array([0.2, 0.3, 8.0, 0.1, 0.3, 0.1])
    # Targets in ns: [200, 1300, 9350, 10100, 350, 600].
    # Last observed event at those target times: [0, 1, 2, 3, 4, 5].
    source_idx = latency_events_to_skip_by_timestamp(frame, latency_us)
    assert_close_array(source_idx, np.array([0, 1, 2, 3, 4, 5]), "timestamp_searchsorted_source_idx")


def test_timestamp_method_resolves_shift_when_event_rate_skip_is_zero() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1"],
            "event_index": [0, 1, 2],
            "timestamp": [0.0, 50.0, 1_150.0],
            "top_depth": [10.0, 20.0, 30.0],
            "bid_sz_1": [5.0, 6.0, 7.0],
            "ask_sz_1": [8.0, 9.0, 10.0],
            "event_rate_past_50": [100.0, 100.0, 100.0],
        }
    )
    latency_us = np.array([0.1, 0.1, 0.1])
    old_skip = latency_events_to_skip(frame, latency_us)
    new_source_idx = latency_events_to_skip_by_timestamp(frame, latency_us)
    assert_close_array(old_skip, np.array([0, 0, 0]), "event_rate_skip_zero")
    assert_close_array(new_source_idx, np.array([1, 1, 2]), "timestamp_detects_dense_local_shift")


def test_session_end_latency_uses_last_observed_book_not_overflow() -> None:
    frame = toy_frame()
    decision = np.array([1, 2, 1, 1], dtype=np.int8)
    norms = FillNorms(top_depth_median=10.0, bid_sz_1_median=5.0, ask_sz_1_median=8.0)
    probs, meta = latency_aware_fill_probabilities(frame, decision, "depth_haircut", norms, 3.0)
    assert_close_array(probs, np.array([0.25, 0.25, 0.25, 1.0 / 11.0]), "session_end_last_observed_book")
    assert meta["latency_no_fill_count"] == 0
    assert math.isclose(meta["latency_no_fill_rate"], 0.0)


def test_latency_shift_uses_last_observed_update_when_update_occurs() -> None:
    frame = toy_frame()
    decision = np.array([1, 2, 0, 0], dtype=np.int8)
    norms = FillNorms(top_depth_median=10.0, bid_sz_1_median=5.0, ask_sz_1_median=8.0)
    probs, meta = latency_aware_fill_probabilities(frame, decision, "depth_haircut", norms, 1.0)
    assert_close_array(probs, np.array([1.0 / 3.0, 0.25, 0.0, 0.0]), "latency_uses_last_observed_update")
    assert meta["latency_no_fill_count"] == 0


def test_sparse_period_no_new_event_keeps_decision_book_state() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1"],
            "event_index": [0, 1],
            "timestamp": [0.0, 10_000_000_000.0],
            "top_depth": [10.0, 999.0],
            "bid_sz_1": [5.0, 999.0],
            "ask_sz_1": [8.0, 999.0],
            "event_rate_past_50": [100.0, 100.0],
        }
    )
    decision = np.array([1, 0], dtype=np.int8)
    norms = FillNorms(top_depth_median=10.0, bid_sz_1_median=5.0, ask_sz_1_median=8.0)
    zero_latency_probs, _ = latency_aware_fill_probabilities(frame, decision, "depth_haircut", norms, 0.0)
    delayed_probs, meta = latency_aware_fill_probabilities(frame, decision, "depth_haircut", norms, 100.0)
    assert_close_array(delayed_probs, zero_latency_probs, "sparse_period_book_unchanged")
    assert meta["latency_no_fill_count"] == 0
    assert meta["max_events_to_skip_selected"] == 0


def main() -> None:
    tests = [
        test_existing_fill_probabilities_regression,
        test_latency_events_to_skip_units,
        test_latency_timestamp_lookup_irregular_searchsorted,
        test_timestamp_method_resolves_shift_when_event_rate_skip_is_zero,
        test_session_end_latency_uses_last_observed_book_not_overflow,
        test_latency_shift_uses_last_observed_update_when_update_occurs,
        test_sparse_period_no_new_event_keeps_decision_book_state,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} latency-aware fill proxy tests")


if __name__ == "__main__":
    main()
