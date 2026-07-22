from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import ROBUSTNESS_DIR, build_actions_with_masks, ensure_dirs, eval_policy_row, load_policy_frame


def run() -> None:
    ensure_dirs()
    frame = load_policy_frame("val")
    actions = build_actions_with_masks(frame, split="val")
    event_thr = float(frame["event_rate_hz"].quantile(0.75))
    policies = [
        {"policy_id": "simple_spread_filter_only_s2", "policy_type": "simple_rule", "rule": "spread_filter_only", "max_spread_ticks": 2.0},
        {"policy_id": "simple_spread_filter_only_s3", "policy_type": "simple_rule", "rule": "spread_filter_only", "max_spread_ticks": 3.0},
        {
            "policy_id": "simple_spread_filter_random_cancel_s3_r10",
            "policy_type": "simple_rule",
            "rule": "spread_filter_random_cancel",
            "max_spread_ticks": 3.0,
            "random_cancel_rate": 0.10,
            "seed": 20260525,
        },
        {
            "policy_id": "simple_spread_filter_queue_imbalance_s3_z1",
            "policy_type": "simple_rule",
            "rule": "spread_filter_queue_imbalance",
            "max_spread_ticks": 3.0,
            "queue_threshold": 1.0,
        },
        {
            "policy_id": "simple_spread_filter_event_rate_s3_q75",
            "policy_type": "simple_rule",
            "rule": "spread_filter_event_rate",
            "max_spread_ticks": 3.0,
            "event_rate_threshold": event_thr,
        },
    ]
    rows = []
    for policy in policies:
        row = eval_policy_row(frame, actions, policy, split="val")
        row["selection_role"] = "non_ml_baseline"
        row["test_evaluated"] = False
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROBUSTNESS_DIR / "simple_rule_baselines.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {ROBUSTNESS_DIR / 'simple_rule_baselines.csv'}")
