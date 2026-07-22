from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import (
    EVALUATION_DIR,
    SEARCH_DIR,
    V18_DIR,
    StressConfig,
    build_actions_with_masks,
    ensure_dirs,
    eval_policy_row,
    json_sanitize,
    load_policy_frame,
    simulate_policy_fast,
)


STRESSES = {
    "base": StressConfig(neutral_fill_haircut=1.0, toxic_multiplier=1.0, cancel_delay_ticks=0),
    "mild": StressConfig(neutral_fill_haircut=0.75, toxic_multiplier=1.25, cancel_delay_ticks=0),
    "moderate": StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=0),
}


def _random_v3_lookup() -> dict[str, float]:
    dist_path = V18_DIR / "random_v3_stress_distribution.csv"
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing {dist_path}. Run 018 final reporting first.")
    dist = pd.read_csv(dist_path)
    out = {}
    for name, stress in STRESSES.items():
        row = dist[
            (dist["neutral_fill_haircut"] == stress.neutral_fill_haircut)
            & (dist["toxic_multiplier"] == stress.toxic_multiplier)
            & (dist["cancel_delay_ticks"] == stress.cancel_delay_ticks)
        ].iloc[0]
        out[name] = float(row["mean"])
    return out


def run(force: bool = False) -> None:
    ensure_dirs()
    out_path = EVALUATION_DIR / "frozen_policy_test_results.csv"
    manifest_path = EVALUATION_DIR / "frozen_test_evaluation_manifest.json"
    if out_path.exists() and not force:
        print(f"Frozen test evaluation already exists: {out_path}")
        return

    selected = json.loads((SEARCH_DIR / "selected_policy.json").read_text(encoding="utf-8"))
    policies = [("selected_main", selected["selected_main_policy"])]
    for i, backup in enumerate(selected.get("backup_policies", []), start=1):
        policies.append((f"backup_{i}", backup))

    frame = load_policy_frame("test")
    actions = build_actions_with_masks(frame, split="test")
    random_v3 = _random_v3_lookup()
    rows = []
    no_cancel_by_stress = {}
    for stress_name, stress in STRESSES.items():
        no_cancel = simulate_policy_fast(frame, actions, np.zeros(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool), stress)
        no_cancel_by_stress[stress_name] = no_cancel.net_pnl
        rows.append(
            {
                "policy_role": "baseline",
                "policy_id": "no_cancel",
                "stress_name": stress_name,
                "neutral_fill_haircut": stress.neutral_fill_haircut,
                "toxic_multiplier": stress.toxic_multiplier,
                "cancel_delay_ticks": stress.cancel_delay_ticks,
                "net_pnl": no_cancel.net_pnl,
                "trade_count": no_cancel.trade_count,
                "max_drawdown": no_cancel.max_drawdown,
                "cancel_count": 0,
                "gap_vs_no_cancel": 0.0,
                "gap_vs_random_v3_mean": no_cancel.net_pnl - random_v3[stress_name],
                "test_used_for_selection": False,
            }
        )
        rows.append(
            {
                "policy_role": "baseline",
                "policy_id": "random_v3_mean",
                "stress_name": stress_name,
                "neutral_fill_haircut": stress.neutral_fill_haircut,
                "toxic_multiplier": stress.toxic_multiplier,
                "cancel_delay_ticks": stress.cancel_delay_ticks,
                "net_pnl": random_v3[stress_name],
                "trade_count": np.nan,
                "max_drawdown": np.nan,
                "cancel_count": np.nan,
                "gap_vs_no_cancel": random_v3[stress_name] - no_cancel.net_pnl,
                "gap_vs_random_v3_mean": 0.0,
                "test_used_for_selection": False,
            }
        )

    for role, entry in policies:
        policy = entry["policy"]
        for stress_name, stress in STRESSES.items():
            row = eval_policy_row(frame, actions, policy, split="test", stress=stress)
            row.update(
                {
                    "policy_role": role,
                    "stress_name": stress_name,
                    "neutral_fill_haircut": stress.neutral_fill_haircut,
                    "toxic_multiplier": stress.toxic_multiplier,
                    "cancel_delay_ticks": stress.cancel_delay_ticks,
                    "gap_vs_no_cancel": float(row["net_pnl"] - no_cancel_by_stress[stress_name]),
                    "gap_vs_random_v3_mean": float(row["net_pnl"] - random_v3[stress_name]),
                    "test_used_for_selection": False,
                    "policy_json": json.dumps(json_sanitize(policy), sort_keys=True),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    manifest = {
        "selection_file": str(SEARCH_DIR / "selected_policy.json"),
        "selection_split": selected["selection_split"],
        "test_used_for_selection": False,
        "evaluated_once": True,
        "stress_definitions": {k: v.__dict__ for k, v in STRESSES.items()},
        "notes": "Frozen selected policy and backups applied to test without changing thresholds or parameters.",
    }
    manifest_path.write_text(json.dumps(json_sanitize(manifest), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen selected policy on test exactly once.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force=args.force)
    print(f"Wrote {EVALUATION_DIR / 'frozen_policy_test_results.csv'}")
