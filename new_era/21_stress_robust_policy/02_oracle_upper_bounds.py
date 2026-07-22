from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stress_robust_utils import (
    ORACLE_DIR,
    STRESSES,
    apply_policy,
    baseline_policy_masks,
    decomp_to_row,
    ensure_dirs,
    json_sanitize,
    load_frames_actions,
    oracle_masks,
    simulate_decomp,
    threshold_for_target_rate,
)


RATES = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
SPREADS = [1.0, 1.5, 2.0, 3.0, 99.0]


def _eval_policy(frame: pd.DataFrame, actions: pd.DataFrame, split: str, policy_name: str, policy: dict) -> list[dict]:
    pull_bid, pull_ask = apply_policy(frame, policy)
    rows = []
    for stress_name, stress in STRESSES.items():
        rows.append(decomp_to_row(policy_name, split, stress_name, simulate_decomp(frame, actions, pull_bid, pull_ask, stress)))
    return rows


def run() -> None:
    ensure_dirs()
    frames = {}
    actions = {}
    for split in ["val", "test"]:
        frames[split], actions[split] = load_frames_actions(split)

    oracle_rows = []
    for split in ["val", "test"]:
        for oracle_type in ["perfect_toxic_cancel", "perfect_quote_pnl_filter"]:
            for stress_name, stress in STRESSES.items():
                pb, pa = oracle_masks(frames[split], actions[split], oracle_type, stress)
                result = simulate_decomp(frames[split], actions[split], pb, pa, stress)
                oracle_rows.append(decomp_to_row(oracle_type, split, stress_name, result))
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(ORACLE_DIR / "oracle_perfect_upper_bounds.csv", index=False)

    val = frames["val"]
    score_rows = []
    for rate in RATES:
        policy = {
            "policy_type": "spread_toxic",
            "policy_id": f"best_val_score_r{rate:g}",
            "max_spread_ticks": 99.0,
            "bid_threshold": threshold_for_target_rate(val["bid_prob_toxic"].to_numpy(dtype=float), rate),
            "ask_threshold": threshold_for_target_rate(val["ask_prob_toxic"].to_numpy(dtype=float), rate),
            "target_cancel_rate": rate,
        }
        score_rows.extend(_eval_policy(frames["val"], actions["val"], "val", policy["policy_id"], policy))
    score_df = pd.DataFrame(score_rows)
    best_score_name = score_df[score_df["stress_name"] == "moderate"].sort_values("net_pnl", ascending=False).iloc[0]["policy"]
    rate = float(best_score_name.split("r")[-1])
    best_score_policy = {
        "policy_type": "spread_toxic",
        "policy_id": f"best_val_score_r{rate:g}",
        "max_spread_ticks": 99.0,
        "bid_threshold": threshold_for_target_rate(val["bid_prob_toxic"].to_numpy(dtype=float), rate),
        "ask_threshold": threshold_for_target_rate(val["ask_prob_toxic"].to_numpy(dtype=float), rate),
        "target_cancel_rate": rate,
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    frozen_score_rows = []
    for split in ["val", "test"]:
        frozen_score_rows.extend(_eval_policy(frames[split], actions[split], split, best_score_policy["policy_id"], best_score_policy))
    pd.DataFrame(frozen_score_rows).to_csv(ORACLE_DIR / "oracle_best_validation_score_threshold.csv", index=False)

    spread_rows = []
    policies = []
    for max_spread in SPREADS:
        for rate in RATES:
            policy = {
                "policy_type": "spread_toxic",
                "policy_id": f"best_val_spread_score_s{max_spread:g}_r{rate:g}",
                "max_spread_ticks": max_spread,
                "bid_threshold": threshold_for_target_rate(val["bid_prob_toxic"].to_numpy(dtype=float), rate),
                "ask_threshold": threshold_for_target_rate(val["ask_prob_toxic"].to_numpy(dtype=float), rate),
                "target_cancel_rate": rate,
                "selection_split": "validation",
                "test_used_for_selection": False,
            }
            policies.append(policy)
            spread_rows.extend(_eval_policy(frames["val"], actions["val"], "val", policy["policy_id"], policy))
    spread_df = pd.DataFrame(spread_rows)
    best_spread_name = spread_df[spread_df["stress_name"] == "moderate"].sort_values("net_pnl", ascending=False).iloc[0]["policy"]
    best_spread_policy = next(p for p in policies if p["policy_id"] == best_spread_name)
    frozen_spread_rows = []
    for split in ["val", "test"]:
        frozen_spread_rows.extend(_eval_policy(frames[split], actions[split], split, best_spread_policy["policy_id"], best_spread_policy))
    pd.DataFrame(frozen_spread_rows).to_csv(ORACLE_DIR / "oracle_best_validation_spread_score_grid.csv", index=False)

    moderate_perfect = oracle_df[(oracle_df["split"] == "test") & (oracle_df["stress_name"] == "moderate")].copy()
    feasibility = bool((moderate_perfect["net_pnl"] > 0).any())
    report = [
        "# Oracle Upper Bound Report",
        "",
        "Oracle rows use future information and are not tradable policies.",
        "",
        f"- Perfect-oracle moderate test feasibility: `{'positive feasible' if feasibility else 'negative infeasible'}`.",
    ]
    for _, row in moderate_perfect.iterrows():
        report.append(f"- {row['policy']} moderate test PnL: `{row['net_pnl']:.0f}`.")
    score_test = pd.read_csv(ORACLE_DIR / "oracle_best_validation_score_threshold.csv")
    spread_test = pd.read_csv(ORACLE_DIR / "oracle_best_validation_spread_score_grid.csv")
    report.extend(
        [
            "",
            "## Validation-Selected Diagnostic Policies",
            "",
            f"- Best validation score threshold: `{best_score_policy['policy_id']}`.",
            f"- Frozen test moderate PnL: `{score_test[(score_test['split']=='test') & (score_test['stress_name']=='moderate')].iloc[0]['net_pnl']:.0f}`.",
            f"- Best validation spread+score grid: `{best_spread_policy['policy_id']}`.",
            f"- Frozen test moderate PnL: `{spread_test[(spread_test['split']=='test') & (spread_test['stress_name']=='moderate')].iloc[0]['net_pnl']:.0f}`.",
            "",
            "If perfect oracles are positive but validation-selected score policies are not, the bottleneck is signal/policy quality rather than mathematical infeasibility under the locked stress assumptions.",
        ]
    )
    (ORACLE_DIR / "ORACLE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (ORACLE_DIR / "oracle_selected_validation_policies.json").write_text(
        json.dumps(json_sanitize({"best_score_policy": best_score_policy, "best_spread_score_policy": best_spread_policy}), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
    print(f"Wrote {ORACLE_DIR / 'ORACLE_REPORT.md'}")
