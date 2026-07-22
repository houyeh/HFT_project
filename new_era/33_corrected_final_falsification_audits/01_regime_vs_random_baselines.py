from __future__ import annotations

import numpy as np
import pandas as pd

from common_033 import (
    RANDOM_TRIALS,
    RESULTS,
    baseline_mask,
    ensure_dirs,
    load_all_eval_actions,
    matched_random_pnls,
    summarize_random_distribution,
)


RANDOM_SCENARIOS = [
    {
        "scenario": "random_all_valid_match_global_count",
        "candidate": "all_valid_actions",
        "group_cols": [],
        "description": "Broader random quotes from all valid bid/ask actions; match only total quote count.",
    },
    {
        "scenario": "random_all_valid_match_session",
        "candidate": "all_valid_actions",
        "group_cols": ["session_id"],
        "description": "Broader random quotes from all valid bid/ask actions; match quote count per session.",
    },
    {
        "scenario": "random_all_valid_match_session_spread_bucket",
        "candidate": "all_valid_actions",
        "group_cols": ["session_id", "spread_bucket"],
        "description": "Broader random quotes from all valid bid/ask actions; match quote count per session and spread bucket.",
    },
    {
        "scenario": "random_bid_all_match_session",
        "candidate": "bid_actions_only",
        "group_cols": ["session_id"],
        "description": "Diagnostic: random bid actions only; saturated when every bid action is selected.",
    },
    {
        "scenario": "random_all_valid_match_session_side",
        "candidate": "all_valid_actions",
        "group_cols": ["session_id", "side"],
        "description": "Diagnostic: matching side makes this equivalent to bid-only selection when all bid actions are selected.",
    },
]


def candidate_mask(actions: pd.DataFrame, candidate: str) -> np.ndarray:
    if candidate == "all_valid_actions":
        return np.ones(len(actions), dtype=bool)
    if candidate == "bid_actions_only":
        return (actions["side"].astype(str) == "bid").to_numpy(dtype=bool)
    raise ValueError(candidate)


def main() -> None:
    ensure_dirs()
    actions = load_all_eval_actions()
    rows = []
    audit_rows = []
    trial_parts = []
    for eval_group, group in actions.groupby("eval_group", sort=True):
        group = group.reset_index(drop=True)
        selected = baseline_mask(group)
        values = group["quote_ev_moderate"].astype(float).to_numpy()
        baseline_pnl = float(values[selected].sum())
        baseline_count = int(selected.sum())
        for i, spec in enumerate(RANDOM_SCENARIOS):
            pnls, audit = matched_random_pnls(
                group,
                selected,
                group_cols=spec["group_cols"],
                candidate_mask=candidate_mask(group, spec["candidate"]),
                trials=RANDOM_TRIALS,
                seed=20260733 + i,
            )
            row = summarize_random_distribution(
                eval_group=str(eval_group),
                scenario=spec["scenario"],
                model_pnl=baseline_pnl,
                model_count=baseline_count,
                pnls=pnls,
                audit=audit,
            )
            row["candidate_universe"] = spec["candidate"]
            row["match_group_cols"] = ",".join(spec["group_cols"]) if spec["group_cols"] else "global"
            row["interpretation"] = spec["description"]
            rows.append(row)
            audit.insert(0, "eval_group", eval_group)
            audit.insert(1, "scenario", spec["scenario"])
            audit_rows.append(audit)
            trial_parts.append(pd.DataFrame({"eval_group": eval_group, "scenario": spec["scenario"], "trial": np.arange(len(pnls)), "random_pnl": pnls}))
    pd.DataFrame(rows).to_csv(RESULTS / "regime_vs_random_baselines.csv", index=False)
    pd.concat(audit_rows, ignore_index=True).to_csv(RESULTS / "regime_vs_random_matching_audit.csv", index=False)
    pd.concat(trial_parts, ignore_index=True).to_csv(RESULTS / "regime_vs_random_trial_pnls.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote regime-vs-random baselines to {RESULTS}")
