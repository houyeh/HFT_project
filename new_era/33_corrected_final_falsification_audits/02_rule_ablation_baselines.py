from __future__ import annotations

import pandas as pd

from common_033 import RESULTS, ensure_dirs, eval_selection, load_all_eval_actions, rule_mask


RULES = [
    "no_quote",
    "all_actions",
    "bid_all",
    "ask_all",
    "spread_ge_0",
    "spread_ge_1",
    "spread_ge_2",
    "spread_ge_3",
    "bid_spread_ge_0",
    "bid_spread_ge_1",
    "bid_spread_ge_2",
    "bid_spread_ge_3",
    "ask_spread_ge_0",
    "ask_spread_ge_1",
    "ask_spread_ge_2",
    "ask_spread_ge_3",
]


def main() -> None:
    ensure_dirs()
    actions = load_all_eval_actions()
    rows = []
    for eval_group, group in actions.groupby("eval_group", sort=True):
        group = group.reset_index(drop=True)
        for rule in RULES:
            row = eval_selection(group, rule_mask(group, rule), rule)
            row["eval_group"] = eval_group
            row["source"] = ",".join(sorted(group["source"].astype(str).unique()))
            if rule in {"bid_all", "bid_spread_ge_1"}:
                row["is_core_controller_equivalent"] = True
                row["interpretation_note"] = "In the current quote-eligible action universe, all bid spreads are >= 1; this ablation is equivalent to the core rule."
            else:
                row["is_core_controller_equivalent"] = False
                row["interpretation_note"] = ""
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rule_ablation_baselines.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote rule ablations to {RESULTS}")
