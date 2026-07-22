from __future__ import annotations

import numpy as np
import pandas as pd

from common_final32 import RESULTS, RANDOM_TRIALS, ensure_dirs, load_eval_actions, matched_random_pnls


def main() -> None:
    ensure_dirs()
    actions = load_eval_actions()
    rows = []
    audits = []
    for split, group in actions.groupby("phase031_split", sort=True):
        selected = group["selected_by_baseline"].astype(bool).to_numpy()
        exact_universe = (group["side"].astype(str).to_numpy() == "bid") & (group["spread_ticks"].astype(float).to_numpy() >= 1.0)
        pnls, audit = matched_random_pnls(
            group,
            selected,
            group_cols=["phase031_split", "session_id", "spread_bucket"],
            candidate_mask=exact_universe,
            trials=RANDOM_TRIALS,
            seed=20260703,
        )
        values = group["quote_ev_moderate"].astype(float).to_numpy()
        model_pnl = float(values[selected].sum())
        rows.append(
            {
                "phase031_split": split,
                "trials": int(len(pnls)),
                "model_pnl": model_pnl,
                "model_selected_count": int(selected.sum()),
                "candidate_universe_count": int(exact_universe.sum()),
                "random_mean": float(np.mean(pnls)),
                "random_std": float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0,
                "random_95p": float(np.quantile(pnls, 0.95)),
                "random_99p": float(np.quantile(pnls, 0.99)),
                "empirical_p_value_random_ge_model": float(np.mean(pnls >= model_pnl)),
                "zero_variance_due_to_saturated_exact_universe": bool(np.std(pnls) == 0.0),
                "match_definition": "candidate universe side=bid and spread>=1; match session and spread_bucket selected counts",
            }
        )
        if "phase031_split" not in audit.columns:
            audit.insert(0, "phase031_split", split)
        audits.append(audit)
        pd.DataFrame({"trial": range(len(pnls)), "random_pnl": pnls}).to_csv(RESULTS / f"matched_random_quote_trials_{split}.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "matched_random_quote_summary.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(RESULTS / "matched_random_quote_matching_audit.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote matched random quote trials to {RESULTS}")
