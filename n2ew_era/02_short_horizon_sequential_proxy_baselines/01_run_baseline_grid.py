from __future__ import annotations

import pandas as pd

from common_phase02 import HORIZONS, RESULTS_DIR, available_splits, ensure_dirs, evaluate_grid, policy_catalog, write_json


def main() -> None:
    ensure_dirs()
    policies = policy_catalog()
    splits = available_splits()
    out = evaluate_grid(policies, splits, HORIZONS)
    out = out.sort_values(["split", "horizon", "family", "policy_id"]).reset_index(drop=True)
    out.to_csv(RESULTS_DIR / "baseline_grid_all_splits.csv", index=False)
    write_json(
        RESULTS_DIR / "baseline_policy_catalog.json",
        {
            "horizons": HORIZONS,
            "splits": splits,
            "policy_count": len(policies),
            "policies": policies,
            "critical_rules": [
                "No ML training.",
                "No test or NEW4 threshold tuning.",
                "Policy decisions use only causal/current L2 features.",
                "Every state maps to at most one action: bid, ask, or no_quote.",
            ],
        },
    )
    print("Baseline grid complete:")
    print(pd.read_csv(RESULTS_DIR / "baseline_grid_all_splits.csv").groupby("split")["policy_id"].nunique().to_string())
    print(f"Wrote {RESULTS_DIR / 'baseline_grid_all_splits.csv'}")


if __name__ == "__main__":
    main()
