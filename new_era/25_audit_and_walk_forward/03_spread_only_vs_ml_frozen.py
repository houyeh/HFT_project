from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "25_audit_and_walk_forward"
RESULTS = ROOT / "results"
V24 = PROJECT_ROOT / "new_era" / "24_calibration_robust_policy"
if str(V24) not in sys.path:
    sys.path.insert(0, str(V24))

from calibration_utils import apply_policy, evaluate_candidate, load_context, policy_from_row  # noqa: E402


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def run() -> None:
    ensure_dirs()
    grid = pd.read_csv(V24 / "results" / "combined_validation_selection_grid.csv")
    spread = grid[grid["score_name"].astype(str) == "spread_only"].sort_values("selected_quote_pnl", ascending=False).iloc[0]
    spread_policy = policy_from_row(spread)
    _, test_actions = load_context("test")
    spread_selected, _ = apply_policy(test_actions, spread_policy)
    spread_metrics = evaluate_candidate(
        test_actions,
        spread_selected,
        policy_id=spread_policy["policy_id"],
        family=spread_policy["family"],
        score_name=spread_policy["score_name"],
        score_col=spread_policy["score_column"],
        window=spread_policy["window"],
        quote_fraction=spread_policy.get("quote_fraction"),
        group_cols=spread_policy["group_cols"],
        higher_is_better=spread_policy["higher_is_better"],
        split="test_diagnostic_burned",
        extra={"policy_role": "spread_only_validation_best", "used_for_main_selection": False},
    )
    ml = pd.read_csv(V24 / "results" / "frozen_calibration_robust_test.csv").iloc[0].to_dict()
    ml["policy_role"] = "024_ml_selected"
    ml["used_for_main_selection"] = True
    rows = [ml, spread_metrics]
    out = pd.DataFrame(rows)
    ml_pnl = float(out.loc[out["policy_role"] == "024_ml_selected", "selected_quote_pnl"].iloc[0])
    spread_pnl = float(out.loc[out["policy_role"] == "spread_only_validation_best", "selected_quote_pnl"].iloc[0])
    out["ml_minus_spread_only_pnl"] = ml_pnl - spread_pnl
    out.to_csv(RESULTS / "spread_only_vs_ml_rolling_frozen_test.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote spread-only vs ML frozen comparison to {RESULTS}")

