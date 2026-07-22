from __future__ import annotations

import pandas as pd

from common_030 import RESULTS, ensure_dirs, threshold_curve


def main() -> None:
    ensure_dirs()
    pred = pd.read_csv("new_era/28_residual_quote_ev_veto/results/residual_ev_predictions_val.csv")
    rows: list[pd.DataFrame] = []
    for model_name, group in pred.groupby("model_name", observed=False):
        curve = threshold_curve(group, "negative_ev_score")
        curve.insert(0, "split", "validation")
        curve.insert(0, "model_name", model_name)
        curve["diagnostic_role"] = "existing_phase028_validation_threshold_sweep"
        curve["test_used_for_selection"] = False
        rows.append(curve)
    pd.concat(rows, ignore_index=True).to_csv(RESULTS / "existing_model_threshold_curves.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote existing model threshold curves to {RESULTS}")

