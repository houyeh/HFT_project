from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common_030 import RANDOM_SEED, RANDOM_TRIALS, RESULTS, ensure_dirs, evaluate_with_random_hurdle, matched_random_veto_after_pnls


def load_predictions(split: str, model_name: str) -> pd.DataFrame:
    pred = pd.read_csv(RESULTS / f"model_family_predictions_{split}.csv")
    return pred[pred["model_name"] == model_name].reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    selected_payload = json.loads((RESULTS / "selected_model_family.json").read_text(encoding="utf-8"))
    selected = selected_payload["selected_model"]
    model_name = selected["model_name"]
    threshold = float(selected["threshold"])

    rows: list[dict] = []
    for split in ["val", "test", "new4"]:
        pred = load_predictions(split, model_name)
        row = evaluate_with_random_hurdle(pred, "negative_ev_score", threshold, trials=RANDOM_TRIALS)
        row.update(
            {
                "split": split,
                "model_name": model_name,
                "objective": selected["objective"],
                "threshold": threshold,
                "selection_split": "validation",
                "test_used_for_selection": False,
                "new4_used_for_selection": False,
                "max_veto_rate": selected_payload["max_veto_rate"],
            }
        )
        rows.append(row)
        veto = pred["negative_ev_score"].astype(float).to_numpy() >= threshold
        random_pnls = matched_random_veto_after_pnls(pred, veto, trials=RANDOM_TRIALS, seed=RANDOM_SEED)
        pd.DataFrame({"trial": range(len(random_pnls)), "matched_random_after_veto_pnl": random_pnls}).to_csv(
            RESULTS / f"final_matched_random_veto_trials_{split}.csv",
            index=False,
        )
    out = pd.DataFrame(rows)
    out["accepted_useful_ml"] = (
        out["improves_baseline"].astype(bool)
        & out["beats_random_veto_95p"].astype(bool)
        & (out["veto_rate"].astype(float) <= out["max_veto_rate"].astype(float))
    )
    out.to_csv(RESULTS / "ml_incremental_value_final.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote final incremental value evaluation to {RESULTS}")

