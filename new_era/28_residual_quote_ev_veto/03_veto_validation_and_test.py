from __future__ import annotations

import numpy as np
import pandas as pd

from common_028 import (
    RESULTS,
    ensure_dirs,
    json_sanitize,
    load_baseline_dataset,
    matched_random_veto_after_pnls,
    write_json,
)


def _load_predictions(split: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"residual_ev_predictions_{split}.csv")


def _threshold_grid(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, group in pred.groupby("model_name", observed=False):
        values = group["quote_ev_moderate"].astype(float).to_numpy()
        scores = group["negative_ev_score"].astype(float).to_numpy()
        base = float(values.sum())
        finite = scores[np.isfinite(scores)]
        no_veto_threshold = float(np.max(finite) + 1e-6) if len(finite) else 1.000001
        thresholds = [no_veto_threshold]
        if len(finite):
            thresholds.extend(np.unique(np.quantile(finite, np.linspace(0.50, 0.99, 50))).tolist())
        for thr in thresholds:
            veto = scores >= float(thr)
            rows.append(
                {
                    "model_name": model_name,
                    "threshold": float(thr),
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "veto_count": int(veto.sum()),
                    "veto_rate": float(veto.mean()) if len(veto) else 0.0,
                    "baseline_pnl": base,
                    "vetoed_quote_ev": float(values[veto].sum()),
                    "after_veto_pnl": base - float(values[veto].sum()),
                    "incremental_vs_baseline": -float(values[veto].sum()),
                }
            )
    return pd.DataFrame(rows)


def _attach_scores(df: pd.DataFrame, pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    p = pred[pred["model_name"] == model_name].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    if len(out) != len(p):
        raise RuntimeError(f"Prediction length mismatch for {model_name}: {len(out)} vs {len(p)}")
    out["negative_ev_score"] = p["negative_ev_score"].astype(float).to_numpy()
    return out


def _eval_split(split: str, policy: dict) -> tuple[dict, np.ndarray]:
    df = load_baseline_dataset("new4" if split == "new4" else split)
    pred = _load_predictions("new4" if split == "new4" else split)
    work = _attach_scores(df, pred, policy["model_name"])
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    veto = work["negative_ev_score"].astype(float).to_numpy() >= float(policy["threshold"])
    baseline_pnl = float(values.sum())
    after = baseline_pnl - float(values[veto].sum())
    random_pnls = matched_random_veto_after_pnls(work, veto, trials=5000, seed=20260629)
    row = {
        "split": split,
        "model_name": policy["model_name"],
        "threshold": float(policy["threshold"]),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "baseline_pnl": baseline_pnl,
        "veto_count": int(veto.sum()),
        "veto_rate": float(veto.mean()) if len(veto) else 0.0,
        "vetoed_quote_ev": float(values[veto].sum()),
        "after_veto_pnl": after,
        "incremental_vs_baseline": after - baseline_pnl,
        "negative_ev_rate_vetoed": float((values[veto] < 0).mean()) if veto.any() else np.nan,
        "random_veto_mean_pnl": float(np.mean(random_pnls)),
        "random_veto_std_pnl": float(np.std(random_pnls, ddof=1)) if len(random_pnls) > 1 else 0.0,
        "random_veto_95p_pnl": float(np.quantile(random_pnls, 0.95)),
        "random_veto_99p_pnl": float(np.quantile(random_pnls, 0.99)),
        "beats_random_veto_95p": bool(after > float(np.quantile(random_pnls, 0.95))),
        "beats_random_veto_99p": bool(after > float(np.quantile(random_pnls, 0.99))),
        "improves_baseline": bool(after > baseline_pnl),
    }
    return row, random_pnls


def main() -> None:
    ensure_dirs()
    val_pred = _load_predictions("val")
    grid = _threshold_grid(val_pred)
    grid.to_csv(RESULTS / "veto_validation_threshold_grid.csv", index=False)
    selected = grid.sort_values(["after_veto_pnl", "veto_count"], ascending=[False, True]).iloc[0].to_dict()
    policy = {
        "policy_id": f"residual_ev_veto__{selected['model_name']}__thr{selected['threshold']:.8f}",
        "model_name": str(selected["model_name"]),
        "threshold": float(selected["threshold"]),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "validation_metrics": json_sanitize(selected),
        "universe": "quotes selected by frozen bid_only_spread_ge_1 baseline",
        "label": "negative_ev_moderate; toxic label not used",
    }
    write_json(RESULTS / "selected_ev_veto_policy.json", policy)

    rows: list[dict] = []
    for split in ["test", "new4"]:
        row, random_pnls = _eval_split(split, policy)
        rows.append(row)
        pd.DataFrame({"trial": range(len(random_pnls)), "random_after_veto_pnl": random_pnls}).to_csv(
            RESULTS / f"matched_random_veto_trials_{split}.csv",
            index=False,
        )
    pd.DataFrame(rows).to_csv(RESULTS / "frozen_veto_test_and_new4.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote frozen residual EV veto evaluation to {RESULTS}")
