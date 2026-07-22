from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_utils import (
    RESULTS_DIR,
    apply_non_ml_policy,
    ensure_dirs,
    load_selected_non_ml_policy,
    matched_random_veto_pnls,
    val_test_actions,
)


VETO_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]


def _threshold(scores: np.ndarray, frac: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan
    return float(np.quantile(scores, 1.0 - float(frac)))


def side_threshold_veto(actions: pd.DataFrame, baseline: np.ndarray, bid_frac: float, ask_frac: float, thresholds: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    side = actions["side"].astype(str).to_numpy()
    score = actions["side_score_016"].astype(float).to_numpy()
    thresholds = dict(thresholds or {})
    if "bid" not in thresholds:
        thresholds["bid"] = _threshold(score[baseline & (side == "bid")], bid_frac)
    if "ask" not in thresholds:
        thresholds["ask"] = _threshold(score[baseline & (side == "ask")], ask_frac)
    veto = baseline & np.where(side == "bid", score >= thresholds["bid"], score >= thresholds["ask"])
    return veto, thresholds


def eval_veto(actions: pd.DataFrame, baseline: np.ndarray, veto: np.ndarray) -> dict:
    values = actions["quote_value_moderate"].astype(float).to_numpy()
    final = baseline & ~veto
    return {
        "baseline_quote_count": int(baseline.sum()),
        "veto_count": int(veto.sum()),
        "final_quote_count": int(final.sum()),
        "baseline_pnl": float(values[baseline].sum()),
        "ml_veto_pnl": float(values[final].sum()),
        "ml_incremental_vs_baseline": float(values[final].sum() - values[baseline].sum()),
        "vetoed_quote_pnl": float(values[veto].sum()),
    }


def run() -> None:
    ensure_dirs()
    policy = load_selected_non_ml_policy()
    val_actions = val_test_actions("val")
    test_actions = val_test_actions("test")
    val_base = apply_non_ml_policy(val_actions, policy)
    test_base = apply_non_ml_policy(test_actions, policy)

    val_rows = []
    best = None
    for bid_frac in VETO_FRACS:
        for ask_frac in VETO_FRACS:
            veto, thresholds = side_threshold_veto(val_actions, val_base, bid_frac, ask_frac)
            metrics = eval_veto(val_actions, val_base, veto)
            row = {"bid_veto_frac": bid_frac, "ask_veto_frac": ask_frac, "bid_threshold": thresholds["bid"], "ask_threshold": thresholds["ask"], **metrics}
            row["selection_objective"] = row["ml_veto_pnl"]
            val_rows.append(row)
            if best is None or row["selection_objective"] > best["selection_objective"]:
                best = row
    pd.DataFrame(val_rows).to_csv(RESULTS_DIR / "ml_incremental_veto_validation_grid.csv", index=False)
    assert best is not None
    veto, thresholds = side_threshold_veto(test_actions, test_base, best["bid_veto_frac"], best["ask_veto_frac"], {"bid": best["bid_threshold"], "ask": best["ask_threshold"]})
    metrics = eval_veto(test_actions, test_base, veto)
    random = matched_random_veto_pnls(test_actions, test_base, veto, trials=1000)
    out = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "bid_veto_frac": best["bid_veto_frac"],
        "ask_veto_frac": best["ask_veto_frac"],
        "bid_threshold": best["bid_threshold"],
        "ask_threshold": best["ask_threshold"],
        **metrics,
        "random_veto_mean_pnl": float(random.mean()),
        "random_veto_95p_pnl": float(np.quantile(random, 0.95)),
        "random_veto_99p_pnl": float(np.quantile(random, 0.99)),
        "ml_beats_random_95p": bool(metrics["ml_veto_pnl"] > np.quantile(random, 0.95)),
    }
    pd.DataFrame([out]).to_csv(RESULTS_DIR / "ml_incremental_value.csv", index=False)

    side_rows = []
    for side_name in ["bid", "ask"]:
        side_mask = test_actions["side"].astype(str).to_numpy() == side_name
        m = eval_veto(test_actions[side_mask].reset_index(drop=True), test_base[side_mask], veto[side_mask])
        m["side"] = side_name
        side_rows.append(m)
    pd.DataFrame(side_rows).to_csv(RESULTS_DIR / "side_specific_ml_value.csv", index=False)
    test_actions.assign(baseline_selected=test_base, ml_veto=veto, ml_final_selected=test_base & ~veto).to_parquet(RESULTS_DIR / "ml_veto_test_actions.parquet", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote ML incremental veto outputs to {RESULTS_DIR}")

