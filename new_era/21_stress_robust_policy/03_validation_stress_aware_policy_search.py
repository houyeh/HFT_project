from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stress_robust_utils import (
    SEARCH_DIR,
    STRESSES,
    apply_policy,
    baseline_policy_masks,
    decomp_to_row,
    ensure_dirs,
    json_sanitize,
    load_frames_actions,
    policy_id_from_dict,
    random_v3_mean_for_policy,
    simulate_decomp,
    threshold_for_target_rate,
)


RATES = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


def _ev_policies() -> list[dict]:
    out = []
    for max_spread in [2.0, 3.0, 99.0]:
        for lam in [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
            for ev_thr in [-25.0, 0.0, 10.0, 25.0]:
                out.append(
                    {
                        "policy_type": "ev_score",
                        "policy_family": "EV_based_quote_selection",
                        "policy_id": f"EV_s{max_spread:g}_lam{lam:g}_thr{ev_thr:g}",
                        "max_spread_ticks": max_spread,
                        "lambda_toxic": lam,
                        "ev_threshold": ev_thr,
                        "hls_complexity_rank": 2,
                    }
                )
    return out


def _lut_policies(frame: pd.DataFrame) -> list[dict]:
    score = np.maximum(frame["bid_prob_toxic"].to_numpy(dtype=float), frame["ask_prob_toxic"].to_numpy(dtype=float))
    thresholds = [float(np.nanquantile(score, q)) for q in [0.80, 0.90, 0.95]]
    out = []
    for min_bucket in [1, 2, 3]:
        for cancel_wide in [False, True]:
            lut = {}
            for side in ["bid", "ask"]:
                for spread_bucket in ["tight_<=2", "wide_>2"]:
                    for b in range(4):
                        cancel = b >= min_bucket or (cancel_wide and spread_bucket == "wide_>2")
                        lut[f"{side}|{spread_bucket}|q{b}"] = bool(cancel)
            out.append(
                {
                    "policy_type": "risk_lut",
                    "policy_family": "spread_bucket_x_risk_bucket_LUT",
                    "policy_id": f"LUT_minq{min_bucket}_wide{int(cancel_wide)}",
                    "risk_thresholds": thresholds,
                    "cancel_lut": lut,
                    "hls_complexity_rank": 2,
                }
            )
    return out


def _disagreement_policies(frame: pd.DataFrame) -> list[dict]:
    out = []
    for max_spread in [2.0, 3.0]:
        for rate in [0.05, 0.075, 0.10, 0.15]:
            bid_thr = threshold_for_target_rate(frame["bid_prob_toxic"].to_numpy(dtype=float), rate)
            ask_thr = threshold_for_target_rate(frame["ask_prob_toxic"].to_numpy(dtype=float), rate)
            for mode in ["both_or_wide", "016_unless_015_disagrees", "either_high"]:
                out.append(
                    {
                        "policy_type": "disagreement_ensemble",
                        "policy_family": "disagreement_aware_015_016_ensemble",
                        "policy_id": f"DIS_{mode}_s{max_spread:g}_r{rate:g}",
                        "max_spread_ticks": max_spread,
                        "safe_spread_ticks": 1.5,
                        "bid_threshold": bid_thr,
                        "ask_threshold": ask_thr,
                        "target_cancel_rate": rate,
                        "agreement_mode": mode,
                        "hls_complexity_rank": 3 if mode != "either_high" else 2,
                    }
                )
    return out


def _regime_safe_policies(frame: pd.DataFrame) -> list[dict]:
    out = []
    for regime_col in ["event_rate_bucket", "coarse_spread_bucket"]:
        regimes = sorted(frame[regime_col].astype(str).unique().tolist())
        for low_rate in [0.03, 0.05, 0.075]:
            for high_rate in [0.10, 0.15, 0.20, 0.30]:
                thresholds = {}
                for regime in regimes:
                    high = ("fast" in regime) or ("wide" in regime) or regime.endswith("q4_fast")
                    rate = high_rate if high else low_rate
                    mask = frame[regime_col].astype(str).to_numpy() == regime
                    thresholds[regime] = {
                        "target_cancel_rate": rate,
                        "bid_threshold": threshold_for_target_rate(frame.loc[mask, "bid_prob_toxic"].to_numpy(dtype=float), rate),
                        "ask_threshold": threshold_for_target_rate(frame.loc[mask, "ask_prob_toxic"].to_numpy(dtype=float), rate),
                    }
                out.append(
                    {
                        "policy_type": "regime_safe",
                        "policy_family": "regime_safe_exposure_scaling",
                        "policy_id": f"REG_{regime_col}_lo{low_rate:g}_hi{high_rate:g}",
                        "regime_col": regime_col,
                        "regime_thresholds": thresholds,
                        "hls_complexity_rank": 3,
                    }
                )
    return out


def _evaluate_candidate(frame: pd.DataFrame, actions: pd.DataFrame, policy: dict, no_cancel_moderate: float) -> dict:
    pull_bid, pull_ask = apply_policy(frame, policy)
    metrics = {}
    rows = {}
    for stress_name, stress in STRESSES.items():
        result = simulate_decomp(frame, actions, pull_bid, pull_ask, stress)
        row = decomp_to_row(policy["policy_id"], "validation", stress_name, result)
        rows[stress_name] = row
        for key in ["net_pnl", "trade_count", "max_drawdown", "any_cancel_rate", "toxic_fill_count", "non_toxic_fill_count"]:
            metrics[f"{stress_name}_{key}"] = row[key]
    random_mod = random_v3_mean_for_policy(frame, actions, pull_bid, pull_ask, STRESSES["moderate"], trials=16, seed=20260528)
    composite_raw = 0.25 * metrics["base_net_pnl"] + 0.25 * metrics["mild_net_pnl"] + 0.50 * metrics["moderate_net_pnl"]
    drawdown_penalty = 0.05 * max(metrics["base_max_drawdown"], metrics["mild_max_drawdown"], metrics["moderate_max_drawdown"])
    over_cancel_penalty = 50_000.0 * max(0.0, metrics["moderate_any_cancel_rate"] - 0.35)
    objective = composite_raw - drawdown_penalty - over_cancel_penalty
    constraints_pass = (
        metrics["moderate_trade_count"] >= 100
        and metrics["moderate_any_cancel_rate"] <= 0.55
        and metrics["moderate_max_drawdown"] <= 200_000.0
        and metrics["moderate_net_pnl"] > no_cancel_moderate
        and metrics["moderate_net_pnl"] > random_mod
    )
    return {
        **policy,
        **metrics,
        "moderate_random_v3_mean_matched": random_mod,
        "moderate_no_cancel_pnl": no_cancel_moderate,
        "beats_no_cancel_moderate": bool(metrics["moderate_net_pnl"] > no_cancel_moderate),
        "beats_random_moderate": bool(metrics["moderate_net_pnl"] > random_mod),
        "composite_raw": composite_raw,
        "drawdown_penalty": drawdown_penalty,
        "over_cancel_penalty": over_cancel_penalty,
        "objective": objective,
        "constraints_pass": bool(constraints_pass),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "policy_json": json.dumps(json_sanitize(policy), sort_keys=True),
    }


def run() -> None:
    ensure_dirs()
    frame, actions = load_frames_actions("val")
    no_bid, no_ask = baseline_policy_masks(frame, "no_cancel")
    no_cancel_moderate = simulate_decomp(frame, actions, no_bid, no_ask, STRESSES["moderate"]).net_pnl
    policies = _ev_policies() + _lut_policies(frame) + _disagreement_policies(frame) + _regime_safe_policies(frame)

    rows = [_evaluate_candidate(frame, actions, p, no_cancel_moderate) for p in policies]
    grid = pd.DataFrame(rows)
    grid.to_csv(SEARCH_DIR / "validation_stress_aware_policy_search.csv", index=False)

    eligible = grid[grid["constraints_pass"]].copy()
    fallback_used = False
    if eligible.empty:
        eligible = grid[(grid["moderate_net_pnl"] > no_cancel_moderate) & (grid["moderate_trade_count"] >= 100)].copy()
        fallback_used = True
    if eligible.empty:
        eligible = grid.copy()
        fallback_used = True
    selected_row = eligible.sort_values(
        ["objective", "moderate_net_pnl", "hls_complexity_rank", "moderate_any_cancel_rate"],
        ascending=[False, False, True, True],
    ).iloc[0]
    selected_policy = json.loads(selected_row["policy_json"])
    selected = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "stress_objective": "0.25*base_val + 0.25*mild_val + 0.50*moderate_val - drawdown_penalty - over_cancel_penalty",
        "constraints": {
            "moderate_trade_count_min": 100,
            "moderate_any_cancel_rate_max": 0.55,
            "moderate_max_drawdown_max": 200000.0,
            "must_beat_no_cancel_moderate": True,
            "must_beat_matched_random_moderate": True,
        },
        "fallback_used": bool(fallback_used),
        "selected_policy": selected_policy,
        "validation_metrics": json_sanitize(selected_row.drop(labels=["policy_json"]).to_dict()),
        "candidate_count": int(len(grid)),
        "constraints_pass_count": int(grid["constraints_pass"].sum()),
    }
    (SEARCH_DIR / "selected_stress_robust_policy.json").write_text(json.dumps(json_sanitize(selected), indent=2), encoding="utf-8")
    lines = [
        "# Stress-Aware Policy Selection Report",
        "",
        "Selection split: validation. Test was not read or used.",
        "",
        f"- Candidates evaluated: `{len(grid)}`",
        f"- Constraint-pass candidates: `{int(grid['constraints_pass'].sum())}`",
        f"- Selected policy: `{selected_policy['policy_id']}`",
        f"- Family: `{selected_policy['policy_family']}`",
        f"- Moderate validation PnL: `{float(selected_row['moderate_net_pnl']):.0f}`",
        f"- Matched random moderate mean: `{float(selected_row['moderate_random_v3_mean_matched']):.0f}`",
        f"- No-cancel moderate PnL: `{no_cancel_moderate:.0f}`",
        f"- Objective: `{float(selected_row['objective']):.3f}`",
        "",
        "The selected policy is frozen by `selected_stress_robust_policy.json` before any test evaluation.",
    ]
    (SEARCH_DIR / "policy_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {SEARCH_DIR / 'selected_stress_robust_policy.json'}")
