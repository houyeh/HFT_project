from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import (
    SEARCH_DIR,
    build_actions_with_masks,
    ensure_dirs,
    eval_policy_row,
    json_sanitize,
    load_policy_frame,
    simulate_policy_fast,
    threshold_for_target_rate,
)


def _target_thresholds(frame: pd.DataFrame, bid_score: np.ndarray, ask_score: np.ndarray, rate: float) -> tuple[float, float]:
    return threshold_for_target_rate(bid_score, rate), threshold_for_target_rate(ask_score, rate)


def _policy_a(frame: pd.DataFrame) -> list[dict[str, object]]:
    policies = []
    bid = frame["bid_prob_toxic"].to_numpy(dtype=float)
    ask = frame["ask_prob_toxic"].to_numpy(dtype=float)
    for max_spread in [1.5, 2.0, 3.0, 99.0]:
        for rate in [0.03, 0.05, 0.075, 0.10, 0.125, 0.15]:
            bthr, athr = _target_thresholds(frame, bid, ask, rate)
            policies.append(
                {
                    "policy_id": f"A_spread_toxic_s{max_spread:g}_r{rate:g}",
                    "policy_family": "A_spread_filter_side_specific_toxic_cancel",
                    "policy_type": "spread_toxic",
                    "is_main_policy": True,
                    "is_diagnostic_only": False,
                    "hls_complexity_rank": 1,
                    "max_spread_ticks": max_spread,
                    "target_cancel_rate": rate,
                    "bid_threshold": bthr,
                    "ask_threshold": athr,
                }
            )
    return policies


def _policy_c(frame: pd.DataFrame) -> list[dict[str, object]]:
    policies = []
    bid016 = frame["bid_prob_toxic"].to_numpy(dtype=float)
    ask016 = frame["ask_prob_toxic"].to_numpy(dtype=float)
    for max_spread in [2.0, 3.0]:
        for rate in [0.05, 0.10, 0.15]:
            bthr, athr = _target_thresholds(frame, bid016, ask016, rate)
            for mode in ["or", "and"]:
                policies.append(
                    {
                        "policy_id": f"C_ensemble_{mode}_s{max_spread:g}_r{rate:g}",
                        "policy_family": "C_015_016_ensemble",
                        "policy_type": "ensemble",
                        "ensemble_mode": mode,
                        "is_main_policy": True,
                        "is_diagnostic_only": False,
                        "hls_complexity_rank": 3 if mode == "or" else 2,
                        "max_spread_ticks": max_spread,
                        "target_cancel_rate": rate,
                        "bid_threshold": bthr,
                        "ask_threshold": athr,
                    }
                )
            for weight in [0.50, 0.75]:
                bid_score = weight * bid016 + (1.0 - weight) * frame["score_bid_015"].to_numpy(dtype=float)
                ask_score = weight * ask016 + (1.0 - weight) * frame["score_ask_015"].to_numpy(dtype=float)
                bw, aw = _target_thresholds(frame, bid_score, ask_score, rate)
                policies.append(
                    {
                        "policy_id": f"C_ensemble_weighted_w{weight:g}_s{max_spread:g}_r{rate:g}",
                        "policy_family": "C_015_016_ensemble",
                        "policy_type": "ensemble",
                        "ensemble_mode": "weighted",
                        "is_main_policy": True,
                        "is_diagnostic_only": False,
                        "hls_complexity_rank": 4,
                        "max_spread_ticks": max_spread,
                        "target_cancel_rate": rate,
                        "weight_016": weight,
                        "bid_threshold": bthr,
                        "ask_threshold": athr,
                        "bid_weighted_threshold": bw,
                        "ask_weighted_threshold": aw,
                    }
                )
    return policies


def _policy_d(frame: pd.DataFrame) -> list[dict[str, object]]:
    policies = []
    regime_col = "coarse_spread_bucket"
    regimes = sorted(frame[regime_col].astype(str).unique().tolist())
    for max_spread in [2.0, 3.0]:
        for tight_rate in [0.05, 0.10]:
            for wide_rate in [0.10, 0.15, 0.20]:
                thresholds = {}
                for regime in regimes:
                    mask = frame[regime_col].astype(str).to_numpy() == regime
                    rate = tight_rate if "tight" in regime else wide_rate
                    thresholds[regime] = {
                        "target_cancel_rate": rate,
                        "bid_threshold": threshold_for_target_rate(frame.loc[mask, "bid_prob_toxic"].to_numpy(dtype=float), rate),
                        "ask_threshold": threshold_for_target_rate(frame.loc[mask, "ask_prob_toxic"].to_numpy(dtype=float), rate),
                    }
                policies.append(
                    {
                        "policy_id": f"D_regime_spread_s{max_spread:g}_tight{tight_rate:g}_wide{wide_rate:g}",
                        "policy_family": "D_regime_aware_threshold",
                        "policy_type": "regime_threshold",
                        "is_main_policy": True,
                        "is_diagnostic_only": False,
                        "hls_complexity_rank": 2,
                        "max_spread_ticks": max_spread,
                        "regime_col": regime_col,
                        "regime_thresholds": thresholds,
                    }
                )
    return policies


def _policy_b_diagnostic(frame: pd.DataFrame) -> list[dict[str, object]]:
    risk = np.maximum(frame["bid_prob_toxic"].to_numpy(dtype=float), frame["ask_prob_toxic"].to_numpy(dtype=float))
    policies = []
    for max_spread in [2.0, 3.0]:
        for rate in [0.05, 0.10, 0.15]:
            thr = threshold_for_target_rate(risk, rate)
            policies.append(
                {
                    "policy_id": f"B_depth_proxy_diag_s{max_spread:g}_r{rate:g}",
                    "policy_family": "B_risk_based_quote_depth_proxy",
                    "policy_type": "depth_proxy_diagnostic",
                    "is_main_policy": False,
                    "is_diagnostic_only": True,
                    "hls_complexity_rank": 2,
                    "max_spread_ticks": max_spread,
                    "target_cancel_rate": rate,
                    "risk_threshold": thr,
                    "diagnostic_note": "Binary cancel equivalent only; not eligible for main positive-PnL claim.",
                }
            )
    return policies


def run() -> None:
    ensure_dirs()
    frame = load_policy_frame("val")
    actions = build_actions_with_masks(frame, split="val")
    policies = _policy_a(frame) + _policy_c(frame) + _policy_d(frame) + _policy_b_diagnostic(frame)

    rows = []
    for policy in policies:
        row = eval_policy_row(frame, actions, policy, split="validation")
        row["selection_split"] = "validation"
        row["test_used_for_selection"] = False
        row["policy_json"] = json.dumps(json_sanitize(policy), sort_keys=True)
        rows.append(row)

    no_cancel = simulate_policy_fast(frame, actions, np.zeros(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool))
    grid = pd.DataFrame(rows)
    grid.to_csv(SEARCH_DIR / "policy_search_grid.csv", index=False)
    manifest = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "num_candidates": int(len(grid)),
        "candidate_families": sorted(grid["policy_family"].unique().tolist()),
        "validation_no_cancel": no_cancel.__dict__,
        "notes": [
            "Policy B depth proxy rows are diagnostic only and are not eligible for the main claim.",
            "All thresholds are derived from validation scores only.",
        ],
    }
    (SEARCH_DIR / "policy_search_manifest.json").write_text(json.dumps(json_sanitize(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {SEARCH_DIR / 'policy_search_grid.csv'}")
