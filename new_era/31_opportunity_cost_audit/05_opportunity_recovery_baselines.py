from __future__ import annotations

import numpy as np
import pandas as pd

from common_031 import (
    RESULTS,
    RANDOM_SEED,
    RANDOM_TRIALS,
    ensure_dirs,
    eval_selection,
    load_full_actions,
    matched_random_selected_pnls,
    split_df,
    write_json,
)


def candidate_recovery_rules(val: pd.DataFrame) -> list[dict]:
    event_cut = float(val["event_rate_past_50"].quantile(0.50))
    vol_cut = float(val["volatility_past_50"].quantile(0.50))
    rules = [{"rule_id": "no_recovery", "kind": "none"}]
    for thr in [1.0, 2.0, 3.0, 5.0, 10.0]:
        rules.append({"rule_id": f"recover_ask_spread_ge_{thr:g}", "kind": "ask_spread_ge", "min_spread": thr})
        rules.append({"rule_id": f"recover_nonbaseline_spread_ge_{thr:g}", "kind": "nonbaseline_spread_ge", "min_spread": thr})
        rules.append({"rule_id": f"recover_ask_spread_ge_{thr:g}_event_low", "kind": "ask_spread_ge_event_low", "min_spread": thr, "event_cut": event_cut})
        rules.append({"rule_id": f"recover_ask_spread_ge_{thr:g}_vol_low", "kind": "ask_spread_ge_vol_low", "min_spread": thr, "vol_cut": vol_cut})
    return rules


def apply_recovery_rule(df: pd.DataFrame, rule: dict) -> np.ndarray:
    nonbase = ~df["selected_by_baseline"].astype(bool).to_numpy()
    side = df["side"].astype(str).to_numpy()
    spread = df["spread_ticks"].astype(float).to_numpy()
    if rule["kind"] == "none":
        return np.zeros(len(df), dtype=bool)
    if rule["kind"] == "ask_spread_ge":
        return nonbase & (side == "ask") & (spread >= float(rule["min_spread"]))
    if rule["kind"] == "nonbaseline_spread_ge":
        return nonbase & (spread >= float(rule["min_spread"]))
    if rule["kind"] == "ask_spread_ge_event_low":
        return nonbase & (side == "ask") & (spread >= float(rule["min_spread"])) & (df["event_rate_past_50"].astype(float).to_numpy() <= float(rule["event_cut"]))
    if rule["kind"] == "ask_spread_ge_vol_low":
        return nonbase & (side == "ask") & (spread >= float(rule["min_spread"])) & (df["volatility_past_50"].astype(float).to_numpy() <= float(rule["vol_cut"]))
    raise ValueError(rule["kind"])


def evaluate_recovery(df: pd.DataFrame, rule: dict, split: str, *, include_random: bool) -> dict:
    baseline = df["selected_by_baseline"].astype(bool).to_numpy()
    recover = apply_recovery_rule(df, rule)
    selected = baseline | recover
    base_eval = eval_selection(df, baseline, "bid_only_spread_ge_1")
    rule_eval = eval_selection(df, selected, rule["rule_id"])
    row = {
        "phase031_split": split,
        "rule_id": rule["rule_id"],
        "selection_split": "validation",
        "test_used_for_selection": False,
        "baseline_pnl": base_eval["selected_quote_pnl"],
        "recovered_count": int(recover.sum()),
        "recovered_quote_ev": float(df["quote_ev_moderate"].astype(float).to_numpy()[recover].sum()),
        "after_recovery_pnl": rule_eval["selected_quote_pnl"],
        "incremental_vs_baseline": rule_eval["selected_quote_pnl"] - base_eval["selected_quote_pnl"],
        "oracle_recall": rule_eval["oracle_recall"],
        "oracle_precision": rule_eval["oracle_precision"],
    }
    if include_random:
        random_pnls = matched_random_selected_pnls(
            df,
            recover,
            group_cols=["side", "session_id", "spread_bucket"],
            candidate_mask=~baseline,
            base_pnl=float(base_eval["selected_quote_pnl"]),
            trials=RANDOM_TRIALS,
            seed=RANDOM_SEED,
        )
        row.update(
            {
                "matched_random_recovery_mean": float(np.mean(random_pnls)),
                "matched_random_recovery_95p": float(np.quantile(random_pnls, 0.95)),
                "matched_random_recovery_99p": float(np.quantile(random_pnls, 0.99)),
                "beats_matched_random_recovery_95p": bool(row["after_recovery_pnl"] > float(np.quantile(random_pnls, 0.95))),
            }
        )
    return row


def main() -> None:
    ensure_dirs()
    actions = load_full_actions()
    val = split_df(actions, "val")
    rules = candidate_recovery_rules(val)
    val_rows = [evaluate_recovery(val, rule, "val", include_random=False) for rule in rules]
    val_grid = pd.DataFrame(val_rows)
    best = val_grid.sort_values(["after_recovery_pnl", "recovered_count"], ascending=[False, True]).iloc[0]
    selected_rule = next(rule for rule in rules if rule["rule_id"] == best["rule_id"])
    write_json(
        RESULTS / "selected_recovery_rule.json",
        {
            "selection_split": "validation",
            "test_used_for_selection": False,
            "selected_rule": selected_rule,
            "validation_metrics": best.to_dict(),
        },
    )

    rows = val_rows
    for split in ["test", "new4"]:
        rows.append(evaluate_recovery(split_df(actions, split), selected_rule, split, include_random=True))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "opportunity_recovery_baselines.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote opportunity recovery baseline audit to {RESULTS}")

