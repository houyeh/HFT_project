from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import SEARCH_DIR, json_sanitize


def _policy_from_row(row: pd.Series) -> dict:
    policy = json.loads(row["policy_json"])
    metrics = {
        "net_pnl": float(row["net_pnl"]),
        "trade_count": int(row["trade_count"]),
        "max_drawdown": float(row["max_drawdown"]),
        "cancel_count": int(row["cancel_count"]),
        "bid_cancel_rate": float(row["bid_cancel_rate"]),
        "ask_cancel_rate": float(row["ask_cancel_rate"]),
        "any_cancel_rate": float(row["any_cancel_rate"]),
    }
    return {"policy": policy, "validation_metrics": metrics}


def run() -> None:
    grid = pd.read_csv(SEARCH_DIR / "policy_search_grid.csv")
    manifest = json.loads((SEARCH_DIR / "policy_search_manifest.json").read_text(encoding="utf-8"))
    no_cancel = manifest["validation_no_cancel"]
    min_trade_count = max(100, int(0.45 * int(no_cancel["trade_count"])))
    max_drawdown = max(50_000.0, 1.25 * float(no_cancel["max_drawdown"]))
    max_any_cancel_rate = 0.40
    max_side_cancel_rate = 0.45

    main = grid[(grid["is_main_policy"] == True) & (grid["is_diagnostic_only"] == False)].copy()  # noqa: E712
    main["eligible"] = (
        (main["trade_count"] >= min_trade_count)
        & (main["max_drawdown"] <= max_drawdown)
        & (main["any_cancel_rate"] <= max_any_cancel_rate)
        & (main["bid_cancel_rate"] <= max_side_cancel_rate)
        & (main["ask_cancel_rate"] <= max_side_cancel_rate)
        & (main["net_pnl"] > 0)
    )
    eligible = main[main["eligible"]].copy()
    if eligible.empty:
        eligible = main[
            (main["trade_count"] >= min_trade_count)
            & (main["any_cancel_rate"] <= max_any_cancel_rate)
            & (main["bid_cancel_rate"] <= max_side_cancel_rate)
            & (main["ask_cancel_rate"] <= max_side_cancel_rate)
        ].copy()
        eligible["eligible"] = True

    eligible["selection_score"] = (
        eligible["net_pnl"]
        - 0.10 * eligible["max_drawdown"]
        - 5_000.0 * eligible["any_cancel_rate"]
        - 500.0 * eligible["hls_complexity_rank"]
    )
    ranked = eligible.sort_values(
        ["selection_score", "net_pnl", "hls_complexity_rank", "any_cancel_rate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    selected = _policy_from_row(ranked.iloc[0])
    backups = [_policy_from_row(row) for _, row in ranked.iloc[1:3].iterrows()]

    audit = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_rule": {
            "min_trade_count": min_trade_count,
            "max_drawdown": max_drawdown,
            "max_any_cancel_rate": max_any_cancel_rate,
            "max_side_cancel_rate": max_side_cancel_rate,
            "score": "net_pnl - 0.10*max_drawdown - 5000*any_cancel_rate - 500*hls_complexity_rank",
            "tie_breaker": "Prefer simpler, more HLS-friendly policies and lower cancel rates.",
        },
        "selected_main_policy": selected,
        "backup_policies": backups,
        "eligible_policy_count": int(len(ranked)),
        "diagnostic_policy_b_excluded_from_main_claim": True,
        "validation_no_cancel": no_cancel,
    }
    (SEARCH_DIR / "selected_policy.json").write_text(json.dumps(json_sanitize(audit), indent=2), encoding="utf-8")
    ranked.to_csv(SEARCH_DIR / "policy_selection_ranked.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {SEARCH_DIR / 'selected_policy.json'}")
