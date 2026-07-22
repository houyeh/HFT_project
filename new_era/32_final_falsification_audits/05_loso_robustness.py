from __future__ import annotations

import pandas as pd

from common_final32 import RESULTS, ensure_dirs, eval_selected, load_eval_actions


def main() -> None:
    ensure_dirs()
    actions = load_eval_actions()
    rows = []
    for split, group in actions.groupby("phase031_split", sort=True):
        selected = group["selected_by_baseline"].astype(bool).to_numpy()
        full = eval_selected(group, selected, "bid_only_spread_ge_1")
        full_pnl = float(full["selected_quote_pnl"])
        rows.append(
            {
                "phase031_split": split,
                "dropped_session": "NONE_FULL_SAMPLE",
                "remaining_session_count": int(group["session_id"].nunique()),
                "full_sample_pnl": full_pnl,
                "pnl_without_session": full_pnl,
                "dropped_session_pnl": 0.0,
                "pnl_retention_ratio": 1.0,
                "selected_count_without_session": int(selected.sum()),
            }
        )
        for session, _ in group.groupby("session_id", sort=True):
            keep = group["session_id"].astype(str).to_numpy() != str(session)
            sub = group.loc[keep].reset_index(drop=True)
            sub_selected = sub["selected_by_baseline"].astype(bool).to_numpy()
            sub_eval = eval_selected(sub, sub_selected, "bid_only_spread_ge_1")
            session_pnl = full_pnl - float(sub_eval["selected_quote_pnl"])
            rows.append(
                {
                    "phase031_split": split,
                    "dropped_session": str(session),
                    "remaining_session_count": int(sub["session_id"].nunique()),
                    "full_sample_pnl": full_pnl,
                    "pnl_without_session": float(sub_eval["selected_quote_pnl"]),
                    "dropped_session_pnl": session_pnl,
                    "pnl_retention_ratio": float(sub_eval["selected_quote_pnl"] / full_pnl) if full_pnl else None,
                    "selected_count_without_session": int(sub_selected.sum()),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "loso_robustness_metrics.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote LOSO robustness metrics to {RESULTS}")

