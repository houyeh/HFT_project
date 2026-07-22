from __future__ import annotations

import pandas as pd

from common_final32 import RESULTS, attach_lagged_spreads, baseline_decision, decision_source_audit, ensure_dirs, eval_selected, load_eval_actions, write_markdown


def main() -> None:
    ensure_dirs()
    actions = attach_lagged_spreads(load_eval_actions(), [0, 1, 2])
    rows = []
    for split, group in actions.groupby("phase031_split", sort=True):
        canonical = group["selected_by_baseline"].astype(bool).to_numpy()
        for lag in [0, 1, 2]:
            selected = baseline_decision(group, f"spread_lag_{lag}")
            row = eval_selected(group, selected, f"bid_only_spread_ge_1_lag_{lag}")
            row.update(
                {
                    "phase031_split": split,
                    "decision_book": "current_row" if lag == 0 else f"{lag}_row_lagged",
                    "lag_ticks": lag,
                    "mismatch_vs_stored_baseline_count": int((selected != canonical).sum()) if lag == 0 else None,
                    "missing_lag_spread_count": int(group[f"spread_lag_{lag}"].isna().sum()),
                    "session_boundary_bad_decision_count": int((selected & (group["local_row_id"].astype(int).to_numpy() < lag)).sum()) if lag else 0,
                }
            )
            rows.append(row)
    src = decision_source_audit()
    out = pd.DataFrame(rows)
    out["source_contains_forbidden_future_tokens"] = src["source_contains_forbidden_future_tokens"]
    out["forbidden_tokens_found"] = src["forbidden_tokens_found"]
    out.to_csv(RESULTS / "no_lookahead_audit.csv", index=False)

    lines = [
        "# No-Lookahead Audit",
        "",
        "Decision rule audited: quote only when `side == bid` and decision-book `spread_ticks >= 1`.",
        "",
        f"- Source function: `{src['source_function']}`.",
        f"- Future/EV token found in decision source: `{src['source_contains_forbidden_future_tokens']}`.",
        f"- Tokens found: `{src['forbidden_tokens_found'] or 'none'}`.",
        "",
        "## Timing Results",
        "",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"- `{row['phase031_split']}` `{row['decision_book']}`: PnL `{row['selected_quote_pnl']:.0f}`, "
            f"selected `{int(row['selected_count'])}`, boundary bad decisions `{int(row['session_boundary_bad_decision_count'])}`."
        )
    lines.extend(
        [
            "",
            "Current-row decision should match the stored baseline. Lagged rows are a deterministic delay/timing stress, not a reselected policy.",
        ]
    )
    write_markdown(RESULTS / "no_lookahead_report.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote no-lookahead audit to {RESULTS}")

