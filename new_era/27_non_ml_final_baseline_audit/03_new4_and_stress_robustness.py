from __future__ import annotations

import numpy as np
import pandas as pd

from common_027 import (
    RESULTS,
    STRESS_SPECS,
    apply_non_ml_policy,
    build_new4_actions,
    build_new4_frame,
    ensure_dirs,
    load_selected_non_ml_policy,
    quote_value_for_stress,
    write_markdown,
)


def _selected_group_summary(actions: pd.DataFrame, group_cols: list[str], group_type: str) -> pd.DataFrame:
    selected = actions[actions["selected_quote"].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame()
    agg = selected.groupby(group_cols, observed=False).agg(
        selected_quote_count=("action_id", "count"),
        pnl_base=("quote_value_base", "sum"),
        pnl_mild=("quote_value_mild", "sum"),
        pnl_moderate=("quote_value_moderate", "sum"),
        avg_moderate_pnl=("quote_value_moderate", "mean"),
        positive_pnl_rate=("quote_value_moderate", lambda s: float((s > 0).mean())),
        toxic_rate=("side_toxic", "mean"),
    ).reset_index()
    agg.insert(0, "group_type", group_type)
    for col in ["session_id", "side", "spread_bucket", "time_bucket"]:
        if col not in agg.columns:
            agg[col] = ""
    return agg[["group_type", "session_id", "side", "spread_bucket", "time_bucket", "selected_quote_count", "pnl_base", "pnl_mild", "pnl_moderate", "avg_moderate_pnl", "positive_pnl_rate", "toxic_rate"]]


def main() -> None:
    ensure_dirs()
    frame, symbol, tick_size = build_new4_frame()
    actions = build_new4_actions(frame)
    policy = load_selected_non_ml_policy()
    selected = apply_non_ml_policy(actions, policy)
    actions["selected_quote"] = selected

    ordered = actions.sort_values(["session_id", "row_index", "side"]).copy()
    ordered["_session_ord"] = ordered.groupby("session_id", observed=False).cumcount()
    ordered["_session_n"] = ordered.groupby("session_id", observed=False)["_session_ord"].transform("max") + 1
    ordered["time_bucket"] = np.where(
        ordered["_session_n"] > 0,
        pd.cut(ordered["_session_ord"] / ordered["_session_n"], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=["q1", "q2", "q3", "q4"]),
        "q1",
    )
    actions = ordered.sort_index()

    summaries = [
        _selected_group_summary(actions, ["session_id", "side"], "session_side"),
        _selected_group_summary(actions, ["spread_bucket"], "spread_bucket"),
        _selected_group_summary(actions, ["session_id", "time_bucket"], "session_time_bucket"),
        _selected_group_summary(actions, ["session_id", "side", "spread_bucket"], "session_side_spread"),
    ]
    breakdown = pd.concat([x for x in summaries if not x.empty], ignore_index=True)
    total_mod = float(actions.loc[selected, "quote_value_moderate"].sum())
    breakdown["moderate_pnl_share"] = breakdown["pnl_moderate"].astype(float) / total_mod if total_mod else np.nan
    breakdown.to_csv(RESULTS / "new4_robustness_breakdown.csv", index=False)

    stress_rows: list[dict] = []
    for stress_name, spec in STRESS_SPECS.items():
        values = quote_value_for_stress(
            actions,
            neutral_fill_haircut=spec["neutral_fill_haircut"],
            toxic_multiplier=spec["toxic_multiplier"],
        )
        stress_rows.append(
            {
                "dataset": "NEW4",
                "symbol": symbol,
                "tick_size": tick_size,
                "policy_id": policy["policy_id"],
                "stress_name": stress_name,
                "neutral_fill_haircut": spec["neutral_fill_haircut"],
                "toxic_multiplier": spec["toxic_multiplier"],
                "selected_quote_count": int(selected.sum()),
                "selected_quote_pnl": float(values[selected].sum()),
                "avg_selected_quote_pnl": float(values[selected].mean()) if selected.any() else np.nan,
                "positive_selected_quote_rate": float((values[selected] > 0).mean()) if selected.any() else np.nan,
            }
        )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(RESULTS / "new4_stress_robustness.csv", index=False)

    pnl = actions.loc[selected, "quote_value_moderate"].astype(float).to_numpy()
    sorted_pos = np.sort(pnl)[::-1]
    concentration = {
        "selected_quote_count": int(len(pnl)),
        "total_moderate_pnl": float(pnl.sum()),
        "top_1pct_pnl_share": float(sorted_pos[: max(1, int(np.ceil(0.01 * len(sorted_pos))))].sum() / pnl.sum()) if len(pnl) and pnl.sum() else np.nan,
        "top_5pct_pnl_share": float(sorted_pos[: max(1, int(np.ceil(0.05 * len(sorted_pos))))].sum() / pnl.sum()) if len(pnl) and pnl.sum() else np.nan,
        "top_10pct_pnl_share": float(sorted_pos[: max(1, int(np.ceil(0.10 * len(sorted_pos))))].sum() / pnl.sum()) if len(pnl) and pnl.sum() else np.nan,
    }
    pd.DataFrame([concentration]).to_csv(RESULTS / "new4_selected_quote_concentration.csv", index=False)

    mod = stress[stress["stress_name"] == "moderate"].iloc[0]
    lines = [
        "# Non-ML Stress Robustness Report",
        "",
        f"- Policy: `{policy['policy_id']}`.",
        f"- NEW4 symbol: `{symbol}`.",
        f"- NEW4 tick size: `{tick_size}`.",
        "- Selection was frozen before NEW4 evaluation; NEW4 is not used for tuning.",
        "",
        "## Stress PnL",
        "",
    ]
    for _, row in stress.iterrows():
        lines.append(
            f"- `{row['stress_name']}`: selected quote PnL `{row['selected_quote_pnl']:.0f}`, "
            f"positive selected quote rate `{row['positive_selected_quote_rate']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Concentration",
            "",
            f"- Moderate selected quote PnL: `{float(mod['selected_quote_pnl']):.0f}`.",
            f"- Top 1/5/10 percent quote contribution shares: `{concentration['top_1pct_pnl_share']:.3f}`, `{concentration['top_5pct_pnl_share']:.3f}`, `{concentration['top_10pct_pnl_share']:.3f}`.",
            "- Detailed NEW4 breakdown is in `new4_robustness_breakdown.csv` by session, spread bucket, and time bucket.",
            "",
            "The NEW4 stress evidence supports the spread/side controller as an external OOS proxy result, not as real exchange profitability.",
        ]
    )
    write_markdown(RESULTS / "non_ml_stress_robustness_report.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote NEW4 stress robustness outputs to {RESULTS}")

