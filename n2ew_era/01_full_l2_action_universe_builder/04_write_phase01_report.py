from __future__ import annotations

import json
import pandas as pd

from common_phase01 import FEATURE_DRIFT_COLUMNS, HORIZONS, MAKER_FEE, POINT_VALUE, RESULTS_DIR, ROOT, DATA_DIR, write_markdown


def fmt_int(value: float) -> str:
    return f"{int(value):,}"


def main() -> None:
    counts = pd.read_csv(RESULTS_DIR / "action_universe_by_split.csv")
    labels = pd.read_csv(RESULTS_DIR / "sequential_proxy_label_summary.csv")
    causality = pd.read_csv(RESULTS_DIR / "causality_audit.csv")
    drift = pd.read_csv(RESULTS_DIR / "feature_drift_train_val_test.csv")
    config = json.loads((DATA_DIR / "build_config.json").read_text(encoding="utf-8"))

    total_raw = int(counts["raw_rows"].sum())
    total_actions = int(counts["candidate_action_rows"].sum())
    train_actions = int(counts[counts["split"].astype(str) == "train"].iloc[0]["candidate_action_rows"])
    old_rows = 820
    causal_cols = causality[causality["causal_or_future"].eq("causal")]["column_name"].tolist()
    future_cols = causality[causality["causal_or_future"].eq("future_label")]["column_name"].tolist()

    display_counts = counts[
        [
            "split",
            "dataset",
            "symbol",
            "raw_rows",
            "valid_book_rows",
            "quote_eligible_rows",
            "candidate_action_rows",
            "bid_actions",
            "ask_actions",
            "no_quote_actions",
            "candidate_action_match",
        ]
    ].copy()

    label_preview = labels[(labels["action_side"].isin(["bid", "ask"])) & (labels["horizon"].isin([1, 10, 50, 100]))].copy()
    new4_meta = config.get("new4_metadata", {})
    new4_loaded = bool(new4_meta.get("new4_loaded"))
    if new4_loaded:
        new4_note = (
            f"NEW4 was included in Phase 01 as split `new4` from `{config.get('source_new4_dir')}`. "
            f"The compatible raw L2 loader detected symbol `{new4_meta.get('symbol')}` with "
            f"`{new4_meta.get('raw_rows'):,}` raw rows and `{new4_meta.get('quote_eligible_rows'):,}` quote-eligible rows."
        )
    else:
        new4_note = (
            "NEW4 was not included in Phase 01 because the current source only contains train/val/test. "
            "It will be added in a separate external-OOS action-universe builder if the raw L2 format is compatible."
        )

    drift_preview = drift[drift["feature"].isin(FEATURE_DRIFT_COLUMNS)].copy()
    drift_preview = drift_preview[["split", "feature", "count", "mean", "std", "p05", "p50", "p95"]]

    lines = [
        "# Full L2 Action Universe Report",
        "",
        "This phase builds a broad full-action L2 universe for short-horizon sequential proxy labels. It does not train ML, tune strategies, use terminal quote EV as a target, or filter actions through future fill logic.",
        "",
        "## 1. Raw Rows vs Candidate Actions",
        "",
        f"Total raw rows loaded from the historical quote-eligible source lineage: `{fmt_int(total_raw)}`.",
        f"Total candidate actions created: `{fmt_int(total_actions)}`.",
        "",
        display_counts.to_markdown(index=False),
        "",
        "Every quote-eligible row emits exactly three candidate actions: `bid`, `ask`, and `no_quote`.",
        "",
        "## NEW4 Status",
        "",
        new4_note,
        "",
        "The NEW4 inclusion check used only raw L2 compatibility: valid 5-level book columns, finite positive bid/ask depth, positive spread, and the same first/last tick boundary removal. It did not use future fill logic or any strategy threshold.",
        "",
        "## 2. Bottleneck Check",
        "",
        f"The old residual ML train universe had `{old_rows:,}` rows. The new train full-action universe has `{train_actions:,}` candidate actions.",
        "This successfully avoids the old 820-row bottleneck and keeps the full quote-eligible decision surface available for later non-ML and ML research.",
        "",
        "## 3. Causal vs Future Columns",
        "",
        "Strictly causal/action-context columns are allowed as future feature candidates only if they are known at the quote decision time.",
        "",
        "Causal/action columns include:",
        "",
        ", ".join(f"`{c}`" for c in causal_cols[:80]),
        "",
        "Future label columns are not allowed as features:",
        "",
        ", ".join(f"`{c}`" for c in future_cols),
        "",
        "The generated future label set includes `negative_pnl_H* = pnl_H* < 0` for each horizon.",
        "",
        "The full column-level audit is in `results/causality_audit.csv`.",
        "",
        "## 4. Sequential Proxy PnL Formula",
        "",
        f"Point value: `{POINT_VALUE}`. Maker fee: `{MAKER_FEE}`.",
        "",
        "PnL unit note: `POINT_VALUE = 200.0` and `MAKER_FEE = 15.0` are inherited from prior project constants in `new_era/models_playaround_15/config.py` and `new_era/18_queue_aware/queue_stress_core.py`. The data symbols are Taiwan futures-style `TXFB6`/`TXFA6`; a 200 TWD point value corresponds to the project assumption for TXF-like contracts, not mini-TXF. The maker fee is treated as a one-way per-contract fee per quote action. The current project source does not verify whether this fee includes tax or the exact exchange/broker schedule. These are proxy PnL units inherited from project assumptions, not exchange-certified values.",
        "",
        "For horizon `H`:",
        "",
        "```text",
        "future_mid_H = mid_price[t + H]",
        "bid pnl_H = (future_mid_H - best_bid[t]) * point_value - maker_fee",
        "ask pnl_H = (best_ask[t] - future_mid_H) * point_value - maker_fee",
        "no_quote pnl_H = 0",
        "markout_ticks_H = side-specific markout divided by tick_size",
        "```",
        "",
        f"Horizons generated: `{config['horizons']}`.",
        "",
        "Rows near the end of a session can have missing `future_mid_H` for bid/ask labels when `t + H` does not exist. `no_quote` remains zero by definition.",
        "",
        "Phase 01 evaluates all possible bid/ask/no_quote actions using future-mid markout labels. It does not model fill probability or queue priority. This is intentional: Phase 01 builds a broad decision universe and avoids the old future-fill-conditioned residual bottleneck. Fill/execution realism will be handled in later phases.",
        "",
        "## 5. Feature Drift Snapshot",
        "",
        "Before Phase 02 policy search, the causal feature distributions were summarized across splits. The full table is in `results/feature_drift_train_val_test.csv`; if NEW4 is available it is included as external OOS context, not as a tuning split.",
        "",
        drift_preview.to_markdown(index=False),
        "",
        "## Label Snapshot",
        "",
        label_preview[
            [
                "split",
                "action_side",
                "horizon",
                "row_count",
                "pnl_mean",
                "pnl_median",
                "pnl_positive_rate",
                "pnl_negative_rate",
                "markout_ticks_mean",
                "markout_ticks_median",
            ]
        ].to_markdown(index=False),
        "",
        "## Figures",
        "",
        "- `figures/action_count_waterfall.png`",
        "- `figures/action_side_distribution.png`",
        "- `figures/horizon_label_distribution.png`",
        "- `figures/feature_distribution_snapshot.png`",
        "- `figures/feature_distribution_by_split.png`",
        "- `figures/old_residual_vs_new_universe.png`",
    ]
    write_markdown(ROOT / "FULL_L2_ACTION_UNIVERSE_REPORT.md", lines)
    print(f"Wrote {ROOT / 'FULL_L2_ACTION_UNIVERSE_REPORT.md'}")


if __name__ == "__main__":
    main()
