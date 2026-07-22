from __future__ import annotations

import pandas as pd

from common_phase02 import RESULTS_DIR, ROOT, read_json, write_markdown


def safe_read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    selected = read_json(RESULTS_DIR / "selected_phase02_policy.json")
    main = selected["main_policy"]
    grid = safe_read_csv("baseline_grid_all_splits.csv")
    frozen = safe_read_csv("frozen_baseline_test_new4.csv")
    random_summary = safe_read_csv("random_baseline_summary.csv")
    loso = safe_read_csv("loso_robustness_by_policy.csv")

    main_id = main["policy_id"]
    horizon = int(main["selected_horizon"])
    val_row = grid[
        (grid["split"].astype(str) == "val")
        & (grid["policy_id"].astype(str) == main_id)
        & (grid["horizon"].astype(int) == horizon)
    ].copy()

    random_matched = random_summary[random_summary["random_baseline"].astype(str).eq("random_quote_matched_count")].copy()
    main_frozen = frozen[frozen["role"].astype(str).eq("main")].copy() if not frozen.empty else pd.DataFrame()
    main_hurdles = main_frozen.merge(
        random_matched[["split", "random_95p", "random_99p", "empirical_p_value_random_ge_model", "beats_random_95p"]],
        on="split",
        how="left",
    ) if not main_frozen.empty and not random_matched.empty else main_frozen

    loso_main = loso[(loso["policy_id"].astype(str) == main_id) & (loso["horizon"].astype(int) == horizon)].copy() if not loso.empty else pd.DataFrame()
    loso_summary = (
        loso_main.groupby("split", as_index=False)
        .agg(
            full_sample_pnl=("full_sample_pnl", "first"),
            min_loso_pnl=("loso_pnl", "min"),
            positive_loso_count=("loso_positive", "sum"),
            session_count=("dropped_session_id", "nunique"),
        )
    ) if not loso_main.empty else pd.DataFrame()

    lines = [
        "# Short-Horizon Sequential Proxy Baseline Report",
        "",
        "Phase 02 evaluates deterministic, non-ML quote-control baselines on the Phase 01 full L2 action universe. No ML was trained, no test/NEW4 tuning was performed, and every state maps to at most one action: `bid`, `ask`, or `no_quote`.",
        "",
        "## 1. Validation-Selected Policy",
        "",
        f"Best main policy selected strictly on validation: `{main_id}` at horizon `H{horizon}`.",
        f"Selection split: `{selected['selection_split']}`. Test used for selection: `{selected['test_used_for_selection']}`. NEW4 used for selection: `{selected['new4_used_for_selection']}`.",
        "",
        "Selection constraints:",
        "",
        pd.DataFrame([selected["selection_constraints"]]).to_markdown(index=False),
        "",
        "Validation metrics for the selected policy:",
        "",
        val_row[
            [
                "split",
                "horizon",
                "policy_id",
                "total_pnl",
                "selected_action_count",
                "selection_rate",
                "win_rate",
                "max_drawdown",
                "avg_pnl_per_selected_action",
            ]
        ].to_markdown(index=False) if not val_row.empty else "Validation row missing.",
        "",
        "Backup policies:",
        "",
        pd.DataFrame(selected.get("backup_policies", [])).reindex(
            columns=["role", "policy_id", "selected_horizon", "policy_type", "family", "threshold"]
        ).to_markdown(index=False) if selected.get("backup_policies") else "No backup policies selected.",
        "",
        "## 2. Frozen Test and NEW4",
        "",
        "The selected policies were evaluated on `test` and `new4` without changing thresholds, horizons, policy family, or weights.",
        "",
        frozen.to_markdown(index=False) if not frozen.empty else "Frozen evaluation is missing.",
        "",
        "## 3. Baseline and Random Hurdles",
        "",
        "Main-policy hurdle answer:",
        "",
        main_hurdles.to_markdown(index=False) if not main_hurdles.empty else "Main-policy hurdle table is missing.",
        "",
        "The matched-count random hurdle is evaluated only for the validation-selected main policy. Backup policies are shown in the frozen table but are not used to rescue the main claim.",
        "",
        random_summary.to_markdown(index=False) if not random_summary.empty else "Random baseline summary is missing.",
        "",
        "## 4. Leave-One-Session-Out Robustness",
        "",
        loso_summary.to_markdown(index=False) if not loso_summary.empty else "LOSO robustness summary is missing.",
        "",
        "## 5. Interpretation",
        "",
        "This phase is a short-horizon non-ML baseline audit. Positive validation PnL is allowed to select a deterministic policy, but frozen test and NEW4 decide whether the rule transfers.",
        "",
        "The reported PnL is short-horizon sequential proxy PnL based on future-mid markout labels. It is not true exchange replay, does not model queue priority, and does not prove real-world profitability.",
        "",
        "## 6. Output Files",
        "",
        "- `results/baseline_grid_all_splits.csv`",
        "- `results/selected_phase02_policy.json`",
        "- `results/frozen_baseline_test_new4.csv`",
        "- `results/random_baseline_summary.csv`",
        "- `results/loso_robustness_by_policy.csv`",
        "- `figures/validation_policy_pnl_heatmap.png`",
        "- `figures/validation_family_pnl_heatmap.png`",
        "- `figures/selected_policy_equity_curves.png`",
        "- `figures/selected_policy_drawdown_curves.png`",
        "- `figures/random_baseline_distributions.png`",
    ]
    write_markdown(ROOT / "SHORT_HORIZON_BASELINE_REPORT.md", lines)
    print(f"Wrote {ROOT / 'SHORT_HORIZON_BASELINE_REPORT.md'}")


if __name__ == "__main__":
    main()
