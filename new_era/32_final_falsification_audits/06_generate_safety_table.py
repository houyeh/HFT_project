from __future__ import annotations

import pandas as pd

from common_final32 import PROJECT_ROOT, RESULTS, ROOT, ensure_dirs, write_markdown


PHASE033_ROOT = PROJECT_ROOT / "new_era" / "33_corrected_final_falsification_audits"
PHASE033_RESULTS = PHASE033_ROOT / "results"


def fmt(v: float) -> str:
    return f"{float(v):,.0f}"


def main() -> None:
    ensure_dirs()
    no_look = pd.read_csv(RESULTS / "no_lookahead_audit.csv")
    delay = pd.read_csv(RESULTS / "delay_sensitivity_grid.csv")
    random = pd.read_csv(RESULTS / "matched_random_quote_summary.csv")
    stress = pd.read_csv(RESULTS / "adverse_stress_grid.csv")
    loso = pd.read_csv(RESULTS / "loso_robustness_metrics.csv")

    source_ok = not bool(no_look["source_contains_forbidden_future_tokens"].astype(bool).any())
    boundary_ok = int(no_look["session_boundary_bad_decision_count"].fillna(0).sum()) == 0
    current = no_look[no_look["lag_ticks"] == 0]
    current_match_ok = int(current["mismatch_vs_stored_baseline_count"].fillna(0).sum()) == 0
    stress_min = stress.groupby("phase031_split")["stress_pnl"].min().reset_index()
    loso_worst = loso[loso["dropped_session"] != "NONE_FULL_SAMPLE"].sort_values("pnl_without_session").groupby("phase031_split").head(1)
    exact_random_saturated = bool(random["zero_variance_due_to_saturated_exact_universe"].astype(bool).all())
    corrected_delay_path = PHASE033_RESULTS / "execution_delay_proxy_grid.csv"
    corrected_dedup_path = PHASE033_RESULTS / "dedup_independence_audit.csv"
    corrected_random_path = PHASE033_RESULTS / "regime_vs_random_baselines.csv"
    corrected_delay_exists = corrected_delay_path.exists()
    corrected_dedup_exists = corrected_dedup_path.exists()
    corrected_random_exists = corrected_random_path.exists()

    rows = [
        {
            "claim": "HLS-friendly deterministic comparator logic",
            "status": "ALLOWED",
            "evidence": "Decision is side==bid and spread_ticks>=1; no floating point model or dynamic state needed.",
            "safe_wording": "The controller is HLS-friendly comparator logic under this proxy dataset.",
        },
        {
            "claim": "No future columns in decision logic",
            "status": "ALLOWED" if source_ok and current_match_ok and boundary_ok else "LIMITED",
            "evidence": f"future-token audit pass={source_ok}; current-row mismatch pass={current_match_ok}; boundary pass={boundary_ok}.",
            "safe_wording": "Decision logic is audited as not using quote_ev/future columns.",
        },
        {
            "claim": "External OOS proxy robustness",
            "status": "ALLOWED_WITH_PROXY_CAVEAT",
            "evidence": "Frozen test and NEW4 remain positive under the base proxy; Phase 033 extends checks to all rebuilt unused OOS and flags NEW/NEW1 duplication risk.",
            "safe_wording": "The rule shows simulator-derived proxy robustness, with duplicate-data caveats.",
        },
        {
            "claim": "Phase 032 delay table as true execution-delay proxy",
            "status": "FORBIDDEN_AS_WRITTEN",
            "evidence": "Phase 032 delay grid only lagged decision features and did not recalculate delayed-entry EV; Phase 033 supplies corrected execution_delay_proxy_grid.csv.",
            "safe_wording": "Treat the Phase 032 delay table as lagged-book timing-artifact stress only.",
        },
        {
            "claim": "Exact-matched random quote-selection edge",
            "status": "FORBIDDEN",
            "evidence": f"Exact matched random saturated={exact_random_saturated}; selected count equals candidate universe count for side=bid and spread>=1.",
            "safe_wording": "The exact-matched random quote baseline is non-informative due to saturation.",
        },
        {
            "claim": "Broader regime-vs-random diagnostic",
            "status": "ALLOWED_DIAGNOSTIC" if corrected_random_exists else "MISSING_CORRECTED_AUDIT",
            "evidence": f"Corrected Phase 033 broader random baselines available={corrected_random_exists}.",
            "safe_wording": "Use broader all-valid-action random controls only as diagnostic evidence of side/regime behavior.",
        },
        {
            "claim": "NEW and NEW1 as independent OOS",
            "status": "FORBIDDEN_IF_DEDUP_FLAGS_DUPLICATE",
            "evidence": f"Corrected Phase 033 dedup audit available={corrected_dedup_exists}; it must be checked before summing OOS PnL.",
            "safe_wording": "Do not sum NEW and NEW1 as independent OOS if dedup flags likely duplication.",
        },
        {
            "claim": "Real-world profitability",
            "status": "FORBIDDEN",
            "evidence": "Data lacks true queue priority, aggressor side, and exchange order-level replay.",
            "safe_wording": "Do not claim real-world profitability.",
        },
        {
            "claim": "FPGA PnL necessity",
            "status": "FORBIDDEN",
            "evidence": "Latency sensitivity is not proof that FPGA deployment is necessary for production PnL.",
            "safe_wording": "Frame FPGA only as a risk-control/latency candidate.",
        },
        {
            "claim": "True queue priority replay",
            "status": "FORBIDDEN",
            "evidence": "The evaluation uses simulator-derived quote EV, not order-id queue replay.",
            "safe_wording": "Do not claim true exchange queue priority replay.",
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "final_claim_safety_table.csv", index=False)

    lines = [
        "# Corrected Final Claim Safety Table",
        "",
        "This file overwrites the earlier Phase 032 interpretation with the Phase 033 corrected interpretation.",
        "",
        "The key correction is that the exact matched random quote baseline is saturated and non-informative: `bid_only_spread_ge_1` selects the entire exact candidate universe `side=bid and spread>=1`.",
        "",
        "## Audit Summary",
        "",
    ]
    for _, row in current.iterrows():
        lines.append(f"- `{row['phase031_split']}` current-row baseline PnL `{fmt(row['selected_quote_pnl'])}`, selected `{int(row['selected_count'])}`.")
    for _, row in delay[delay["delay_ticks"].isin([0, 1, 2, 5, 10])].iterrows():
        lines.append(
            f"- Phase 032 lagged-book timing-artifact delay `{int(row['delay_ticks'])}` `{row['phase031_split']}` PnL `{fmt(row['selected_quote_pnl'])}`, "
            f"gap vs delay0 `{fmt(row['gap_vs_delay0'])}`; this is not delayed-entry EV."
        )
    for _, row in random.iterrows():
        lines.append(
            f"- Exact matched random `{row['phase031_split']}` p-value `{row['empirical_p_value_random_ge_model']:.3f}`, "
            f"zero-variance exact universe `{bool(row['zero_variance_due_to_saturated_exact_universe'])}`; non-informative saturation diagnostic only."
        )
    for _, row in stress_min.iterrows():
        lines.append(f"- Worst stress-grid PnL `{row['phase031_split']}`: `{fmt(row['stress_pnl'])}`.")
    for _, row in loso_worst.iterrows():
        lines.append(
            f"- Worst LOSO `{row['phase031_split']}` drops `{row['dropped_session']}`: remaining PnL `{fmt(row['pnl_without_session'])}`, "
            f"dropped session contribution `{fmt(row['dropped_session_pnl'])}`."
        )
    lines.extend(["", "## Claim Table", ""])
    for row in rows:
        lines.append(f"- `{row['claim']}`: `{row['status']}`. {row['safe_wording']} Evidence: {row['evidence']}")
    lines.extend(
        [
            "",
            "## Corrected Phase 033 Artifacts",
            "",
            f"- Broader random controls available: `{corrected_random_exists}` at `{corrected_random_path}`.",
            f"- Corrected delayed-entry proxy grid available: `{corrected_delay_exists}` at `{corrected_delay_path}`.",
            f"- NEW/NEW1 dedup audit available: `{corrected_dedup_exists}` at `{corrected_dedup_path}`.",
            "",
            "## Final Boundary",
            "",
            "Allowed claims: HLS-friendly deterministic logic and proxy OOS robustness with duplicate-data caveats.",
            "",
            "Forbidden claims: real-world profitability, FPGA PnL necessity, true queue priority replay, true execution-delay proof from the old Phase 032 delay table, and beating exact-matched random quote selection.",
        ]
    )
    write_markdown(ROOT / "FINAL_CLAIM_SAFETY_TABLE.md", lines)
    write_markdown(
        ROOT / "matched_random_saturation_report.md",
        [
            "# Matched Random Saturation Report",
            "",
            "The Phase 032 exact matched random quote test is non-informative for quote-selection skill.",
            "",
            "Reason: `bid_only_spread_ge_1` selects the entire exact universe `side=bid and spread>=1`. A random baseline constrained to the same exact universe and same session/spread-bucket counts has no alternative quotes to sample.",
            "",
            "Correct interpretation: zero variance and p-value `1.0` mean saturation, not a win over matched random.",
            "",
            "Forbidden wording: do not claim that `bid_only_spread_ge_1` beats the exact-matched random quote baseline.",
        ],
    )


if __name__ == "__main__":
    main()
    print(f"Wrote final claim safety table to {ROOT / 'FINAL_CLAIM_SAFETY_TABLE.md'}")
