from __future__ import annotations

import pandas as pd

from common_033 import PHASE032_RANDOM_SUMMARY, RESULTS, ROOT, ensure_dirs, write_markdown


def fmt_num(x: float) -> str:
    return f"{x:,.0f}"


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required result missing: {path}")
    return pd.read_csv(path)


def make_saturation_report() -> None:
    lines = [
        "# Matched Random Saturation Report",
        "",
        "The Phase 032 exact matched random quote test is non-informative for quote-selection skill.",
        "",
        "Reason: the tested policy is `bid_only_spread_ge_1`, and in the quote-eligible action universe every bid action has `spread_ticks >= 1`. Therefore the policy selects the entire exact universe `side=bid and spread>=1`.",
        "",
        "When the random baseline is constrained to the exact same candidate universe and must match the same session/spread-bucket counts, there are no alternative quotes left to sample. The random baseline is forced to select the same set as the policy, producing zero variance and empirical p-value `1.0`.",
        "",
        "**Correct interpretation:** this exact matched random result is a saturation diagnostic, not evidence that the controller beats matched random quote selection.",
        "",
        "**Forbidden wording:** do not claim that `bid_only_spread_ge_1` beats the exact-matched random quote baseline.",
    ]
    if PHASE032_RANDOM_SUMMARY.exists():
        phase032 = pd.read_csv(PHASE032_RANDOM_SUMMARY)
        lines.extend(["", "## Phase 032 Saturation Evidence", ""])
        for row in phase032.itertuples(index=False):
            lines.append(
                f"- `{row.phase031_split}`: selected `{int(row.model_selected_count)}`, exact candidate universe `{int(row.candidate_universe_count)}`, random std `{float(row.random_std):.6f}`, p-value `{float(row.empirical_p_value_random_ge_model):.3f}`."
            )
    write_markdown(ROOT / "matched_random_saturation_report.md", lines)


def main() -> None:
    ensure_dirs()
    make_saturation_report()
    randoms = read_csv("regime_vs_random_baselines.csv")
    ablation = read_csv("rule_ablation_baselines.csv")
    delay = read_csv("execution_delay_proxy_grid.csv")
    lagged = read_csv("lagged_book_decision_sensitivity.csv")
    timing = read_csv("strict_timing_rebuild_audit.csv")
    boundary = read_csv("strict_timing_boundary_audit.csv")
    source = read_csv("strict_timing_source_audit.csv")
    dedup = read_csv("dedup_independence_audit.csv")

    broad = randoms[randoms["scenario"] == "random_all_valid_match_session_spread_bucket"].copy()
    broad = broad.sort_values("baseline_minus_random_95p", ascending=False)
    core = ablation[ablation["policy"] == "bid_spread_ge_1"].copy()
    ask = ablation[ablation["policy"] == "ask_all"].copy()
    all_actions = ablation[ablation["policy"] == "all_actions"].copy()
    strict_c = timing[timing["setup"] == "C_previous_row_decision_delayed_entry"].copy()
    delay10 = delay[delay["delay_ticks"] == 10].copy()
    likely_dupes = dedup[dedup["classification"] == "likely_duplicate"].copy()

    safety_rows = [
        {
            "claim": "HLS-friendly deterministic comparator logic",
            "status": "ALLOWED",
            "supporting_evidence": "Core controller is side==bid and spread_ticks>=1; rule ablation confirms bid_all and bid_spread_ge_1 are equivalent in this universe.",
            "safe_wording": "The controller is HLS-friendly comparator logic under this proxy dataset.",
        },
        {
            "claim": "Proxy OOS robustness",
            "status": "ALLOWED_WITH_CAVEATS",
            "supporting_evidence": "Falsification metrics were applied to phase031 test/new4 and all rebuilt unused OOS groups. NEW/NEW1 duplication is audited separately.",
            "safe_wording": "The rule remains robust in simulator-derived proxy OOS checks, after separating duplicate-risk datasets.",
        },
        {
            "claim": "Strict previous-row timing feasibility",
            "status": "ALLOWED_WITH_CAVEATS",
            "supporting_evidence": f"Source no-future-token pass={bool(source.loc[0, 'passes_no_future_token_check'])}; session-boundary lag reset pass={bool(boundary['session_boundary_lag_reset_pass'].all())}.",
            "safe_wording": "A previous-row decision version can be evaluated without using current/future outcome columns.",
        },
        {
            "claim": "Beat exact-matched random quote selection",
            "status": "FORBIDDEN",
            "supporting_evidence": "The exact universe random baseline is saturated because the policy selects all bid+spread>=1 actions.",
            "safe_wording": "The exact-matched random quote baseline is non-informative due to saturation.",
        },
        {
            "claim": "Real-world profitability",
            "status": "FORBIDDEN",
            "supporting_evidence": "Evaluation uses simulator-derived quote EV and lacks true exchange queue replay/aggressor-side reconstruction.",
            "safe_wording": "Do not claim real-world profitability.",
        },
        {
            "claim": "FPGA PnL necessity",
            "status": "FORBIDDEN",
            "supporting_evidence": "Latency sensitivity motivates risk-control analysis but does not prove production FPGA necessity.",
            "safe_wording": "Frame FPGA only as a latency risk-control implementation candidate.",
        },
        {
            "claim": "True queue priority replay",
            "status": "FORBIDDEN",
            "supporting_evidence": "No order-id queue replay or true fill priority is available in this proxy evaluation.",
            "safe_wording": "Do not claim true exchange queue replay.",
        },
        {
            "claim": "NEW and NEW1 are independent OOS",
            "status": "FORBIDDEN_IF_DEDUP_FLAGS_DUPLICATE",
            "supporting_evidence": "Dedup audit checks price-sequence hashes; likely duplicates must not be summed as independent evidence.",
            "safe_wording": "Report NEW and NEW1 separately unless dedup classifies them independent.",
        },
    ]
    pd.DataFrame(safety_rows).to_csv(RESULTS / "corrected_claim_safety_table.csv", index=False)

    lines = [
        "# Corrected Final Falsification Audit Report",
        "",
        "## 1. Correction Scope",
        "",
        "Phase 033 corrects the portions of Phase 032 that were too easy to over-interpret. The exact matched random quote test is saturated and non-informative because `bid_only_spread_ge_1` selects every action in the exact candidate universe `side=bid and spread>=1`.",
        "",
        "This report therefore separates four concepts: exact-universe random saturation, broader regime-vs-random baselines, rule ablations, and stricter timing/proxy-delay stress.",
        "",
        "## 2. Matched Random Saturation",
        "",
        "The exact matched random baseline must not be used as an alpha claim. It is now documented in `matched_random_saturation_report.md` as a saturation diagnostic only.",
        "",
        "## 3. Broader Regime-vs-Random Baselines",
        "",
        "Broader random baselines restore discriminative power by sampling from all valid bid/ask actions while matching counts by session or session/spread bucket. These are diagnostic controls for side/regime selection, not replacements for true queue replay.",
        "",
    ]
    for row in broad.head(10).itertuples(index=False):
        lines.append(
            f"- `{row.eval_group}` vs `{row.scenario}`: baseline PnL `{fmt_num(row.baseline_pnl)}`, random 95p `{fmt_num(row.random_95p)}`, gap `{fmt_num(row.baseline_minus_random_95p)}`, p-value `{row.empirical_p_value_random_ge_baseline:.4f}`."
        )
    lines.extend(
        [
            "",
            "## 4. Rule Ablation",
            "",
            "The ablation confirms the core rule is effectively `bid_all` in the current quote-eligible universe, because all candidate spreads are at least one tick. The edge should therefore be described as a bid-side regime effect under the proxy, not as proof that the spread threshold alone creates alpha.",
            "",
        ]
    )
    for df_name, df in [("core", core), ("ask_all", ask), ("all_actions", all_actions)]:
        if not df.empty:
            total = df["selected_quote_pnl"].sum()
            lines.append(f"- `{df_name}` aggregate across reported groups: `{fmt_num(total)}` proxy PnL.")
    lines.extend(
        [
            "",
            "## 5. Timing and Delay Rebuild",
            "",
            "The lagged-book test is now labeled as a timing-artifact stress: it changes decision features but does not recalculate delayed execution value. The new execution-delay proxy does recalculate value from the delayed entry row.",
            "",
        ]
    )
    for row in strict_c.head(12).itertuples(index=False):
        lines.append(
            f"- Strict C `{row.eval_group}`: delayed-entry proxy PnL `{fmt_num(row.selected_quote_pnl)}`, selected `{int(row.selected_quote_count)}`, missing delayed entries `{int(row.missing_delayed_entry_count)}`."
        )
    lines.append("")
    for row in delay10.head(12).itertuples(index=False):
        lines.append(
            f"- Delay 10 `{row.eval_group}`: proxy PnL `{fmt_num(row.proxy_entry_pnl)}`, gap vs delay0 `{fmt_num(row.gap_vs_delay0)}`, effective delayed entries `{int(row.effective_entry_count)}` of `{int(row.decision_selected_count)}`."
        )
    lines.extend(
        [
            "",
            "Delay caveat: the available action universe is a sparse set of quote-candidate rows, not a full raw-tick EV table. When `local_row_id + delay` is not present in the action universe, the script records a missing delayed entry and contributes zero proxy PnL. This is a conservative proxy-execution stress, not a true queue replay.",
        ]
    )
    lines.extend(
        [
            "",
            "## 6. NEW/NEW1/NEW3/NEW4 Dedup and Independence",
            "",
        ]
    )
    if likely_dupes.empty:
        lines.append("No pair was classified as `likely_duplicate`; still, all OOS groups should be reported with source provenance.")
    else:
        for row in likely_dupes.itertuples(index=False):
            lines.append(
                f"- `{row.data_dir_left}` vs `{row.data_dir_right}` classified `{row.classification}` with `{int(row.identical_price_sequence_hash_count)}` identical price-sequence hashes out of `{int(row.common_symbol_session_count)}` common symbol/session pairs."
            )
        lines.append("")
        lines.append("Do not sum likely-duplicate OOS PnL as independent evidence.")
    lines.extend(
        [
            "",
            "## 7. All Unused OOS Coverage",
            "",
            "The corrected falsification scripts include every rebuilt unused-OOS group available from NEW, NEW1, NEW3, and NEW4, in addition to the Phase031 frozen test/new4 rows. NEW and NEW1 are reported separately and interpreted through the dedup audit.",
            "",
            "## 8. Corrected Claim Boundary",
            "",
            "Allowed claims: HLS-friendly comparator logic and proxy OOS robustness with duplicate-data caveats.",
            "",
            "Forbidden claims: real-world profitability, FPGA PnL necessity, true queue replay, and beating exact-matched random selection.",
            "",
            "Conservative conclusion: `bid_only_spread_ge_1` remains a simple, auditable, FPGA-friendly proxy controller candidate, but the corrected audit narrows the claim to proxy robustness and side-regime behavior. The exact matched-random result is saturated and cannot support a quote-selection skill claim.",
        ]
    )
    write_markdown(ROOT / "CORRECTED_FINAL_AUDIT_REPORT.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote corrected final audit report to {ROOT}")
