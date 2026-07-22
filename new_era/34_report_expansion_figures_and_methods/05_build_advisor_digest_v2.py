from __future__ import annotations

from pathlib import Path

from common_034 import ROOT, copy_if_exists, find_existing_digest, markdown_to_html, write_markdown


MODULES = [
    ("Baseline Method", "baseline_method_explanation.md"),
    ("Quote Bid/Ask Mechanics", "quote_bid_ask_explanation.md"),
    ("L1 vs L2 Data Usage", "l1_vs_l2_data_usage.md"),
    ("ML Filtering Waterfall", "ml_filtering_waterfall.md"),
    ("Table Column Definitions", "table_column_definitions.md"),
    ("Exact Matched Random", "exact_matched_random_explanation.md"),
    ("ML Results", "ml_model_results_explanation.md"),
    ("ML Feature Metadata", "ml_feature_metadata.md"),
    ("ML Comeback", "ml_comeback_future_work.md"),
    ("OOS and Stress Methodology", "oos_stress_methodology.md"),
    ("Queue-Aware Stress", "queue_aware_stress_explanation.md"),
    ("Quote EV Formula Audit", "quote_ev_formula_audit.md"),
    ("Oracle Upper Bound", "oracle_upper_bound_explanation.md"),
    ("Simulation Environment", "simulation_environment_and_limitations.md"),
    ("CPU vs FPGA Latency", "cpu_fpga_latency_plan.md"),
]


FIGURE_BLOCKS = [
    ("Baseline Mechanics", "figures/baseline_flowchart.png"),
    ("Quote Bid/Ask Mechanics", "figures/quote_bid_ask_diagram.png"),
    ("L1 vs L2 Book Usage", "figures/l1_l2_book_diagram.png"),
    ("ML Filtering Waterfall", "figures/ml_filtering_waterfall.png"),
    ("Spread Distribution", "figures/spread_distribution_by_split_side.png"),
    ("Rule Ablation", "figures/rule_ablation_pnl_bar.png"),
    ("ML vs Baseline", "figures/ml_vs_baseline_summary.png"),
    ("ML Classification Metrics", "figures/ml_model_family_metrics.png"),
    ("ML Validation Threshold Curves", "figures/ml_validation_threshold_curves.png"),
    ("ML Epoch Loss Curves", "figures/ml_epoch_loss_curves.png"),
    ("ML Epoch Metric Curves", "figures/ml_epoch_metric_curves.png"),
    ("ML Overfit Diagnostic", "figures/ml_overfit_diagnostic.png"),
    ("ML Confusion Matrices", "figures/ml_confusion_matrices.png"),
    ("ML Future Work Decision Tree", "figures/ml_future_work_decision_tree.png"),
    ("Queue Stress Heatmap", "figures/queue_stress_heatmap_summary.png"),
    ("LOSO Robustness", "figures/loso_robustness_bar.png"),
    ("LOSO Session Concentration", "figures/loso_session_concentration.png"),
    ("Adverse Stress Grid", "figures/adverse_stress_grid_summary.png"),
    ("Oracle Gap", "figures/oracle_vs_causal_gap.png"),
    ("Simulation Environment Table", "figures/simulation_environment_table.png"),
    ("Latency Path Diagram", "figures/latency_path_diagram.png"),
]


def read_module(path: Path) -> list[str]:
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    return [f"# Missing Module: {path.name}", "", f"Required module was not generated at `{path}`."]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    md_src, html_src = find_existing_digest()
    md_dst = ROOT / "Week14_HFT_FPGA_Advisor_Digest_v2.md"
    html_dst = ROOT / "Week14_HFT_FPGA_Advisor_Digest_v2.html"
    if md_src:
        base = md_src.read_text(encoding="utf-8", errors="replace")
        copy_if_exists(md_src, ROOT / "Week14_HFT_FPGA_Advisor_Digest_source_copy.md")
    else:
        base = (
            "# Week 14 HFT FPGA Advisor Digest\n\n"
            "Source `Week14_HFT_FPGA_Advisor_Digest.md` was not found in the workspace, so this v2 digest was rebuilt from the current audited artifacts.\n\n"
        )
    if html_src:
        copy_if_exists(html_src, ROOT / "Week14_HFT_FPGA_Advisor_Digest_source_copy.html")

    lines: list[str] = [base.rstrip(), "", "---", "", "# Week 14 Advisor Digest v2 Addendum", ""]
    lines.extend(
        [
            "This addendum is presentation-ready but deliberately conservative.",
            "",
            "Hard boundaries:",
            "",
            "- Do not claim real-world profitability.",
            "- Do not claim FPGA improves PnL.",
            "- Do not claim true queue replay.",
            "- Do not claim the controller beats exact-matched random quote selection; that test is saturated.",
            "",
            "## Figure Index",
            "",
        ]
    )
    for title, rel in FIGURE_BLOCKS:
        if (ROOT / rel).exists():
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    lines.extend(["## Explanatory Modules", ""])
    for _, module in MODULES:
        lines.extend(read_module(ROOT / module))
        lines.extend(["", "---", ""])

    write_markdown(md_dst, lines)
    html_dst.write_text(markdown_to_html("\n".join(lines), "Week 14 HFT FPGA Advisor Digest v2"), encoding="utf-8")


if __name__ == "__main__":
    main()
    print(f"Wrote Advisor Digest v2 to {ROOT}")
