from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_034 import FIGURES, RESULTS, V33, ensure_dirs, fmt_money, read_csv, set_style, write_markdown


POLICY_MAP = {
    "bid_all": "Bid all",
    "ask_all": "Ask all",
    "all_actions": "Both sides",
    "bid_spread_ge_1": "Bid spread>=1",
}


def main() -> None:
    ensure_dirs()
    set_style()
    src = V33 / "results" / "rule_ablation_baselines.csv"
    if not src.exists():
        from common_034 import save_placeholder_figure

        save_placeholder_figure(FIGURES / "rule_ablation_pnl_bar.png", "Rule ablation missing", f"Missing {src}")
        write_markdown(RESULTS / "rule_ablation_missing.md", [f"Missing source file: `{src}`."])
        return

    df = read_csv(src)
    df = df[df["policy"].isin(POLICY_MAP)].copy()
    df["policy_label"] = df["policy"].map(POLICY_MAP)
    selected_groups = [
        "phase031_test",
        "phase031_new4",
        "unused_NEW3_TXFA6",
        "unused_NEW4_TXFA6",
    ]
    plot_df = df[df["eval_group"].isin(selected_groups)].copy()
    if plot_df.empty:
        plot_df = df.copy()
    plot_df.to_csv(RESULTS / "rule_ablation_plot_data.csv", index=False)

    pivot = plot_df.pivot_table(index="eval_group", columns="policy_label", values="selected_quote_pnl", aggfunc="sum").fillna(0.0)
    policy_order = ["Bid all", "Bid spread>=1", "Ask all", "Both sides"]
    pivot = pivot[[c for c in policy_order if c in pivot.columns]]
    x = range(len(pivot.index))
    width = 0.18
    colors = {
        "Bid all": "#276ef1",
        "Bid spread>=1": "#7aa6ff",
        "Ask all": "#c43c35",
        "Both sides": "#6b7280",
    }
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, col in enumerate(pivot.columns):
        offsets = [v + (i - (len(pivot.columns) - 1) / 2) * width for v in x]
        bars = ax.bar(offsets, pivot[col].values, width=width, label=col, color=colors.get(col, "#888"))
        for bar, val in zip(bars, pivot[col].values):
            if abs(val) > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val, fmt_money(val), ha="center", va="bottom" if val >= 0 else "top", fontsize=7, rotation=90)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("phase031_", "").replace("unused_", "") for s in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("Proxy quote PnL")
    ax.set_title("Rule Ablation: edge is bid-side regime behavior, not exact random skill")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "rule_ablation_pnl_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    agg = df.groupby("policy_label", observed=False)["selected_quote_pnl"].sum().reindex(policy_order)
    write_markdown(
        RESULTS / "rule_ablation_interpretation.md",
        [
            "# Rule Ablation Interpretation",
            "",
            f"Aggregate `Bid spread>=1` proxy PnL across available groups: `{fmt_money(float(agg.get('Bid spread>=1', 0.0)))}`.",
            f"Aggregate `Bid all` proxy PnL across available groups: `{fmt_money(float(agg.get('Bid all', 0.0)))}`.",
            f"Aggregate `Ask all` proxy PnL across available groups: `{fmt_money(float(agg.get('Ask all', 0.0)))}`.",
            "",
            "Because every quote-eligible bid action has spread >= 1, `Bid spread>=1` and `Bid all` are equivalent in this universe. The safe claim is bid-side regime filtering, not a precise spread-threshold alpha.",
        ],
    )


if __name__ == "__main__":
    main()
    print(f"Wrote rule ablation plot to {FIGURES}")
