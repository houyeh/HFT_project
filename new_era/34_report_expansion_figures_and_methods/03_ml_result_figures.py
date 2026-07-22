from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common_034 import FIGURES, RESULTS, V28, V30, ensure_dirs, fmt_money, read_csv, save_placeholder_figure, set_style, write_markdown


MODEL_FAMILY_COLORS = {
    "Linear Models": "#276ef1",
    "Tree Boosting": "#2f9e44",
    "Neural Networks": "#c43c35",
    "Sequence Models": "#7a4cc2",
    "Other": "#6b7280",
}


def model_family(model_name: str) -> str:
    name = str(model_name).lower()
    if "logistic" in name or "ridge" in name:
        return "Linear Models"
    if "hist_gradient" in name or "hgb" in name or "boost" in name:
        return "Tree Boosting"
    if "tcn" in name:
        return "Sequence Models"
    if "mlp" in name or "torch" in name:
        return "Neural Networks"
    return "Other"


def plot_ml_vs_baseline() -> None:
    final_path = V30 / "results" / "ml_incremental_value_final.csv"
    if not final_path.exists():
        save_placeholder_figure(FIGURES / "ml_vs_baseline_summary.png", "ML final value missing", f"Missing {final_path}")
        return
    df = read_csv(final_path)
    df.to_csv(RESULTS / "ml_incremental_value_final_snapshot.csv", index=False)
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.32
    model_family_label = model_family(df["model_name"].iloc[0]) if "model_name" in df.columns and len(df) else "Neural Networks"
    ml_color = MODEL_FAMILY_COLORS.get(model_family_label, MODEL_FAMILY_COLORS["Other"])
    ax.bar([i - width / 2 for i in x], df["baseline_pnl"], width=width, label="Non-ML baseline", color="#9ca3af")
    ax.bar([i + width / 2 for i in x], df["after_veto_pnl"], width=width, label=f"Baseline + ML veto ({model_family_label})", color=ml_color)
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, max(row.baseline_pnl, row.after_veto_pnl), f"delta {fmt_money(row.incremental_vs_baseline)}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["split"].astype(str).tolist())
    ax.set_ylabel("Proxy quote PnL")
    ax.set_title("Residual EV ML veto did not pass frozen-test / NEW4 incremental criteria")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_vs_baseline_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary() -> None:
    metrics_path = V30 / "results" / "model_family_metrics.csv"
    if not metrics_path.exists():
        save_placeholder_figure(FIGURES / "ml_model_family_metrics.png", "ML metrics missing", f"Missing {metrics_path}")
        return
    metrics = read_csv(metrics_path)
    metrics.to_csv(RESULTS / "model_family_metrics_snapshot.csv", index=False)
    clf = metrics[(metrics["objective"] == "classification_negative_ev") & metrics["split"].isin(["val", "test", "new4"])].copy()
    if clf.empty:
        save_placeholder_figure(FIGURES / "ml_model_family_metrics.png", "No classification rows", "model_family_metrics.csv had no val/test/new4 classification rows.")
        return
    clf["label"] = clf["model_name"].astype(str).str.replace("_", " ", regex=False) + "\n" + clf["split"].astype(str)
    clf["model_family"] = clf["model_name"].map(model_family)
    clf["color"] = clf["model_family"].map(MODEL_FAMILY_COLORS).fillna(MODEL_FAMILY_COLORS["Other"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(clf["label"], clf["pr_auc_negative_ev"], color=clf["color"].tolist())
    ax.axhline(clf["negative_ev_rate"].mean(), color="#c43c35", linestyle="--", lw=1.5, label="mean negative-EV rate")
    ax.set_ylabel("PR-AUC for negative EV")
    ax.set_title("Classification-only residual ML metrics: negative-EV PR-AUC by split")
    ax.tick_params(axis="x", rotation=45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=family) for family, color in MODEL_FAMILY_COLORS.items() if family in set(clf["model_family"])]
    handles.append(plt.Line2D([0], [0], color="#c43c35", linestyle="--", label="mean negative-EV rate"))
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_model_family_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve_or_placeholder() -> None:
    curves_path = V30 / "results" / "model_family_validation_threshold_curves.csv"
    if not curves_path.exists():
        save_placeholder_figure(
            FIGURES / "ml_training_curves_placeholder.png",
            "Training curves unavailable",
            "No epoch-level training curve file was found in Phase 28 or 30.",
        )
        write_markdown(RESULTS / "ml_training_curves_missing.md", [f"Missing threshold/training source: `{curves_path}`."])
        return
    curves = read_csv(curves_path)
    curves.to_csv(RESULTS / "model_family_validation_threshold_curves_snapshot.csv", index=False)
    top_models = curves["model_name"].drop_duplicates().head(5).tolist()
    plot = curves[curves["model_name"].isin(top_models)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, group in plot.groupby("model_name", sort=False):
        group = group.sort_values("veto_rate")
        ax.plot(group["veto_rate"], group["incremental_vs_baseline"], marker="o", markersize=2.8, lw=1.3, label=name.replace("_", " "))
    ax.axhline(0, color="#111827", lw=1)
    ax.axvline(0.05, color="#c43c35", linestyle="--", lw=1.2, label="5% max veto target")
    ax.set_xlabel("Validation veto rate")
    ax.set_ylabel("Validation incremental PnL vs baseline")
    ax.set_title("Validation threshold curves: gains did not transfer to frozen test/NEW4")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_validation_threshold_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    loss_files = list((V28 / "results").glob("*loss*curve*.csv")) + list((V30 / "results").glob("*loss*curve*.csv"))
    if not loss_files:
        save_placeholder_figure(
            FIGURES / "ml_training_curves_placeholder.png",
            "Epoch training curves unavailable",
            "No epoch-level loss history was produced by the earlier residual ML phases. Validation threshold curves are provided instead.",
        )
        write_markdown(
            RESULTS / "ml_training_curves_missing.md",
            [
                "# ML Training Curves Missing",
                "",
                "No epoch-level loss history file was found in Phase 28 or Phase 30 results.",
                "",
                "Generated substitute figure: `figures/ml_validation_threshold_curves.png`.",
            ],
        )


def write_explanation() -> None:
    final = read_csv(V30 / "results" / "ml_incremental_value_final.csv") if (V30 / "results" / "ml_incremental_value_final.csv").exists() else pd.DataFrame()
    rows = []
    if not final.empty:
        for row in final.itertuples(index=False):
            rows.append(
                f"- `{row.split}`: baseline `{fmt_money(row.baseline_pnl)}`, after ML veto `{fmt_money(row.after_veto_pnl)}`, incremental `{fmt_money(row.incremental_vs_baseline)}`, accepted useful ML `{bool(row.accepted_useful_ml)}`."
            )
    else:
        rows.append("- Final ML incremental CSV missing; see placeholder figure.")
    write_markdown(
        RESULTS / "ml_result_key_numbers.md",
        [
            "# ML Result Key Numbers",
            "",
            *rows,
            "",
            "The safe conclusion is that residual ML did not pass the incremental value hurdle. It should remain a future-work route, not part of the final core controller.",
            "",
            "Epoch-level training curves are shown as a placeholder because earlier Phase 28/30 scripts did not save epoch logs. The validation threshold curve is provided as the available training-selection diagnostic.",
        ],
    )


def main() -> None:
    ensure_dirs()
    set_style()
    plot_ml_vs_baseline()
    plot_metric_summary()
    plot_threshold_curve_or_placeholder()
    write_explanation()


if __name__ == "__main__":
    main()
    print(f"Wrote ML figures to {FIGURES}")
