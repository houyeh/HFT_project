from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common_034 import FIGURES, PROJECT_ROOT, RESULTS, ROOT, V32, ensure_dirs, fmt_money, read_csv, set_style, write_markdown


COMMON_028_DIR = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
if str(COMMON_028_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_028_DIR))

from common_028 import FEATURE_COLUMNS  # noqa: E402


FAMILY_COLORS = {
    "Linear Models (Logistic)": "#276ef1",
    "Tree Boosting (HGB)": "#2f9e44",
    "Neural Networks (MLP)": "#c43c35",
    "Sequence Models (TCN)": "#7a4cc2",
}


def plot_loso_concentration() -> None:
    path = V32 / "results" / "loso_robustness_metrics.csv"
    df = read_csv(path)
    plot = df[df["dropped_session"] != "NONE_FULL_SAMPLE"].copy()
    plot["label"] = plot["phase031_split"].astype(str) + "\n" + plot["dropped_session"].astype(str)
    x = np.arange(len(plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.8))
    full_bars = ax.bar(x - width / 2, plot["full_sample_pnl"], width=width, color="#9ca3af", label="Full-sample PnL")
    loso_colors = ["#276ef1" if v >= 0 else "#c43c35" for v in plot["pnl_without_session"]]
    loso_bars = ax.bar(x + width / 2, plot["pnl_without_session"], width=width, color=loso_colors, label="Leave-one-session-out PnL")
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["label"], rotation=35, ha="right")
    ax.set_ylabel("Proxy PnL")
    ax.set_title("Session Concentration: full sample vs leave-one-session-out")
    ax.legend()
    for i, row in enumerate(plot.itertuples(index=False)):
        full = float(row.full_sample_pnl)
        loso = float(row.pnl_without_session)
        gap = full - loso
        top = max(full, loso)
        bottom = min(full, loso)
        ax.plot([i + width / 2, i + width / 2], [bottom, top], color="#111827", lw=1.0, alpha=0.65)
        ax.text(i + width / 2 + 0.05, (top + bottom) / 2, f"drop contrib\n{fmt_money(gap)}", fontsize=7, va="center")
    for bars in [full_bars, loso_bars]:
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val, fmt_money(val), ha="center", va="bottom" if val >= 0 else "top", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(FIGURES / "loso_session_concentration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices() -> None:
    pred_path = RESULTS / "small_mlp_64_32_epoch_rerun_predictions.csv"
    frozen_path = RESULTS / "frozen_incremental_value_from_rerun.csv"
    pred = read_csv(pred_path)
    frozen = read_csv(frozen_path)
    threshold = float(frozen.iloc[0]["threshold"])
    splits = ["val", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    rows = []
    for ax, split in zip(axes, splits):
        g = pred[pred["phase034_split"].astype(str) == split].copy()
        y = g["negative_ev_moderate"].astype(int).to_numpy()
        yhat = (g["negative_ev_score"].astype(float).to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y, yhat, labels=[1, 0])
        # Rows are actual positive/negative, columns are predicted positive/negative.
        display = np.array([[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]])
        im = ax.imshow(display, cmap="Blues")
        ax.set_title(f"{split} | threshold={threshold:.6f}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred negative EV", "Pred safe"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual negative EV", "Actual safe"])
        for i in range(2):
            for j in range(2):
                label = [["TP", "FN"], ["FP", "TN"]][i][j]
                ax.text(j, i, f"{label}\n{display[i, j]}", ha="center", va="center", fontsize=11, color="#111827")
        rows.append(
            {
                "split": split,
                "threshold": threshold,
                "tp": int(display[0, 0]),
                "fn": int(display[0, 1]),
                "fp": int(display[1, 0]),
                "tn": int(display[1, 1]),
                "sample_count": int(len(g)),
                "predicted_positive_count": int(yhat.sum()),
                "actual_positive_count": int(y.sum()),
            }
        )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.80)
    fig.suptitle("Residual MLP confusion matrices: validation vs frozen test", y=1.02)
    fig.savefig(FIGURES / "ml_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "ml_confusion_matrices.csv", index=False)


def write_feature_metadata() -> None:
    l2_depth = ["queue_imbalance_calc", "deep_imbalance_calc", "top_depth", "depth_ratio"]
    rolling = ["volatility_past_50", "abs_return_past_50", "event_rate_past_50", "queue_imbalance_change_past"]
    spread = ["spread_ticks", "normalized_spread_calc"]
    lines = [
        "# ML Feature Metadata",
        "",
        "All residual ML families in Phase 28/30 used the same causal residual EV tabular feature set unless noted otherwise.",
        "",
        "## Shared Tabular Feature Set",
        "",
        *[f"- `{col}`" for col in FEATURE_COLUMNS],
        "",
        "## Feature Categories",
        "",
        f"- L2 depth features: {', '.join(f'`{x}`' for x in l2_depth)}.",
        f"- Imbalance ratios: `queue_imbalance_calc`, `deep_imbalance_calc`, `depth_ratio`.",
        f"- Rolling/lagged features: {', '.join(f'`{x}`' for x in rolling)}.",
        f"- Spread features: {', '.join(f'`{x}`' for x in spread)}.",
        "",
        "## Model Family Feature Table",
        "",
        "| Model family | Color | Input format | Uses L2 depth | Uses imbalance ratios | Uses rolling/lagged features | Sequence window | Feature columns |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| Linear Models (Logistic) | `{FAMILY_COLORS['Linear Models (Logistic)']}` | Tabular | Yes | Yes | Yes | None | {', '.join(f'`{x}`' for x in FEATURE_COLUMNS)} |",
        f"| Tree Boosting (HGB) | `{FAMILY_COLORS['Tree Boosting (HGB)']}` | Tabular | Yes | Yes | Yes | None | {', '.join(f'`{x}`' for x in FEATURE_COLUMNS)} |",
        f"| Neural Networks (MLP) | `{FAMILY_COLORS['Neural Networks (MLP)']}` | Tabular | Yes | Yes | Yes | None | {', '.join(f'`{x}`' for x in FEATURE_COLUMNS)} |",
        f"| Sequence Models (TCN) | `{FAMILY_COLORS['Sequence Models (TCN)']}` | Causal sequence | Yes | Yes | Yes | Default `50` rows | {', '.join(f'`{x}`' for x in FEATURE_COLUMNS)} over a past-only window |",
        "",
        "## Model Family Notes",
        "",
        "- Linear Models (Logistic): calibrated logistic classifier on the shared tabular feature set.",
        "- Tree Boosting (HGB): histogram gradient boosting classifier/regressor on the shared tabular feature set.",
        "- Neural Networks (MLP): sklearn/PyTorch MLPs, including `small_mlp_64_32_epoch_rerun`, on the shared tabular feature set.",
        "- Sequence Models (TCN): causal TCN in the exported RTX pack, using the same feature columns over a default causal window of `50` rows.",
        "",
        "## TCN Sequence Details",
        "",
        "- Script: `new_era/30_ml_training_adequacy_audit/export/ml_training_pack/05_tcn_sequence_training.py`.",
        "- Default window: `--window 50`.",
        "- Causal padding is used; the target row sees only historical rows in its window.",
        "- The full sequence tensor is not expanded into RAM; windows are sliced dynamically by the PyTorch Dataset.",
        "",
        "Conservative note: richer feature use does not imply deployable value. Residual ML still failed the frozen incremental-value hurdle.",
    ]
    write_markdown(ROOT / "ml_feature_metadata.md", lines)


def main() -> None:
    ensure_dirs()
    set_style()
    plot_loso_concentration()
    plot_confusion_matrices()
    write_feature_metadata()


if __name__ == "__main__":
    main()
    print(f"Wrote PM follow-up figures and metadata to {ROOT}")
