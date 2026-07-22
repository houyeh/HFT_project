"""Generate charts and markdown assets for the CME validation update."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "artifacts" / "cme_eval_transfer" / "20260407_161938"
OUTPUT_DIR = REPO_ROOT / "meeting" / "w7" / "w6_result"


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


COLORS = {
    "raw": "#5875A4",
    "gated": "#CC8963",
    "es": "#5F9E6E",
    "cl": "#C85C5C",
}


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_summary() -> dict:
    with (SOURCE_DIR / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_overview(summary: dict) -> None:
    raw_metrics = summary["raw_metrics"]
    gated_metrics = summary["gated_metrics"]
    raw_signal = summary["raw_signal_summary"]
    gated_signal = summary["gated_signal_summary"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metric_names = ["accuracy", "macro_f1"]
    metric_labels = ["Accuracy", "Macro F1"]
    axes[0, 0].bar(
        np.arange(len(metric_names)) - 0.18,
        [raw_metrics[name] for name in metric_names],
        width=0.36,
        color=COLORS["raw"],
        label="Raw",
    )
    axes[0, 0].bar(
        np.arange(len(metric_names)) + 0.18,
        [gated_metrics[name] for name in metric_names],
        width=0.36,
        color=COLORS["gated"],
        label="Gated",
    )
    axes[0, 0].set_xticks(np.arange(len(metric_names)), metric_labels)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Classification Metrics")
    axes[0, 0].legend(frameon=False)

    trade_names = ["trades", "trade_rate", "hit_rate"]
    trade_labels = ["Trades", "Trade Rate", "Hit Rate"]
    raw_trade_values = [
        raw_signal["trades"],
        raw_signal["trade_rate"],
        raw_signal["hit_rate"],
    ]
    gated_trade_values = [
        gated_signal["trades"],
        gated_signal["trade_rate"],
        gated_signal["hit_rate"],
    ]
    axes[0, 1].bar(
        np.arange(len(trade_names)) - 0.18,
        raw_trade_values,
        width=0.36,
        color=COLORS["raw"],
        label="Raw",
    )
    axes[0, 1].bar(
        np.arange(len(trade_names)) + 0.18,
        gated_trade_values,
        width=0.36,
        color=COLORS["gated"],
        label="Gated",
    )
    axes[0, 1].set_xticks(np.arange(len(trade_names)), trade_labels)
    axes[0, 1].set_title("Signal Trading Summary")
    axes[0, 1].legend(frameon=False)

    pnl_names = ["gross_pnl_ticks", "net_pnl_ticks"]
    pnl_labels = ["Gross PnL", "Net PnL"]
    axes[1, 0].bar(
        np.arange(len(pnl_names)) - 0.18,
        [raw_signal[name] for name in pnl_names],
        width=0.36,
        color=COLORS["raw"],
        label="Raw",
    )
    axes[1, 0].bar(
        np.arange(len(pnl_names)) + 0.18,
        [gated_signal[name] for name in pnl_names],
        width=0.36,
        color=COLORS["gated"],
        label="Gated",
    )
    axes[1, 0].axhline(0.0, color="#444444", linewidth=1)
    axes[1, 0].set_xticks(np.arange(len(pnl_names)), pnl_labels)
    axes[1, 0].set_title("Backtest PnL (ticks)")
    axes[1, 0].legend(frameon=False)

    sharpe_names = ["avg_net_pnl_ticks", "sharpe_like"]
    sharpe_labels = ["Avg Net PnL", "Sharpe-like"]
    axes[1, 1].bar(
        np.arange(len(sharpe_names)) - 0.18,
        [raw_signal[name] for name in sharpe_names],
        width=0.36,
        color=COLORS["raw"],
        label="Raw",
    )
    axes[1, 1].bar(
        np.arange(len(sharpe_names)) + 0.18,
        [gated_signal[name] for name in sharpe_names],
        width=0.36,
        color=COLORS["gated"],
        label="Gated",
    )
    axes[1, 1].axhline(0.0, color="#444444", linewidth=1)
    axes[1, 1].set_xticks(np.arange(len(sharpe_names)), sharpe_labels)
    axes[1, 1].set_title("Trade Quality")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("CME Logic20 Direct Validation Overview", fontsize=16, y=1.02)
    _save_figure(fig, "p01_cme_validation_overview.svg")


def plot_by_symbol() -> None:
    raw_df = pd.read_csv(SOURCE_DIR / "signal_backtest_raw_by_symbol.csv")
    gated_df = pd.read_csv(SOURCE_DIR / "signal_backtest_gated_by_symbol.csv")
    merged = raw_df.merge(gated_df, on="symbol", suffixes=("_raw", "_gated"))

    x = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(x - 0.18, merged["trades_raw"], width=0.36, color=COLORS["raw"], label="Raw")
    axes[0].bar(x + 0.18, merged["trades_gated"], width=0.36, color=COLORS["gated"], label="Gated")
    axes[0].set_xticks(x, merged["symbol"])
    axes[0].set_title("Trades by Symbol")
    axes[0].legend(frameon=False)

    axes[1].bar(x - 0.18, merged["net_pnl_ticks_raw"], width=0.36, color=COLORS["raw"], label="Raw")
    axes[1].bar(x + 0.18, merged["net_pnl_ticks_gated"], width=0.36, color=COLORS["gated"], label="Gated")
    axes[1].axhline(0.0, color="#444444", linewidth=1)
    axes[1].set_xticks(x, merged["symbol"])
    axes[1].set_title("Net PnL by Symbol (ticks)")
    axes[1].legend(frameon=False)

    fig.suptitle("Raw vs Gated Signal Backtest by Symbol", fontsize=16, y=1.04)
    _save_figure(fig, "p02_cme_validation_by_symbol.svg")


def plot_confusion_matrix(summary: dict) -> None:
    matrix = np.array(summary["gated_metrics"]["confusion_matrix"], dtype=float)
    labels = ["Down", "Stationary", "Up"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Gated Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="#222222")

    _save_figure(fig, "p03_cme_gated_confusion_matrix.svg")


def plot_data_volume() -> None:
    file_summary = pd.read_csv(SOURCE_DIR / "cme_file_summary.csv")
    agg = file_summary.groupby("symbol", as_index=False).agg(
        quotes=("quotes", "sum"),
        trades=("trades", "sum"),
        files=("file", "size"),
    )

    x = np.arange(len(agg))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(x - 0.18, agg["quotes"], width=0.36, color="#6CA6CD", label="Quotes")
    axes[0].bar(x + 0.18, agg["trades"], width=0.36, color="#E39C5F", label="Trades")
    axes[0].set_xticks(x, agg["symbol"])
    axes[0].set_title("Total CME Logic20 Volume")
    axes[0].legend(frameon=False)

    axes[1].bar(x, agg["files"], width=0.5, color="#88B04B")
    axes[1].set_xticks(x, agg["symbol"])
    axes[1].set_title("CSV Files Contributing to Each Symbol")

    fig.suptitle("Input Data Scale for Direct Validation", fontsize=16, y=1.04)
    _save_figure(fig, "p04_cme_input_data_scale.svg")


def plot_label_vs_prediction() -> None:
    labels = pd.read_csv(SOURCE_DIR / "label_distribution.csv")
    preds = pd.read_csv(SOURCE_DIR / "prediction_distribution.csv")

    def pivot_counts(df: pd.DataFrame, cls_col: str) -> pd.DataFrame:
        pivot = df.pivot(index="symbol", columns=cls_col, values="count").fillna(0.0)
        pivot = pivot.reindex(columns=[0, 1, 2], fill_value=0.0)
        pivot.columns = ["Down", "Stationary", "Up"]
        return pivot

    label_pivot = pivot_counts(labels.assign(y_true=labels["y_true"].astype(int)), "y_true")
    pred_pivot = pivot_counts(preds.assign(y_pred=preds["y_pred"].astype(int)), "y_pred")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    bottom = np.zeros(len(label_pivot))
    for column, color in zip(label_pivot.columns, ["#C85C5C", "#909090", "#5F9E6E"], strict=True):
        axes[0].bar(label_pivot.index, label_pivot[column], bottom=bottom, label=column, color=color)
        bottom += label_pivot[column].to_numpy(dtype=float)
    axes[0].set_title("True Label Distribution")

    bottom = np.zeros(len(pred_pivot))
    for column, color in zip(pred_pivot.columns, ["#C85C5C", "#909090", "#5F9E6E"], strict=True):
        axes[1].bar(pred_pivot.index, pred_pivot[column], bottom=bottom, label=column, color=color)
        bottom += pred_pivot[column].to_numpy(dtype=float)
    axes[1].set_title("Raw Prediction Distribution")
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle("Label vs Prediction Distribution by Symbol", fontsize=16, y=1.03)
    _save_figure(fig, "p05_cme_label_prediction_distribution.svg")


def main() -> None:
    _ensure_output_dir()
    summary = _load_summary()

    plot_overview(summary)
    plot_by_symbol()
    plot_confusion_matrix(summary)
    plot_data_volume()
    plot_label_vs_prediction()

    manifest = {
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "generated_files": [
            "p01_cme_validation_overview.svg",
            "p02_cme_validation_by_symbol.svg",
            "p03_cme_gated_confusion_matrix.svg",
            "p04_cme_input_data_scale.svg",
            "p05_cme_label_prediction_distribution.svg",
        ],
    }
    with (OUTPUT_DIR / "chart_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
