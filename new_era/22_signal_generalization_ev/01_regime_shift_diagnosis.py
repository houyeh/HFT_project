from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_utils import PLOTS_DIR, RESULTS_DIR, ensure_dirs, psi, quote_rows_for_split


FEATURES = [
    "spread_ticks",
    "event_rate_hz",
    "volatility",
    "queue_imbalance",
    "score_015",
    "score_016",
    "ev_score_021",
    "toxic",
    "fill_proxy",
    "realized_quote_pnl_moderate",
]


def _summary_row(name: str, val: pd.Series, test: pd.Series) -> dict:
    return {
        "variable": name,
        "val_mean": float(np.nanmean(val)),
        "test_mean": float(np.nanmean(test)),
        "mean_diff_test_minus_val": float(np.nanmean(test) - np.nanmean(val)),
        "val_median": float(np.nanmedian(val)),
        "test_median": float(np.nanmedian(test)),
        "val_std": float(np.nanstd(val)),
        "test_std": float(np.nanstd(test)),
        "psi": psi(val.to_numpy(dtype=float), test.to_numpy(dtype=float)),
    }


def _plot_feature(name: str, val: pd.Series, test: pd.Series) -> None:
    vv = val.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    tt = test.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vv) == 0 or len(tt) == 0:
        return
    if name == "event_rate_hz":
        vv = np.log10(np.clip(vv, 1.0, np.nanpercentile(vv, 99.5)))
        tt = np.log10(np.clip(tt, 1.0, np.nanpercentile(tt, 99.5)))
        xlabel = "log10(event_rate_hz clipped)"
    else:
        lo = np.nanpercentile(np.concatenate([vv, tt]), 1)
        hi = np.nanpercentile(np.concatenate([vv, tt]), 99)
        vv = np.clip(vv, lo, hi)
        tt = np.clip(tt, lo, hi)
        xlabel = name
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vv, bins=60, density=True, alpha=0.55, label="validation")
    ax.hist(tt, bins=60, density=True, alpha=0.55, label="test_diagnostic")
    ax.set_title(f"Val/Test Distribution: {name}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"distribution_{name}.png", dpi=150)
    plt.close(fig)


def run() -> None:
    ensure_dirs()
    val = quote_rows_for_split("val")
    test = quote_rows_for_split("test")
    rows = []
    for feature in FEATURES:
        rows.append(_summary_row(feature, val[feature], test[feature]))
        _plot_feature(feature, val[feature], test[feature])

    failed_region_val = val[(val["spread_ticks"] <= 2.0) & (val["ev_score_021"] >= 0.0)]
    failed_region_test = test[(test["spread_ticks"] <= 2.0) & (test["ev_score_021"] >= 0.0)]
    region_rows = [
        {
            "variable": "021_selected_region",
            "val_mean": float(failed_region_val["realized_quote_pnl_moderate"].mean()),
            "test_mean": float(failed_region_test["realized_quote_pnl_moderate"].mean()),
            "mean_diff_test_minus_val": float(failed_region_test["realized_quote_pnl_moderate"].mean() - failed_region_val["realized_quote_pnl_moderate"].mean()),
            "val_sample_count": int(len(failed_region_val)),
            "test_sample_count": int(len(failed_region_test)),
            "val_total_stress_quote_pnl": float(failed_region_val["realized_quote_pnl_moderate"].sum()),
            "test_total_stress_quote_pnl": float(failed_region_test["realized_quote_pnl_moderate"].sum()),
            "interpretation": "Region quoted by failed 021 EV_s2_lam5_thr0 policy.",
        }
    ]
    summary = pd.DataFrame(rows).sort_values("psi", ascending=False)
    summary.to_csv(RESULTS_DIR / "regime_shift_summary.csv", index=False)
    summary[["variable", "psi"]].to_csv(RESULTS_DIR / "feature_distribution_psi.csv", index=False)
    pd.DataFrame(region_rows).to_csv(RESULTS_DIR / "failed_021_region_val_test.csv", index=False)

    top = summary.head(5)
    lines = [
        "# Regime Shift Diagnosis",
        "",
        "The current test set is treated as a burned diagnostic set, not as a pristine holdout for tuning.",
        "",
        "## Largest PSI Variables",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['variable']}: PSI `{row['psi']:.4f}`, val mean `{row['val_mean']:.6g}`, test mean `{row['test_mean']:.6g}`.")
    fr = region_rows[0]
    lines.extend(
        [
            "",
            "## Failed 021 Region",
            "",
            f"The EV_s2_lam5_thr0 quoted region has validation avg stress quote PnL `{fr['val_mean']:.6f}` and test avg stress quote PnL `{fr['test_mean']:.6f}`.",
            f"Total proxy stress quote PnL changes from `{fr['val_total_stress_quote_pnl']:.0f}` on validation quotes to `{fr['test_total_stress_quote_pnl']:.0f}` on diagnostic test quotes.",
        ]
    )
    (RESULTS_DIR / "regime_shift_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'regime_shift_summary.csv'}")
