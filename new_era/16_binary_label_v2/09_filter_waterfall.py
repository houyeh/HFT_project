from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"


def _markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    headers = [str(c) for c in show.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in show.columns) + " |")
    return "\n".join(lines)


def build_waterfall(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    report = pd.read_csv(OUTPUTS / "quote_eligible_filter_report.csv")
    labels = pd.read_parquet(OUTPUTS / "binary_labels_v2.parquet")
    session_rows = report[report["scope"] == "session"].copy()

    rows = []
    for split, g in session_rows.groupby("split", sort=True):
        raw = int(g["raw_rows"].sum())
        invalid = int(g["dropped_invalid_bid_ask"].sum())
        spread = int(g["dropped_nonpositive_spread"].sum())
        depth = int(g["dropped_missing_depth"].sum())
        boundary = int(g["dropped_session_boundary"].sum())
        feature_na = int(g["dropped_feature_na"].sum())
        eligible = int(g["eligible_rows"].sum())
        after_invalid = raw - invalid
        after_spread = after_invalid - spread
        after_depth = after_spread - depth
        after_boundary = after_depth - boundary
        after_feature = after_boundary - feature_na
        rows.append(
            {
                "split": split,
                "raw_rows": raw,
                "after_valid_bid_ask": after_invalid,
                "after_positive_spread": after_spread,
                "after_complete_depth": after_depth,
                "after_session_boundary": after_boundary,
                "after_feature_finite": after_feature,
                "quote_eligible_rows": eligible,
                "dropped_invalid_bid_ask": invalid,
                "dropped_nonpositive_spread": spread,
                "dropped_missing_depth": depth,
                "dropped_session_boundary": boundary,
                "dropped_feature_na": feature_na,
                "retention_rate": eligible / max(raw, 1),
            }
        )

    wf = pd.DataFrame(rows)
    wf.to_csv(results_dir / "filter_waterfall_by_split.csv", index=False)

    label_rows = []
    for split, g in labels.groupby("split", sort=True):
        for side in ["bid", "ask"]:
            fill = g[f"{side}_fill_proxy"].astype(bool)
            toxic = g[f"{side}_toxic"].astype(bool)
            label_rows.append(
                {
                    "split": split,
                    "side": side,
                    "eligible_rows": int(len(g)),
                    "fill_proxy_count": int(fill.sum()),
                    "fill_proxy_rate": float(fill.mean()),
                    "toxic_count": int(toxic.sum()),
                    "toxic_rate": float(toxic.mean()),
                    "toxic_rate_given_fill": float((toxic & fill).sum() / max(fill.sum(), 1)),
                    "median_spread_ticks": float(g["spread_ticks"].median()),
                    "p95_spread_ticks": float(g["spread_ticks"].quantile(0.95)),
                }
            )
    label_df = pd.DataFrame(label_rows)

    lines = [
        "# Quote-Eligible Distribution Report",
        "",
        "The filter did not remove the hard/toxic event population; retention is nearly complete and the post-filter set still contains fill-proxy and toxic examples in every split.",
        "",
        "## Waterfall",
        "",
        _markdown_table(wf),
        "",
        "## Toxic and Fill Proxy Distribution After Filtering",
        "",
        _markdown_table(label_df),
        "",
        "Interpretation: the eligibility filter is primarily a market-validity and boundary filter. It removes invalid/depth/boundary rows while preserving natural train/val/test distributions for evaluation.",
    ]
    (results_dir / "quote_eligible_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quote-eligible filter waterfall artifacts.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_waterfall(args.results_dir)
    print(f"Filter waterfall written to {args.results_dir}")
