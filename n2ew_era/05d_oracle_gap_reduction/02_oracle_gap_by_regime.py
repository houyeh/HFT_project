from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common_05d as c
import common_05a as p05a


REGIME_FEATURES = [
    "spread_ticks",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
    "time_in_session",
]


def buckets_for_payload(payload: dict) -> dict[str, np.ndarray]:
    return {
        "spread_ticks": c.spread_bucket(c.raw_feature(payload, "spread_ticks")),
        "queue_imbalance": c.quantile_bucket(c.raw_feature(payload, "queue_imbalance"), "queue_imb"),
        "deep_imbalance": c.quantile_bucket(c.raw_feature(payload, "deep_imbalance"), "deep_imb"),
        "volatility_past_50": c.quantile_bucket(c.raw_feature(payload, "volatility_past_50"), "volatility"),
        "event_rate_past_50": c.quantile_bucket(c.raw_feature(payload, "event_rate_past_50"), "event_rate"),
        "inter_arrival_ms": c.quantile_bucket(c.raw_feature(payload, "inter_arrival_ms"), "inter_arrival"),
        "imbalance_change_past": c.quantile_bucket(c.raw_feature(payload, "imbalance_change_past"), "imb_change"),
        "spread_change_past": c.quantile_bucket(c.raw_feature(payload, "spread_change_past"), "spread_change"),
        "time_in_session": c.time_in_session_bucket(payload),
    }


def regime_rows() -> pd.DataFrame:
    rows = []
    for split in c.SPLITS:
        payload = p05a.load_sequences(split)
        oracle_action, oracle_pnl, oracle_value = c.oracle_decision_and_pnl(payload)
        bucket_map = buckets_for_payload(payload)
        decisions = c.model_decisions(payload)
        for model_id, (decision, scores) in decisions.items():
            err = c.error_type(decision, oracle_action)
            pnl = c.model_pnl(payload, decision)
            selected_score = c.selected_scores(scores, decision)
            for feature_name, bucket in bucket_map.items():
                frame = pd.DataFrame(
                    {
                        "bucket": bucket,
                        "error_type": err,
                        "oracle_pnl": oracle_pnl,
                        "model_pnl": pnl,
                        "oracle_value": oracle_value,
                        "model_score": selected_score,
                        "missed": err == "missed_opportunity",
                        "false_positive": err == "false_positive",
                        "side_error": err == "side_error",
                    }
                )
                grouped = frame.groupby("bucket").agg(
                    state_count=("bucket", "size"),
                    oracle_pnl=("oracle_pnl", "sum"),
                    model_pnl=("model_pnl", "sum"),
                    missed_opportunity_count=("missed", "sum"),
                    false_positive_count=("false_positive", "sum"),
                    side_error_count=("side_error", "sum"),
                    avg_model_score=("model_score", "mean"),
                    avg_oracle_value=("oracle_value", "mean"),
                ).reset_index()
                for _, row in grouped.iterrows():
                    rows.append(
                        {
                            "split": split,
                            "model_id": model_id,
                            "regime_feature": feature_name,
                            "bucket": row["bucket"],
                            "state_count": int(row["state_count"]),
                            "oracle_pnl": float(row["oracle_pnl"]),
                            "model_pnl": float(row["model_pnl"]),
                            "capture_ratio": float(row["model_pnl"] / row["oracle_pnl"]) if row["oracle_pnl"] else 0.0,
                            "missed_opportunity_count": int(row["missed_opportunity_count"]),
                            "false_positive_count": int(row["false_positive_count"]),
                            "side_error_count": int(row["side_error_count"]),
                            "missed_opportunity_rate": float(row["missed_opportunity_count"] / max(row["state_count"], 1)),
                            "false_positive_rate": float(row["false_positive_count"] / max(row["state_count"], 1)),
                            "side_error_rate": float(row["side_error_count"] / max(row["state_count"], 1)),
                            "avg_model_score": float(row["avg_model_score"]) if pd.notna(row["avg_model_score"]) else None,
                            "avg_oracle_value": float(row["avg_oracle_value"]),
                            "oracle_uses_future_labels": True,
                            "new5_used": False,
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "oracle_gap_by_regime.csv", index=False)
    return out


def plot_regime(out: pd.DataFrame, feature: str, path_name: str) -> None:
    subset = out[(out["model_id"].eq("phc_lite_v1")) & (out["regime_feature"].eq(feature))]
    if subset.empty:
        return
    pivot = subset.pivot_table(index="bucket", columns="split", values="capture_ratio", aggfunc="mean")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title(f"PHC-lite Oracle Capture by {feature}")
    plt.ylabel("capture ratio")
    plt.tight_layout()
    plt.savefig(c.FIGURES / path_name, dpi=160)
    plt.close()


def plot_error_stack(out: pd.DataFrame) -> None:
    subset = out[(out["model_id"].eq("phc_lite_v1")) & (out["regime_feature"].eq("volatility_past_50"))]
    if subset.empty:
        return
    grouped = subset.groupby("bucket")[["missed_opportunity_count", "false_positive_count", "side_error_count"]].sum()
    grouped.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("PHC-lite Oracle Error Type Stack by Volatility Bucket")
    plt.ylabel("state count")
    plt.tight_layout()
    plt.savefig(c.FIGURES / "oracle_gap_error_type_stack.png", dpi=160)
    plt.close()


def report(out: pd.DataFrame) -> None:
    phc = out[out["model_id"].eq("phc_lite_v1")]
    feature_summary = phc.groupby("regime_feature").agg(
        min_capture=("capture_ratio", "min"),
        max_capture=("capture_ratio", "max"),
        capture_range=("capture_ratio", lambda s: float(s.max() - s.min())),
        total_missed=("missed_opportunity_count", "sum"),
        total_false_positive=("false_positive_count", "sum"),
        total_side_error=("side_error_count", "sum"),
    ).reset_index().sort_values("capture_range", ascending=False)
    feature_summary.to_csv(c.RESULTS / "oracle_gap_regime_feature_summary.csv", index=False)
    top_gap = phc.assign(gap=lambda df: df["oracle_pnl"] - df["model_pnl"]).sort_values("gap", ascending=False).head(20)
    c.write_markdown(
        c.ROOT / "ORACLE_GAP_BY_REGIME_REPORT.md",
        [
            "# Oracle Gap by Market Regime Report",
            "",
            "Universe: PHC sequence-window terminal states on val/NEW3/NEW4. NEW5 is not used. Oracle is diagnostic only.",
            "",
            "## PHC Regime Feature Summary",
            "",
            feature_summary.to_markdown(index=False),
            "",
            "## Largest PHC Regime Gaps",
            "",
            top_gap.to_markdown(index=False),
            "",
            "Interpretation guide: high `missed_opportunity_count` indicates threshold/no_quote errors; high `side_error_count` indicates bid/ask ranking errors; high `false_positive_count` indicates over-quoting states where oracle chooses no_quote.",
        ],
    )


def main() -> None:
    c.ensure_dirs()
    out = regime_rows()
    plot_regime(out, "volatility_past_50", "oracle_gap_by_volatility_bucket.png")
    plot_regime(out, "event_rate_past_50", "oracle_gap_by_event_rate_bucket.png")
    plot_error_stack(out)
    report(out)
    print("oracle_gap_by_regime_complete")


if __name__ == "__main__":
    main()
