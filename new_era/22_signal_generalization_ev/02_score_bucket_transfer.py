from __future__ import annotations

import numpy as np
import pandas as pd

from ev_utils import RESULTS_DIR, bucket_by_validation_quantiles, ensure_dirs, quote_rows_for_split


SCORE_SPECS = [
    ("015", "score_015", True),
    ("016_v2", "score_016", True),
    ("EV_s2_lam5_thr0", "ev_score_021", False),
]


def _rows_for_score(val: pd.DataFrame, other: pd.DataFrame, score_name: str, score_col: str, higher_is_risk: bool, split: str) -> list[dict]:
    buckets = bucket_by_validation_quantiles(val[score_col].to_numpy(dtype=float), other[score_col].to_numpy(dtype=float), higher_is_risk=higher_is_risk)
    rows = []
    for b in range(1, 11):
        mask = buckets == b
        rows.append(
            {
                "score_name": score_name,
                "split": split,
                "bucket": b,
                "bucket_orientation": "10=highest_risk" if higher_is_risk else "10=lowest_EV",
                "sample_count": int(mask.sum()),
                "mean_score": float(np.nanmean(other.loc[mask, score_col])) if mask.any() else np.nan,
                "toxic_rate": float(other.loc[mask, "toxic"].mean()) if mask.any() else np.nan,
                "fill_proxy_rate": float(other.loc[mask, "fill_proxy"].mean()) if mask.any() else np.nan,
                "avg_quote_pnl": float((other.loc[mask, "fill_proxy"] * (0.5 * other.loc[mask, "spread_ticks"] * 200 + other.loc[mask, "side_markout_ticks"] * 200 - 15)).mean()) if mask.any() else np.nan,
                "stress_adjusted_avg_quote_pnl": float(other.loc[mask, "realized_quote_pnl_moderate"].mean()) if mask.any() else np.nan,
                "stress_adjusted_total_quote_pnl": float(other.loc[mask, "realized_quote_pnl_moderate"].sum()) if mask.any() else 0.0,
            }
        )
    return rows


def run() -> None:
    ensure_dirs()
    val = quote_rows_for_split("val")
    test = quote_rows_for_split("test")
    rows = []
    for score_name, score_col, higher_is_risk in SCORE_SPECS:
        rows.extend(_rows_for_score(val, val, score_name, score_col, higher_is_risk, "val"))
        rows.extend(_rows_for_score(val, test, score_name, score_col, higher_is_risk, "test_diagnostic"))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "score_bucket_transfer.csv", index=False)

    lines = [
        "# Score Bucket Transfer Report",
        "",
        "Buckets are defined by validation quantiles and then applied unchanged to the burned diagnostic test set.",
        "",
    ]
    for score_name, _, _ in SCORE_SPECS:
        sub = out[out["score_name"] == score_name]
        val_top = sub[(sub["split"] == "val") & (sub["bucket"] == 10)].iloc[0]
        test_top = sub[(sub["split"] == "test_diagnostic") & (sub["bucket"] == 10)].iloc[0]
        lines.append(
            f"- {score_name}: bucket 10 stress avg PnL val `{val_top['stress_adjusted_avg_quote_pnl']:.6f}`, "
            f"test `{test_top['stress_adjusted_avg_quote_pnl']:.6f}`; toxic rate val `{val_top['toxic_rate']:.6f}`, test `{test_top['toxic_rate']:.6f}`."
        )
    lines.append("")
    lines.append("Economic transfer is considered weak when bucket ordering or stress-adjusted PnL sign changes between validation and diagnostic test.")
    (RESULTS_DIR / "score_bucket_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'score_bucket_transfer.csv'}")
