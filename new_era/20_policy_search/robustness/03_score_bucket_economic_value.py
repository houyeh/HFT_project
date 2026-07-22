from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import MAKER_FEE, POINT_VALUE, ROBUSTNESS_DIR, ensure_dirs, load_policy_frame


def _side_quote_pnl(frame: pd.DataFrame, side: str) -> np.ndarray:
    fut = frame["future_mid_return_H_ticks"].to_numpy(dtype=float)
    if side == "bid":
        fill = frame["bid_fill_proxy"].to_numpy(dtype=bool)
        ticks = fut
    else:
        fill = frame["ask_fill_proxy"].to_numpy(dtype=bool)
        ticks = -fut
    return np.where(fill, ticks * POINT_VALUE - MAKER_FEE, 0.0)


def _bucket_rows(frame: pd.DataFrame, split: str, side: str, score_col: str) -> list[dict[str, object]]:
    score = frame[score_col].to_numpy(dtype=float)
    pnl = _side_quote_pnl(frame, side)
    toxic = frame[f"{side}_toxic"].to_numpy(dtype=bool)
    rows = []
    finite = np.isfinite(score)
    base = pnl[finite]
    base_toxic = toxic[finite]
    for q in range(10):
        lo = np.nanquantile(score[finite], q / 10)
        hi = np.nanquantile(score[finite], (q + 1) / 10)
        if q < 9:
            mask = finite & (score >= lo) & (score < hi)
        else:
            mask = finite & (score >= lo) & (score <= hi)
        rows.append(_summ(split, side, f"decile_{q+1}", score, pnl, toxic, mask, base, base_toxic))
    for frac, name in [(0.10, "top_10pct"), (0.05, "top_5pct"), (0.01, "top_1pct")]:
        thr = np.sort(score[finite])[::-1][max(0, min(finite.sum() - 1, int(np.ceil(frac * finite.sum())) - 1))]
        mask = finite & (score >= thr)
        rows.append(_summ(split, side, name, score, pnl, toxic, mask, base, base_toxic))
    return rows


def _summ(split: str, side: str, bucket: str, score: np.ndarray, pnl: np.ndarray, toxic: np.ndarray, mask: np.ndarray, base: np.ndarray, base_toxic: np.ndarray) -> dict[str, object]:
    return {
        "split": split,
        "side": side,
        "bucket": bucket,
        "num_samples": int(mask.sum()),
        "mean_score": float(np.nanmean(score[mask])) if mask.any() else np.nan,
        "expected_quote_pnl_if_not_cancelled": float(np.nanmean(pnl[mask])) if mask.any() else np.nan,
        "total_quote_pnl_if_not_cancelled": float(np.nansum(pnl[mask])) if mask.any() else 0.0,
        "base_expected_quote_pnl": float(np.nanmean(base)) if len(base) else np.nan,
        "economic_lift_vs_base": float(np.nanmean(pnl[mask]) - np.nanmean(base)) if mask.any() and len(base) else np.nan,
        "toxic_rate": float(np.mean(toxic[mask])) if mask.any() else np.nan,
        "base_toxic_rate": float(np.mean(base_toxic)) if len(base_toxic) else np.nan,
    }


def run() -> None:
    ensure_dirs()
    rows = []
    for split in ["val", "test"]:
        frame = load_policy_frame(split)
        rows.extend(_bucket_rows(frame, split, "bid", "bid_prob_toxic"))
        rows.extend(_bucket_rows(frame, split, "ask", "ask_prob_toxic"))
    pd.DataFrame(rows).to_csv(ROBUSTNESS_DIR / "score_bucket_economic_value.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {ROBUSTNESS_DIR / 'score_bucket_economic_value.csv'}")
