from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from round2_audit_utils import RESULTS, load_test_frame


SIDES = ["bid", "ask"]


def _bucket_stats(
    *,
    side: str,
    bucket_type: str,
    bucket: str,
    scores: np.ndarray,
    toxic: np.ndarray,
    mask: np.ndarray,
    base_rate: float,
    monotonic_deciles: bool | None,
) -> dict:
    toxic_rate = float(np.mean(toxic[mask])) if int(mask.sum()) else float("nan")
    return {
        "side": side,
        "split": "test",
        "bucket_type": bucket_type,
        "bucket": bucket,
        "num_samples": int(mask.sum()),
        "mean_score": float(np.mean(scores[mask])) if int(mask.sum()) else float("nan"),
        "toxic_rate": toxic_rate,
        "base_toxic_rate": base_rate,
        "lift": float(toxic_rate / base_rate) if base_rate > 0 and np.isfinite(toxic_rate) else float("nan"),
        "higher_than_base_rate": bool(toxic_rate > base_rate) if np.isfinite(toxic_rate) else False,
        "monotonic_deciles_non_decreasing": monotonic_deciles,
    }


def _side_rows(frame: pd.DataFrame, side: str) -> list[dict]:
    scores = frame[f"{side}_prob_toxic"].to_numpy(dtype=float)
    toxic = frame[f"{side}_toxic"].to_numpy(dtype=int)
    base_rate = float(np.mean(toxic))
    rows: list[dict] = []

    rank = pd.Series(scores).rank(method="first")
    decile_ids = pd.qcut(rank, q=10, labels=False, duplicates="drop").to_numpy()
    decile_rates = []
    for decile in sorted(pd.unique(decile_ids)):
        mask = decile_ids == decile
        decile_rates.append(float(np.mean(toxic[mask])))
    monotonic = bool(np.all(np.diff(decile_rates) >= -1e-12)) if len(decile_rates) > 1 else True

    max_decile = int(np.nanmax(decile_ids))
    for decile in sorted(pd.unique(decile_ids)):
        label = f"decile_{int(decile) + 1:02d}"
        if int(decile) == max_decile:
            label += "_top10pct"
        rows.append(
            _bucket_stats(
                side=side,
                bucket_type="decile",
                bucket=label,
                scores=scores,
                toxic=toxic,
                mask=decile_ids == decile,
                base_rate=base_rate,
                monotonic_deciles=monotonic,
            )
        )

    for frac, label in [(0.10, "top_10pct"), (0.05, "top_5pct"), (0.01, "top_1pct")]:
        threshold = np.sort(scores)[::-1][max(0, min(len(scores) - 1, int(np.ceil(frac * len(scores))) - 1))]
        rows.append(
            _bucket_stats(
                side=side,
                bucket_type="top_tail",
                bucket=label,
                scores=scores,
                toxic=toxic,
                mask=scores >= threshold,
                base_rate=base_rate,
                monotonic_deciles=monotonic,
            )
        )

    top_checks = [r for r in rows if r["bucket"] in {"top_10pct", "top_5pct", "top_1pct"}]
    passes_minimum = bool(all(r["higher_than_base_rate"] for r in top_checks))
    for row in rows:
        row["minimum_requirement_top_10_5_1_gt_base"] = passes_minimum
    return rows


def _plot_pos_neg(frame: pd.DataFrame, results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, side in zip(axes, SIDES):
        scores = frame[f"{side}_prob_toxic"].to_numpy(dtype=float)
        toxic = frame[f"{side}_toxic"].to_numpy(dtype=int).astype(bool)
        ax.hist(scores[~toxic], bins=80, alpha=0.55, density=True, label="non-toxic")
        ax.hist(scores[toxic], bins=80, alpha=0.65, density=True, label="toxic")
        ax.set_title(f"{side.capitalize()} Test Score Distribution")
        ax.set_xlabel("P(toxic)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(results_dir / "score_distribution_pos_neg.png", dpi=150)
    plt.close(fig)


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = load_test_frame()
    rows = []
    for side in SIDES:
        rows.extend(_side_rows(frame, side))
    pd.DataFrame(rows).to_csv(results_dir / "score_bucket_toxic_lift.csv", index=False)
    _plot_pos_neg(frame, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit toxic lift by model-score bucket.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"Score-bucket toxic lift written to {args.results_dir}")
