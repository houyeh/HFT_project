from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03a_ml_readiness_excess_markout"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EXCESS_DATA = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy" / "data" / "full_action_universe_with_excess_markout.parquet"

SPLITS = ["train", "val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
ACTION_SIDES = ["bid", "ask", "no_quote"]
TARGET = "excess_pnl_H20"
FEATURES_REQUESTED = [
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "volatility_past_50",
]
DRIFT_FEATURES = [
    "spread_ticks",
    "queue_imbalance",
    "deep_imbalance",
    "top_depth",
    "bid_sz_1",
    "ask_sz_1",
    "event_rate_past_50",
    "volatility_past_50",
]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def part_paths(split: str) -> list[Path]:
    return sorted((EXCESS_DATA / f"split={split}").glob("*.parquet"))


def existing_columns() -> list[str]:
    cached = getattr(existing_columns, "_cached", None)
    if cached is not None:
        return cached
    for split in SPLITS:
        paths = part_paths(split)
        if paths:
            cols = pd.read_parquet(paths[0]).columns.tolist()
            setattr(existing_columns, "_cached", cols)
            return cols
    setattr(existing_columns, "_cached", [])
    return []


def available_columns(cols: Iterable[str]) -> list[str]:
    existing = set(existing_columns())
    return [col for col in cols if col in existing]


def read_part(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=[col for col in columns if col in existing_columns()])


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed)


def stats_for_series(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
        }
    q = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
    }


def autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= lag + 2:
        return np.nan
    x = values[:-lag]
    y = values[lag:]
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def effective_sample_size(n: int, rho: float) -> float:
    if not np.isfinite(rho):
        return np.nan
    rho = min(max(float(rho), -0.99), 0.99)
    return float(n * (1.0 - rho) / (1.0 + rho))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1.0), 1e-6, 1.0)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1.0), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_stat(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    actual = np.asarray(actual, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    actual = np.sort(actual[np.isfinite(actual)])
    if len(reference) == 0 or len(actual) == 0:
        return np.nan
    values = np.sort(np.concatenate([reference, actual]))
    ref_cdf = np.searchsorted(reference, values, side="right") / len(reference)
    act_cdf = np.searchsorted(actual, values, side="right") / len(actual)
    return float(np.max(np.abs(ref_cdf - act_cdf)))


def collect_sample(split: str, columns: list[str], per_part: int = 50_000, action_sides: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(part_paths(split)):
        df = read_part(path, columns)
        if action_sides is not None and "action_side" in df:
            df = df[df["action_side"].astype(str).isin(action_sides)]
        frames.append(sample_frame(df, per_part, seed=20260601 + idx))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
