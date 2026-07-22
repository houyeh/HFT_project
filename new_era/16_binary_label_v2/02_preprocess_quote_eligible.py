from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))


from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob
from src.utils.tw_train_helpers import build_tw_feature_frame


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
DEFAULT_SYMBOL = "TXFB6"

SELECTED_FEATURES = [
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
]
Z_FEATURES = [f"{name}_z" for name in SELECTED_FEATURES]
DEPTH_LEVELS = 5


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(str(s) for s in session_ids)
    total = len(ordered)
    if total <= 1:
        return ordered, [], []

    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def _finite_positive(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        mask &= pd.Series(np.isfinite(arr) & (arr > 0), index=df.index)
    return mask


def _past_only_rolling_zscore(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    min_periods: int,
    clip_value: float = 10.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        past = pd.to_numeric(out[col], errors="coerce").shift(1)
        mean = past.rolling(window=window, min_periods=min_periods).mean()
        std = past.rolling(window=window, min_periods=min_periods).std()
        std = std.replace(0.0, np.nan)
        z = (pd.to_numeric(out[col], errors="coerce") - mean) / std
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_value, clip_value)
        out[f"{col}_z"] = z.astype(np.float32)
    return out


def _sequential_drop_counts(rule_masks: dict[str, np.ndarray]) -> tuple[dict[str, int], np.ndarray]:
    if not rule_masks:
        return {}, np.array([], dtype=bool)

    n = len(next(iter(rule_masks.values())))
    remaining = np.ones(n, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in rule_masks.items():
        mask = np.asarray(mask, dtype=bool)
        dropped = remaining & ~mask
        counts[f"dropped_{name}"] = int(dropped.sum())
        remaining &= mask
    return counts, remaining


def _build_rule_masks(
    lob_session: pd.DataFrame,
    feature_session: pd.DataFrame,
    first_ticks: int,
    last_ticks: int,
) -> dict[str, np.ndarray]:
    lob = lob_session.reset_index(drop=True)
    feature = feature_session.reset_index(drop=True)
    n = len(lob)

    invalid_bid_ask_mask = _finite_positive(lob, ["bid_px_1", "ask_px_1"]).to_numpy(dtype=bool)
    spread = pd.to_numeric(lob["ask_px_1"], errors="coerce") - pd.to_numeric(lob["bid_px_1"], errors="coerce")
    positive_spread_mask = (np.isfinite(spread.to_numpy(dtype=np.float64)) & (spread > 0))

    depth_cols: list[str] = []
    for level in range(1, DEPTH_LEVELS + 1):
        depth_cols.extend([f"bid_px_{level}", f"ask_px_{level}", f"bid_vol_{level}", f"ask_vol_{level}"])
    complete_depth_mask = _finite_positive(lob, depth_cols).to_numpy(dtype=bool)

    idx = np.arange(n)
    boundary_mask = (idx >= int(first_ticks)) & (idx < max(n - int(last_ticks), 0))

    feature_cols = SELECTED_FEATURES + Z_FEATURES + ["mid_price"]
    feature_finite_mask = np.ones(n, dtype=bool)
    for col in feature_cols:
        values = pd.to_numeric(feature[col], errors="coerce").to_numpy(dtype=np.float64)
        feature_finite_mask &= np.isfinite(values)

    return {
        "invalid_bid_ask": invalid_bid_ask_mask,
        "nonpositive_spread": positive_spread_mask,
        "missing_depth": complete_depth_mask,
        "session_boundary": boundary_mask,
        "feature_na": feature_finite_mask,
    }


def _assemble_output_frame(
    *,
    dataset_name: str,
    symbol: str,
    split: str,
    session_id: str,
    tick_size: float,
    lob_session: pd.DataFrame,
    feature_session: pd.DataFrame,
    eligible_mask: np.ndarray,
) -> pd.DataFrame:
    lob = lob_session.reset_index(drop=True)
    feature = feature_session.reset_index(drop=True)

    quote_cols = ["timestamp"]
    for level in range(1, DEPTH_LEVELS + 1):
        quote_cols.extend([f"bid_px_{level}", f"bid_vol_{level}", f"ask_px_{level}", f"ask_vol_{level}"])

    out = lob.loc[eligible_mask, quote_cols].copy()
    local_idx = np.flatnonzero(eligible_mask)
    out.insert(0, "dataset", dataset_name)
    out.insert(1, "symbol", symbol)
    out.insert(2, "session_id", session_id)
    out.insert(3, "split", split)
    out.insert(4, "local_row_id", local_idx.astype(np.int64))
    out["tick_size"] = float(tick_size)
    out["mid_price"] = feature.loc[eligible_mask, "mid_price"].to_numpy(dtype=np.float64)
    out["spread_ticks"] = (
        (out["ask_px_1"].astype(float) - out["bid_px_1"].astype(float)) / max(float(tick_size), 1e-9)
    )

    for col in SELECTED_FEATURES + Z_FEATURES:
        out[col] = feature.loc[eligible_mask, col].to_numpy(dtype=np.float32)

    out["quote_eligible"] = True
    return out.reset_index(drop=True)


def preprocess_quote_eligible(
    *,
    data_dir: Path,
    symbol: str,
    out_dir: Path,
    first_ticks: int,
    last_ticks: int,
    zscore_window: int,
    zscore_min_periods: int,
    train_frac: float,
    val_frac: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"Symbol {symbol} not found under {data_dir}")

    lob = lob_frames[symbol].copy()
    trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()
    dataset_name = data_dir.name

    sessions = sorted(str(s) for s in lob["session_id"].unique().tolist())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(sessions, train_frac=train_frac, val_frac=val_frac)
    split_by_session = {sid: "train" for sid in train_sessions}
    split_by_session.update({sid: "val" for sid in val_sessions})
    split_by_session.update({sid: "test" for sid in test_sessions})

    output_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    for raw_session_id, lob_session in lob.groupby("session_id", sort=True):
        session_id = str(raw_session_id)
        split = split_by_session.get(session_id, "unassigned")
        trade_session = trades[trades["session_id"].astype(str) == session_id]
        lob_session = lob_session.sort_values("timestamp").reset_index(drop=True)
        trade_session = trade_session.sort_values("timestamp").reset_index(drop=True)

        tick_size = float(infer_tick_size_from_lob(lob_session))
        raw_features = build_tw_feature_frame(
            lob_session.reset_index(drop=True),
            trade_session.reset_index(drop=True),
            symbol=symbol,
            tick_size=tick_size,
        ).reset_index(drop=True)
        feature_session = _past_only_rolling_zscore(
            raw_features,
            SELECTED_FEATURES,
            window=zscore_window,
            min_periods=zscore_min_periods,
        )

        rule_masks = _build_rule_masks(
            lob_session=lob_session,
            feature_session=feature_session,
            first_ticks=first_ticks,
            last_ticks=last_ticks,
        )
        drop_counts, eligible_mask = _sequential_drop_counts(rule_masks)

        output_frames.append(
            _assemble_output_frame(
                dataset_name=dataset_name,
                symbol=symbol,
                split=split,
                session_id=session_id,
                tick_size=tick_size,
                lob_session=lob_session,
                feature_session=feature_session,
                eligible_mask=eligible_mask,
            )
        )

        raw_rows = int(len(lob_session))
        eligible_rows = int(eligible_mask.sum())
        report_rows.append(
            {
                "scope": "session",
                "dataset": dataset_name,
                "symbol": symbol,
                "split": split,
                "session_id": session_id,
                "raw_rows": raw_rows,
                **drop_counts,
                "eligible_rows": eligible_rows,
                "eligible_rate": float(eligible_rows / max(raw_rows, 1)),
                "tick_size": tick_size,
                "boundary_first_ticks": int(first_ticks),
                "boundary_last_ticks": int(last_ticks),
                "zscore_window": int(zscore_window),
                "zscore_min_periods": int(zscore_min_periods),
            }
        )

    if not output_frames:
        raise RuntimeError("No quote-eligible frames were produced.")

    eligible_df = pd.concat(output_frames, ignore_index=True)
    report_df = pd.DataFrame(report_rows)

    numeric_drop_cols = [col for col in report_df.columns if col.startswith("dropped_")]
    aggregate_rows: list[dict[str, object]] = []
    for scope, group_cols in [
        ("split", ["dataset", "symbol", "split"]),
        ("all", ["dataset", "symbol"]),
    ]:
        grouped = report_df.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            raw_rows = int(group["raw_rows"].sum())
            eligible_rows = int(group["eligible_rows"].sum())
            aggregate_rows.append(
                {
                    "scope": scope,
                    **base,
                    "session_id": "__ALL__",
                    "raw_rows": raw_rows,
                    **{col: int(group[col].sum()) for col in numeric_drop_cols},
                    "eligible_rows": eligible_rows,
                    "eligible_rate": float(eligible_rows / max(raw_rows, 1)),
                    "tick_size": float(group["tick_size"].median()),
                    "boundary_first_ticks": int(first_ticks),
                    "boundary_last_ticks": int(last_ticks),
                    "zscore_window": int(zscore_window),
                    "zscore_min_periods": int(zscore_min_periods),
                }
            )

    report_df = pd.concat([report_df, pd.DataFrame(aggregate_rows)], ignore_index=True)

    parquet_path = out_dir / "quote_eligible_events.parquet"
    report_path = out_dir / "quote_eligible_filter_report.csv"
    eligible_df.to_parquet(parquet_path, index=False)
    report_df.to_csv(report_path, index=False)

    metadata = {
        "data_dir": str(data_dir),
        "dataset": dataset_name,
        "symbol": symbol,
        "selected_features": SELECTED_FEATURES,
        "z_features": Z_FEATURES,
        "zscore_definition": "past-only rolling z-score: (x_t - mean(x_<t)) / std(x_<t), clipped to +/-10",
        "zscore_window": int(zscore_window),
        "zscore_min_periods": int(zscore_min_periods),
        "quote_eligible_rules_in_order": [
            "valid best bid and ask: bid_px_1 > 0, ask_px_1 > 0, finite",
            "positive spread: ask_px_1 - bid_px_1 > 0",
            "complete 5-level depth: bid/ask prices and volumes at levels 1..5 are finite and > 0",
            f"session boundary removal: first {int(first_ticks)} ticks and last {int(last_ticks)} ticks per session",
            "finite selected feature and z-score values",
        ],
        "split_policy": "chronological session split on raw sessions before filtering",
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
        "raw_rows": int(report_df[report_df["scope"] == "session"]["raw_rows"].sum()),
        "eligible_rows": int(len(eligible_df)),
    }
    (out_dir / "quote_eligible_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    definition_lines = [
        "# Quote-Eligible Universe Definition",
        "",
        "Rows are retained only when all rules pass in order:",
        "",
        "1. Best bid and ask are finite and strictly positive.",
        "2. Spread is strictly positive.",
        "3. All five bid and ask price/volume levels are finite and strictly positive.",
        f"4. The row is not in the first {int(first_ticks)} ticks or last {int(last_ticks)} ticks of its session.",
        "5. The selected model features and past-only z-scores are finite.",
        "",
        "Z-scores use only prior rows from the same session: `shift(1).rolling(...)` supplies the mean and standard deviation.",
    ]
    (out_dir / "quote_eligible_definition.md").write_text("\n".join(definition_lines), encoding="utf-8")

    return parquet_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the quote-eligible event universe for binary_label_v2.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--first_ticks", type=int, default=100)
    parser.add_argument("--last_ticks", type=int, default=10)
    parser.add_argument("--zscore_window", type=int, default=2000)
    parser.add_argument("--zscore_min_periods", type=int, default=100)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path, report_path = preprocess_quote_eligible(
        data_dir=args.data_dir,
        symbol=args.symbol,
        out_dir=args.out_dir,
        first_ticks=args.first_ticks,
        last_ticks=args.last_ticks,
        zscore_window=args.zscore_window,
        zscore_min_periods=args.zscore_min_periods,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("Quote-eligible preprocessing complete:")
    print(f"- {parquet_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
