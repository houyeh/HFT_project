from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs"
DEFAULT_INPUT = DEFAULT_OUT_DIR / "quote_eligible_events.parquet"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
DEFAULT_SYMBOL = "TXFB6"


def _future_window_min(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return (
        series.iloc[::-1]
        .shift(1)
        .rolling(window=int(window), min_periods=1)
        .min()
        .iloc[::-1]
        .to_numpy(dtype=np.float64)
    )


def _future_window_max(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return (
        series.iloc[::-1]
        .shift(1)
        .rolling(window=int(window), min_periods=1)
        .max()
        .iloc[::-1]
        .to_numpy(dtype=np.float64)
    )


def _future_mid_return_ticks(
    mid: np.ndarray,
    tick_size: float,
    horizon: int,
) -> np.ndarray:
    out = np.full(len(mid), np.nan, dtype=np.float64)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(mid) > horizon:
        out[:-horizon] = (mid[horizon:] - mid[:-horizon]) / max(float(tick_size), 1e-9)
    return out


def _trade_extrema_by_future_quote_window(
    *,
    quote_timestamps: np.ndarray,
    trade_df: pd.DataFrame,
    fill_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(quote_timestamps)
    min_trade_at_quote = np.full(n, np.inf, dtype=np.float64)
    max_trade_at_quote = np.full(n, -np.inf, dtype=np.float64)

    if not trade_df.empty:
        trades = trade_df.sort_values("timestamp")
        trade_ts = trades["timestamp"].to_numpy(dtype=np.int64, copy=False)
        trade_px = trades["trade_price"].to_numpy(dtype=np.float64, copy=False)

        quote_idx = np.searchsorted(quote_timestamps, trade_ts, side="left")
        valid = (quote_idx >= 0) & (quote_idx < n) & np.isfinite(trade_px)
        if np.any(valid):
            np.minimum.at(min_trade_at_quote, quote_idx[valid], trade_px[valid])
            np.maximum.at(max_trade_at_quote, quote_idx[valid], trade_px[valid])

    future_min_trade = _future_window_min(min_trade_at_quote, fill_window)
    future_max_trade = _future_window_max(max_trade_at_quote, fill_window)
    return future_min_trade, future_max_trade


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _label_one_session(
    *,
    eligible_session: pd.DataFrame,
    lob_session: pd.DataFrame,
    trade_session: pd.DataFrame,
    fill_window: int,
    horizon: int,
    adverse_threshold_ticks: float,
) -> pd.DataFrame:
    eligible = eligible_session.sort_values("local_row_id").copy()
    lob = lob_session.sort_values("timestamp").reset_index(drop=True)

    raw_idx = eligible["local_row_id"].to_numpy(dtype=np.int64, copy=False)
    raw_n = len(lob)
    if raw_idx.size and (raw_idx.min() < 0 or raw_idx.max() >= raw_n):
        raise IndexError("local_row_id is outside raw session bounds")

    tick_size = float(eligible["tick_size"].median())
    if not np.isfinite(tick_size) or tick_size <= 0:
        tick_size = float(infer_tick_size_from_lob(lob))

    raw_bid = lob["bid_px_1"].to_numpy(dtype=np.float64, copy=False)
    raw_ask = lob["ask_px_1"].to_numpy(dtype=np.float64, copy=False)
    raw_mid = (raw_bid + raw_ask) / 2.0
    quote_ts = lob["timestamp"].to_numpy(dtype=np.int64, copy=False)

    future_mid_return_all = _future_mid_return_ticks(raw_mid, tick_size, horizon)
    future_min_ask_all = _future_window_min(raw_ask, fill_window)
    future_max_bid_all = _future_window_max(raw_bid, fill_window)
    future_min_trade_all, future_max_trade_all = _trade_extrema_by_future_quote_window(
        quote_timestamps=quote_ts,
        trade_df=trade_session,
        fill_window=fill_window,
    )

    current_bid = eligible["bid_px_1"].to_numpy(dtype=np.float64, copy=False)
    current_ask = eligible["ask_px_1"].to_numpy(dtype=np.float64, copy=False)

    future_mid_return = future_mid_return_all[raw_idx]
    future_min_ask = future_min_ask_all[raw_idx]
    future_max_bid = future_max_bid_all[raw_idx]
    future_min_trade = future_min_trade_all[raw_idx]
    future_max_trade = future_max_trade_all[raw_idx]

    bid_trade_hit = future_min_trade <= current_bid
    bid_price_cross = future_min_ask <= current_bid
    bid_fill_proxy = bid_trade_hit | bid_price_cross

    ask_trade_hit = future_max_trade >= current_ask
    ask_price_cross = future_max_bid >= current_ask
    ask_fill_proxy = ask_trade_hit | ask_price_cross

    label_valid = np.isfinite(future_mid_return)
    bid_adverse = label_valid & (future_mid_return <= -float(adverse_threshold_ticks))
    ask_adverse = label_valid & (future_mid_return >= float(adverse_threshold_ticks))

    eligible["future_mid_return_H_ticks"] = future_mid_return.astype(np.float32)
    eligible["future_min_ask_fill_window"] = future_min_ask.astype(np.float32)
    eligible["future_max_bid_fill_window"] = future_max_bid.astype(np.float32)
    eligible["future_min_trade_price_fill_window"] = np.where(
        np.isfinite(future_min_trade), future_min_trade, np.nan
    ).astype(np.float32)
    eligible["future_max_trade_price_fill_window"] = np.where(
        np.isfinite(future_max_trade), future_max_trade, np.nan
    ).astype(np.float32)

    eligible["bid_trade_hits_quote"] = bid_trade_hit.astype(bool)
    eligible["bid_price_cross_hits_quote"] = bid_price_cross.astype(bool)
    eligible["bid_fill_proxy"] = bid_fill_proxy.astype(bool)
    eligible["bid_adverse_move"] = bid_adverse.astype(bool)
    eligible["bid_toxic"] = (bid_fill_proxy & bid_adverse).astype(np.int8)

    eligible["ask_trade_hits_quote"] = ask_trade_hit.astype(bool)
    eligible["ask_price_cross_hits_quote"] = ask_price_cross.astype(bool)
    eligible["ask_fill_proxy"] = ask_fill_proxy.astype(bool)
    eligible["ask_adverse_move"] = ask_adverse.astype(bool)
    eligible["ask_toxic"] = (ask_fill_proxy & ask_adverse).astype(np.int8)

    eligible["label_valid_horizon"] = label_valid.astype(bool)
    eligible["label_horizon_ticks"] = int(horizon)
    eligible["fill_proxy_window_ticks"] = int(fill_window)
    eligible["adverse_threshold_ticks"] = float(adverse_threshold_ticks)
    return eligible


def _summarize_side(
    labels: pd.DataFrame,
    side: str,
    scope: str,
    split_value: str,
) -> dict[str, object]:
    fill_col = f"{side}_fill_proxy"
    toxic_col = f"{side}_toxic"
    trade_col = f"{side}_trade_hits_quote"
    cross_col = f"{side}_price_cross_hits_quote"
    adverse_col = f"{side}_adverse_move"

    total = int(len(labels))
    valid = int(labels["label_valid_horizon"].sum())
    valid_mask = labels["label_valid_horizon"].astype(bool)
    fill_mask = labels[fill_col].astype(bool)
    toxic_mask = labels[toxic_col].astype(bool)
    fill_count = int(fill_mask.sum())
    valid_fill_count = int((valid_mask & fill_mask).sum())
    toxic_count = int(labels[toxic_col].sum())
    adverse_count = int(labels[adverse_col].sum())
    trade_hit_count = int(labels[trade_col].sum())
    price_cross_count = int(labels[cross_col].sum())
    non_toxic_fill_count = int((valid_mask & fill_mask & ~toxic_mask).sum())

    return {
        "scope": scope,
        "split": split_value,
        "side": side,
        "quote_eligible_rows": total,
        "label_valid_rows": valid,
        "fill_proxy_count": fill_count,
        "fill_proxy_count_valid_rows": valid_fill_count,
        "fill_proxy_rate_all_rows": _safe_ratio(fill_count, total),
        "fill_proxy_rate_valid_rows": _safe_ratio(valid_fill_count, valid),
        "trade_hit_count": trade_hit_count,
        "price_cross_hit_count": price_cross_count,
        "adverse_move_count": adverse_count,
        "toxic_count": toxic_count,
        "non_toxic_fill_count": non_toxic_fill_count,
        "toxic_rate_all_rows": _safe_ratio(toxic_count, total),
        "toxic_rate_valid_rows": _safe_ratio(toxic_count, valid),
        "toxic_rate_given_fill": _safe_ratio(toxic_count, valid_fill_count),
    }


def _build_toxic_rate_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in labels.groupby("split", sort=True):
        for side in ["bid", "ask"]:
            rows.append(_summarize_side(group, side=side, scope="split", split_value=str(split)))

    for side in ["bid", "ask"]:
        rows.append(_summarize_side(labels, side=side, scope="all", split_value="__ALL__"))

    return pd.DataFrame(rows)


def _write_label_definition(
    *,
    out_path: Path,
    input_path: Path,
    data_dir: Path,
    symbol: str,
    fill_window: int,
    horizon: int,
    adverse_threshold_ticks: float,
) -> None:
    lines = [
        "# Binary Labels V2 Definition",
        "",
        "## Inputs",
        f"- Quote-eligible events: `{input_path}`",
        f"- Raw LOB/trade source for future proxy construction: `{data_dir}`",
        f"- Symbol: `{symbol}`",
        "",
        "## Fill Proxy",
        f"- Fill window: next {int(fill_window)} raw quote ticks within the same session.",
        "- `bid_fill_proxy = bid_trade_hits_quote OR bid_price_cross_hits_quote`.",
        "- `bid_trade_hits_quote`: a future trade in the fill window has `trade_price <= current_bid_px_1`.",
        "- `bid_price_cross_hits_quote`: future best ask in the fill window has `ask_px_1 <= current_bid_px_1`.",
        "- `ask_fill_proxy = ask_trade_hits_quote OR ask_price_cross_hits_quote`.",
        "- `ask_trade_hits_quote`: a future trade in the fill window has `trade_price >= current_ask_px_1`.",
        "- `ask_price_cross_hits_quote`: future best bid in the fill window has `bid_px_1 >= current_ask_px_1`.",
        "",
        "This is a proxy because true aggressor-side and queue-position fills are not available.",
        "",
        "## Toxicity",
        f"- Label horizon: {int(horizon)} raw quote ticks within the same session.",
        f"- Adverse threshold: {float(adverse_threshold_ticks):.6g} ticks.",
        "- `future_mid_return_H_ticks = (mid_price[t + H] - mid_price[t]) / tick_size`.",
        "- `bid_toxic = 1` iff `bid_fill_proxy == True` and `future_mid_return_H_ticks <= -adverse_threshold_ticks`.",
        "- `ask_toxic = 1` iff `ask_fill_proxy == True` and `future_mid_return_H_ticks >= adverse_threshold_ticks`.",
        "",
        "## Leakage Control",
        "- Labels use future information by definition, but only as targets.",
        "- Model features come from `quote_eligible_events.parquet`, whose z-scores were computed with past-only rolling statistics.",
        "- No threshold selection or model tuning is performed in this script.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_binary_labels(
    *,
    input_path: Path,
    data_dir: Path,
    symbol: str,
    out_dir: Path,
    fill_window: int,
    horizon: int,
    adverse_threshold_ticks: float,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing quote-eligible parquet: {input_path}")

    eligible = pd.read_parquet(input_path)
    required_cols = {"session_id", "local_row_id", "bid_px_1", "ask_px_1", "tick_size", "split"}
    missing = sorted(required_cols - set(eligible.columns))
    if missing:
        raise KeyError(f"Input parquet is missing required columns: {missing}")

    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"Symbol {symbol} not found under {data_dir}")
    lob = lob_frames[symbol].copy()
    trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()

    labeled_frames: list[pd.DataFrame] = []
    for session_id, eligible_session in eligible.groupby("session_id", sort=True):
        sid = str(session_id)
        lob_session = lob[lob["session_id"].astype(str) == sid]
        if lob_session.empty:
            raise RuntimeError(f"Raw LOB session not found for session_id={sid}")
        trade_session = trades[trades["session_id"].astype(str) == sid]
        labeled_frames.append(
            _label_one_session(
                eligible_session=eligible_session,
                lob_session=lob_session,
                trade_session=trade_session,
                fill_window=fill_window,
                horizon=horizon,
                adverse_threshold_ticks=adverse_threshold_ticks,
            )
        )

    labels = pd.concat(labeled_frames, ignore_index=True)

    labels_path = out_dir / "binary_labels_v2.parquet"
    summary_path = out_dir / "toxic_rate_by_side.csv"
    definition_path = out_dir / "label_definition.md"

    labels.to_parquet(labels_path, index=False)
    summary = _build_toxic_rate_summary(labels)
    summary["fill_proxy_window_ticks"] = int(fill_window)
    summary["label_horizon_ticks"] = int(horizon)
    summary["adverse_threshold_ticks"] = float(adverse_threshold_ticks)
    summary.to_csv(summary_path, index=False)

    _write_label_definition(
        out_path=definition_path,
        input_path=input_path,
        data_dir=data_dir,
        symbol=symbol,
        fill_window=fill_window,
        horizon=horizon,
        adverse_threshold_ticks=adverse_threshold_ticks,
    )

    metadata = {
        "input_path": str(input_path),
        "data_dir": str(data_dir),
        "symbol": symbol,
        "labels_path": str(labels_path),
        "summary_path": str(summary_path),
        "definition_path": str(definition_path),
        "fill_proxy_window_ticks": int(fill_window),
        "label_horizon_ticks": int(horizon),
        "adverse_threshold_ticks": float(adverse_threshold_ticks),
        "row_count": int(len(labels)),
        "label_valid_rows": int(labels["label_valid_horizon"].sum()),
    }
    (out_dir / "binary_labels_v2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return labels_path, summary_path, definition_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-specific binary cancel-risk labels.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fill_window", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--adverse_threshold_ticks", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path, summary_path, definition_path = make_binary_labels(
        input_path=args.input_path,
        data_dir=args.data_dir,
        symbol=args.symbol,
        out_dir=args.out_dir,
        fill_window=args.fill_window,
        horizon=args.horizon,
        adverse_threshold_ticks=args.adverse_threshold_ticks,
    )
    print("Binary labels v2 complete:")
    print(f"- {labels_path}")
    print(f"- {summary_path}")
    print(f"- {definition_path}")


if __name__ == "__main__":
    main()
