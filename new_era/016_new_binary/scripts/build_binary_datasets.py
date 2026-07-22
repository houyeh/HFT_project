from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.tcn_pipeline_utils import apply_rolling_zscore
from src.data.tw_replay import infer_tick_size_from_lob
from src.utils.tw_train_helpers import build_tw_feature_frame

from common import (
    DATASETS_DIR,
    DATA_AUDIT_DIR,
    DEFAULT_ZSCORE_WINDOW,
    EVAL_SAMPLE_STRIDE,
    LABELS_DIR,
    LOGS_DIR,
    NEW3_DIR,
    NEW3_SYMBOL,
    NEW_ALL_DIR,
    NEW_SYMBOL,
    REPORTS_DIR,
    SELECTED_FEATURES,
    TRAIN_SAMPLE_STRIDE,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZE,
    Z_FEATURES,
    ensure_output_dirs,
    save_json,
    split_ordered_session_ids,
)


FILL_WINDOWS = [1, 3, 5]
HORIZONS = [5, 10, 20]
PRIMARY_FILL_WINDOW = 1
PRIMARY_HORIZON = 10
TOXIC_THRESHOLD_TICKS = -0.5
PROXY_TYPE = "price_level_removed"


def _load_symbol(data_dir: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"Missing LOB symbol={symbol} in {data_dir}")
    lob = lob_frames[symbol].copy()
    trade = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()
    return lob, trade


def _build_session_features(
    lob_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    tick_size: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_feat = build_tw_feature_frame(
        lob_df.reset_index(drop=True),
        trade_df.reset_index(drop=True),
        symbol=symbol,
        tick_size=float(tick_size),
    )
    rows_before_dropna = int(len(raw_feat))
    raw_feat = raw_feat.dropna().reset_index(drop=True)
    rows_after_dropna = int(len(raw_feat))
    zdf = apply_rolling_zscore(raw_feat, SELECTED_FEATURES, window=DEFAULT_ZSCORE_WINDOW)
    zdf = zdf.reset_index(drop=True)
    return zdf, {
        "rows_before_dropna": rows_before_dropna,
        "rows_after_dropna": rows_after_dropna,
    }


def _compute_markouts_ticks(df: pd.DataFrame, tick_size: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    bid_px = df["bid_px_1"].to_numpy(dtype=np.float64)
    ask_px = df["ask_px_1"].to_numpy(dtype=np.float64)
    mid = df["mid_price"].to_numpy(dtype=np.float64)
    n = len(df)
    bid_m = np.full(n, np.nan, dtype=np.float64)
    ask_m = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return bid_m, ask_m
    future_mid = mid[horizon:]
    bid_m[:-horizon] = (future_mid - bid_px[:-horizon]) / max(tick_size, 1e-9)
    ask_m[:-horizon] = (ask_px[:-horizon] - future_mid) / max(tick_size, 1e-9)
    return bid_m, ask_m


def _compute_proxy_masks(
    df: pd.DataFrame,
    fill_window: int,
    min_size_decrease: float = 1.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    bid_px = df["bid_px_1"].to_numpy(dtype=np.float64)
    ask_px = df["ask_px_1"].to_numpy(dtype=np.float64)
    bid_vol = df["bid_vol_1"].to_numpy(dtype=np.float64)
    ask_vol = df["ask_vol_1"].to_numpy(dtype=np.float64)
    n = len(df)

    b_queue = np.zeros(n, dtype=bool)
    a_queue = np.zeros(n, dtype=bool)
    b_removed = np.zeros(n, dtype=bool)
    a_removed = np.zeros(n, dtype=bool)
    b_through = np.zeros(n, dtype=bool)
    a_through = np.zeros(n, dtype=bool)

    for i in range(0, n - fill_window):
        cbid = bid_px[i]
        cask = ask_px[i]
        cbv = bid_vol[i]
        cav = ask_vol[i]

        j0 = i + 1
        j1 = i + fill_window + 1
        f_bid = bid_px[j0:j1]
        f_ask = ask_px[j0:j1]
        f_bv = bid_vol[j0:j1]
        f_av = ask_vol[j0:j1]

        same_bid = np.abs(f_bid - cbid) <= 1e-9
        same_ask = np.abs(f_ask - cask) <= 1e-9
        if np.any(same_bid & (f_bv < (cbv - min_size_decrease))):
            b_queue[i] = True
        if np.any(same_ask & (f_av < (cav - min_size_decrease))):
            a_queue[i] = True

        if np.any(f_bid < cbid):
            b_removed[i] = True
        if np.any(f_ask > cask):
            a_removed[i] = True

        if np.any(f_ask <= cbid):
            b_through[i] = True
        if np.any(f_bid >= cask):
            a_through[i] = True

    return {
        "queue_depletion_proxy": (b_queue, a_queue),
        "price_level_removed_proxy": (b_removed, a_removed),
        "price_through_proxy": (b_through, a_through),
    }


def _aggregate_label_distribution(
    split_name: str,
    session_frames: dict[str, pd.DataFrame],
    session_tick: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for fill_window in FILL_WINDOWS:
        proxy_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for sid, fdf in session_frames.items():
            proxy_cache[sid] = _compute_proxy_masks(fdf, fill_window=fill_window, min_size_decrease=1.0)

        for horizon in HORIZONS:
            for side in ["bid", "ask"]:
                for proxy_name in ["queue_depletion_proxy", "price_level_removed_proxy", "price_through_proxy"]:
                    total_events = 0
                    total_after = 0
                    fills_markout: list[np.ndarray] = []

                    for sid, fdf in session_frames.items():
                        tick = session_tick[sid]
                        bid_markout, ask_markout = _compute_markouts_ticks(fdf, tick_size=tick, horizon=horizon)
                        max_lookahead = max(fill_window, horizon)
                        n = len(fdf)
                        start = WINDOW_SIZE - 1
                        end = n - max_lookahead
                        if end <= start:
                            continue
                        idx = np.arange(start, end, 1, dtype=np.int64)
                        total_events += int(len(idx))

                        bid_proxy, ask_proxy = proxy_cache[sid][proxy_name]
                        if side == "bid":
                            proxy_mask = bid_proxy[idx]
                            m = bid_markout[idx]
                        else:
                            proxy_mask = ask_proxy[idx]
                            m = ask_markout[idx]
                        valid = proxy_mask & np.isfinite(m)
                        total_after += int(valid.sum())
                        if valid.any():
                            fills_markout.append(m[valid])

                    if total_after > 0:
                        mark = np.concatenate(fills_markout, axis=0)
                        toxic = mark < TOXIC_THRESHOLD_TICKS
                        good = mark > 0.5
                        neutral = (~toxic) & (~good)
                        toxic_count = int(toxic.sum())
                        good_count = int(good.sum())
                        neutral_count = int(neutral.sum())
                        toxic_rate = float(toxic.mean())
                        mean_mark = float(np.mean(mark))
                        median_mark = float(np.median(mark))
                        p10 = float(np.percentile(mark, 10))
                        p90 = float(np.percentile(mark, 90))
                    else:
                        toxic_count = 0
                        good_count = 0
                        neutral_count = 0
                        toxic_rate = np.nan
                        mean_mark = np.nan
                        median_mark = np.nan
                        p10 = np.nan
                        p90 = np.nan

                    rows.append(
                        {
                            "side": side,
                            "proxy_type": proxy_name,
                            "fill_window": int(fill_window),
                            "horizon": int(horizon),
                            "split": split_name,
                            "num_events_before_conditioning": int(total_events),
                            "num_samples_after_conditioning": int(total_after),
                            "fill_proxy_rate": float(total_after / max(total_events, 1)),
                            "toxic_count": int(toxic_count),
                            "non_toxic_count": int(good_count + neutral_count),
                            "toxic_rate": float(toxic_rate),
                            "mean_markout": float(mean_mark),
                            "median_markout": float(median_mark),
                            "p10_markout": float(p10),
                            "p90_markout": float(p90),
                        }
                    )
    return rows


def _build_primary_side_arrays(
    side: str,
    split_name: str,
    session_frames: dict[str, pd.DataFrame],
    session_tick: dict[str, float],
    sample_stride: int,
    fill_window: int = PRIMARY_FILL_WINDOW,
    horizon: int = PRIMARY_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []
    sess_list: list[np.ndarray] = []
    ask_list: list[np.ndarray] = []
    bid_list: list[np.ndarray] = []

    total_events = 0
    total_after = 0
    toxic_count = 0

    for sid in sorted(session_frames):
        fdf = session_frames[sid]
        tick = session_tick[sid]
        n = len(fdf)
        max_lookahead = max(fill_window, horizon)
        start = WINDOW_SIZE - 1
        end = n - max_lookahead
        if end <= start:
            continue
        idx = np.arange(start, end, int(sample_stride), dtype=np.int64)
        total_events += int(len(idx))

        proxies = _compute_proxy_masks(fdf, fill_window=fill_window, min_size_decrease=1.0)
        bid_proxy, ask_proxy = proxies["price_level_removed_proxy"]
        bid_mark, ask_mark = _compute_markouts_ticks(fdf, tick_size=tick, horizon=horizon)

        if side == "bid":
            valid = bid_proxy[idx] & np.isfinite(bid_mark[idx])
            labels = (bid_mark[idx] < TOXIC_THRESHOLD_TICKS).astype(np.int64)
        else:
            valid = ask_proxy[idx] & np.isfinite(ask_mark[idx])
            labels = (ask_mark[idx] < TOXIC_THRESHOLD_TICKS).astype(np.int64)

        selected_idx = idx[valid]
        if selected_idx.size == 0:
            continue
        labels = labels[valid]
        total_after += int(selected_idx.size)
        toxic_count += int(labels.sum())

        feat = fdf[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
        ask = fdf["ask_px_1"].to_numpy(dtype=np.float32, copy=True)
        bid = fdf["bid_px_1"].to_numpy(dtype=np.float32, copy=True)

        x = np.zeros((selected_idx.size, len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
        for k, label_idx in enumerate(selected_idx):
            st = int(label_idx) - WINDOW_SIZE + 1
            ed = int(label_idx) + 1
            x[k] = feat[st:ed].T

        x_list.append(x)
        y_list.append(labels.astype(np.int64))
        idx_list.append(selected_idx.astype(np.int64))
        sess_list.append(np.array([sid] * selected_idx.size, dtype=object))
        ask_list.append(ask[selected_idx])
        bid_list.append(bid[selected_idx])

    if x_list:
        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        idx_all = np.concatenate(idx_list, axis=0)
        sess_all = np.concatenate(sess_list, axis=0)
        ask_all = np.concatenate(ask_list, axis=0)
        bid_all = np.concatenate(bid_list, axis=0)
    else:
        x_all = np.zeros((0, len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
        y_all = np.zeros((0,), dtype=np.int64)
        idx_all = np.zeros((0,), dtype=np.int64)
        sess_all = np.array([], dtype=object)
        ask_all = np.zeros((0,), dtype=np.float32)
        bid_all = np.zeros((0,), dtype=np.float32)

    stats = {
        "split": split_name,
        "side": side,
        "num_events_before_conditioning": int(total_events),
        "num_samples_after_conditioning": int(total_after),
        "fill_proxy_rate": float(total_after / max(total_events, 1)),
        "toxic_rate": float(toxic_count / max(total_after, 1)),
    }
    return x_all, y_all, idx_all, sess_all, ask_all, bid_all, stats


def _collect_feature_sessions(
    lob: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, dict[str, int]], dict[str, int]]:
    feature_by_session: dict[str, pd.DataFrame] = {}
    tick_by_session: dict[str, float] = {}
    audit_by_session: dict[str, dict[str, int]] = {}
    counts = {
        "raw_lob_rows": 0,
        "rows_after_feature_dropna": 0,
    }

    for sid, lob_s in lob.groupby("session_id", sort=True):
        trd_s = trades[trades["session_id"] == sid]
        tick = float(infer_tick_size_from_lob(lob_s))
        fdf, audit = _build_session_features(lob_s, trd_s, symbol=symbol, tick_size=tick)
        if len(fdf) < WINDOW_SIZE + PRIMARY_HORIZON + PRIMARY_FILL_WINDOW:
            continue
        feature_by_session[str(sid)] = fdf
        tick_by_session[str(sid)] = tick
        audit_by_session[str(sid)] = audit
        counts["raw_lob_rows"] += int(len(lob_s))
        counts["rows_after_feature_dropna"] += int(len(fdf))
    return feature_by_session, tick_by_session, audit_by_session, counts


def main() -> None:
    ensure_output_dirs()

    new_lob, new_trade = _load_symbol(NEW_ALL_DIR, NEW_SYMBOL)
    new3_lob, new3_trade = _load_symbol(NEW3_DIR, NEW3_SYMBOL)

    # Data source audit
    audit_lines = []
    audit_lines.append("# Data Source Audit")
    audit_lines.append("")
    audit_lines.append(f"- NEW_ALL exists: {NEW_ALL_DIR.exists()}")
    audit_lines.append(f"- NEW3 exists: {NEW3_DIR.exists()}")
    audit_lines.append(f"- NEW symbol loaded: {NEW_SYMBOL} (rows={len(new_lob):,})")
    audit_lines.append(f"- NEW3 symbol loaded: {NEW3_SYMBOL} (rows={len(new3_lob):,})")
    audit_lines.append("")
    audit_lines.append("## LOB Columns")
    audit_lines.append(", ".join(new_lob.columns.tolist()))
    audit_lines.append("")
    audit_lines.append("## Trade Columns")
    audit_lines.append(", ".join(new_trade.columns.tolist()))
    audit_lines.append("")
    audit_lines.append("## Fill Truth Limitation")
    audit_lines.append("- aggressor_side available: NO")
    audit_lines.append("- this pipeline is proxy-based, not true fill truth.")
    (DATA_AUDIT_DIR / "data_source_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    # Build features per session
    new_features, new_ticks, new_audit, new_counts = _collect_feature_sessions(new_lob, new_trade, NEW_SYMBOL)
    new3_features, new3_ticks, new3_audit, new3_counts = _collect_feature_sessions(new3_lob, new3_trade, NEW3_SYMBOL)

    new_sessions = sorted(new_features.keys())
    train_s, val_s, test_s = split_ordered_session_ids(new_sessions, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)

    split_frames_new = {
        "NEW_train": {sid: new_features[sid] for sid in train_s if sid in new_features},
        "NEW_val": {sid: new_features[sid] for sid in val_s if sid in new_features},
        "NEW_test": {sid: new_features[sid] for sid in test_s if sid in new_features},
    }
    split_ticks_new = {
        "NEW_train": {sid: new_ticks[sid] for sid in train_s if sid in new_ticks},
        "NEW_val": {sid: new_ticks[sid] for sid in val_s if sid in new_ticks},
        "NEW_test": {sid: new_ticks[sid] for sid in test_s if sid in new_ticks},
    }
    split_frames_new3 = {"NEW3_full": new3_features}
    split_ticks_new3 = {"NEW3_full": new3_ticks}

    # Label distribution (B/C/D, fill windows, horizons)
    rows: list[dict[str, object]] = []
    for split in ["NEW_train", "NEW_val", "NEW_test"]:
        rows.extend(_aggregate_label_distribution(split, split_frames_new[split], split_ticks_new[split]))
    rows.extend(_aggregate_label_distribution("NEW3_full", split_frames_new3["NEW3_full"], split_ticks_new3["NEW3_full"]))
    label_dist_df = pd.DataFrame(rows)
    label_dist_df.to_csv(LABELS_DIR / "binary_label_distribution.csv", index=False)

    # Primary training datasets (Proxy C, fill_window=1, horizon=10)
    dataset_meta: dict[str, dict[str, object]] = {}
    for side in ["bid", "ask"]:
        train_pack = _build_primary_side_arrays(
            side=side,
            split_name="NEW_train",
            session_frames=split_frames_new["NEW_train"],
            session_tick=split_ticks_new["NEW_train"],
            sample_stride=TRAIN_SAMPLE_STRIDE,
        )
        val_pack = _build_primary_side_arrays(
            side=side,
            split_name="NEW_val",
            session_frames=split_frames_new["NEW_val"],
            session_tick=split_ticks_new["NEW_val"],
            sample_stride=EVAL_SAMPLE_STRIDE,
        )
        test_pack = _build_primary_side_arrays(
            side=side,
            split_name="NEW_test",
            session_frames=split_frames_new["NEW_test"],
            session_tick=split_ticks_new["NEW_test"],
            sample_stride=EVAL_SAMPLE_STRIDE,
        )
        new3_pack = _build_primary_side_arrays(
            side=side,
            split_name="NEW3_full",
            session_frames=split_frames_new3["NEW3_full"],
            session_tick=split_ticks_new3["NEW3_full"],
            sample_stride=EVAL_SAMPLE_STRIDE,
        )

        npz_path = DATASETS_DIR / f"{side}_primary_fw1_h10.npz"
        np.savez_compressed(
            npz_path,
            X_train=train_pack[0],
            y_train=train_pack[1],
            idx_train=train_pack[2],
            session_train=train_pack[3],
            ask_train=train_pack[4],
            bid_train=train_pack[5],
            X_val=val_pack[0],
            y_val=val_pack[1],
            idx_val=val_pack[2],
            session_val=val_pack[3],
            ask_val=val_pack[4],
            bid_val=val_pack[5],
            X_test=test_pack[0],
            y_test=test_pack[1],
            idx_test=test_pack[2],
            session_test=test_pack[3],
            ask_test=test_pack[4],
            bid_test=test_pack[5],
            X_new3=new3_pack[0],
            y_new3=new3_pack[1],
            idx_new3=new3_pack[2],
            session_new3=new3_pack[3],
            ask_new3=new3_pack[4],
            bid_new3=new3_pack[5],
        )

        dataset_meta[side] = {
            "npz_path": str(npz_path),
            "train_stats": train_pack[6],
            "val_stats": val_pack[6],
            "test_stats": test_pack[6],
            "new3_stats": new3_pack[6],
        }

    save_json(DATASETS_DIR / "primary_dataset_metadata.json", dataset_meta)

    # Save compact session split definitions for reuse
    split_meta = {
        "train_sessions": train_s,
        "val_sessions": val_s,
        "test_sessions": test_s,
    }
    save_json(DATASETS_DIR / "new_session_split.json", split_meta)
    save_json(
        DATASETS_DIR / "dataset_build_metadata.json",
        {
            "window_size": WINDOW_SIZE,
            "train_sample_stride": TRAIN_SAMPLE_STRIDE,
            "eval_sample_stride": EVAL_SAMPLE_STRIDE,
            "primary_proxy_type": PROXY_TYPE,
            "primary_fill_window": PRIMARY_FILL_WINDOW,
            "primary_horizon": PRIMARY_HORIZON,
            "toxic_threshold_ticks": TOXIC_THRESHOLD_TICKS,
            "selected_features": SELECTED_FEATURES,
            "z_features": Z_FEATURES,
            "new_counts": new_counts,
            "new3_counts": new3_counts,
            "new_audit_by_session": new_audit,
            "new3_audit_by_session": new3_audit,
        },
    )

    # Reports
    report_lines = []
    report_lines.append("# Binary Label Definition Report")
    report_lines.append("")
    report_lines.append("## Why old binary labels failed")
    report_lines.append("- toxic_bid/ask_binary was not fill-conditional.")
    report_lines.append("- adverse_bid/ask_fill_binary used immediate-cross fill proxy and became too sparse.")
    report_lines.append("- active_buy/sell labels target taker alpha, not maker cancel risk.")
    report_lines.append("")
    report_lines.append("## New definition")
    report_lines.append("- side-specific fill-conditional quote-risk labels.")
    report_lines.append("- primary proxy: price_level_removed (Proxy C).")
    report_lines.append("- primary setting: fill_window=1, horizon=10, toxic if markout_ticks < -0.5.")
    report_lines.append("- proxy limitation: level removed can be fill or cancel.")
    report_lines.append("- distribution-shift limitation: training conditioned on proxy-filled events.")
    report_lines.append("")

    for side in ["bid", "ask"]:
        md = dataset_meta[side]
        report_lines.append(f"### {side} toxic rates (primary)")
        for key in ["train_stats", "val_stats", "test_stats", "new3_stats"]:
            st = md[key]
            report_lines.append(
                f"- {key.replace('_stats','')}: samples={int(st['num_samples_after_conditioning']):,}, "
                f"fill_proxy_rate={st['fill_proxy_rate']:.6f}, toxic_rate={st['toxic_rate']:.6f}"
            )
        report_lines.append("")

    (REPORTS_DIR / "binary_label_definition_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    index_lines = []
    index_lines.append("# Index Alignment Report")
    index_lines.append("")
    index_lines.append("Pipeline alignment:")
    index_lines.append("1. raw LOB/trade -> feature frame")
    index_lines.append("2. dropna on feature frame")
    index_lines.append("3. rolling zscore (causal)")
    index_lines.append("4. windowing with window_size=100, label index=i=end-1")
    index_lines.append("5. horizon lookahead for markout (future i+H)")
    index_lines.append("6. fill proxy conditioning")
    index_lines.append("")
    index_lines.append(f"- NEW raw rows aggregated: {new_counts['raw_lob_rows']:,}")
    index_lines.append(f"- NEW rows after feature/dropna: {new_counts['rows_after_feature_dropna']:,}")
    index_lines.append(f"- NEW3 raw rows aggregated: {new3_counts['raw_lob_rows']:,}")
    index_lines.append(f"- NEW3 rows after feature/dropna: {new3_counts['rows_after_feature_dropna']:,}")
    index_lines.append("")
    for side in ["bid", "ask"]:
        md = dataset_meta[side]
        index_lines.append(f"## {side} primary dataset counts")
        for key in ["train_stats", "val_stats", "test_stats", "new3_stats"]:
            st = md[key]
            index_lines.append(
                f"- {key.replace('_stats','')}: before={int(st['num_events_before_conditioning']):,}, "
                f"after={int(st['num_samples_after_conditioning']):,}, fill_proxy_rate={st['fill_proxy_rate']:.6f}"
            )
        index_lines.append("")
    (REPORTS_DIR / "index_alignment_report.md").write_text("\n".join(index_lines), encoding="utf-8")

    print("Dataset build done.")
    print(f"Output root: {DATASETS_DIR.parent}")
    print(f"Label distribution: {LABELS_DIR / 'binary_label_distribution.csv'}")
    print(f"Primary datasets: {DATASETS_DIR}")


if __name__ == "__main__":
    main()
