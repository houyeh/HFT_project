from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fix_common import (
    DATASETS_DIR,
    HORIZON,
    NEW3_SYMBOL,
    NEW_SYMBOL,
    WINDOW_SIZE,
    Z_FEATURES,
    build_processed_frame,
    ensure_fix_dirs,
    load_new_split_frames,
    save_json,
)


FILL_WINDOW = 1
TOXIC_THRESHOLD = -0.5
TRAIN_STRIDE = 5
EVAL_STRIDE = 1


def _proxy_masks_price_level_removed(frame: pd.DataFrame, fill_window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    bid_px = frame["bid_px_1"].to_numpy(dtype=np.float64)
    ask_px = frame["ask_px_1"].to_numpy(dtype=np.float64)
    n = len(frame)
    bid_proxy = np.zeros(n, dtype=bool)
    ask_proxy = np.zeros(n, dtype=bool)
    for i in range(0, n - fill_window):
        j0 = i + 1
        j1 = i + fill_window + 1
        if np.any(bid_px[j0:j1] < bid_px[i]):
            bid_proxy[i] = True
        if np.any(ask_px[j0:j1] > ask_px[i]):
            ask_proxy[i] = True
    return bid_proxy, ask_proxy


def _side_markouts_ticks(frame: pd.DataFrame, tick_size: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    n = len(frame)
    bm = np.full(n, np.nan, dtype=np.float64)
    am = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return bm, am
    fut = mid[horizon:]
    bm[:-horizon] = (fut - bid[:-horizon]) / max(float(tick_size), 1e-9)
    am[:-horizon] = (ask[:-horizon] - fut) / max(float(tick_size), 1e-9)
    return bm, am


def _build_side_arrays(
    *,
    side: str,
    frame: pd.DataFrame,
    tick_size: float,
    sample_stride: int,
    fill_window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = len(frame)
    start = WINDOW_SIZE - 1
    end = n - max(fill_window, horizon)
    if end <= start:
        empty_x = np.zeros((0, len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
        empty = np.zeros((0,), dtype=np.float32)
        return empty_x, empty.astype(np.int64), empty.astype(np.int64), np.array([], dtype=object), empty, empty, {
            "num_samples": 0,
            "positive_count": 0,
            "positive_rate": float("nan"),
            "fill_proxy_rate": float("nan"),
            "toxic_given_fill_rate": float("nan"),
            "mean_markout_for_positive": float("nan"),
        }

    idx = np.arange(start, end, int(sample_stride), dtype=np.int64)
    bid_proxy, ask_proxy = _proxy_masks_price_level_removed(frame, fill_window=fill_window)
    bid_mark, ask_mark = _side_markouts_ticks(frame, tick_size=tick_size, horizon=horizon)

    if side == "bid":
        fill_proxy = bid_proxy[idx]
        mark = bid_mark[idx]
    else:
        fill_proxy = ask_proxy[idx]
        mark = ask_mark[idx]

    valid = np.isfinite(mark)
    idx = idx[valid]
    fill_proxy = fill_proxy[valid]
    mark = mark[valid]

    y = ((fill_proxy) & (mark < TOXIC_THRESHOLD)).astype(np.int64)
    x_feat = frame[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
    ask = frame["ask_px_1"].to_numpy(dtype=np.float32, copy=True)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float32, copy=True)
    sid = frame["session_id"].to_numpy(copy=True)

    X = np.zeros((len(idx), len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
    for k, ii in enumerate(idx):
        st = int(ii) - WINDOW_SIZE + 1
        ed = int(ii) + 1
        X[k] = x_feat[st:ed].T

    pos_mask = y == 1
    stats = {
        "num_samples": int(len(y)),
        "positive_count": int(pos_mask.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "fill_proxy_rate": float(fill_proxy.mean()) if len(fill_proxy) else float("nan"),
        "toxic_given_fill_rate": float(((mark < TOXIC_THRESHOLD) & fill_proxy).sum() / max(fill_proxy.sum(), 1)),
        "mean_markout_for_positive": float(mark[pos_mask].mean()) if pos_mask.any() else float("nan"),
    }
    return X, y, idx, sid[idx], ask[idx], bid[idx], stats


def _save_side_npz(side: str, packs: dict[str, tuple[np.ndarray, ...]]) -> Path:
    out_path = DATASETS_DIR / f"{side}_all_event_fw1_h10.npz"
    np.savez_compressed(
        out_path,
        X_train=packs["NEW_train"][0],
        y_train=packs["NEW_train"][1],
        idx_train=packs["NEW_train"][2],
        session_train=packs["NEW_train"][3],
        ask_train=packs["NEW_train"][4],
        bid_train=packs["NEW_train"][5],
        X_val=packs["NEW_val"][0],
        y_val=packs["NEW_val"][1],
        idx_val=packs["NEW_val"][2],
        session_val=packs["NEW_val"][3],
        ask_val=packs["NEW_val"][4],
        bid_val=packs["NEW_val"][5],
        X_test=packs["NEW_test"][0],
        y_test=packs["NEW_test"][1],
        idx_test=packs["NEW_test"][2],
        session_test=packs["NEW_test"][3],
        ask_test=packs["NEW_test"][4],
        bid_test=packs["NEW_test"][5],
        X_new3=packs["NEW3_full"][0],
        y_new3=packs["NEW3_full"][1],
        idx_new3=packs["NEW3_full"][2],
        session_new3=packs["NEW3_full"][3],
        ask_new3=packs["NEW3_full"][4],
        bid_new3=packs["NEW3_full"][5],
    )
    return out_path


def main() -> None:
    ensure_fix_dirs()
    split_frames = load_new_split_frames()
    rows: list[dict[str, object]] = []
    side_meta: dict[str, dict[str, object]] = {}

    for side in ["bid", "ask"]:
        packs = {}
        for split_name, (frame, tick) in split_frames.items():
            stride = TRAIN_STRIDE if split_name == "NEW_train" else EVAL_STRIDE
            arrs = _build_side_arrays(
                side=side,
                frame=frame,
                tick_size=tick,
                sample_stride=stride,
                fill_window=FILL_WINDOW,
                horizon=HORIZON,
            )
            packs[split_name] = arrs
            st = arrs[6]
            rows.append(
                {
                    "side": side,
                    "split": split_name,
                    "num_samples": int(st["num_samples"]),
                    "positive_count": int(st["positive_count"]),
                    "negative_count": int(st["num_samples"] - st["positive_count"]),
                    "positive_rate": float(st["positive_rate"]),
                    "fill_proxy_rate": float(st["fill_proxy_rate"]),
                    "toxic_given_fill_rate": float(st["toxic_given_fill_rate"]),
                    "mean_markout_for_positive": float(st["mean_markout_for_positive"]),
                    "notes": "",
                }
            )
        npz_path = _save_side_npz(side, packs)
        side_meta[side] = {
            "npz_path": str(npz_path),
            "splits": {
                k: {
                    "num_samples": int(v[6]["num_samples"]),
                    "positive_count": int(v[6]["positive_count"]),
                    "positive_rate": float(v[6]["positive_rate"]),
                    "fill_proxy_rate": float(v[6]["fill_proxy_rate"]),
                    "toxic_given_fill_rate": float(v[6]["toxic_given_fill_rate"]),
                }
                for k, v in packs.items()
            },
        }

    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(DATASETS_DIR / "all_event_label_distribution.csv", index=False)
    save_json(
        DATASETS_DIR / "all_event_dataset_metadata.json",
        {
            "label_type": "all_event_quote_risk_binary_proxyC",
            "proxy_type": "price_level_removed_proxy",
            "fill_window": FILL_WINDOW,
            "horizon": HORIZON,
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "eval_stride": EVAL_STRIDE,
            "toxic_threshold": TOXIC_THRESHOLD,
            "sides": side_meta,
        },
    )
    print("Wrote:")
    print(DATASETS_DIR / "all_event_label_distribution.csv")
    print(DATASETS_DIR / "all_event_dataset_metadata.json")


if __name__ == "__main__":
    main()

