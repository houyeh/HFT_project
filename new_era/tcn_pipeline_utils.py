from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


TOP_8_FEATURES = [
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
]
Z_FEATURES = [f"{name}_z" for name in TOP_8_FEATURES]

WINDOW_SIZE = 100
HORIZON = 10
THRESHOLD_TICKS = 1.0
SAMPLE_STRIDE = 5
TRAIN_SESSION_FRAC = 0.70
VAL_SESSION_FRAC = 0.15

POINT_VALUE_TWD = 200
MAKER_FEE_TWD = 15
TAKER_FEE_TWD = 50


@dataclass(frozen=True)
class SequenceRecord:
    key: str
    symbol: str
    session_id: str
    x: np.ndarray
    y: np.ndarray


def safe_session_key(session_id: str) -> str:
    """Convert folder/file session names into stable npz key fragments."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", session_id).strip("_")


def apply_rolling_zscore(df: pd.DataFrame, columns: list[str], window: int = 2000) -> pd.DataFrame:
    """
    對指定的特徵欄位計算滾動 Z-Score。
    這個函式會在單一 session 內使用，避免跨日或跨盤別把分布混在一起。
    """
    df_out = df.copy()
    for col in columns:
        r_mean = df[col].rolling(window=window, min_periods=100).mean()
        r_std = df[col].rolling(window=window, min_periods=100).std().replace(0, 1e-5)
        df_out[f"{col}_z"] = ((df[col] - r_mean) / r_std).fillna(0.0).clip(-10, 10)
    return df_out


def parse_sequence_key(key: str) -> tuple[str, str]:
    base = key[:-2] if key.endswith("_X") else key
    if "__" in base:
        symbol, session_id = base.split("__", 1)
        return symbol, session_id
    return base, "legacy_full_stream"


def load_sequence_records(npz_path: Path) -> list[SequenceRecord]:
    data = np.load(npz_path, allow_pickle=False)
    records: list[SequenceRecord] = []
    for key in data.files:
        if not key.endswith("_X"):
            continue
        y_key = f"{key[:-2]}_y"
        if y_key not in data.files:
            raise KeyError(f"Missing label array for {key}: expected {y_key}")
        symbol, session_id = parse_sequence_key(key)
        records.append(
            SequenceRecord(
                key=key[:-2],
                symbol=symbol,
                session_id=session_id,
                x=data[key],
                y=data[y_key],
            )
        )
    if not records:
        raise ValueError(f"No *_X arrays found in {npz_path}")
    return records


def split_records_by_session(
    records: list[SequenceRecord],
    train_frac: float = TRAIN_SESSION_FRAC,
    val_frac: float = VAL_SESSION_FRAC,
) -> tuple[list[SequenceRecord], list[SequenceRecord], list[SequenceRecord]]:
    """
    以 session 為單位切分資料，避免同一個交易時段同時出現在 train/val/test。
    """
    train_records: list[SequenceRecord] = []
    val_records: list[SequenceRecord] = []
    test_records: list[SequenceRecord] = []
    records_by_symbol: dict[str, list[SequenceRecord]] = defaultdict(list)

    for record in records:
        records_by_symbol[record.symbol].append(record)

    for symbol in sorted(records_by_symbol):
        symbol_records = sorted(records_by_symbol[symbol], key=lambda item: item.session_id)
        total = len(symbol_records)
        if total == 1:
            train_records.extend(symbol_records)
            continue

        train_end = max(1, int(total * train_frac))
        val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
        val_end = min(val_end, total - 1)

        train_records.extend(symbol_records[:train_end])
        val_records.extend(symbol_records[train_end:val_end])
        test_records.extend(symbol_records[val_end:])

    return train_records, val_records, test_records


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = TRAIN_SESSION_FRAC,
    val_frac: float = VAL_SESSION_FRAC,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def concatenate_records(records: list[SequenceRecord], channel_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No sequence records were provided.")
    x = np.concatenate([record.x for record in records], axis=0)
    y = np.concatenate([record.y for record in records], axis=0)
    if channel_first:
        x = np.transpose(x, (0, 2, 1))
    return x, y


def compute_class_weight_tensor(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def normalize_price(price: float) -> float:
    return float(price) / 10000.0 if price > 100000 else float(price)


def run_market_making_simulation(
    all_probs,
    all_ask_px,
    all_bid_px,
    strategy: str = "baseline",
    point_value: int = POINT_VALUE_TWD,
    maker_fee: int = MAKER_FEE_TWD,
    taker_fee: int = TAKER_FEE_TWD,
) -> dict[str, float | int]:
    """
    使用同一套成交邏輯比較 baseline / gated / alpha。
    Alpha 的主動買賣會在當下 best ask/bid 成交，不再等待下一筆報價才決定是否成交。
    """
    inventory = 0
    max_inventory = 3
    cash = 0.0
    maker_trades = 0
    taker_trades = 0

    if len(all_probs) < 2:
        return {"pnl": 0.0, "trades": 0, "maker_trades": 0, "taker_trades": 0, "ending_inventory": 0}

    for i in range(len(all_probs) - 1):
        prob_down, _, prob_up = all_probs[i]
        ask = normalize_price(all_ask_px[i])
        bid = normalize_price(all_bid_px[i])
        next_ask = normalize_price(all_ask_px[i + 1])
        next_bid = normalize_price(all_bid_px[i + 1])

        if min(ask, bid, next_ask, next_bid) <= 0:
            continue

        quote_ask = inventory > -max_inventory
        quote_bid = inventory < max_inventory
        my_ask = ask
        my_bid = bid

        if strategy == "tcn_gated":
            if prob_down > 0.8:
                quote_bid = False
            if prob_up > 0.8:
                quote_ask = False

        elif strategy == "tcn_alpha":
            if prob_up > 0.6 and inventory < max_inventory:
                quote_ask = False
                quote_bid = False
                inventory += 1
                cash -= ask * point_value
                cash -= taker_fee
                taker_trades += 1
            elif prob_down > 0.6 and inventory > -max_inventory:
                quote_ask = False
                quote_bid = False
                inventory -= 1
                cash += bid * point_value
                cash -= taker_fee
                taker_trades += 1
            else:
                if inventory >= 1:
                    quote_bid = False
                    my_ask = bid + 1
                elif inventory <= -1:
                    quote_ask = False
                    my_bid = ask - 1

        if quote_bid and next_ask <= my_bid:
            inventory += 1
            cash -= my_bid * point_value
            cash -= maker_fee
            maker_trades += 1

        if quote_ask and next_bid >= my_ask:
            inventory -= 1
            cash += my_ask * point_value
            cash -= maker_fee
            maker_trades += 1

    final_mid = (normalize_price(all_ask_px[-1]) + normalize_price(all_bid_px[-1])) / 2.0
    final_pnl = cash + inventory * final_mid * point_value
    total_trades = maker_trades + taker_trades
    return {
        "pnl": final_pnl,
        "trades": total_trades,
        "maker_trades": maker_trades,
        "taker_trades": taker_trades,
        "ending_inventory": inventory,
    }
