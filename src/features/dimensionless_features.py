"""
無因次特徵工程模組 (Dimensionless Feature Engineering)

負責將原始 LOB 快照轉換為跨資產通用的無因次特徵：
  1. Normalized Spread (正規化價差)
  2. Mid-price Micro Return (微觀動能報酬)
  3. Order Book Imbalance, OBI (委託簿失衡)
  4. Weighted Mid-price (加權中間價)
  5. Volume Imbalance across levels (多檔量失衡)

所有特徵均以向量化 (Vectorized) 方式計算，避免 Python for-loop。
"""

import numpy as np
import pandas as pd

# Module note: the functions below build the base scale-invariant feature set
# used across both the Taiwan replay and CME transfer experiments.

from src.utils.constants import TICK_SIZES, LOB_DEPTH


def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    計算中間價 (Mid Price)：
      Mid = (Best Ask + Best Bid) / 2
    """
    return (df["ask_px_1"] + df["bid_px_1"]) / 2


def compute_normalized_spread(df: pd.DataFrame, tick_size: float) -> pd.Series:
    """
    正規化價差 (Normalized Spread)：
      Spread_norm = (Ask_1 - Bid_1) / Tick_Size

    結果為「幾個 Tick 寬」，跨資產可比。
    """
    spread = df["ask_px_1"] - df["bid_px_1"]
    return spread / tick_size


def compute_micro_return(df: pd.DataFrame, tick_size: float) -> pd.Series:
    """
    微觀動能報酬 (Micro-price Return)：
      Return_mid = (Mid_t - Mid_{t-1}) / Tick_Size

    以 Tick 為單位衡量價格變動幅度，跨資產可比。
    第一筆設為 0（沒有前一筆）。
    """
    mid = compute_mid_price(df)
    ret = mid.diff() / tick_size
    ret.iloc[0] = 0.0
    return ret


def compute_obi(df: pd.DataFrame) -> pd.Series:
    """
    委託簿失衡 (Order Book Imbalance, OBI)：
      OBI = (V_bid1 - V_ask1) / (V_bid1 + V_ask1)

    數值範圍 [-1, 1]：
      +1 表示完全由買方主導（即將上漲的機率高）
      -1 表示完全由賣方主導
    """
    v_bid = df["bid_vol_1"].astype(float)
    v_ask = df["ask_vol_1"].astype(float)
    total = v_bid + v_ask
    obi = (v_bid - v_ask) / total
    # 分母為 0 時設為 0
    obi = obi.fillna(0.0)
    return obi


def compute_weighted_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    加權中間價 (Volume-Weighted Mid Price)：
      Weighted_Mid = (Ask_1 * V_bid1 + Bid_1 * V_ask1) / (V_bid1 + V_ask1)

    比簡單中間價更能反映真實的「均衡價格」。
    """
    v_bid = df["bid_vol_1"].astype(float)
    v_ask = df["ask_vol_1"].astype(float)
    total = v_bid + v_ask
    wmid = (df["ask_px_1"] * v_bid + df["bid_px_1"] * v_ask) / total
    wmid = wmid.fillna(compute_mid_price(df))
    return wmid


def compute_deep_obi(df: pd.DataFrame, depth: int = LOB_DEPTH) -> pd.Series:
    """
    多檔量失衡 (Deep Order Book Imbalance)：
      Deep_OBI = (sum(V_bid_1..N) - sum(V_ask_1..N)) / (sum(V_bid_1..N) + sum(V_ask_1..N))

    納入更多深度的掛單量，更穩健地判斷供需壓力。
    """
    bid_vol_sum = sum(df[f"bid_vol_{i}"].astype(float) for i in range(1, depth + 1))
    ask_vol_sum = sum(df[f"ask_vol_{i}"].astype(float) for i in range(1, depth + 1))
    total = bid_vol_sum + ask_vol_sum
    deep_obi = (bid_vol_sum - ask_vol_sum) / total
    deep_obi = deep_obi.fillna(0.0)
    return deep_obi


def build_feature_dataframe(
    df: pd.DataFrame,
    symbol: str,
    tick_size: float | None = None,
) -> pd.DataFrame:
    """
    對某商品的清理好的 LOB DataFrame，計算所有無因次特徵，
    回傳一個只包含特徵欄位的新 DataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        已經過清理 (load_and_clean) 的原始 LOB 資料。
    symbol : str
        商品代碼 ("ES", "NQ", "CL")。若提供 tick_size，則 symbol 只作為識別用途。
    tick_size : float | None
        若為 None，使用 constants.py 中對應商品的 tick size；
        否則使用傳入值，方便擴充到非 CME 資料來源。

    Returns
    -------
    pd.DataFrame
        包含以下欄位：
        - normalized_spread
        - micro_return
        - obi
        - weighted_mid_price
        - deep_obi
    """
    tick_size = TICK_SIZES[symbol] if tick_size is None else tick_size

    features = pd.DataFrame({
        "timestamp":          df["timestamp"].values,
        "mid_price":          compute_mid_price(df).values,
        "normalized_spread":  compute_normalized_spread(df, tick_size).values,
        "micro_return":       compute_micro_return(df, tick_size).values,
        "obi":                compute_obi(df).values,
        "weighted_mid_price": compute_weighted_mid_price(df).values,
        "deep_obi":           compute_deep_obi(df).values,
    })

    return features
