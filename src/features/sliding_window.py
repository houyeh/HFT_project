"""
滑動視窗與張量轉換模組 (Sliding Window & Tensor Conversion)

負責：
  1. 將 2D 特徵 DataFrame 切割為 3D 張量 (Samples, Sequence, Features)
  2. 標記未來 N-tick 的價格方向 (Label: 漲=2, 平=1, 跌=0)

這是將表格式資料轉換為 TCN 模型可接受輸入的最關鍵步驟。
"""

import numpy as np
import pandas as pd

# Module note: this file defines how event-level features are converted into
# fixed-length windows and how future price movement labels are assigned.

from src.utils.constants import DEFAULT_WINDOW_SIZE, DEFAULT_HORIZON, DEFAULT_THRESHOLD


def label_direction(mid_prices: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """
    根據未來 N-tick 的 mid-price 變動方向標記 Label。

    規則：
      - 未來 mid-price 漲超過 threshold 個 tick → Label = 2 (Up)
      - 未來 mid-price 跌超過 threshold 個 tick → Label = 0 (Down)
      - 其餘 → Label = 1 (Stationary)

    Parameters
    ----------
    mid_prices : np.ndarray
        中間價序列。
    horizon : int
        看未來幾個 Tick。
    threshold : float
        漲跌判定閾值（以 Tick 為單位的變動量）。

    Returns
    -------
    np.ndarray
        長度為 len(mid_prices) - horizon 的 int 陣列。
    """
    n = len(mid_prices)
    if n <= horizon:
        raise ValueError(
            f"資料不足: 需要超過 {horizon} 筆資料來計算 label，"
            f"但只有 {n} 筆。"
        )
    labels = np.empty(n - horizon, dtype=np.int64)

    future_mid = mid_prices[horizon:]
    current_mid = mid_prices[:n - horizon]
    diff = future_mid - current_mid

    labels[diff > threshold] = 2   # Up
    labels[diff < -threshold] = 0  # Down
    labels[(diff >= -threshold) & (diff <= threshold)] = 1  # Stationary

    return labels


def label_direction_dynamic(
    mid_prices: np.ndarray,
    horizon: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Dynamic-threshold version of label_direction.

    thresholds can be length N or N-horizon. In both cases we use the threshold
    at the current decision time t to label the move from t to t+horizon.
    """

    n = len(mid_prices)
    if n <= horizon:
        raise ValueError(
            f"資料不足: 需要超過 {horizon} 筆資料來計算 label，"
            f"但只有 {n} 筆。"
        )

    thresholds = np.asarray(thresholds, dtype=np.float64)
    if len(thresholds) == n:
        current_thresholds = thresholds[: n - horizon]
    elif len(thresholds) == n - horizon:
        current_thresholds = thresholds
    else:
        raise ValueError(
            f"thresholds 長度必須是 {n} 或 {n - horizon}，但收到 {len(thresholds)}。"
        )

    labels = np.empty(n - horizon, dtype=np.int64)
    future_mid = mid_prices[horizon:]
    current_mid = mid_prices[: n - horizon]
    diff = future_mid - current_mid

    labels[diff > current_thresholds] = 2
    labels[diff < -current_thresholds] = 0
    labels[(diff >= -current_thresholds) & (diff <= current_thresholds)] = 1
    return labels


def create_sliding_windows(
    feature_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
    feature_cols: list = None,
) -> tuple:
    """
    將 2D 的特徵 DataFrame 轉換為 3D 張量與對應的標籤。

    Parameters
    ----------
    feature_df : pd.DataFrame
        由 build_feature_dataframe() 產出的特徵 DataFrame。
    window_size : int
        每個樣本的時間序列長度 (多少個 Ticks)。
    horizon : int
        預測距離 N-ticks ahead。
    threshold : float
        漲跌分類閾值。
    feature_cols : list[str] or None
        要使用的特徵欄位名稱。None 則使用所有數值特徵。

    Returns
    -------
    X : np.ndarray, shape = (num_samples, window_size, num_features)
        3D 特徵張量。
    y : np.ndarray, shape = (num_samples,)
        每個樣本的 Label (0=Down, 1=Stationary, 2=Up)。
    """
    if feature_cols is None:
        # 預設使用所有非 timestamp 的數值特徵
        feature_cols = [c for c in feature_df.columns
                        if c not in ("timestamp", "mid_price")]

    # 取出特徵矩陣 (2D)
    feat_matrix = feature_df[feature_cols].values.astype(np.float32)
    mid_prices = feature_df["mid_price"].values.astype(np.float64)

    # 計算 Label (長度 = 總行數 - horizon)
    labels = label_direction(mid_prices, horizon, threshold)

    # 可產生的樣本數 = 總行數 - window_size - horizon + 1
    num_samples = len(labels) - window_size + 1
    if num_samples <= 0:
        raise ValueError(
            f"資料不足: 需要至少 {window_size + horizon} 筆資料，"
            f"但只有 {len(feature_df)} 筆。"
        )

    num_features = feat_matrix.shape[1]
    X = np.empty((num_samples, window_size, num_features), dtype=np.float32)
    y = np.empty(num_samples, dtype=np.int64)

    for i in range(num_samples):
        X[i] = feat_matrix[i: i + window_size]
        y[i] = labels[i + window_size - 1]  # 用視窗最後一筆的 Label

    return X, y


def print_tensor_summary(X: np.ndarray, y: np.ndarray, symbol: str = "") -> None:
    """
    列印 3D 張量與標籤的摘要資訊。

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    symbol : str
    """
    prefix = f"[{symbol}] " if symbol else ""
    print(f"{prefix}X shape: {X.shape}  (Samples, Sequence, Features)")
    print(f"{prefix}y shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "Down", 1: "Stationary", 2: "Up"}
    for u, c in zip(unique, counts):
        pct = c / len(y) * 100
        print(f"  Label {u} ({label_names.get(u, '?')}): {c} ({pct:.1f}%)")
