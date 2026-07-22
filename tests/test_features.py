"""
Week 1-2 單元測試
驗證資料前處理、特徵工程、滑動視窗的正確性。
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Module note: these tests guard the earliest preprocessing path that turns mock
# order-book CSV files into features and sliding-window tensors.

# 確保 import 路徑正確
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import (
    load_and_clean,
    validate_columns,
    handle_missing_values,
)
from src.features.dimensionless_features import (
    compute_mid_price,
    compute_normalized_spread,
    compute_micro_return,
    compute_obi,
    compute_deep_obi,
    build_feature_dataframe,
)
from src.features.sliding_window import (
    label_direction,
    create_sliding_windows,
)
from src.utils.constants import TICK_SIZES


# ===== 測試用小型 DataFrame =====
def make_tiny_df():
    """產生一個 20 筆的迷你 LOB DataFrame，用於單元測試。"""
    np.random.seed(42)
    n = 20
    data = {"timestamp": np.arange(1, n + 1) * 1000}
    base_ask = 100.0
    base_bid = 99.75
    for i in range(1, 6):
        data[f"ask_px_{i}"] = base_ask + (i - 1) * 0.25
        data[f"ask_vol_{i}"] = np.random.randint(1, 50, size=n)
        data[f"bid_px_{i}"] = base_bid - (i - 1) * 0.25
        data[f"bid_vol_{i}"] = np.random.randint(1, 50, size=n)
    return pd.DataFrame(data)


# ---------- Loader 測試 ----------

class TestLoader:
    def test_validate_columns_pass(self):
        df = make_tiny_df()
        # 應該不會拋出例外
        validate_columns(df)

    def test_validate_columns_fail(self):
        df = make_tiny_df().drop(columns=["ask_px_1"])
        with pytest.raises(ValueError, match="缺少必要欄位"):
            validate_columns(df)

    def test_handle_missing_values_fills_nan(self):
        df = make_tiny_df()
        df.loc[5, "ask_px_1"] = np.nan
        df.loc[10, "bid_vol_1"] = np.nan
        cleaned = handle_missing_values(df)
        assert not cleaned.isnull().any().any(), "不應有任何 NaN 殘留"

    def test_load_and_clean_es(self):
        """如果有假資料，實際測試載入 ES。"""
        data_dir = os.path.join(os.path.dirname(__file__), "..",  "data", "raw")
        if not os.path.exists(os.path.join(data_dir, "mock_ES.csv")):
            pytest.skip("mock_ES.csv 尚未產生，跳過")
        df = load_and_clean("ES", data_dir)
        assert len(df) > 0
        assert "ask_px_1" in df.columns


# ---------- 特徵工程測試 ----------

class TestFeatures:
    def test_mid_price_range(self):
        df = make_tiny_df()
        mid = compute_mid_price(df)
        assert (mid >= df["bid_px_1"]).all()
        assert (mid <= df["ask_px_1"]).all()

    def test_normalized_spread_positive(self):
        df = make_tiny_df()
        ns = compute_normalized_spread(df, tick_size=0.25)
        assert (ns >= 0).all(), "正規化價差不能為負值"

    def test_micro_return_first_zero(self):
        df = make_tiny_df()
        mr = compute_micro_return(df, tick_size=0.25)
        assert mr.iloc[0] == 0.0, "第一筆微觀報酬必須是 0"

    def test_obi_range(self):
        df = make_tiny_df()
        obi = compute_obi(df)
        assert (obi >= -1).all() and (obi <= 1).all(), "OBI 須在 [-1, 1] 之間"

    def test_deep_obi_range(self):
        df = make_tiny_df()
        dobi = compute_deep_obi(df)
        assert (dobi >= -1).all() and (dobi <= 1).all()

    def test_build_feature_no_nan(self):
        df = make_tiny_df()
        feat = build_feature_dataframe(df, "ES")
        assert not feat.isnull().any().any(), "特徵 DataFrame 不應有 NaN"


# ---------- 滑動視窗測試 ----------

class TestSlidingWindow:
    def test_label_direction_shapes(self):
        mid = np.array([100.0, 100.5, 101.0, 100.0, 99.5, 100.0, 101.0, 100.5,
                        100.0, 99.0])
        labels = label_direction(mid, horizon=2, threshold=0.3)
        assert len(labels) == len(mid) - 2

    def test_label_values_valid(self):
        mid = np.linspace(100, 110, 50)
        labels = label_direction(mid, horizon=5, threshold=0.1)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_sliding_window_shapes(self):
        df = make_tiny_df()
        feat = build_feature_dataframe(df, "ES")
        window_size = 5
        horizon = 2
        X, y = create_sliding_windows(feat, window_size=window_size,
                                      horizon=horizon, threshold=0.5)
        expected_samples = len(feat) - window_size - horizon + 1
        assert X.shape[0] == expected_samples
        assert X.shape[1] == window_size
        assert y.shape[0] == expected_samples

    def test_sliding_window_too_small_raises(self):
        """資料量不夠大時，應該拋出 ValueError。"""
        df = make_tiny_df().head(3)
        feat = build_feature_dataframe(df, "ES")
        with pytest.raises(ValueError, match="資料不足"):
            create_sliding_windows(feat, window_size=100, horizon=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
