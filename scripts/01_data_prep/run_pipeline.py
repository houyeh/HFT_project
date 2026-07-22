"""
Week 1-2 End-to-End Pipeline Runner
====================================
一鍵執行完整的資料前處理 → 特徵工程 → 滑動視窗張量轉換管線。
產出的 3D 張量將儲存至 data/processed/ 目錄。

使用方式：
  python scripts/01_data_prep/run_pipeline.py
"""

import os
import sys
import numpy as np

# Module note: this runner reflects the earliest version of the project
# pipeline. It uses the mock CSV loader instead of the replay/CME adapters.

# 確保可以 import src 模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.loader import load_and_clean
from src.features.dimensionless_features import build_feature_dataframe
from src.features.sliding_window import create_sliding_windows, print_tensor_summary
from src.utils.constants import DEFAULT_WINDOW_SIZE, DEFAULT_HORIZON, DEFAULT_THRESHOLD


def run_pipeline(symbols=None, data_dir="data/raw", output_dir="data/processed"):
    """
    針對指定商品列表執行完整管線，並儲存 3D 張量。

    Parameters
    ----------
    symbols : list[str]
        商品代碼列表。
    data_dir : str
        原始資料目錄。
    output_dir : str
        處理後資料輸出目錄。
    """
    if symbols is None:
        symbols = ["ES", "NQ", "CL"]

    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  Processing: {symbol}")
        print(f"{'='*60}")

        # Step 1: 資料讀取與清理
        print("\n[Step 1] 資料讀取與清理...")
        df = load_and_clean(symbol, data_dir)

        # Step 2: 無因次特徵工程
        print("[Step 2] 計算無因次特徵...")
        features = build_feature_dataframe(df, symbol)
        print(f"  特徵欄位: {list(features.columns)}")
        print(f"  特徵矩陣大小: {features.shape}")

        # Step 3: 滑動視窗 → 3D 張量
        print(f"[Step 3] 滑動視窗轉換 (Window={DEFAULT_WINDOW_SIZE}, "
              f"Horizon={DEFAULT_HORIZON}, Threshold={DEFAULT_THRESHOLD})...")
        X, y = create_sliding_windows(
            features,
            window_size=DEFAULT_WINDOW_SIZE,
            horizon=DEFAULT_HORIZON,
            threshold=DEFAULT_THRESHOLD,
        )
        print_tensor_summary(X, y, symbol)

        # Step 4: 儲存為 .npz
        save_path = os.path.join(output_dir, f"{symbol}_tensor.npz")
        np.savez_compressed(save_path, X=X, y=y)
        print(f"\n💾 已儲存至: {save_path}")

    print(f"\n{'='*60}")
    print("  ✅ 所有商品處理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()
