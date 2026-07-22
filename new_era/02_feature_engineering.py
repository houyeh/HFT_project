import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.utils.tw_train_helpers import build_tw_feature_frame
from src.training.datasets import WindowedLOBSequenceDataset
from src.data.tw_replay import infer_tick_size_from_lob
from new_era.tcn_pipeline_utils import (
    HORIZON,
    SAMPLE_STRIDE,
    THRESHOLD_TICKS,
    TOP_8_FEATURES,
    WINDOW_SIZE,
    Z_FEATURES,
    apply_rolling_zscore,
    safe_session_key,
)

# Import the loader from the previous step dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("feat_imp", str(project_root / "new_era" / "01_feature_importance.py"))
feat_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_imp)
load_new_tw_data = feat_imp.load_new_tw_data

def main():
    """
    主程式：
    1. 讀取台指期原始 LOB 與成交資料。
    2. 挑出特徵重要性 (Step 1) 排序最高的前 8 個特徵。
    3. 計算這些特徵的滾動 Z-Score，適應市場變動。
    4. 透過 WindowedLOBSequenceDataset 進行切窗 (Sliding Window)，
       把連續的時間序列切割成「長度 100」的序列特徵 (X) 與對應的未來趨勢標籤 (y)。
    5. 儲存為 .npz 壓縮檔供後續模型訓練。
    """
    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6", "MXFB6"]
    
    print(f"Loading raw CSV data from {data_dir}...")
    # 呼叫 01_feature_importance.py 的資料讀取函式
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    
    output_data = {}
    
    for symbol in target_symbols:
        if symbol not in lob_frames: continue
        print(f"\nProcessing features for {symbol}...")
        symbol_lob = lob_frames[symbol]
        symbol_trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))

        for session_id, lob_df in symbol_lob.groupby("session_id", sort=True):
            if len(lob_df) < WINDOW_SIZE + HORIZON:
                print(f"  Skip {session_id}: insufficient rows ({len(lob_df)})")
                continue

            trade_df = symbol_trades[symbol_trades["session_id"] == session_id]
            tick_size = infer_tick_size_from_lob(lob_df)

            # 1. 建立單一 session 的基礎特徵，避免 window/label 跨日串接
            feature_frame = build_tw_feature_frame(lob_df.reset_index(drop=True), trade_df.reset_index(drop=True), symbol, tick_size).dropna()
            print(f"  Session {session_id} base feature frame shape: {feature_frame.shape}")

            # 2. Rolling Z-score 僅在該 session 內計算，避免跨日分布外洩
            zscore_frame = apply_rolling_zscore(feature_frame, TOP_8_FEATURES, window=2000)

            # 3. 建立 Dataset 進行切窗 (Sliding Window) 擷取序列特徵與標籤
            print("  Extracting 100-tick sequences and labels...")
            dataset = WindowedLOBSequenceDataset(
                feature_df=zscore_frame,
                symbol=symbol,
                tick_size=tick_size,
                window_size=WINDOW_SIZE,
                horizon=HORIZON,
                threshold_ticks=THRESHOLD_TICKS,
                sample_stride=SAMPLE_STRIDE,
                feature_cols=Z_FEATURES
            )

            X_list, y_list = [], []
            for i in range(len(dataset)):
                res = dataset[i]
                X_list.append(res["x"].numpy())
                y_list.append(res["y"])
                if (i + 1) % 50000 == 0:
                    print(f"    extracted {i + 1} sequences...")

            if X_list:
                session_key = f"{symbol}__{safe_session_key(session_id)}"
                X_np = np.stack(X_list)
                y_np = np.array(y_list)
                output_data[f"{session_key}_X"] = X_np
                output_data[f"{session_key}_y"] = y_np
                print(f"  {session_key} completed: X shape {X_np.shape}, y shape {y_np.shape}")
            
    # 4. 將全部結果打包存入 單一的壓縮 .npz 檔案，方便 Step 3 訓練時快速讀取
    npz_path = project_root / "new_era" / "002_zscore_dataset.npz"
    np.savez_compressed(npz_path, **output_data)
    print(f"\nSaved dataset to {npz_path}")

if __name__ == "__main__":
    # 程式進入點
    main()
