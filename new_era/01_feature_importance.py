import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 設定專案根目錄，並將其加入系統路徑以便匯入自定義模組
project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

# 匯入資料處理與特徵工程相關的輔助函式
from src.utils.tw_train_helpers import build_tw_feature_frame, feature_columns_from_name
from src.training.datasets import WindowedLOBSequenceDataset
from src.data.tw_replay import infer_tick_size_from_lob

SESSION_NS_SPACING = 36 * 60 * 60 * 1_000_000_000

def parse_new_tw_timestamp(raw_str: str) -> int:
    """
    將字串格式的時間戳記 (HH:MM:SS:µs) 轉換為以奈秒 (nanoseconds) 為單位的整數。
    例如："08:45:00:000000" 會轉換為自午夜起的奈秒數。
    """
    parts = raw_str.split(":")
    # hours * 3.6e12 + minutes * 6e10 + seconds * 1e9 + microseconds * 1e3 = nanoseconds
    return int(parts[0]) * 3600000000000 + int(parts[1]) * 60000000000 + int(parts[2]) * 1000000000 + int(parts[3]) * 1000000

def load_new_tw_data(data_dir: Path, target_symbols: list[str]):
    """
    掃描指定目錄下的所有 CSV 檔案，讀取目標商品的 LOB (限價委託簿) 與成交紀錄 (Trade)。
    每個資料夾視為一個獨立交易 session，並把 session offset 加到 timestamp，
    避免多天資料只用盤中時間排序而互相交錯。
    """
    lob_records = defaultdict(list)
    trade_records = defaultdict(list)
    print(f"Scanning {data_dir} for CSVs...")
    
    # 遞迴尋找所有 CSV 檔案
    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda path: str(path.relative_to(data_dir)))
    session_order_map = {}
    for csv_path in csv_paths:
        session_id = str(csv_path.parent.relative_to(data_dir)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem
        session_order = session_order_map.setdefault(session_id, len(session_order_map))
        session_offset = session_order * SESSION_NS_SPACING

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳過標頭列 (Header)
            
            for row in reader:
                # 確保資料列完整 (至少需要 34 個欄位以包含所有 LOB 資訊)
                if len(row) < 34: continue
                symbol = row[3].strip()
                # 只處理我們關心的目標商品 (如 TXFB6, MXFB6)
                if symbol not in target_symbols: continue
                
                # 解析時間戳記
                timestamp = session_offset + parse_new_tw_timestamp(row[0].strip())
                last_trade_qty = int(float(row[7]))
                last_trade_price = float(row[6])
                
                # 如果有成交量，則記錄為一筆成交資訊
                if last_trade_qty > 0:
                    trade_records[symbol].append({
                        "timestamp": timestamp,
                        "session_id": session_id,
                        "trade_price": last_trade_price,
                        "trade_volume": last_trade_qty,
                    })
                
                # 嘗試解析 LOB 資訊 (5檔報價與委託量)
                try:
                    # 確保最佳買賣價皆大於 0 (過濾異常或無效報價)
                    if float(row[21]) > 0 and float(row[22]) > 0:
                        lob_records[symbol].append({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "bid_px_1": float(row[21]), "bid_vol_1": int(float(row[20])),
                            "bid_px_2": float(row[19]), "bid_vol_2": int(float(row[18])),
                            "bid_px_3": float(row[17]), "bid_vol_3": int(float(row[16])),
                            "bid_px_4": float(row[15]), "bid_vol_4": int(float(row[14])),
                            "bid_px_5": float(row[13]), "bid_vol_5": int(float(row[12])),
                            "ask_px_1": float(row[22]), "ask_vol_1": int(float(row[23])),
                            "ask_px_2": float(row[24]), "ask_vol_2": int(float(row[25])),
                            "ask_px_3": float(row[26]), "ask_vol_3": int(float(row[27])),
                            "ask_px_4": float(row[28]), "ask_vol_4": int(float(row[29])),
                            "ask_px_5": float(row[30]), "ask_vol_5": int(float(row[31])),
                        })
                except ValueError:
                    # 忽略轉換失敗的資料列
                    continue
                    
    # 將收集到的字典轉換為 Pandas DataFrame，並依時間排序
    lob_frames = {
        s: pd.DataFrame(lob_records[s]).sort_values(["session_id", "timestamp"]).reset_index(drop=True)
        for s in target_symbols
        if lob_records[s]
    }
    trade_frames = {
        s: pd.DataFrame(trade_records[s]).sort_values(["session_id", "timestamp"]).reset_index(drop=True)
        for s in target_symbols
        if trade_records[s]
    }
    return lob_frames, trade_frames

def main():
    """
    主程式：
    1. 讀取並整理原始資料
    2. 針對各目標商品進行特徵工程與資料切窗 (Sliding Window)
    3. 收集特徵與標籤
    4. 訓練隨機森林 (Random Forest) 模型以評估特徵重要性
    5. 儲存結果
    """
    # 設定資料來源與目標商品 (大台指 TXF, 小台指 MXF)
    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6", "MXFB6"]
    
    # 讀取 LOB 與成交資料
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    
    # 取得特徵欄位名稱清單 (使用 'execution_aware' 模式)
    feature_cols = feature_columns_from_name("execution_aware")
    X_list, y_list = [], []
    
    # 針對每個目標商品進行資料處理
    for symbol in target_symbols:
        if symbol not in lob_frames: continue
        
        symbol_lob = lob_frames[symbol]
        symbol_trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))

        for session_id, lob_df in symbol_lob.groupby("session_id", sort=True):
            trade_df = symbol_trades[symbol_trades["session_id"] == session_id]
            if len(lob_df) < 120:
                continue

            tick_size = infer_tick_size_from_lob(lob_df)
            feature_frame = build_tw_feature_frame(lob_df.reset_index(drop=True), trade_df.reset_index(drop=True), symbol, tick_size).dropna()

            dataset = WindowedLOBSequenceDataset(
                feature_df=feature_frame, symbol=symbol, tick_size=tick_size,
                window_size=100, horizon=10, threshold_ticks=1.0, sample_stride=10, feature_cols=feature_cols
            )
            print(f"Extracting features for {symbol} / {session_id} ({len(dataset)} samples)...")

            for i in range(len(dataset)):
                res = dataset[i]
                X_seq = res["x"]
                y_val = res["y"]

                # 隨機森林不吃序列資料，只取最新 snapshot 作為 feature-importance 估計。
                X_list.append(X_seq[-1, :].numpy())
                y_list.append(y_val)

                if (i+1) % 100000 == 0:
                    print(f"  Processed {i+1} samples...")
    
    # 確保有提取到資料
    if not X_list:
        print("No data extracted.")
        return

    # 將 List 轉換為 Numpy Array
    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    print("Training RandomForestClassifier...")
    # 初始化隨機森林分類器
    # 使用 max_depth=10 以保持訓練速度，並防止模型對雜訊過度擬合 (Over-fitting)
    # class_weight="balanced" 幫助處理樣本類別不平衡的問題
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, class_weight="balanced", random_state=42)
    rf.fit(X, y)
    
    # 將特徵名稱與其對應的 Feature Importance (特徵重要性) 整理成 DataFrame 並由高至低排序
    results = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    
    # 輸出結果
    print("\n=== Feature Importance Ranking ===")
    try:
        # 嘗試使用 Markdown 格式印出表格
        print(results.to_markdown(index=False))
    except ImportError:
        # 若未安裝 tabulate，則使用預設字串格式印出
        print(results.to_string(index=False))
        
    # 將結果儲存成 CSV 檔案，供後續分析或簡報使用
    out_file = project_root / "new_era" / "001_feature_importance.csv"
    results.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    # 程式進入點
    main()
