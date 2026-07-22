import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

# Add project root to sys path
project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.utils.tw_train_helpers import build_tw_feature_frame, feature_columns_from_name
from src.backtest.baseline import run_signal_backtest
from src.backtest.signals import apply_probability_gating
from src.models.tcn import build_default_model
from src.training.datasets import WindowedLOBSequenceDataset
from src.training.metrics import classification_metrics
from src.training.trainer import predict_model, select_device
from src.data.tw_replay import infer_tick_size_from_lob

def parse_new_tw_timestamp(raw_str: str) -> int:
    parts = raw_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2])
    millis = int(parts[3])
    # Ignore sequence
    return hour * 3600000000000 + minute * 60000000000 + second * 1000000000 + millis * 1000000

def load_new_tw_data(data_dir: Path, target_symbols: list[str]):
    lob_records = defaultdict(list)
    trade_records = defaultdict(list)

    csv_files = list(data_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for csv_path in csv_files:
        print(f"Parsing {csv_path}...")
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) < 34: continue
                symbol = row[3].strip()
                if symbol not in target_symbols:
                    continue
                
                timestamp = parse_new_tw_timestamp(row[0].strip())
                
                last_trade_qty = int(float(row[7]))
                last_trade_price = float(row[6])
                
                if last_trade_qty > 0:
                    trade_records[symbol].append({
                        "timestamp": timestamp,
                        "trade_price": last_trade_price,
                        "trade_volume": last_trade_qty
                    })
                
                # Check if it's a valid quote by looking at 1st bid/ask price
                try:
                    bid_px_1 = float(row[21])
                    ask_px_1 = float(row[22])
                    if bid_px_1 > 0 and ask_px_1 > 0:
                        lob_records[symbol].append({
                            "timestamp": timestamp,
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
                    continue

    lob_frames = {}
    trade_frames = {}
    
    for symbol in target_symbols:
        if lob_records[symbol]:
            df = pd.DataFrame(lob_records[symbol]).sort_values("timestamp").reset_index(drop=True)
            lob_frames[symbol] = df
        if trade_records[symbol]:
            df = pd.DataFrame(trade_records[symbol]).sort_values("timestamp").reset_index(drop=True)
            trade_frames[symbol] = df
            
    return lob_frames, trade_frames

def main():
    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6", "MXFB6"]
    
    print("Loading data...")
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    
    datasets = []
    feature_cols = feature_columns_from_name("execution_aware")
    results = {}
    
    for symbol in target_symbols:
        if symbol not in lob_frames:
            print(f"No data for {symbol}")
            continue
            
        lob_df = lob_frames[symbol]
        trade_df = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
        
        tick_size = infer_tick_size_from_lob(lob_df)
        print(f"Symbol {symbol} inferred tick size: {tick_size}")
        
        print(f"Building features for {symbol}...")
        feature_frame = build_tw_feature_frame(
            lob_df=lob_df,
            trade_df=trade_df,
            symbol=symbol,
            tick_size=tick_size
        )
        
        # Remove NaNs
        feature_frame = feature_frame.dropna()
        if len(feature_frame) == 0:
            print(f"No valid features for {symbol}")
            continue
            
        dataset = WindowedLOBSequenceDataset(
            feature_df=feature_frame,
            symbol=symbol,
            tick_size=tick_size,
            window_size=100,
            horizon=10,
            threshold_ticks=1.0,
            sample_stride=10,
            feature_cols=feature_cols,
        )
        datasets.append(dataset)
        
    if not datasets:
        print("No valid datasets built.")
        return
        
    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    dataloader = DataLoader(full_dataset, batch_size=512, shuffle=False)
    
    print("Loading model...")
    model_path = project_root / "artifacts" / "03_latency_proof" / "latency_proof" / "frozen_tcn_best_model.pt"
    model = build_default_model(num_inputs=len(feature_cols))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    device = select_device()
    model.to(device)
    
    print("Predicting...")
    predictions = predict_model(model, dataloader, device=device)
    
    raw_metrics = classification_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )
    
    # Gating and backtest
    gated_predictions = apply_probability_gating(
        predictions,
        probability_threshold=0.334,
        margin_threshold=0.005,
    )
    
    gated_metrics = classification_metrics(
        gated_predictions["y_true"].to_numpy(),
        gated_predictions["gated_signal"].to_numpy(),
    )
    
    gated_trades, gated_signal_summary, gated_signal_by_symbol = run_signal_backtest(
        gated_predictions,
        spread_cost_multiplier=1.0,
        signal_column="gated_signal",
    )
    
    # Write report
    report_path = project_root / "0428_test.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 0428 新台指期資料測試報告 (NEW Data)\n\n")
        f.write("## 1. 資料格式與對齊結果\n")
        f.write(f"- 解析了 `data/NEW` 中的新格式 CSV。\n")
        f.write(f"- 提取的 Symbol：{', '.join(target_symbols)}\n")
        for symbol in target_symbols:
            lob_len = len(lob_frames.get(symbol, []))
            trd_len = len(trade_frames.get(symbol, []))
            f.write(f"  - **{symbol}**: LOB Quotes: {lob_len}, Trades: {trd_len}\n")
            
        f.write("\n## 2. 預測結果 (Raw Metrics)\n")
        f.write("未經過濾的原始分類表現：\n")
        f.write("```json\n")
        f.write(json.dumps(raw_metrics, indent=2) + "\n")
        f.write("```\n")
        
        f.write("\n## 3. Gating 後的可交易訊號表現 (Probability Gating)\n")
        f.write("應用機率門檻與邊際門檻過濾後的分類表現：\n")
        f.write("```json\n")
        f.write(json.dumps(gated_metrics, indent=2) + "\n")
        f.write("```\n")
        
        f.write("\n## 4. 向量化回測結果 (Signal Backtest)\n")
        f.write("依據 Gating 後的訊號進行的快速回測：\n")
        f.write("```json\n")
        f.write(json.dumps(gated_signal_summary, indent=2) + "\n")
        f.write("```\n")
        
        f.write("\n### 各商品回測拆解\n")
        f.write("```\n")
        f.write(gated_signal_by_symbol.to_string() + "\n")
        f.write("```\n")
        
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
