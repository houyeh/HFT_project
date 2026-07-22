import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from src.utils.tw_train_helpers import build_tw_feature_frame
from src.training.datasets import WindowedLOBSequenceDataset, chronological_split_indices
from src.data.tw_replay import infer_tick_size_from_lob

import importlib.util
spec = importlib.util.spec_from_file_location("feat_imp", str(project_root / "new_era" / "01_feature_importance.py"))
feat_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_imp)
load_new_tw_data = feat_imp.load_new_tw_data

def apply_rolling_zscore(df: pd.DataFrame, columns: list[str], window: int = 2000) -> pd.DataFrame:
    df_out = df.copy()
    for col in columns:
        r_mean = df[col].rolling(window=window, min_periods=100).mean()
        r_std = df[col].rolling(window=window, min_periods=100).std().replace(0, 1e-5)
        df_out[f"{col}_z"] = ((df[col] - r_mean) / r_std).fillna(0.0).clip(-10, 10)
    return df_out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    
    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6"]  
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    symbol = target_symbols[0]
    
    lob_df = lob_frames[symbol]
    trade_df = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
    tick_size = infer_tick_size_from_lob(lob_df)
    
    feature_frame = build_tw_feature_frame(lob_df, trade_df, symbol, tick_size).dropna()
    top_8_features = ["micro_return", "trade_intensity_short", "queue_imbalance_change", 
                      "deep_obi", "net_cancel_pressure", "queue_imbalance", 
                      "weighted_mid_deviation", "obi"]
    zscore_frame = apply_rolling_zscore(feature_frame, top_8_features, window=2000)
    z_cols = [f"{c}_z" for c in top_8_features]
    
    dataset = WindowedLOBSequenceDataset(
        feature_df=zscore_frame, symbol=symbol, tick_size=tick_size,
        window_size=100, horizon=10, threshold_ticks=1.0, sample_stride=5, feature_cols=z_cols
    )
    
    _, _, test_idx = chronological_split_indices(len(dataset), 0.8, 0.0) 
    test_subset = Subset(dataset, list(test_idx))
    test_loader = DataLoader(test_subset, batch_size=2048, shuffle=False)
    
    model = HLSFriendlyTCNClassifier(num_inputs=8, num_classes=3, channel_list=[16, 16, 16]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_probs = []
    all_ask_px = []
    all_bid_px = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch["x"].transpose(1, 2).to(device) 
            probs = torch.softmax(model(batch_X), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_ask_px.extend(batch["ask_px_1"].numpy())
            all_bid_px.extend(batch["bid_px_1"].numpy())
            all_labels.extend(batch["y"].numpy())
            
    print("\n5. Starting Fixed Event-Driven Backtest Simulation...")
    
    PROB_THRESHOLD = 0.80  
    HOLDING_TICKS = 10     
    FEE_PER_TRADE_TWD = 50 
    POINT_VALUE = 200      
    
    position = 0 
    entry_price = 0.0
    ticks_held = 0
    total_pnl_twd = 0.0
    trades_executed = 0
    winning_trades = 0
    
    for i in range(len(all_probs)):
        prob_down, prob_stat, prob_up = all_probs[i]
        
        # FIX: Ensure price is not 0 or corrupted. Taiwan futures shouldn't be under 10000 currently.
        ask_px = all_ask_px[i] / 10000.0 if all_ask_px[i] > 100000 else all_ask_px[i] 
        bid_px = all_bid_px[i] / 10000.0 if all_bid_px[i] > 100000 else all_bid_px[i] 
        
        # If the price is completely bogus, close position at breakeven and skip
        if ask_px < 10000 or bid_px < 10000:
            if position != 0:
                position = 0
                ticks_held = 0
            continue
        
        if position != 0:
            ticks_held += 1
            if ticks_held >= HOLDING_TICKS:
                if position == 1:
                    gross_profit_pts = bid_px - entry_price
                else: 
                    gross_profit_pts = entry_price - ask_px
                
                # FIX: Realism check. If profit > 200 points in 10 ticks, it's a data gap anomaly. Skip it.
                if abs(gross_profit_pts) > 200:
                    position = 0
                    ticks_held = 0
                    continue
                    
                net_pnl = (gross_profit_pts * POINT_VALUE) - (FEE_PER_TRADE_TWD * 2) 
                total_pnl_twd += net_pnl
                trades_executed += 1
                if net_pnl > 0: winning_trades += 1
                
                position = 0
                ticks_held = 0
                
        if position == 0:
            if prob_up > PROB_THRESHOLD:
                position = 1
                entry_price = ask_px
                ticks_held = 0
            elif prob_down > PROB_THRESHOLD:
                position = -1
                entry_price = bid_px
                ticks_held = 0
                
    print("\n" + "="*50)
    print("📊 FIXED TCN Event-Driven Backtest Results")
    print("="*50)
    print(f"Total Ticks Evaluated: {len(all_probs):,}")
    print(f"Trades Executed:       {trades_executed:,}")
    if trades_executed > 0:
        print(f"Win Rate:              {(winning_trades/trades_executed)*100:.2f}%")
        print(f"Avg PnL per Trade:     {(total_pnl_twd/trades_executed):.2f} TWD")
        print(f"🔥 Net PnL (TWD):      {total_pnl_twd:,.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
