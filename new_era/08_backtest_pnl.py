import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    """
    動態計算特徵的滾動 Z-Score，使模型能適應盤中波動度的變化。
    """
    df_out = df.copy()
    for col in columns:
        r_mean = df[col].rolling(window=window, min_periods=100).mean()
        r_std = df[col].rolling(window=window, min_periods=100).std().replace(0, 1e-5)
        df_out[f"{col}_z"] = ((df[col] - r_mean) / r_std).fillna(0.0).clip(-10, 10)
    return df_out

def main():
    """
    主程式：Event-Driven 事件驅動回測 (預測方向並進行交易)
    1. 重新讀取原始資料，模擬真實時間流逝。
    2. 萃取最新 20% (Out-of-Sample) 的資料來驗證交易策略。
    3. 載入 TCN 模型，產出每一時刻的漲跌預測機率。
    4. 依據設定的門檻 (PROB_THRESHOLD) 觸發進場交易，並設定固定時間後出場 (HOLDING_TICKS)。
    5. 結算最終 PnL (考慮手續費與跳動點值)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    
    print("1. Loading RAW Data for Backtesting...")
    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6"]  # 這裡只拿大台來回測
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    symbol = target_symbols[0]
    
    lob_df = lob_frames[symbol]
    trade_df = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
    tick_size = infer_tick_size_from_lob(lob_df)
    
    # Debug: 確認原始報價是正常的
    print(f"Debug: Raw TXF Prices -> Ask_1: {lob_df['ask_px_1'].iloc[100]}, Bid_1: {lob_df['bid_px_1'].iloc[100]}")
    
    print("2. Re-applying Feature Engineering...")
    # 重新計算微結構特徵
    feature_frame = build_tw_feature_frame(lob_df, trade_df, symbol, tick_size).dropna()
    top_8_features = ["micro_return", "trade_intensity_short", "queue_imbalance_change", 
                      "deep_obi", "net_cancel_pressure", "queue_imbalance", 
                      "weighted_mid_deviation", "obi"]
    # Z-score 正規化
    zscore_frame = apply_rolling_zscore(feature_frame, top_8_features, window=2000)
    z_cols = [f"{c}_z" for c in top_8_features]
    
    print("3. Building Sequential Dataset (No Shuffling)...")
    # 切窗時，必須按照時間順序 (不可打亂)，否則無法進行回測
    dataset = WindowedLOBSequenceDataset(
        feature_df=zscore_frame, symbol=symbol, tick_size=tick_size,
        window_size=100, horizon=10, threshold_ticks=1.0, sample_stride=5, feature_cols=z_cols
    )
    
    # 擷取最後 20% 的 Out-of-Sample 資料作為回測區間 (絕對不可看到未來的訓練資料)
    _, _, test_idx = chronological_split_indices(len(dataset), 0.8, 0.0) 
    test_subset = Subset(dataset, list(test_idx))
    test_loader = DataLoader(test_subset, batch_size=2048, shuffle=False)
    
    print(f"Test Set Size: {len(test_subset)} ticks.")
    
    print("4. Loading TCN Model & Running Inference...")
    # 載入 TCN 模型進行預測
    model = HLSFriendlyTCNClassifier(num_inputs=8, num_classes=3, channel_list=[16, 16, 16]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_probs = []
    all_current_mids = []
    all_ask_px = []
    all_bid_px = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch["x"].transpose(1, 2).to(device) 
            probs = torch.softmax(model(batch_X), dim=1).cpu().numpy() # 轉為機率
            all_probs.extend(probs)
            all_current_mids.extend(batch["current_mid"].numpy())
            all_ask_px.extend(batch["ask_px_1"].numpy()) # 記錄當下的最佳賣價 (供進場用)
            all_bid_px.extend(batch["bid_px_1"].numpy()) # 記錄當下的最佳買價 (供出場用)
            all_labels.extend(batch["y"].numpy())
            
    # --- Confusion Matrix (檢驗模型是否有真正在預測，而不是全都猜盤整) ---
    all_preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*50)
    print(f"🔥 Sequential Test Accuracy: {acc*100:.2f}%")
    print("="*50)
    print("📊 Confusion Matrix:")
    print("               Predicted Down | Predicted Stat | Predicted Up")
    print(f"Actual Down  | {cm[0][0]:14d} | {cm[0][1]:14d} | {cm[0][2]:12d}")
    print(f"Actual Stat  | {cm[1][0]:14d} | {cm[1][1]:14d} | {cm[1][2]:12d}")
    print(f"Actual Up    | {cm[2][0]:14d} | {cm[2][1]:14d} | {cm[2][2]:12d}")
            
    # --- 5. Event-Driven Backtest Logic 事件驅動回測邏輯 ---
    print("\n5. Starting Event-Driven Backtest Simulation...")
    
    PROB_THRESHOLD = 0.80  # 模型非常有把握 (機率 > 80%) 才出手
    HOLDING_TICKS = 10     # 每次進場固定持有 10 個 tick 後平倉
    FEE_PER_TRADE_TWD = 50 # 假設每口單邊手續費與稅金共 50 元 (保守估計)
    POINT_VALUE = 200      # 大台指每點 200 元
    
    position = 0 # 0: 空手, 1: 做多, -1: 做空
    entry_price = 0.0
    ticks_held = 0
    total_pnl_twd = 0.0
    trades_executed = 0
    winning_trades = 0
    
    # 逐筆掃描 (Tick by Tick)
    for i in range(len(all_probs)):
        prob_down, prob_stat, prob_up = all_probs[i]
        # 處理資料中可能的報價放大問題 (如果有)
        ask_px = all_ask_px[i] / 10000.0 if all_ask_px[i] > 100000 else all_ask_px[i] 
        bid_px = all_bid_px[i] / 10000.0 if all_bid_px[i] > 100000 else all_bid_px[i] 
        
        # 1. 出場邏輯 (Exit Logic)
        if position != 0:
            ticks_held += 1
            # 持有時間到了，強制市價平倉
            if ticks_held >= HOLDING_TICKS:
                if position == 1:
                    # 做多出場：以當下的最佳買價 (Bid) 賣出
                    gross_profit_pts = bid_px - entry_price
                else: 
                    # 做空出場：以當下的最佳賣價 (Ask) 買回
                    gross_profit_pts = entry_price - ask_px
                    
                # 計算淨損益 (扣掉來回兩次的手續費)
                net_pnl = (gross_profit_pts * POINT_VALUE) - (FEE_PER_TRADE_TWD * 2) 
                total_pnl_twd += net_pnl
                trades_executed += 1
                if net_pnl > 0: winning_trades += 1
                
                # 恢復空手
                position = 0
                ticks_held = 0
                
        # 2. 進場邏輯 (Entry Logic)
        if position == 0:
            # 強烈看漲：市價買進 (敲 Ask)
            if prob_up > PROB_THRESHOLD:
                position = 1
                entry_price = ask_px
                ticks_held = 0
            # 強烈看跌：市價賣出 (敲 Bid)
            elif prob_down > PROB_THRESHOLD:
                position = -1
                entry_price = bid_px
                ticks_held = 0
                
    # 輸出回測報表
    print("\n" + "="*50)
    print("📊 TCN Event-Driven Backtest Results (Out-of-Sample)")
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
