import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
import importlib.util
spec = importlib.util.spec_from_file_location("qat_module", r"c:\Users\holbo\00VS_projjects\00_HFT_project\new_era\12_qat_tcn.py")
qat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qat_module)
QATTCNClassifier = qat_module.QATTCNClassifier
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

def run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="baseline"):
    """
    模擬高頻造市 (Market Making) 策略。
    
    參數 `strategy`:
    - 'baseline': 無腦雙向掛單，賺取固定價差，但容易被趨勢輾壓。
    - 'tcn_gated': 只有防禦。當 TCN 預測方向強烈時，撤掉對向掛單以閃避損失。
    - 'tcn_alpha': 攻擊型造市。結合部位傾斜 (Inventory Skewing) 與動能捕捉。
    """
    inventory = 0 # 初始庫存為 0
    max_inventory = 3  # 收緊庫存限制，避免被單邊趨勢打穿 (最多只持有 3 口)
    realized_pnl = 0.0
    POINT_VALUE = 200 # 大台指一點 200 元
    FEE_PER_TRADE = 15 # Maker (掛單方) 手續費較低
    
    trades_executed = 0
    
    # 逐筆掃描 (Tick by Tick)
    for i in range(len(all_probs) - 1):
        prob_down, prob_stat, prob_up = all_probs[i]
        
        # 取得當前市場的最佳買賣價
        my_ask = all_ask_px[i] / 10000.0 if all_ask_px[i] > 100000 else all_ask_px[i]
        my_bid = all_bid_px[i] / 10000.0 if all_bid_px[i] > 100000 else all_bid_px[i]
        
        if my_ask < 10000 or my_bid < 10000:
            continue
            
        # 預設：只要庫存沒滿，就雙向掛單
        quote_ask = True if inventory > -max_inventory else False
        quote_bid = True if inventory < max_inventory else False
        
        if strategy == "tcn_gated":
            # 【防禦型】：預防不良逆向選擇 (Adverse Selection)
            # 若 TCN 預測大跌 (>80%)，則不掛買單 (Bid)，防止接到下殺的刀子
            if prob_down > 0.8: quote_bid = False
            # 若 TCN 預測大漲 (>80%)，則不掛賣單 (Ask)，防止被往上軋空
            if prob_up > 0.8: quote_ask = False
            
        elif strategy == "tcn_alpha":
            # 【攻擊型造市】 (Asymmetric Quoting)
            # 1. 預測強烈上漲：撤銷賣單，同時把買價往上提 (加價搶買)，試圖累積多頭部位
            if prob_up > 0.6:
                quote_ask = False
                my_bid = my_ask # 跨越價差主動吃單 (轉為 Taker)
            # 2. 預測強烈下跌：撤銷買單，同時把賣價往下壓 (降價搶賣)，試圖累積空頭部位
            elif prob_down > 0.6:
                quote_bid = False
                my_ask = my_bid # 跨越價差主動吃單 (轉為 Taker)
            # 3. 預測盤整：乖乖在兩邊掛單賺價差，並根據庫存調整價格 (Inventory Skewing)
            else:
                if inventory >= 1: 
                    quote_bid = False # 手上有多單了，不再接刀，專心賣
                    my_ask = my_bid + 1 # 壓低賣價求售
                elif inventory <= -1:
                    quote_ask = False # 手上有空單了，不再擋車，專心買
                    my_bid = my_ask - 1 # 提高買價回補

        # 判定下一刻的成交 (看下一個 Tick 的報價有沒有掃到我們的單)
        next_ask = all_ask_px[i+1] / 10000.0 if all_ask_px[i+1] > 100000 else all_ask_px[i+1]
        next_bid = all_bid_px[i+1] / 10000.0 if all_bid_px[i+1] > 100000 else all_bid_px[i+1]
        
        if next_ask < 10000 or next_bid < 10000:
            continue
            
        # 如果下一刻的市場買單 (next_ask) 下殺吃到我們的買單 (my_bid)
        if quote_bid and next_ask <= my_bid:
            inventory += 1 # 買到一口，庫存加一
            realized_pnl -= my_bid * POINT_VALUE # 帳戶扣款
            realized_pnl -= FEE_PER_TRADE # 扣手續費
            trades_executed += 1
            
        # 如果下一刻的市場賣單 (next_bid) 上攻吃到我們的賣單 (my_ask)
        if quote_ask and next_bid >= my_ask:
            inventory -= 1 # 賣出一口，庫存減一
            realized_pnl += my_ask * POINT_VALUE # 帳戶進帳
            realized_pnl -= FEE_PER_TRADE # 扣手續費
            trades_executed += 1
            
    # 回測結束，以最後一刻的中間價強制結算剩餘庫存
    final_mid = (all_ask_px[-1] + all_bid_px[-1]) / 2.0
    if all_ask_px[-1] > 100000: final_mid /= 10000.0
    final_pnl = realized_pnl + (inventory * final_mid * POINT_VALUE)
    
    return final_pnl, trades_executed

def main():
    """
    主程式：造市商 (Market Maker) 策略比較
    讀取 OOS 資料，由 TCN 模型產生機率，
    並對比三種造市策略的最終損益 (Baseline vs Gated vs Alpha)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "013_qat_finetuned_weights.pth"
    
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
    
    # 切割出 OOS (Out-of-sample) 測試集
    _, _, test_idx = chronological_split_indices(len(dataset), 0.8, 0.0) 
    test_subset = Subset(dataset, list(test_idx))
    test_loader = DataLoader(test_subset, batch_size=2048, shuffle=False)
    
    model = QATTCNClassifier(num_inputs=8, num_classes=3, channel_list=[16, 16, 16], seq_length=100).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_probs = []
    all_ask_px = []
    all_bid_px = []
    
    # 執行推論，收集所有時步的報價與預測機率
    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch["x"].transpose(1, 2).to(device) 
            probs = torch.softmax(model(batch_X), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_ask_px.extend(batch["ask_px_1"].numpy())
            all_bid_px.extend(batch["bid_px_1"].numpy())
            
    print("\n" + "="*60)
    print("🛡️ 進階造市商回測：無腦造市 vs. 預測防禦 vs. 攻擊型造市")
    print("="*60)
    
    # 策略 1 測試
    pnl_base, t_base = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="baseline")
    print(f"\n[1] Baseline 無腦造市 (永遠雙向掛單)")
    print(f"  - 交易次數: {t_base:,}")
    print(f"  - 淨損益: {pnl_base:,.2f} TWD (被大盤趨勢無情輾壓)")
    
    # 策略 2 測試
    pnl_gated, t_gated = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_gated")
    print(f"\n[2] TCN 防禦造市 (預測大漲跌時抽單閃避)")
    print(f"  - 交易次數: {t_gated:,}")
    print(f"  - 淨損益: {pnl_gated:,.2f} TWD (少賠了 {(pnl_gated - pnl_base):,.2f} TWD)")
    
    # 策略 3 測試
    pnl_alpha, t_alpha = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_alpha")
    print(f"\n[3] 🔥 TCN 攻擊型造市 (Inventory Skewing + 動能捕捉)")
    print(f"  - 交易次數: {t_alpha:,}")
    print(f"  - 淨損益: {pnl_alpha:,.2f} TWD")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
