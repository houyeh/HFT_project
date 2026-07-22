import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 設定專案根目錄
project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from src.utils.tw_train_helpers import build_tw_feature_frame
from src.training.datasets import WindowedLOBSequenceDataset
from src.data.tw_replay import load_tw_replay_as_lob, load_tw_replay_trades, infer_tick_size_from_lob
from new_era.tcn_pipeline_utils import (
    HORIZON,
    THRESHOLD_TICKS,
    TOP_8_FEATURES,
    WINDOW_SIZE,
    Z_FEATURES,
    apply_rolling_zscore,
    run_market_making_simulation,
)


def print_strategy_result(title, result, baseline_pnl=None):
    improvement = "" if baseline_pnl is None else f" | 相對 baseline: {result['pnl'] - baseline_pnl:,.2f} TWD"
    print(f"\n{title}")
    print(f"  - 交易次數: {result['trades']:,} (maker={result['maker_trades']:,}, taker={result['taker_trades']:,})")
    print(f"  - 結束庫存: {result['ending_inventory']}")
    print(f"  - 淨損益: {result['pnl']:,.2f} TWD{improvement}")

def main():
    print("🚀 啟動舊資料造市回測 (Aligning features to new_era model)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    old_data_path = project_root / "data" / "raw" / "TW" / "replay1_TW.csv"
    
    if not old_data_path.exists():
        print(f"❌ 找不到舊資料: {old_data_path}")
        return

    # 1. 載入舊資料
    target_symbol = "TXFJ5"
    print(f"Step 1: Loading {target_symbol} from {old_data_path}...")
    lob_map = load_tw_replay_as_lob(old_data_path, symbols=[target_symbol])
    trade_map = load_tw_replay_trades(old_data_path, symbols=[target_symbol])
    
    if target_symbol not in lob_map:
        print(f"❌ 找不到目標商品 {target_symbol} 資料")
        return
        
    lob_df = lob_map[target_symbol]
    trade_df = trade_map.get(target_symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
    tick_size = infer_tick_size_from_lob(lob_df)
    print(f"  - Loaded {len(lob_df)} quotes, {len(trade_df)} trades. Tick size: {tick_size}")

    # 2. 特徵工程 (對齊 new_era 格式)
    print("Step 2: Building features and applying rolling Z-score...")
    feature_frame = build_tw_feature_frame(lob_df, trade_df, target_symbol, tick_size).dropna()
    
    zscore_frame = apply_rolling_zscore(feature_frame, TOP_8_FEATURES, window=2000)
    
    print("  Feature Stats (Z-score):")
    print(zscore_frame[Z_FEATURES].describe().loc[['mean', 'std', 'min', 'max']])
    print("  Sample features (first 5 rows):")
    print(zscore_frame[Z_FEATURES].head())
    
    # 3. 建立 Dataset (Sliding Window)
    print("Step 3: Extracting sequences...")
    dataset = WindowedLOBSequenceDataset(
        feature_df=zscore_frame, symbol=target_symbol, tick_size=tick_size,
        window_size=WINDOW_SIZE, horizon=HORIZON, threshold_ticks=THRESHOLD_TICKS, sample_stride=1, feature_cols=Z_FEATURES
    )
    
    # 全部跑回測 (不切 OOS，因為這是驗證舊資料的表現)
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    # 4. 載入模型
    print("Step 4: Loading TCN model weights...")
    model = HLSFriendlyTCNClassifier(num_inputs=8, num_classes=3, channel_list=[16, 16, 16]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_probs = []
    all_ask_px = []
    all_bid_px = []
    all_labels = []
    
    print("Step 5: Running inference...")
    with torch.no_grad():
        for batch in data_loader:
            batch_X = batch["x"].transpose(1, 2).to(device) 
            probs = torch.softmax(model(batch_X), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_ask_px.extend(batch["ask_px_1"].numpy())
            all_bid_px.extend(batch["bid_px_1"].numpy())
            all_labels.extend(batch["y"].numpy())

    from sklearn.metrics import confusion_matrix, accuracy_score
    all_preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
            
    print("\n" + "="*60)
    print(f"📈 舊資料 ({target_symbol}) 造市商回測結果")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("="*60)
    
    # 策略 1: Baseline
    baseline = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="baseline")
    gated = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_gated")
    alpha = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_alpha")

    print_strategy_result("[1] Baseline 無腦造市", baseline)
    print_strategy_result("[2] TCN 防禦造市 (Gated)", gated, baseline_pnl=baseline["pnl"])
    print_strategy_result("[3] TCN Alpha 造市", alpha, baseline_pnl=baseline["pnl"])
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
