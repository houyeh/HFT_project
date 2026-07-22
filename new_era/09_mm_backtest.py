import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from src.utils.tw_train_helpers import build_tw_feature_frame
from src.training.datasets import WindowedLOBSequenceDataset
from src.data.tw_replay import infer_tick_size_from_lob
from new_era.tcn_pipeline_utils import (
    HORIZON,
    THRESHOLD_TICKS,
    TOP_8_FEATURES,
    WINDOW_SIZE,
    Z_FEATURES,
    apply_rolling_zscore,
    run_market_making_simulation,
    split_ordered_session_ids,
)

spec = importlib.util.spec_from_file_location("feat_imp", str(project_root / "new_era" / "01_feature_importance.py"))
feat_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_imp)
load_new_tw_data = feat_imp.load_new_tw_data


def build_session_dataset(lob_df, trade_df, symbol, tick_size):
    feature_frame = build_tw_feature_frame(lob_df.reset_index(drop=True), trade_df.reset_index(drop=True), symbol, tick_size).dropna()
    zscore_frame = apply_rolling_zscore(feature_frame, TOP_8_FEATURES, window=2000)
    return WindowedLOBSequenceDataset(
        feature_df=zscore_frame,
        symbol=symbol,
        tick_size=tick_size,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        threshold_ticks=THRESHOLD_TICKS,
        sample_stride=1,  # 回測需要逐 quote 檢查成交，不能沿用訓練用的 stride=5。
        feature_cols=Z_FEATURES,
    )


def collect_predictions(model, datasets, device):
    all_probs = []
    all_labels = []
    all_ask_px = []
    all_bid_px = []

    with torch.no_grad():
        for session_id, dataset in datasets:
            print(f"Running inference on test session {session_id}: {len(dataset)} samples")
            loader = DataLoader(dataset, batch_size=2048, shuffle=False)
            for batch in loader:
                batch_X = batch["x"].transpose(1, 2).to(device)
                probs = torch.softmax(model(batch_X), dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch["y"].numpy())
                all_ask_px.extend(batch["ask_px_1"].numpy())
                all_bid_px.extend(batch["bid_px_1"].numpy())

    return np.asarray(all_probs), np.asarray(all_labels), np.asarray(all_ask_px), np.asarray(all_bid_px)


def print_strategy_result(title, result, baseline_pnl=None):
    improvement = "" if baseline_pnl is None else f" | 相對 baseline: {result['pnl'] - baseline_pnl:,.2f} TWD"
    print(f"\n{title}")
    print(f"  - 交易次數: {result['trades']:,} (maker={result['maker_trades']:,}, taker={result['taker_trades']:,})")
    print(f"  - 結束庫存: {result['ending_inventory']}")
    print(f"  - 淨損益: {result['pnl']:,.2f} TWD{improvement}")


def main():
    """
    使用真正未見過的 session 做造市回測。
    訓練資料仍可用 stride=5 降低樣本量，但回測改用 stride=1 逐 quote 檢查成交。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"

    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6"]
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    symbol = target_symbols[0]

    if symbol not in lob_frames:
        print(f"Error: no LOB data found for {symbol}")
        return

    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(session_ids)

    print(f"Session split for {symbol}:")
    print(f"  train: {train_sessions}")
    print(f"  val:   {val_sessions}")
    print(f"  test:  {test_sessions}")

    if not test_sessions:
        print("Error: no test sessions available for market-making backtest.")
        return

    tick_size = infer_tick_size_from_lob(symbol_lob)
    test_datasets = []
    for session_id in test_sessions:
        lob_df = symbol_lob[symbol_lob["session_id"] == session_id]
        trade_df = symbol_trades[symbol_trades["session_id"] == session_id]
        if len(lob_df) < WINDOW_SIZE + HORIZON:
            print(f"Skip {session_id}: insufficient rows ({len(lob_df)})")
            continue
        dataset = build_session_dataset(lob_df, trade_df, symbol, tick_size)
        test_datasets.append((session_id, dataset))

    if not test_datasets:
        print("Error: test sessions did not contain enough rows for a 100-tick window.")
        return

    model = HLSFriendlyTCNClassifier(num_inputs=8, num_classes=3, channel_list=[16, 16, 16]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels, all_ask_px, all_bid_px = collect_predictions(model, test_datasets, device)
    all_preds = all_probs.argmax(axis=1)

    print("\n" + "="*60)
    print("Session-aware TCN market-making backtest")
    print("="*60)
    print(f"Accuracy on test sessions: {accuracy_score(all_labels, all_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]))

    baseline = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="baseline")
    gated = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_gated")
    alpha = run_market_making_simulation(all_probs, all_ask_px, all_bid_px, strategy="tcn_alpha")

    print_strategy_result("[1] Baseline 無腦造市", baseline)
    print_strategy_result("[2] TCN 防禦造市", gated, baseline_pnl=baseline["pnl"])
    print_strategy_result("[3] TCN Alpha 造市", alpha, baseline_pnl=baseline["pnl"])
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
