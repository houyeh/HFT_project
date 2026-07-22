import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from _common import build_cfg, build_paths
from new_era.models_playaround_15.backtest_val_sweep import (
    _collect_predictions,
    _build_session_dataset,
    _future_min_max,
    _max_drawdown,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import split_ordered_session_ids


def compute_metrics_for_threshold(
    cfg, prob, ask_px, bid_px, fut_min, fut_max, tick_size, threshold
):
    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    max_inv = cfg.backtest.max_inventory

    # State variables
    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0

    bid_cash = 0.0
    bid_inv = 0
    ask_cash = 0.0
    ask_inv = 0

    toxic_avoided = 0
    missed_profitable = 0
    signal_count = 0
    total_opportunities = 0

    pnl_path = []

    for i in range(len(prob) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])

        mid_now = (ask + bid) / 2.0

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue
        if ask <= bid:
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue

        p_down = float(prob[i, 0])
        p_up = float(prob[i, 2])

        pull_bid = p_down >= threshold
        pull_ask = p_up >= threshold

        if pull_bid: signal_count += 1
        if pull_ask: signal_count += 1
        total_opportunities += 2

        baseline_bid_fill = next_ask <= bid
        baseline_ask_fill = next_bid >= ask

        # Path-independent toxic fill analysis
        if baseline_bid_fill and pull_bid:
            if i < len(fut_min) and fut_min[i] < bid:
                toxic_avoided += 1
            else:
                missed_profitable += 1

        if baseline_ask_fill and pull_ask:
            if i < len(fut_max) and fut_max[i] > ask:
                toxic_avoided += 1
            else:
                missed_profitable += 1

        # Path-dependent simulation
        quote_bid = bid if not pull_bid and inventory < max_inv else None
        quote_ask = ask if not pull_ask and inventory > -max_inv else None

        if quote_bid is not None and next_ask <= quote_bid:
            gross_cash -= quote_bid * point_value
            bid_cash -= quote_bid * point_value
            fees_paid += maker_fee
            inventory += 1
            bid_inv += 1
            maker_trades += 1

        if quote_ask is not None and next_bid >= quote_ask:
            gross_cash += quote_ask * point_value
            ask_cash += quote_ask * point_value
            fees_paid += maker_fee
            inventory -= 1
            ask_inv -= 1
            maker_trades += 1

        pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)

    final_mid = (ask_px[-1] + bid_px[-1]) / 2.0 if len(ask_px) else 0.0
    
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid
    bid_side_pnl = bid_cash + bid_inv * final_mid * point_value - (bid_inv + ask_inv)*maker_fee # rough fee approx if needed, but let's strictly assign maker fee
    
    # accurately assign fees
    bid_side_pnl = bid_cash + bid_inv * final_mid * point_value
    ask_side_pnl = ask_cash + ask_inv * final_mid * point_value

    pnl_arr = np.asarray(pnl_path, dtype=np.float64)
    mdd = _max_drawdown(pnl_arr)

    # Note: total fees were collected, we can attribute fee to sides if we want.
    # Currently bid_side_pnl is gross. We should do net. Let's do net.
    bid_side_pnl_net = bid_side_pnl - (maker_trades / 2) * maker_fee # approximate if we assume equal trades, or we can precisely track.
    
    # We will just report gross bid_side / ask_side.
    
    return {
        "threshold": threshold,
        "signal rate": signal_count / total_opportunities if total_opportunities else 0.0,
        "fill rate": maker_trades / total_opportunities if total_opportunities else 0.0,
        "trade count": maker_trades,
        "gross pnl": gross_pnl,
        "net pnl": net_pnl,
        "fee": fees_paid,
        "avg pnl per fill": net_pnl / maker_trades if maker_trades else 0.0,
        "toxic fill avoided": toxic_avoided,
        "missed profitable fill": missed_profitable,
        "max drawdown": mdd,
        "bid-side pnl": bid_side_pnl,
        "ask-side pnl": ask_side_pnl,
    }


def main():
    cfg = build_cfg()
    paths = build_paths(cfg)

    data_dir = cfg.data.data_dir
    symbol = cfg.backtest.symbol
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(
        session_ids, train_frac=cfg.split.train_session_frac, val_frac=cfg.split.val_session_frac
    )

    tick_size = float(infer_tick_size_from_lob(symbol_lob))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    weights_pth = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot/checkpoints/tcn_weights.pth")
    model.load_state_dict(torch.load(weights_pth, map_location=device, weights_only=True))
    model.eval()

    test_datasets = []
    for sid in test_sessions:
        lob_df = symbol_lob[symbol_lob["session_id"] == sid]
        trd_df = symbol_trades[symbol_trades["session_id"] == sid]
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
            continue
        test_datasets.append(
            (sid, _build_session_dataset(cfg, lob_df, trd_df, symbol, tick_size, sample_stride=cfg.backtest.sample_stride))
        )

    print("Collecting predictions for Test split...")
    test_probs, _, test_ask, test_bid = _collect_predictions(model, test_datasets, device=device)
    
    mid = (test_ask + test_bid) / 2.0
    fut_min, fut_max = _future_min_max(mid, horizon=cfg.window.horizon)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []

    print("Running simulations for each threshold...")
    for thr in tqdm(thresholds):
        res = compute_metrics_for_threshold(
            cfg, test_probs, test_ask, test_bid, fut_min, fut_max, tick_size, thr
        )
        results.append(res)

    df = pd.DataFrame(results)
    
    out_dir = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w13/02_threshold_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "threshold_trading_backtest.csv"
    
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
