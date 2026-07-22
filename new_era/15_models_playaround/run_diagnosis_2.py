import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from _common import build_cfg, build_paths
from new_era.models_playaround_15.backtest_val_sweep import _collect_predictions, _build_session_dataset
from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import split_ordered_session_ids
from run_diagnosis_1 import simulate_with_metrics

def simulate_with_latency(cfg, prob, ask_px, bid_px, tick_size, threshold, delay_steps=0):
    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    max_inv = cfg.backtest.max_inventory
    H = cfg.window.horizon
    
    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0
    missed_toxic_pnl_avoided = 0.0
    toxic_avoided_count = 0
    
    mid = (ask_px + bid_px) / 2.0
    pnl_path = []
    
    for i in range(len(prob) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])
        mid_now = mid[i]
        
        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue
        if ask <= bid:
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue
            
        eff_i = max(0, i - delay_steps)
        p_down = float(prob[eff_i, 0])
        p_up = float(prob[eff_i, 2])
        
        pull_bid = p_down >= threshold
        pull_ask = p_up >= threshold
        
        # Calculate toxic avoided exactly as before but with delayed pull
        baseline_bid_fill = next_ask <= bid
        baseline_ask_fill = next_bid >= ask
        fut_mid = mid[i + H] if i + H < len(mid) else mid[-1]
        
        if baseline_bid_fill and pull_bid:
            p = (fut_mid - bid) * point_value
            if p <= 0:
                toxic_avoided_count += 1
                missed_toxic_pnl_avoided += (-p)
                
        if baseline_ask_fill and pull_ask:
            p = (ask - fut_mid) * point_value
            if p <= 0:
                toxic_avoided_count += 1
                missed_toxic_pnl_avoided += (-p)
                
        quote_bid = bid if not pull_bid and inventory < max_inv else None
        quote_ask = ask if not pull_ask and inventory > -max_inv else None
        
        if quote_bid is not None and next_ask <= quote_bid:
            gross_cash -= quote_bid * point_value
            fees_paid += maker_fee
            inventory += 1
            maker_trades += 1
            
        if quote_ask is not None and next_bid >= quote_ask:
            gross_cash += quote_ask * point_value
            fees_paid += maker_fee
            inventory -= 1
            maker_trades += 1
            
        pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
        
    final_mid = mid[-1] if len(mid) else 0.0
    net_pnl = gross_cash + inventory * final_mid * point_value - fees_paid
    mdd = float(np.max(np.maximum.accumulate(pnl_path) - pnl_path)) if pnl_path else 0.0
    
    return {
        "threshold": threshold,
        "delay_events": delay_steps,
        "trade_count": maker_trades,
        "net_pnl": net_pnl,
        "avg_pnl_per_fill": net_pnl / max(1, maker_trades),
        "toxic_fill_avoided_count": toxic_avoided_count,
        "missed_toxic_pnl_avoided": missed_toxic_pnl_avoided,
        "max_drawdown": mdd
    }

def run_task_7(cfg, device, model, out_dir):
    print("Running Task 7: OOS Stability")
    datasets = ["NEW", "NEW1", "NEW2", "NEW3"]
    results = []
    
    for ds in datasets:
        ds_path = cfg.data.data_dir.parent / ds
        if not ds_path.exists():
            continue
        try:
            lob_frames, trade_frames = load_new_tw_data(ds_path, [cfg.backtest.symbol])
            if cfg.backtest.symbol not in lob_frames: continue
            
            symbol_lob = lob_frames[cfg.backtest.symbol]
            symbol_trades = trade_frames.get(cfg.backtest.symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
            tick_size = float(infer_tick_size_from_lob(symbol_lob))
            
            session_ids = list(symbol_lob["session_id"].drop_duplicates())
            for sid in tqdm(session_ids, desc=f"OOS {ds}"):
                lob_df = symbol_lob[symbol_lob["session_id"] == sid]
                trd_df = symbol_trades[symbol_trades["session_id"] == sid]
                if len(lob_df) < cfg.window.window_size + cfg.window.horizon: continue
                
                dataset = _build_session_dataset(cfg, lob_df, trd_df, cfg.backtest.symbol, tick_size, sample_stride=cfg.backtest.sample_stride)
                probs, _, ask_px, bid_px = _collect_predictions(model, [(sid, dataset)], device=device)
                
                for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
                    r = simulate_with_metrics(cfg, probs, ask_px, bid_px, tick_size, thr)
                    results.append({
                        "dataset": ds,
                        "date_or_split": sid,
                        "threshold": thr,
                        "signal_rate": r["signal_rate"],
                        "trade_count": r["trade_count"],
                        "net_pnl": r["net_pnl"],
                        "avg_pnl_per_fill": r["avg_pnl_per_fill"],
                        "max_drawdown": r["max_drawdown"],
                        "toxic_fill_avoided_count": r["toxic_fill_avoided"],
                        "missed_profitable_pnl": r["missed_profitable_pnl"],
                        "net_cancel_value": r["net_cancel_value"],
                        "realized_bid_pnl": r["realized_bid_pnl"],
                        "realized_ask_pnl": r["realized_ask_pnl"]
                    })
        except Exception as e:
            print(f"Error in {ds}: {e}")
            
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(out_dir / "oos_threshold_stability.csv", index=False)
        
        summary = []
        for thr in df["threshold"].unique():
            sub = df[df["threshold"] == thr]
            summary.append({
                "threshold": thr,
                "num_splits": len(sub),
                "mean_net_pnl": sub["net_pnl"].mean(),
                "median_net_pnl": sub["net_pnl"].median(),
                "std_net_pnl": sub["net_pnl"].std(),
                "min_net_pnl": sub["net_pnl"].min(),
                "max_net_pnl": sub["net_pnl"].max(),
                "positive_pnl_rate": (sub["net_pnl"] > 0).mean(),
                "mean_max_drawdown": sub["max_drawdown"].mean(),
                "mean_trade_count": sub["trade_count"].mean(),
                "mean_net_cancel_value": sub["net_cancel_value"].mean()
            })
        pd.DataFrame(summary).to_csv(out_dir / "oos_threshold_summary.csv", index=False)


def run_task_8(cfg, prob, ask_px, bid_px, tick_size, out_dir):
    print("Running Task 8: Latency Sensitivity")
    delays = [0, 1, 2, 3, 5, 10]
    thresholds = [0.55, 0.60]
    results = []
    
    for thr in thresholds:
        for d in delays:
            # We assume prediction delay + cancel delay = total delay
            r = simulate_with_latency(cfg, prob, ask_px, bid_px, tick_size, thr, delay_steps=d)
            # just arbitrary breakdown for the CSV
            r["prediction_delay_events"] = d
            r["cancel_delay_events"] = 0
            r["notes"] = f"Total delay {d} ticks"
            results.append(r)
            
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "latency_sensitivity.csv", index=False)
    
    fig, ax = plt.subplots()
    for thr in thresholds:
        sub = df[df["threshold"] == thr]
        ax.plot(sub["delay_events"], sub["net_pnl"], marker='o', label=f"thr={thr}")
    ax.set_title("Latency vs Net PnL")
    ax.legend()
    fig.savefig(out_dir / "latency_vs_net_pnl.png")
    plt.close(fig)

def run_task_9(cfg, probs, ask_px, bid_px, y_true, tick_size, out_dir):
    print("Running Task 9: Calibration Diagnostics")
    
    mid = (ask_px + bid_px) / 2.0
    fut_mid = np.zeros_like(mid)
    H = cfg.window.horizon
    for i in range(len(mid)):
        fut_mid[i] = mid[i + H] if i + H < len(mid) else mid[-1]
    
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    rows = []
    for cls, name in [(0, "Down"), (2, "Up")]:
        p = probs[:, cls]
        y_bin = (y_true == cls).astype(int)
        
        for i in range(len(bins)-1):
            mask = (p >= bins[i]) & (p < bins[i+1])
            if i == len(bins)-2:
                mask = (p >= bins[i]) & (p <= bins[i+1])
                
            n = mask.sum()
            if n > 0:
                mean_p = p[mask].mean()
                act = y_bin[mask].mean()
                if cls == 0: # Down
                    ret = (mid[mask] - fut_mid[mask]).mean() / tick_size
                    pnl_if_used = (fut_mid[mask] - bid_px[mask]).mean() * cfg.backtest.point_value_twd_per_point
                else: # Up
                    ret = (fut_mid[mask] - mid[mask]).mean() / tick_size
                    pnl_if_used = (ask_px[mask] - fut_mid[mask]).mean() * cfg.backtest.point_value_twd_per_point
                    
                rows.append({
                    "class": name,
                    "bin_left": bins[i],
                    "bin_right": bins[i+1],
                    "num_samples": n,
                    "mean_pred_prob": mean_p,
                    "actual_positive_rate": act,
                    "avg_future_return_ticks": ret,
                    "avg_realized_pnl_if_used_for_cancel": pnl_if_used,
                    "notes": ""
                })
    pd.DataFrame(rows).to_csv(out_dir / "probability_bin_diagnostics.csv", index=False)


if __name__ == "__main__":
    out_dir = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot/eval_outputs/diagnosis")
    cfg = build_cfg()
    paths = build_paths(cfg)
    
    from dataclasses import replace
    paths = replace(paths, root_dir=Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"))
    paths = replace(paths, eval_dir=paths.root_dir / "eval_outputs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs, num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list), kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout, seq_length=cfg.window.window_size,
    ).to(device)
    model.load_state_dict(torch.load(paths.root_dir / "checkpoints/tcn_weights.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Load test split for Task 8 & 9
    lob_frames, trade_frames = load_new_tw_data(cfg.data.data_dir, [cfg.backtest.symbol])
    symbol_lob = lob_frames[cfg.backtest.symbol]
    symbol_trades = trade_frames.get(cfg.backtest.symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    tick_size = float(infer_tick_size_from_lob(symbol_lob))
    
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    _, _, test_sessions = split_ordered_session_ids(session_ids, train_frac=cfg.split.train_session_frac, val_frac=cfg.split.val_session_frac)
    
    test_datasets = []
    for sid in test_sessions:
        lob_df = symbol_lob[symbol_lob["session_id"] == sid]
        trd_df = symbol_trades[symbol_trades["session_id"] == sid]
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon: continue
        test_datasets.append((sid, _build_session_dataset(cfg, lob_df, trd_df, cfg.backtest.symbol, tick_size, sample_stride=cfg.backtest.sample_stride)))
        
    test_probs, test_y, test_ask, test_bid = _collect_predictions(model, test_datasets, device=device)
    
    run_task_8(cfg, test_probs, test_ask, test_bid, tick_size, out_dir)
    run_task_9(cfg, test_probs, test_ask, test_bid, test_y, tick_size, out_dir)
    
    # Task 7 can take a while so we run it last
    run_task_7(cfg, device, model, out_dir)
