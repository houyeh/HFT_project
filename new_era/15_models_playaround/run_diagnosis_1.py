import os
import json
import numpy as np
import pandas as pd
import torch
import numbers
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from _common import build_cfg, build_paths
from new_era.models_playaround_15.backtest_val_sweep import _collect_predictions, _build_session_dataset, _max_drawdown
from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import split_ordered_session_ids

def run_task_1(cfg, paths, out_dir):
    print("Running Task 1: File Check")
    files_to_check = [
        "train_outputs/train_history.csv",
        "eval_outputs/classification_report_test.csv",
        "eval_outputs/evaluation_summary.json",
        "eval_outputs/confusion_matrix_count_test.png",
        "eval_outputs/confusion_matrix_percent_test.png",
        "eval_outputs/precision_recall_curve_test.png",
        "eval_outputs/roc_curve_test.png",
        "eval_outputs/calibration_curve_test.png",
        "eval_outputs/threshold_diagnostics_test.csv",
        "eval_outputs/threshold_diagnostics_plot_test.png",
    ]
    
    rows = []
    base_dir = paths.root_dir
    for f in files_to_check:
        p = base_dir / f
        rows.append({
            "file_name": f,
            "exists": p.exists(),
            "path": str(p),
            "note": ""
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_file_check.csv", index=False)
    
def run_task_2(paths, out_dir, test_y, test_probs):
    print("Running Task 2: Classification / Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    
    y_pred = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(test_y, y_pred, labels=[0, 1, 2])
    
    cm_df = pd.DataFrame(cm, index=["True Down", "True Stat", "True Up"], columns=["Pred Down", "Pred Stat", "Pred Up"])
    cm_df.to_csv(out_dir / "confusion_matrix_count.csv")
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row_pct = cm / np.where(row_sums == 0, 1, row_sums)
    cm_row_df = pd.DataFrame(cm_row_pct, index=["True Down", "True Stat", "True Up"], columns=["Pred Down", "Pred Stat", "Pred Up"])
    cm_row_df.to_csv(out_dir / "confusion_matrix_row_percent.csv")
    
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_col_pct = cm / np.where(col_sums == 0, 1, col_sums)
    cm_col_df = pd.DataFrame(cm_col_pct, index=["True Down", "True Stat", "True Up"], columns=["Pred Down", "Pred Stat", "Pred Up"])
    cm_col_df.to_csv(out_dir / "confusion_matrix_column_percent.csv")
    
    cr_path = paths.eval_dir / "classification_report_test.csv"
    if cr_path.exists():
        cr_df = pd.read_csv(cr_path)
        col_name = cr_df.columns[0]
        classes_df = cr_df[cr_df[col_name].astype(str).isin(['0', '1', '2'])].copy()
        classes_df[col_name] = classes_df[col_name].astype(str).map({'0':'Down', '1':'Stationary', '2':'Up'})
        classes_df.to_csv(out_dir / "classification_summary_clean.csv", index=False)
        
        avg_df = cr_df[~cr_df[col_name].astype(str).isin(['0', '1', '2'])].copy()
        avg_df.rename(columns={col_name: 'metric_type'}, inplace=True)
        avg_df.to_csv(out_dir / "average_metrics_clean.csv", index=False)

def simulate_with_metrics(
    cfg,
    prob,
    ask_px,
    bid_px,
    tick_size,
    threshold,
    random_cancel_rate=0.0,
    prediction_delay_events=0,
    cancel_delay_events=0,
    force_pull_bid_mask=None,
    force_pull_ask_mask=None,
    active_mask=None,
    seed=None,
):
    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    max_inv = cfg.backtest.max_inventory
    H = cfg.window.horizon

    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0

    signal_count = 0
    total_opportunities = 0

    missed_fill_count = 0
    missed_profitable_fill_count = 0
    missed_profitable_pnl = 0.0
    missed_toxic_fill_count = 0
    missed_toxic_pnl_avoided = 0.0

    pnl_path = []
    effective_pull_bid_state = False
    effective_pull_ask_state = False
    pending_bid_delay = 0
    pending_ask_delay = 0
    delayed_fill_count = 0
    delayed_bid_fill_count = 0
    delayed_ask_fill_count = 0
    delayed_toxic_fill_count = 0
    delayed_toxic_loss = 0.0
    
    # FIFO for realized side PnL
    open_trades = [] # list of (price, side, qty)
    realized_bid_pnl = 0.0
    realized_ask_pnl = 0.0
    bid_fill_count = 0
    ask_fill_count = 0
    closed_trade_pnls = []

    mid = (ask_px + bid_px) / 2.0
    rng = np.random.default_rng(seed)
    
    def process_fill(fill_px, side):
        nonlocal inventory, gross_cash, fees_paid, maker_trades, realized_bid_pnl, realized_ask_pnl
        nonlocal bid_fill_count, ask_fill_count, open_trades
        
        gross_cash += fill_px * point_value if side == 'ask' else -fill_px * point_value
        fees_paid += maker_fee
        maker_trades += 1
        
        if side == 'bid': bid_fill_count += 1
        if side == 'ask': ask_fill_count += 1
        
        if side == 'bid':
            if inventory < 0:
                # closing a short (which was an ask fill)
                open_px = open_trades.pop(0)
                # pnl of this round trip goes to the initiated side ('ask')
                pnl_closed = (open_px - fill_px) * point_value - 2 * maker_fee
                realized_ask_pnl += pnl_closed
                closed_trade_pnls.append(pnl_closed)
            else:
                open_trades.append(fill_px)
            inventory += 1
        else: # ask
            if inventory > 0:
                # closing a long (which was a bid fill)
                open_px = open_trades.pop(0)
                pnl_closed = (fill_px - open_px) * point_value - 2 * maker_fee
                realized_bid_pnl += pnl_closed
                closed_trade_pnls.append(pnl_closed)
            else:
                open_trades.append(fill_px)
            inventory -= 1

    n_events = min(len(ask_px), len(bid_px))
    if prob is not None:
        n_events = min(n_events, len(prob))
    if force_pull_bid_mask is not None:
        n_events = min(n_events, len(force_pull_bid_mask))
    if force_pull_ask_mask is not None:
        n_events = min(n_events, len(force_pull_ask_mask))
    if active_mask is not None:
        n_events = min(n_events, len(active_mask))

    for i in range(n_events - 1):
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
            
        score_idx = i - int(prediction_delay_events)
        if prob is not None and score_idx >= 0:
            p_down = float(prob[score_idx, 0])
            p_up = float(prob[score_idx, 2])
        else:
            p_down = 0.0
            p_up = 0.0
        
        if random_cancel_rate > 0:
            raw_pull_bid = bool(rng.random() < random_cancel_rate)
            raw_pull_ask = bool(rng.random() < random_cancel_rate)
        else:
            raw_pull_bid = p_down >= threshold if prob is not None else False
            raw_pull_ask = p_up >= threshold if prob is not None else False

        # True cancel-delay model:
        # - Cancel request (pull=True) takes N events to become effective.
        # - Re-quote (pull=False) is immediate.
        # This is intentionally asymmetric and not equivalent to prediction delay.
        if cancel_delay_events <= 0:
            effective_pull_bid = bool(raw_pull_bid)
            effective_pull_ask = bool(raw_pull_ask)
            effective_pull_bid_state = effective_pull_bid
            effective_pull_ask_state = effective_pull_ask
            pending_bid_delay = 0
            pending_ask_delay = 0
        else:
            # Bid side
            if raw_pull_bid:
                if effective_pull_bid_state:
                    pass
                elif pending_bid_delay <= 0:
                    pending_bid_delay = int(cancel_delay_events)
                else:
                    pending_bid_delay -= 1
                    if pending_bid_delay <= 0:
                        effective_pull_bid_state = True
            else:
                effective_pull_bid_state = False
                pending_bid_delay = 0

            # Ask side
            if raw_pull_ask:
                if effective_pull_ask_state:
                    pass
                elif pending_ask_delay <= 0:
                    pending_ask_delay = int(cancel_delay_events)
                else:
                    pending_ask_delay -= 1
                    if pending_ask_delay <= 0:
                        effective_pull_ask_state = True
            else:
                effective_pull_ask_state = False
                pending_ask_delay = 0

            effective_pull_bid = bool(effective_pull_bid_state)
            effective_pull_ask = bool(effective_pull_ask_state)

        if active_mask is not None and not bool(active_mask[i]):
            effective_pull_bid = True
            effective_pull_ask = True
        if force_pull_bid_mask is not None and bool(force_pull_bid_mask[i]):
            effective_pull_bid = True
        if force_pull_ask_mask is not None and bool(force_pull_ask_mask[i]):
            effective_pull_ask = True

        if effective_pull_bid:
            signal_count += 1
        if effective_pull_ask:
            signal_count += 1
        total_opportunities += 2

        baseline_bid_fill = next_ask <= bid
        baseline_ask_fill = next_bid >= ask

        if i + H < len(mid):
            fut_mid = mid[i + H]
        else:
            fut_mid = mid[-1]

        if baseline_bid_fill and effective_pull_bid:
            missed_fill_count += 1
            markout_pnl = (fut_mid - bid) * point_value
            if markout_pnl > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout_pnl
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += (-markout_pnl)
        elif baseline_bid_fill and raw_pull_bid and not effective_pull_bid:
            delayed_fill_count += 1
            delayed_bid_fill_count += 1
            delayed_markout = (fut_mid - bid) * point_value
            if delayed_markout < 0:
                delayed_toxic_fill_count += 1
                delayed_toxic_loss += (-delayed_markout)

        if baseline_ask_fill and effective_pull_ask:
            missed_fill_count += 1
            markout_pnl = (ask - fut_mid) * point_value
            if markout_pnl > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout_pnl
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += (-markout_pnl)
        elif baseline_ask_fill and raw_pull_ask and not effective_pull_ask:
            delayed_fill_count += 1
            delayed_ask_fill_count += 1
            delayed_markout = (ask - fut_mid) * point_value
            if delayed_markout < 0:
                delayed_toxic_fill_count += 1
                delayed_toxic_loss += (-delayed_markout)

        quote_bid = bid if not effective_pull_bid and inventory < max_inv else None
        quote_ask = ask if not effective_pull_ask and inventory > -max_inv else None

        if quote_bid is not None and next_ask <= quote_bid:
            process_fill(quote_bid, 'bid')
        if quote_ask is not None and next_bid >= quote_ask:
            process_fill(quote_ask, 'ask')

        pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)

    final_mid = mid[-1] if len(mid) else 0.0
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid

    pnl_arr = np.asarray(pnl_path, dtype=np.float64)
    mdd = _max_drawdown(pnl_arr)
    closed_arr = np.asarray(closed_trade_pnls, dtype=np.float64)
    median_pnl_per_fill = float(np.median(closed_arr)) if closed_arr.size > 0 else 0.0
    win_rate = float((closed_arr > 0).mean()) if closed_arr.size > 0 else 0.0

    return {
        "threshold": threshold,
        "signal_rate": signal_count / total_opportunities if total_opportunities else 0.0,
        "cancel_count": int(signal_count),
        "fill_rate": maker_trades / total_opportunities if total_opportunities else 0.0,
        "trade_count": maker_trades,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fee": fees_paid,
        "avg_pnl_per_fill": net_pnl / maker_trades if maker_trades else 0.0,
        "median_pnl_per_fill": median_pnl_per_fill,
        "win_rate": win_rate,
        "toxic_fill_count": missed_toxic_fill_count,
        "toxic_fill_avoided_count": missed_toxic_fill_count,
        "toxic_fill_avoided": missed_toxic_fill_count,
        "missed_fill_count": missed_fill_count,
        "missed_profitable_fill_count": missed_profitable_fill_count,
        "missed_profitable_pnl": missed_profitable_pnl,
        "missed_toxic_fill_count": missed_toxic_fill_count,
        "missed_toxic_pnl_avoided": missed_toxic_pnl_avoided,
        "net_cancel_value": missed_toxic_pnl_avoided - missed_profitable_pnl,
        "max_drawdown": mdd,
        "bid_fill_count": bid_fill_count,
        "ask_fill_count": ask_fill_count,
        "realized_bid_pnl": realized_bid_pnl,
        "realized_ask_pnl": realized_ask_pnl,
        "prediction_delay_events": int(prediction_delay_events),
        "cancel_delay_events": int(cancel_delay_events),
        "delayed_fill_count": delayed_fill_count,
        "delayed_bid_fill_count": delayed_bid_fill_count,
        "delayed_ask_fill_count": delayed_ask_fill_count,
        "delayed_toxic_fill_count": delayed_toxic_fill_count,
        "delayed_toxic_loss": delayed_toxic_loss,
    }

def _to_serializable_row(experiment_name, threshold, signal_rate, random_seed, metrics, notes=""):
    row = {
        "experiment_name": str(experiment_name),
        "threshold": float(threshold) if threshold is not None else np.nan,
        "signal_rate": float(signal_rate) if signal_rate is not None else np.nan,
        "random_seed": int(random_seed) if random_seed is not None else np.nan,
        "fill_rate": float(metrics.get("fill_rate", np.nan)),
        "trade_count": int(metrics.get("trade_count", 0)),
        "gross_pnl": float(metrics.get("gross_pnl", np.nan)),
        "net_pnl": float(metrics.get("net_pnl", np.nan)),
        "fee": float(metrics.get("fee", np.nan)),
        "avg_pnl_per_fill": float(metrics.get("avg_pnl_per_fill", np.nan)),
        "median_pnl_per_fill": float(metrics.get("median_pnl_per_fill", np.nan)),
        "win_rate": float(metrics.get("win_rate", np.nan)),
        "toxic_fill_count": int(metrics.get("toxic_fill_count", 0)),
        "toxic_fill_avoided_count": int(metrics.get("toxic_fill_avoided_count", 0)),
        "missed_profitable_pnl": float(metrics.get("missed_profitable_pnl", np.nan)),
        "missed_toxic_pnl_avoided": float(metrics.get("missed_toxic_pnl_avoided", np.nan)),
        "net_cancel_value": float(metrics.get("net_cancel_value", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "realized_bid_pnl": float(metrics.get("realized_bid_pnl", np.nan)),
        "realized_ask_pnl": float(metrics.get("realized_ask_pnl", np.nan)),
        "cancel_count": int(metrics.get("cancel_count", 0)),
        "missed_fill_count": int(metrics.get("missed_fill_count", 0)),
        "missed_profitable_fill_count": int(metrics.get("missed_profitable_fill_count", 0)),
        "missed_toxic_fill_count": int(metrics.get("missed_toxic_fill_count", 0)),
        "delayed_fill_count": int(metrics.get("delayed_fill_count", 0)),
        "delayed_toxic_fill_count": int(metrics.get("delayed_toxic_fill_count", 0)),
        "delayed_toxic_loss": float(metrics.get("delayed_toxic_loss", 0.0)),
        "notes": str(notes),
    }
    return row


def _summarize_random_group(rows_df, group_name):
    gdf = rows_df[(rows_df["experiment_group"] == group_name) & (rows_df["random_seed"].notna())].copy()
    return {
        "experiment_group": group_name,
        "num_runs": int(len(gdf)),
        "mean_net_pnl": float(gdf["net_pnl"].mean()),
        "std_net_pnl": float(gdf["net_pnl"].std(ddof=0)),
        "min_net_pnl": float(gdf["net_pnl"].min()),
        "max_net_pnl": float(gdf["net_pnl"].max()),
        "positive_pnl_rate": float((gdf["net_pnl"] > 0).mean()),
        "mean_trade_count": float(gdf["trade_count"].mean()),
        "mean_max_drawdown": float(gdf["max_drawdown"].mean()),
        "mean_toxic_fill_avoided": float(gdf["toxic_fill_avoided_count"].mean()),
        "mean_net_cancel_value": float(gdf["net_cancel_value"].mean()),
    }


def run_baseline_random_fixed(cfg, test_probs, test_ask, test_bid, tick_size, out_dir):
    print("Running fixed baseline/random diagnostics...")
    rows = []
    summary_rows = []

    baseline = simulate_with_metrics(cfg, None, test_ask, test_bid, tick_size, threshold=1.1)
    model55 = simulate_with_metrics(cfg, test_probs, test_ask, test_bid, tick_size, threshold=0.55)
    model60 = simulate_with_metrics(cfg, test_probs, test_ask, test_bid, tick_size, threshold=0.60)
    model70 = simulate_with_metrics(cfg, test_probs, test_ask, test_bid, tick_size, threshold=0.70)

    model_rows = [
        ("Baseline_A_NoModel", None, None, baseline),
        ("Model_0.55", 0.55, model55["signal_rate"], model55),
        ("Model_0.60", 0.60, model60["signal_rate"], model60),
        ("Model_0.70", 0.70, model70["signal_rate"], model70),
    ]
    for name, thr, sr, metrics in model_rows:
        row = _to_serializable_row(name, thr, sr, None, metrics, notes="deterministic single run")
        row["experiment_group"] = name
        rows.append(row)
        summary_rows.append(
            {
                "experiment_group": name,
                "num_runs": 1,
                "mean_net_pnl": float(row["net_pnl"]),
                "std_net_pnl": 0.0,
                "min_net_pnl": float(row["net_pnl"]),
                "max_net_pnl": float(row["net_pnl"]),
                "positive_pnl_rate": float(row["net_pnl"] > 0),
                "mean_trade_count": float(row["trade_count"]),
                "mean_max_drawdown": float(row["max_drawdown"]),
                "mean_toxic_fill_avoided": float(row["toxic_fill_avoided_count"]),
                "mean_net_cancel_value": float(row["net_cancel_value"]),
            }
        )

    random_configs = [
        ("Random_0.55", 0.55, float(model55["signal_rate"])),
        ("Random_0.60", 0.60, float(model60["signal_rate"])),
        ("Random_0.70", 0.70, float(model70["signal_rate"])),
    ]
    for group_name, thr, sr in random_configs:
        per_seed = []
        for seed in range(50):
            metrics = simulate_with_metrics(
                cfg,
                None,
                test_ask,
                test_bid,
                tick_size,
                threshold=1.1,
                random_cancel_rate=sr,
                seed=seed,
            )
            row = _to_serializable_row(
                group_name,
                thr,
                sr,
                seed,
                metrics,
                notes=f"random cancel, target signal_rate={sr:.6f}",
            )
            row["experiment_group"] = group_name
            rows.append(row)
            per_seed.append(metrics)

        avg_res = {}
        std_res = {}
        for k in per_seed[0].keys():
            vals = []
            for r in per_seed:
                v = r.get(k, np.nan)
                if isinstance(v, numbers.Number):
                    vals.append(float(v))
            if len(vals) > 0:
                avg_res[k] = float(np.nanmean(vals))
                std_res[k] = float(np.nanstd(vals))
        avg_note = (
            f"mean over 50 seeds, "
            f"mean_net_pnl={avg_res.get('net_pnl', float('nan')):.2f}, "
            f"std_net_pnl={std_res.get('net_pnl', float('nan')):.2f}"
        )
        rows.append(
            {
                **_to_serializable_row(group_name + "_MEAN", thr, sr, None, avg_res, notes=avg_note),
                "experiment_group": group_name,
            }
        )
        summary_rows.append(_summarize_random_group(pd.DataFrame(rows), group_name))

    baseline_fixed = pd.DataFrame(rows)
    baseline_fixed.to_csv(out_dir / "baseline_comparison_fixed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "baseline_summary_fixed.csv", index=False)
    return baseline_fixed, pd.DataFrame(summary_rows)


def run_latency_sensitivity_enhanced(cfg, test_probs, test_ask, test_bid, tick_size, out_dir):
    print("Running enhanced latency sensitivity...")
    rows = []
    thresholds = [0.55, 0.60, 0.70]
    delays = [0, 1, 2, 3, 5, 10]
    modes = [
        ("prediction_delay", lambda d: (d, 0)),
        ("cancel_delay", lambda d: (0, d)),
        ("combined_delay", lambda d: (d, d)),
    ]

    for mode_name, mapper in modes:
        for thr in thresholds:
            for delay in delays:
                pred_delay, cancel_delay = mapper(delay)
                r = simulate_with_metrics(
                    cfg,
                    test_probs,
                    test_ask,
                    test_bid,
                    tick_size,
                    threshold=thr,
                    prediction_delay_events=pred_delay,
                    cancel_delay_events=cancel_delay,
                )
                rows.append(
                    {
                        "mode": mode_name,
                        "threshold": float(thr),
                        "prediction_delay_events": int(pred_delay),
                        "cancel_delay_events": int(cancel_delay),
                        "trade_count": int(r["trade_count"]),
                        "net_pnl": float(r["net_pnl"]),
                        "avg_pnl_per_fill": float(r["avg_pnl_per_fill"]),
                        "toxic_fill_count": int(r["toxic_fill_count"]),
                        "toxic_fill_avoided_count": int(r["toxic_fill_avoided_count"]),
                        "delayed_fill_count": int(r["delayed_fill_count"]),
                        "delayed_toxic_fill_count": int(r["delayed_toxic_fill_count"]),
                        "delayed_toxic_loss": float(r["delayed_toxic_loss"]),
                        "missed_toxic_pnl_avoided": float(r["missed_toxic_pnl_avoided"]),
                        "missed_profitable_pnl": float(r["missed_profitable_pnl"]),
                        "net_cancel_value": float(r["net_cancel_value"]),
                        "max_drawdown": float(r["max_drawdown"]),
                        "notes": "",
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "latency_sensitivity_enhanced.csv", index=False)

    def plot_metric(mode_name, x_col, y_col, out_name, title):
        fig, ax = plt.subplots()
        sub_all = df[df["mode"] == mode_name]
        for thr in sorted(sub_all["threshold"].unique()):
            sub = sub_all[sub_all["threshold"] == thr].sort_values(x_col)
            ax.plot(sub[x_col], sub[y_col], marker="o", label=f"thr={thr:.2f}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        fig.savefig(out_dir / out_name)
        plt.close(fig)

    plot_metric("prediction_delay", "prediction_delay_events", "net_pnl", "latency_prediction_delay_vs_net_pnl.png", "Prediction Delay vs Net PnL")
    plot_metric("cancel_delay", "cancel_delay_events", "net_pnl", "latency_cancel_delay_vs_net_pnl.png", "Cancel Delay vs Net PnL")
    plot_metric("combined_delay", "prediction_delay_events", "net_pnl", "latency_combined_delay_vs_net_pnl.png", "Combined Delay vs Net PnL")
    plot_metric("cancel_delay", "cancel_delay_events", "delayed_toxic_loss", "latency_cancel_delay_vs_delayed_toxic_loss.png", "Cancel Delay vs Delayed Toxic Loss")
    return df


def run_missed_opportunity_fixed(cfg, test_probs, test_ask, test_bid, tick_size, baseline_fixed_df, out_dir):
    print("Running missed opportunity diagnostics (fixed)...")
    rows = []
    for thr in [0.55, 0.60, 0.70]:
        r = simulate_with_metrics(cfg, test_probs, test_ask, test_bid, tick_size, threshold=thr)
        rows.append(
            {
                "threshold": float(thr),
                "signal_rate": float(r["signal_rate"]),
                "cancel_count": int(r["cancel_count"]),
                "missed_fill_count": int(r["missed_fill_count"]),
                "missed_profitable_fill_count": int(r["missed_profitable_fill_count"]),
                "missed_profitable_pnl": float(r["missed_profitable_pnl"]),
                "missed_toxic_fill_count": int(r["missed_toxic_fill_count"]),
                "missed_toxic_pnl_avoided": float(r["missed_toxic_pnl_avoided"]),
                "net_cancel_value": float(r["net_cancel_value"]),
                "avg_net_cancel_value_per_cancel": float(r["net_cancel_value"] / max(1, r["cancel_count"])),
                "notes": "model threshold",
            }
        )

    baseline = simulate_with_metrics(cfg, None, test_ask, test_bid, tick_size, threshold=1.1)
    rows.append(
        {
            "threshold": np.nan,
            "signal_rate": float(baseline["signal_rate"]),
            "cancel_count": int(baseline["cancel_count"]),
            "missed_fill_count": int(baseline["missed_fill_count"]),
            "missed_profitable_fill_count": int(baseline["missed_profitable_fill_count"]),
            "missed_profitable_pnl": float(baseline["missed_profitable_pnl"]),
            "missed_toxic_fill_count": int(baseline["missed_toxic_fill_count"]),
            "missed_toxic_pnl_avoided": float(baseline["missed_toxic_pnl_avoided"]),
            "net_cancel_value": float(baseline["net_cancel_value"]),
            "avg_net_cancel_value_per_cancel": float(baseline["net_cancel_value"] / max(1, baseline["cancel_count"])),
            "notes": "no-model baseline",
        }
    )

    for random_group in ["Random_0.55", "Random_0.60"]:
        gdf = baseline_fixed_df[
            (baseline_fixed_df["experiment_group"] == random_group)
            & (baseline_fixed_df["random_seed"].notna())
        ]
        rows.append(
            {
                "threshold": float(random_group.split("_")[1]),
                "signal_rate": float(gdf["signal_rate"].mean()),
                "cancel_count": float(gdf["cancel_count"].mean()),
                "missed_fill_count": float(gdf["missed_fill_count"].mean()),
                "missed_profitable_fill_count": float(gdf["missed_profitable_fill_count"].mean()),
                "missed_profitable_pnl": float(gdf["missed_profitable_pnl"].mean()),
                "missed_toxic_fill_count": float(gdf["missed_toxic_fill_count"].mean()),
                "missed_toxic_pnl_avoided": float(gdf["missed_toxic_pnl_avoided"].mean()),
                "net_cancel_value": float(gdf["net_cancel_value"].mean()),
                "avg_net_cancel_value_per_cancel": float(
                    gdf["net_cancel_value"].sum() / max(1.0, gdf["cancel_count"].sum())
                ),
                "notes": f"{random_group} mean over {len(gdf)} seeds",
            }
        )

    df = pd.DataFrame(rows)
    if (df["missed_fill_count"].fillna(0.0) == 0).all() or (df["missed_profitable_fill_count"].fillna(0.0) == 0).all():
        df["notes"] = df["notes"].astype(str) + " | BUG: missed opportunity counts all zero"
    df.to_csv(out_dir / "missed_opportunity_diagnostics_fixed.csv", index=False)
    return df


def _rank_to_unit(series, higher_better=True):
    if higher_better:
        return series.rank(pct=True)
    return (-series).rank(pct=True)


def run_validation_threshold_selection_v2(cfg, val_probs, val_ask, val_bid, tick_size, out_dir):
    print("Running validation threshold selection v2...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []
    for thr in thresholds:
        r = simulate_with_metrics(cfg, val_probs, val_ask, val_bid, tick_size, threshold=thr)
        rows.append(
            {
                "threshold": float(thr),
                "val_signal_rate": float(r["signal_rate"]),
                "val_trade_count": int(r["trade_count"]),
                "val_net_pnl": float(r["net_pnl"]),
                "val_max_drawdown": float(r["max_drawdown"]),
                "val_net_cancel_value": float(r["net_cancel_value"]),
            }
        )
    df = pd.DataFrame(rows)

    pnl_score = _rank_to_unit(df["val_net_pnl"], higher_better=True)
    dd_score = _rank_to_unit(df["val_max_drawdown"], higher_better=False)
    trade_score = _rank_to_unit(df["val_trade_count"], higher_better=True)
    cancel_score = _rank_to_unit(df["val_net_cancel_value"], higher_better=True)
    signal_target = 0.05
    signal_penalty = (df["val_signal_rate"] - signal_target).abs()
    signal_score = 1.0 - (signal_penalty / max(float(signal_penalty.max()), 1e-9))

    df["selection_score"] = (
        0.40 * pnl_score
        + 0.20 * dd_score
        + 0.15 * trade_score
        + 0.15 * cancel_score
        + 0.10 * signal_score
    )

    # Penalize thresholds that produce too few trades.
    df.loc[df["val_trade_count"] < 100, "selection_score"] -= 0.15
    df.loc[df["val_signal_rate"] > 0.20, "selection_score"] -= 0.10

    df = df.sort_values(["selection_score", "val_net_pnl"], ascending=[False, False]).reset_index(drop=True)
    best = df.iloc[0]
    df["selection_reason"] = ""
    df.loc[0, "selection_reason"] = (
        f"best composite score: pnl={best['val_net_pnl']:.2f}, "
        f"dd={best['val_max_drawdown']:.2f}, trades={int(best['val_trade_count'])}, "
        f"net_cancel={best['val_net_cancel_value']:.2f}, signal_rate={best['val_signal_rate']:.4f}"
    )
    df.to_csv(out_dir / "validation_threshold_selection_v2.csv", index=False)
    # Keep legacy file for compatibility.
    df.rename(
        columns={
            "val_net_pnl": "val_net_pnl",
            "val_trade_count": "val_trade_count",
            "val_signal_rate": "val_signal_rate",
            "val_max_drawdown": "val_max_drawdown",
            "val_net_cancel_value": "val_net_cancel_value",
        }
    ).to_csv(out_dir / "validation_threshold_selection.csv", index=False)
    return float(best["threshold"]), df


def run_val_threshold_selection(cfg, val_probs, val_ask, val_bid, tick_size, out_dir):
    print("Running validation threshold selection...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []
    for thr in thresholds:
        r = simulate_with_metrics(cfg, val_probs, val_ask, val_bid, tick_size, threshold=thr)
        rows.append(
            {
                "threshold": float(thr),
                "val_net_pnl": float(r["net_pnl"]),
                "val_trade_count": int(r["trade_count"]),
                "val_signal_rate": float(r["signal_rate"]),
                "val_fill_rate": float(r["fill_rate"]),
                "val_max_drawdown": float(r["max_drawdown"]),
            }
        )

    df = pd.DataFrame(rows).sort_values(["val_net_pnl", "val_trade_count"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(out_dir / "validation_threshold_selection.csv", index=False)
    return float(df.iloc[0]["threshold"])


def _detect_symbol_from_data_dir(data_dir: Path, preferred_symbol: str) -> str:
    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir)))
    symbol_counts = {}
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        for row in lines[1:]:
            parts = [item.strip() for item in row.split(",")]
            if len(parts) < 34:
                continue
            symbol = parts[3]
            if not symbol:
                continue
            try:
                best_bid = float(parts[21])
                best_ask = float(parts[22])
            except ValueError:
                continue
            if best_bid <= 0 or best_ask <= 0:
                continue
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    if preferred_symbol in symbol_counts:
        return preferred_symbol
    if not symbol_counts:
        raise RuntimeError(f"No valid symbols detected in {data_dir}")
    return sorted(symbol_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def _build_datasets_for_sessions(cfg, symbol_lob, symbol_trades, symbol, tick_size, session_ids):
    datasets = []
    for sid in session_ids:
        lob_df = symbol_lob[symbol_lob["session_id"] == sid]
        trd_df = symbol_trades[symbol_trades["session_id"] == sid]
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
            continue
        datasets.append(
            (
                sid,
                _build_session_dataset(
                    cfg,
                    lob_df,
                    trd_df,
                    symbol,
                    tick_size,
                    sample_stride=cfg.backtest.sample_stride,
                ),
            )
        )
    return datasets


@torch.no_grad()
def _collect_predictions_extended(model, datasets, device):
    probs_all = []
    y_all = []
    ask_all = []
    bid_all = []
    ts_all = []
    sess_all = []
    mid_all = []
    spread_all = []
    bid_vol1_all = []
    ask_vol1_all = []
    qimb_all = []
    rvol_all = []

    for session_id, dataset in datasets:
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        for batch in loader:
            batch_x = batch["x"].transpose(1, 2).to(device)
            probs = torch.softmax(model(batch_x), dim=1).detach().cpu().numpy()
            n = int(probs.shape[0])
            probs_all.append(probs)
            y_all.append(batch["y"].cpu().numpy())
            ask = batch["ask_px_1"].cpu().numpy()
            bid = batch["bid_px_1"].cpu().numpy()
            ask_all.append(ask)
            bid_all.append(bid)
            ts_all.append(batch["timestamp"].cpu().numpy() if "timestamp" in batch else np.arange(n))
            sess_all.append(np.array([session_id] * n, dtype=object))
            mid_all.append(batch["current_mid"].cpu().numpy() if "current_mid" in batch else (ask + bid) / 2.0)
            spread_all.append(batch["spread_ticks"].cpu().numpy() if "spread_ticks" in batch else (ask - bid))
            bid_vol1_all.append(batch["bid_vol_1"].cpu().numpy() if "bid_vol_1" in batch else np.zeros(n))
            ask_vol1_all.append(batch["ask_vol_1"].cpu().numpy() if "ask_vol_1" in batch else np.zeros(n))
            qimb_all.append(batch["queue_imbalance"].cpu().numpy() if "queue_imbalance" in batch else np.zeros(n))
            rvol_all.append(batch["rolling_volatility"].cpu().numpy() if "rolling_volatility" in batch else np.zeros(n))

    if not probs_all:
        empty = np.array([], dtype=np.float64)
        return {
            "probs": np.zeros((0, 0), dtype=np.float64),
            "y": np.zeros((0,), dtype=np.int64),
            "ask": empty,
            "bid": empty,
            "timestamp": np.array([], dtype=np.int64),
            "session_id": np.array([], dtype=object),
            "current_mid": empty,
            "spread_ticks": empty,
            "bid_vol_1": empty,
            "ask_vol_1": empty,
            "queue_imbalance": empty,
            "rolling_volatility": empty,
        }

    return {
        "probs": np.concatenate(probs_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "ask": np.concatenate(ask_all, axis=0),
        "bid": np.concatenate(bid_all, axis=0),
        "timestamp": np.concatenate(ts_all, axis=0).astype(np.int64),
        "session_id": np.concatenate(sess_all, axis=0),
        "current_mid": np.concatenate(mid_all, axis=0),
        "spread_ticks": np.concatenate(spread_all, axis=0),
        "bid_vol_1": np.concatenate(bid_vol1_all, axis=0),
        "ask_vol_1": np.concatenate(ask_vol1_all, axis=0),
        "queue_imbalance": np.concatenate(qimb_all, axis=0),
        "rolling_volatility": np.concatenate(rvol_all, axis=0),
    }


def _build_event_frame(pred_pack, tick_size):
    df = pd.DataFrame(
        {
            "session_id": pred_pack["session_id"],
            "timestamp": pred_pack["timestamp"],
            "prob_down": pred_pack["probs"][:, 0],
            "prob_stationary": pred_pack["probs"][:, 1],
            "prob_up": pred_pack["probs"][:, 2],
            "ask_px_1": pred_pack["ask"],
            "bid_px_1": pred_pack["bid"],
            "current_mid": pred_pack["current_mid"],
            "spread_ticks_raw": pred_pack["spread_ticks"],
            "bid_vol_1": pred_pack["bid_vol_1"],
            "ask_vol_1": pred_pack["ask_vol_1"],
            "queue_imbalance": pred_pack["queue_imbalance"],
            "rolling_volatility_raw": pred_pack["rolling_volatility"],
        }
    )
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    df["spread_ticks"] = (df["ask_px_1"] - df["bid_px_1"]) / max(float(tick_size), 1e-9)
    df["volume_imbalance"] = (df["bid_vol_1"] - df["ask_vol_1"]) / np.maximum(
        df["bid_vol_1"] + df["ask_vol_1"], 1e-9
    )
    df["mid_diff_1"] = df.groupby("session_id")["current_mid"].diff(1)
    df["rolling_mid_return"] = df.groupby("session_id")["current_mid"].diff(200)
    df["cumulative_return"] = df.groupby("session_id")["current_mid"].transform(lambda s: s - float(s.iloc[0]))
    df["trend_strength"] = df["rolling_mid_return"].abs()
    df["rolling_volatility"] = df.groupby("session_id")["mid_diff_1"].transform(
        lambda s: s.rolling(200, min_periods=50).std()
    )
    ts_mod = df["timestamp"] % (24 * 60 * 60 * 1_000_000_000)
    hour = (ts_mod // (60 * 60 * 1_000_000_000)).astype(int)
    df["time_bucket"] = np.where(hour < 12, "morning", "afternoon")
    return df


def _run_regime_bucket_sim(cfg, event_df, mask, threshold, tick_size):
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.sum() < 2:
        return None
    prob = event_df.loc[mask_arr, ["prob_down", "prob_stationary", "prob_up"]].to_numpy(dtype=np.float64)
    ask = event_df.loc[mask_arr, "ask_px_1"].to_numpy(dtype=np.float64)
    bid = event_df.loc[mask_arr, "bid_px_1"].to_numpy(dtype=np.float64)
    base = simulate_with_metrics(cfg, None, ask, bid, tick_size, threshold=1.1)
    model = simulate_with_metrics(cfg, prob, ask, bid, tick_size, threshold=threshold)
    return {
        "num_events": int(mask_arr.sum()),
        "num_trades": int(model["trade_count"]),
        "baseline_net_pnl": float(base["net_pnl"]),
        "model_net_pnl": float(model["net_pnl"]),
        "improvement": float(model["net_pnl"] - base["net_pnl"]),
        "signal_rate": float(model["signal_rate"]),
        "toxic_fill_avoided": int(model["toxic_fill_avoided_count"]),
        "missed_profitable_pnl": float(model["missed_profitable_pnl"]),
        "missed_toxic_pnl_avoided": float(model["missed_toxic_pnl_avoided"]),
        "net_cancel_value": float(model["net_cancel_value"]),
        "max_drawdown": float(model["max_drawdown"]),
    }


def run_regime_diagnostics(cfg, dataset_event_map, selected_threshold, out_dir):
    print("Running regime diagnostics...")
    rows = []
    for dataset_name, payload in dataset_event_map.items():
        event_df = payload["event_df"]
        tick_size = payload["tick_size"]

        # trend direction regime
        trend_thr = 5.0 * tick_size
        trend_bucket = np.where(
            event_df["rolling_mid_return"] <= -trend_thr,
            "down",
            np.where(event_df["rolling_mid_return"] >= trend_thr, "up", "flat"),
        )
        for bucket in ["down", "flat", "up"]:
            res = _run_regime_bucket_sim(cfg, event_df, trend_bucket == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "trend_direction",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "rolling_mid_return +/-5 ticks",
                }
            )

        # trend strength regime
        ts = event_df["trend_strength"].fillna(0.0)
        q1, q2 = ts.quantile([0.33, 0.67]).tolist()
        strength_bucket = np.where(ts <= q1, "low", np.where(ts <= q2, "mid", "high"))
        for bucket in ["low", "mid", "high"]:
            res = _run_regime_bucket_sim(cfg, event_df, strength_bucket == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "trend_strength",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "abs rolling_mid_return terciles",
                }
            )

        # volatility regime
        vol = event_df["rolling_volatility"].fillna(0.0)
        v1, v2 = vol.quantile([0.33, 0.67]).tolist()
        vol_bucket = np.where(vol <= v1, "low", np.where(vol <= v2, "mid", "high"))
        for bucket in ["low", "mid", "high"]:
            res = _run_regime_bucket_sim(cfg, event_df, vol_bucket == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "volatility",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "rolling volatility terciles",
                }
            )

        # spread regime
        spread = event_df["spread_ticks"].fillna(0.0)
        spread_bucket = np.where(spread < 1.0, "<1", np.where(spread < 2.0, "1-2", ">=2"))
        for bucket in ["<1", "1-2", ">=2"]:
            res = _run_regime_bucket_sim(cfg, event_df, spread_bucket == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "spread",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "spread bucket by ticks",
                }
            )

        # imbalance regime
        imb = event_df["volume_imbalance"].fillna(0.0)
        imb_bucket = np.where(imb <= -0.2, "sell_heavy", np.where(imb >= 0.2, "buy_heavy", "neutral"))
        for bucket in ["sell_heavy", "neutral", "buy_heavy"]:
            res = _run_regime_bucket_sim(cfg, event_df, imb_bucket == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "order_imbalance",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "volume imbalance buckets",
                }
            )

        # time regime
        for bucket in ["morning", "afternoon"]:
            res = _run_regime_bucket_sim(cfg, event_df, event_df["time_bucket"] == bucket, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "time_of_day",
                    "regime_bucket": bucket,
                    **res,
                    "notes": "time bucket from synthetic timestamp",
                }
            )

        # per-session regime
        for sid in sorted(event_df["session_id"].dropna().unique().tolist()):
            res = _run_regime_bucket_sim(cfg, event_df, event_df["session_id"] == sid, selected_threshold, tick_size)
            if res is None:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_type": "session",
                    "regime_bucket": str(sid),
                    **res,
                    "notes": "per-session split",
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "regime_diagnostics.csv", index=False)
    return pd.DataFrame(rows)


def _run_gate_eval(cfg, probs, ask, bid, tick_size, threshold, active_mask=None, force_bid=None, force_ask=None):
    baseline = simulate_with_metrics(
        cfg,
        None,
        ask,
        bid,
        tick_size,
        threshold=1.1,
        active_mask=active_mask,
        force_pull_bid_mask=force_bid,
        force_pull_ask_mask=force_ask,
    )
    model = simulate_with_metrics(
        cfg,
        probs,
        ask,
        bid,
        tick_size,
        threshold=threshold,
        active_mask=active_mask,
        force_pull_bid_mask=force_bid,
        force_pull_ask_mask=force_ask,
    )
    return baseline, model


def run_regime_gate_sweep(cfg, dataset_event_map, out_dir):
    print("Running regime gate sweep...")
    rows = []
    model_thresholds = [0.55, 0.60, 0.70]

    for dataset_name, payload in dataset_event_map.items():
        # Focus NEW3 first; NEW is retained for reference.
        if dataset_name not in {"NEW", "NEW3"}:
            continue
        event_df = payload["event_df"]
        tick_size = payload["tick_size"]
        probs = event_df[["prob_down", "prob_stationary", "prob_up"]].to_numpy(dtype=np.float64)
        ask = event_df["ask_px_1"].to_numpy(dtype=np.float64)
        bid = event_df["bid_px_1"].to_numpy(dtype=np.float64)
        rolling_ret = event_df["rolling_mid_return"].fillna(0.0).to_numpy(dtype=np.float64)
        rolling_vol = event_df["rolling_volatility"].fillna(0.0).to_numpy(dtype=np.float64)
        spread_ticks = event_df["spread_ticks"].fillna(0.0).to_numpy(dtype=np.float64)

        vol_levels = sorted(set(float(v) for v in np.quantile(rolling_vol, [0.60, 0.75, 0.90])))
        spread_levels = [1.0, 1.5, 2.0]
        trend_levels = [5.0, 10.0, 20.0]

        gate_definitions = []
        for t in trend_levels:
            thr_abs = t * tick_size
            force_bid = rolling_ret <= -thr_abs
            force_ask = rolling_ret >= thr_abs
            gate_definitions.append(("trend_filter", {"trend_ticks": t}, None, force_bid, force_ask))
        for v in vol_levels:
            active = rolling_vol <= v
            gate_definitions.append(("volatility_filter", {"vol_threshold": float(v)}, active, None, None))
        for s in spread_levels:
            active = spread_ticks >= s
            gate_definitions.append(("spread_filter", {"spread_ticks_min": float(s)}, active, None, None))
        for t, v, s in zip(trend_levels, vol_levels, spread_levels):
            thr_abs = t * tick_size
            active = (rolling_vol <= v) & (spread_ticks >= s)
            force_bid = rolling_ret <= -thr_abs
            force_ask = rolling_ret >= thr_abs
            gate_definitions.append(
                (
                    "combined_filter",
                    {"trend_ticks": float(t), "vol_threshold": float(v), "spread_ticks_min": float(s)},
                    active,
                    force_bid,
                    force_ask,
                )
            )

        for gate_type, params, active, force_bid, force_ask in gate_definitions:
            active_event_rate = float(np.mean(active)) if active is not None else 1.0
            for thr in model_thresholds:
                baseline, model = _run_gate_eval(
                    cfg,
                    probs,
                    ask,
                    bid,
                    tick_size,
                    threshold=thr,
                    active_mask=active,
                    force_bid=force_bid,
                    force_ask=force_ask,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "gate_type": gate_type,
                        "gate_params": json.dumps(params, ensure_ascii=False),
                        "model_threshold": float(thr),
                        "active_event_rate": active_event_rate,
                        "trade_count": int(model["trade_count"]),
                        "gross_pnl": float(model["gross_pnl"]),
                        "net_pnl": float(model["net_pnl"]),
                        "baseline_net_pnl": float(baseline["net_pnl"]),
                        "improvement": float(model["net_pnl"] - baseline["net_pnl"]),
                        "max_drawdown": float(model["max_drawdown"]),
                        "avg_pnl_per_fill": float(model["avg_pnl_per_fill"]),
                        "toxic_fill_avoided": int(model["toxic_fill_avoided_count"]),
                        "net_cancel_value": float(model["net_cancel_value"]),
                        "notes": "",
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regime_gate_sweep.csv", index=False)
    return df


def run_oos_threshold_stability(cfg, model, device, selected_threshold, out_dir, new_test_sessions):
    print("Running OOS threshold stability scan...")
    project_root = Path(__file__).resolve().parents[2]
    dataset_specs = [
        ("NEW", Path(cfg.data.data_dir), cfg.backtest.symbol, "test_split"),
        ("NEW3", project_root / "data" / "NEW3", cfg.backtest.symbol, "full"),
    ]
    rows = []
    for ds_name, ds_data_dir, preferred_symbol, mode in dataset_specs:
        symbol = _detect_symbol_from_data_dir(ds_data_dir, preferred_symbol)
        lob_frames, trade_frames = load_new_tw_data(ds_data_dir, [symbol])
        symbol_lob = lob_frames[symbol]
        symbol_trades = trade_frames.get(
            symbol,
            pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
        )
        tick_size = float(infer_tick_size_from_lob(symbol_lob))

        if mode == "test_split" and symbol == cfg.backtest.symbol:
            session_ids = list(new_test_sessions)
        else:
            session_ids = list(symbol_lob["session_id"].drop_duplicates())

        datasets = _build_datasets_for_sessions(cfg, symbol_lob, symbol_trades, symbol, tick_size, session_ids)
        probs, _, ask_px, bid_px = _collect_predictions(model, datasets, device=device)
        baseline = simulate_with_metrics(cfg, None, ask_px, bid_px, tick_size, threshold=1.1)
        model_res = simulate_with_metrics(cfg, probs, ask_px, bid_px, tick_size, threshold=selected_threshold)

        rows.append(
            {
                "dataset": ds_name,
                "data_dir": str(ds_data_dir),
                "symbol": symbol,
                "mode": mode,
                "session_count": int(len(session_ids)),
                "sample_count": int(len(probs)),
                "selected_threshold": float(selected_threshold),
                "baseline_net_pnl": float(baseline["net_pnl"]),
                "model_net_pnl": float(model_res["net_pnl"]),
                "improvement": float(model_res["net_pnl"] - baseline["net_pnl"]),
                "model_trade_count": int(model_res["trade_count"]),
                "model_signal_rate": float(model_res["signal_rate"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "oos_threshold_stability.csv", index=False)
    return df


def _evaluate_random_same_signal_rate(cfg, ask, bid, tick_size, signal_rate, seeds=50):
    vals = []
    for seed in range(seeds):
        r = simulate_with_metrics(
            cfg,
            None,
            ask,
            bid,
            tick_size,
            threshold=1.1,
            random_cancel_rate=float(signal_rate),
            seed=seed,
        )
        vals.append(float(r["net_pnl"]))
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def generate_report_v2(
    out_dir,
    baseline_fixed,
    baseline_summary,
    latency_df,
    regime_df,
    gate_df,
    val_sel_df,
    final_test_v2_row,
    oos_df,
):
    def _safe_get(df, key, col):
        sub = df[df["experiment_group"] == key]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0][col])

    model55 = _safe_get(baseline_summary, "Model_0.55", "mean_net_pnl")
    model60 = _safe_get(baseline_summary, "Model_0.60", "mean_net_pnl")
    model70 = _safe_get(baseline_summary, "Model_0.70", "mean_net_pnl")
    r55_mean = _safe_get(baseline_summary, "Random_0.55", "mean_net_pnl")
    r55_std = _safe_get(baseline_summary, "Random_0.55", "std_net_pnl")
    r60_mean = _safe_get(baseline_summary, "Random_0.60", "mean_net_pnl")
    r60_std = _safe_get(baseline_summary, "Random_0.60", "std_net_pnl")
    r70_mean = _safe_get(baseline_summary, "Random_0.70", "mean_net_pnl")
    r70_std = _safe_get(baseline_summary, "Random_0.70", "std_net_pnl")

    new3_row = oos_df[oos_df["dataset"] == "NEW3"].head(1)
    new3_model = float(new3_row["model_net_pnl"].iloc[0]) if not new3_row.empty else float("nan")
    new3_improvement = float(new3_row["improvement"].iloc[0]) if not new3_row.empty else float("nan")

    selected_threshold = float(final_test_v2_row["selected_threshold"])
    test_net = float(final_test_v2_row["test_net_pnl"])
    test_impr = float(final_test_v2_row["test_vs_baseline_improvement"])
    test_vs_random = float(final_test_v2_row["test_vs_random_improvement"])

    # Latency verdict cues
    cancel_mode = latency_df[latency_df["mode"] == "cancel_delay"]
    cancel_d0 = cancel_mode[cancel_mode["cancel_delay_events"] == 0].groupby("threshold")["net_pnl"].mean()
    cancel_d3 = cancel_mode[cancel_mode["cancel_delay_events"] == 3].groupby("threshold")["net_pnl"].mean()
    cancel_worse = (cancel_d3 - cancel_d0).mean() < 0 if len(cancel_d0) and len(cancel_d3) else False

    pred_mode = latency_df[latency_df["mode"] == "prediction_delay"]
    pred_d3 = pred_mode[pred_mode["prediction_delay_events"] == 3].groupby("threshold")["net_pnl"].mean()
    pred_d0 = pred_mode[pred_mode["prediction_delay_events"] == 0].groupby("threshold")["net_pnl"].mean()
    pred_better = (pred_d3 - pred_d0).mean() > 0 if len(pred_d3) and len(pred_d0) else False

    # Regime clues
    new3_reg = regime_df[regime_df["dataset"] == "NEW3"]
    trend_down = new3_reg[(new3_reg["regime_type"] == "trend_direction") & (new3_reg["regime_bucket"] == "down")]
    trend_up = new3_reg[(new3_reg["regime_type"] == "trend_direction") & (new3_reg["regime_bucket"] == "up")]
    trend_flat = new3_reg[(new3_reg["regime_type"] == "trend_direction") & (new3_reg["regime_bucket"] == "flat")]

    gate_best_new3 = gate_df[gate_df["dataset"] == "NEW3"].sort_values("net_pnl", ascending=False).head(1)
    gate_best_txt = "N/A"
    if not gate_best_new3.empty:
        row = gate_best_new3.iloc[0]
        gate_best_txt = (
            f"{row['gate_type']} {row['gate_params']} @thr={row['model_threshold']}: "
            f"net_pnl={row['net_pnl']:.2f}, improvement={row['improvement']:.2f}, "
            f"active_rate={row['active_event_rate']:.3f}"
        )

    # Grade logic
    grade = "D"
    if np.isfinite(r55_mean) and np.isfinite(r60_mean) and np.isfinite(r70_mean):
        model_beats_random = (model55 > r55_mean + r55_std) and (model60 > r60_mean + r60_std) and (model70 > r70_mean + r70_std)
        if model_beats_random:
            grade = "B" if new3_model < 0 else "A"
        else:
            grade = "C"
    else:
        grade = "C"
    if new3_model < 0 and grade == "A":
        grade = "B"

    report = f"""# 1. Executive Summary
目前模型不適合方向交易；它可能作為 maker-only cancel filter，但是否有真實 edge 取決於 random cancel / OOS / true cancel delay 結果。

# 2. Confirmed Findings
- 模型是 risk score，不是 calibrated probability。
- 在 test split 上，Model_0.55/0.60 高於 no-model baseline。
- Validation-selected threshold (v2) = {selected_threshold:.2f}，test once net_pnl = {test_net:.2f}。
- NEW3 OOS 仍為負損益：model_net_pnl = {new3_model:.2f}，僅小幅改善 {new3_improvement:.2f}。

# 3. Critical Checks
- Random cancel baseline：已修正並產出 baseline_comparison_fixed.csv / baseline_summary_fixed.csv。
- Missed opportunity：已修正並產出 missed_opportunity_diagnostics_fixed.csv。
- True cancel delay：已拆分 prediction / cancel / combined 三種延遲。
- OOS / NEW3：策略在不利 regime 下仍失效（虧損）。
- Regime diagnostics：已產出 regime_diagnostics.csv 與 regime_gate_sweep.csv。

# 4. Random Baseline Verdict
- Model 0.55 net_pnl={model55:.2f} vs Random_0.55 mean+1std={r55_mean + r55_std:.2f}
- Model 0.60 net_pnl={model60:.2f} vs Random_0.60 mean+1std={r60_mean + r60_std:.2f}
- Model 0.70 net_pnl={model70:.2f} vs Random_0.70 mean+1std={r70_mean + r70_std:.2f}
- 若上述未穩定勝出，模型訊號尚未證明有 trading edge。

# 5. Latency Verdict
- prediction delay 3/5 事件表現變好：{pred_better}（較可能是減少 false-positive cancel，不是低延遲優勢）。
- true cancel delay 增加時 PnL 是否下降：{cancel_worse}。
- 結論：不可把 prediction delay 當作 cancel delay；FPGA必要性應以 true cancel delay 敏感度判定。

# 6. Regime Verdict
- NEW3 強趨勢情境下，maker-only + cancel filter 改善有限，策略仍虧損。
- 趨勢/高波動 regime 需要 macro regime gate。
- 最佳 gate（目前掃描）: {gate_best_txt}

# 7. Final Strategy Recommendation
Regime Gate
→ Maker-only Quote Engine
→ TCN Risk Score
→ Cancel / Skew Decision
→ Latency-aware Backtest

# 8. Final Grade
{grade}

# 9. Next Actions
- 強化 random baseline 顯著性檢定（bootstrap / block bootstrap）。
- 修正 label 與 horizon，分離 directional vs toxicity objective。
- 擴充 regime gate（session-level + volatility-aware）。
- 強化 backtest queue/cancel realism（true cancel delay / queue position decay）。
- 擴充 feature set（order-flow imbalance, cancel pressure, volatility state）。
"""
    (out_dir / "model_usability_diagnosis_report_v2.md").write_text(report, encoding="utf-8")

if __name__ == "__main__":
    out_dir = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot/eval_outputs/diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg()
    paths = build_paths(cfg)
    
    # Temp modify exp_dir if it doesn't match
    from dataclasses import replace
    paths = replace(paths, root_dir=Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"))
    paths = replace(paths, eval_dir=paths.root_dir / "eval_outputs")
    
    run_task_1(cfg, paths, out_dir)
    
    # Load model and data for test split
    print("Loading test data...")
    data_dir = cfg.data.data_dir
    symbol = cfg.backtest.symbol
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    _, val_sessions, test_sessions = split_ordered_session_ids(
        session_ids, train_frac=cfg.split.train_session_frac, val_frac=cfg.split.val_session_frac
    )
    
    tick_size = float(infer_tick_size_from_lob(symbol_lob))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs, num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list), kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout, seq_length=cfg.window.window_size,
    ).to(device)
    model.load_state_dict(torch.load(paths.root_dir / "checkpoints/tcn_weights.pth", map_location=device, weights_only=True))
    model.eval()

    val_datasets = _build_datasets_for_sessions(cfg, symbol_lob, symbol_trades, symbol, tick_size, val_sessions)
    test_datasets = _build_datasets_for_sessions(cfg, symbol_lob, symbol_trades, symbol, tick_size, test_sessions)

    val_probs, _, val_ask, val_bid = _collect_predictions(model, val_datasets, device=device)
    test_probs, test_y, test_ask, test_bid = _collect_predictions(model, test_datasets, device=device)

    run_task_2(paths, out_dir, test_y, test_probs)
    baseline_fixed_df, baseline_summary_df = run_baseline_random_fixed(
        cfg, test_probs, test_ask, test_bid, tick_size, out_dir
    )

    # Keep legacy baseline_comparison.csv for compatibility.
    legacy_groups = ["Baseline_A_NoModel", "Model_0.55", "Model_0.60", "Model_0.70", "Random_0.55", "Random_0.60", "Random_0.70"]
    legacy_rows = []
    for group in legacy_groups:
        if group.startswith("Random_"):
            row = baseline_summary_df[baseline_summary_df["experiment_group"] == group].head(1)
            if row.empty:
                continue
            legacy_rows.append(
                {
                    "experiment_name": group,
                    "net_pnl": float(row["mean_net_pnl"].iloc[0]),
                    "std_net_pnl": float(row["std_net_pnl"].iloc[0]),
                }
            )
        else:
            row = baseline_fixed_df[baseline_fixed_df["experiment_group"] == group].head(1)
            if row.empty:
                continue
            legacy_rows.append(
                {
                    "experiment_name": group,
                    "net_pnl": float(row["net_pnl"].iloc[0]),
                    "std_net_pnl": 0.0,
                }
            )
    pd.DataFrame(legacy_rows).to_csv(out_dir / "baseline_comparison.csv", index=False)

    latency_df = run_latency_sensitivity_enhanced(cfg, test_probs, test_ask, test_bid, tick_size, out_dir)
    missed_df = run_missed_opportunity_fixed(cfg, test_probs, test_ask, test_bid, tick_size, baseline_fixed_df, out_dir)

    selected_threshold, val_sel_df = run_validation_threshold_selection_v2(
        cfg, val_probs, val_ask, val_bid, tick_size, out_dir
    )
    final_test = simulate_with_metrics(cfg, test_probs, test_ask, test_bid, tick_size, threshold=selected_threshold)
    baseline_test = simulate_with_metrics(cfg, None, test_ask, test_bid, tick_size, threshold=1.1)
    rand_mean_same_sr, rand_std_same_sr = _evaluate_random_same_signal_rate(
        cfg, test_ask, test_bid, tick_size, signal_rate=final_test["signal_rate"], seeds=50
    )
    final_test_row_v2 = {
        "selected_threshold": float(selected_threshold),
        "test_signal_rate": float(final_test["signal_rate"]),
        "test_trade_count": int(final_test["trade_count"]),
        "test_net_pnl": float(final_test["net_pnl"]),
        "test_max_drawdown": float(final_test["max_drawdown"]),
        "test_net_cancel_value": float(final_test["net_cancel_value"]),
        "test_vs_baseline_improvement": float(final_test["net_pnl"] - baseline_test["net_pnl"]),
        "test_vs_random_improvement": float(final_test["net_pnl"] - rand_mean_same_sr),
        "notes": (
            f"random_same_signal_rate_mean={rand_mean_same_sr:.2f}, "
            f"std={rand_std_same_sr:.2f}, signal_rate={final_test['signal_rate']:.6f}"
        ),
    }
    pd.DataFrame([final_test_row_v2]).to_csv(out_dir / "final_test_once_result_v2.csv", index=False)
    # legacy compatibility output
    pd.DataFrame(
        [
            {
                "selected_threshold": float(selected_threshold),
                "test_net_pnl": float(final_test["net_pnl"]),
                "test_trade_count": int(final_test["trade_count"]),
                "test_signal_rate": float(final_test["signal_rate"]),
                "test_fill_rate": float(final_test["fill_rate"]),
                "test_max_drawdown": float(final_test["max_drawdown"]),
                "test_missed_fill_count": int(final_test["missed_fill_count"]),
                "test_delayed_fill_count": int(final_test["delayed_fill_count"]),
            }
        ]
    ).to_csv(out_dir / "final_test_once_result.csv", index=False)

    oos_df = run_oos_threshold_stability(cfg, model, device, selected_threshold, out_dir, test_sessions)

    # Build event frames for regime diagnostics and gate sweeps.
    new_pack = _collect_predictions_extended(model, test_datasets, device=device)
    new_event_df = _build_event_frame(new_pack, tick_size=tick_size)

    project_root = Path(__file__).resolve().parents[2]
    new3_dir = project_root / "data" / "NEW3"
    new3_symbol = _detect_symbol_from_data_dir(new3_dir, cfg.backtest.symbol)
    new3_lob_frames, new3_trade_frames = load_new_tw_data(new3_dir, [new3_symbol])
    new3_lob = new3_lob_frames[new3_symbol]
    new3_trades = new3_trade_frames.get(
        new3_symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])
    )
    new3_tick_size = float(infer_tick_size_from_lob(new3_lob))
    new3_sessions = list(new3_lob["session_id"].drop_duplicates())
    new3_datasets = _build_datasets_for_sessions(cfg, new3_lob, new3_trades, new3_symbol, new3_tick_size, new3_sessions)
    new3_pack = _collect_predictions_extended(model, new3_datasets, device=device)
    new3_event_df = _build_event_frame(new3_pack, tick_size=new3_tick_size)

    dataset_event_map = {
        "NEW": {"event_df": new_event_df, "tick_size": tick_size},
        "NEW3": {"event_df": new3_event_df, "tick_size": new3_tick_size},
    }
    regime_df = run_regime_diagnostics(cfg, dataset_event_map, selected_threshold, out_dir)
    gate_df = run_regime_gate_sweep(cfg, dataset_event_map, out_dir)

    generate_report_v2(
        out_dir=out_dir,
        baseline_fixed=baseline_fixed_df,
        baseline_summary=baseline_summary_df,
        latency_df=latency_df,
        regime_df=regime_df,
        gate_df=gate_df,
        val_sel_df=val_sel_df,
        final_test_v2_row=final_test_row_v2,
        oos_df=oos_df,
    )

    # Immediate sanity answers for random-edge checks.
    def _summary_row(group):
        row = baseline_summary_df[baseline_summary_df["experiment_group"] == group].head(1)
        return None if row.empty else row.iloc[0]

    r55 = _summary_row("Random_0.55")
    r60 = _summary_row("Random_0.60")
    r70 = _summary_row("Random_0.70")
    m55 = baseline_fixed_df[baseline_fixed_df["experiment_group"] == "Model_0.55"].head(1)
    m60 = baseline_fixed_df[baseline_fixed_df["experiment_group"] == "Model_0.60"].head(1)
    m70 = baseline_fixed_df[baseline_fixed_df["experiment_group"] == "Model_0.70"].head(1)

    if not m55.empty and r55 is not None:
        print(
            f"[CHECK] Model_0.55={float(m55['net_pnl'].iloc[0]):.2f}, "
            f"Random_0.55 mean+1std={float(r55['mean_net_pnl'] + r55['std_net_pnl']):.2f}"
        )
    if not m60.empty and r60 is not None:
        print(
            f"[CHECK] Model_0.60={float(m60['net_pnl'].iloc[0]):.2f}, "
            f"Random_0.60 mean+1std={float(r60['mean_net_pnl'] + r60['std_net_pnl']):.2f}"
        )
    if not m70.empty and r70 is not None:
        print(
            f"[CHECK] Model_0.70={float(m70['net_pnl'].iloc[0]):.2f}, "
            f"Random_0.70 mean+1std={float(r70['mean_net_pnl'] + r70['std_net_pnl']):.2f}"
        )

    print("Diagnosis v2 complete!")
