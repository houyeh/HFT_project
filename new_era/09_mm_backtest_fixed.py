import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    split_ordered_session_ids,
)

# Reuse the data loader from 01_feature_importance.py.
spec = importlib.util.spec_from_file_location(
    "feat_imp",
    str(project_root / "new_era" / "01_feature_importance.py"),
)
feat_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_imp)
load_new_tw_data = feat_imp.load_new_tw_data


# =========================
# Backtest assumptions
# =========================
CONTRACT_MULTIPLIER = 200.0     # TXF: 1 point = 200 TWD
MAKER_FEE_TWD = 0.0             # Conservative setting can be 10~25; set yours here.
TAKER_FEE_TWD = 50.0            # Kept for diagnostics; main fixed strategies are maker-only.
MAX_INVENTORY = 3               # Avoid strategy becoming an unlimited directional bet.

# TCN classes: 0 = Down, 1 = Stationary, 2 = Up
DOWN, STAT, UP = 0, 1, 2
CLASS_NAMES = ["Down", "Stationary", "Up"]


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_session_dataset(lob_df, trade_df, symbol, tick_size):
    """
    Build stride=1 evaluation dataset for realistic quote-by-quote replay.
    Training can use stride=5, but backtest should not skip quote events.
    """
    feature_frame = build_tw_feature_frame(
        lob_df.reset_index(drop=True),
        trade_df.reset_index(drop=True),
        symbol,
        tick_size,
    ).dropna()
    zscore_frame = apply_rolling_zscore(feature_frame, TOP_8_FEATURES, window=2000)
    return WindowedLOBSequenceDataset(
        feature_df=zscore_frame,
        symbol=symbol,
        tick_size=tick_size,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        threshold_ticks=THRESHOLD_TICKS,
        sample_stride=1,
        feature_cols=Z_FEATURES,
    )


def collect_predictions(model, datasets, device):
    all_probs, all_labels, all_ask_px, all_bid_px, all_session = [], [], [], [], []

    with torch.no_grad():
        for session_id, dataset in datasets:
            print(f"Running inference on test session {session_id}: {len(dataset):,} samples")
            loader = DataLoader(dataset, batch_size=2048, shuffle=False)
            for batch in loader:
                # Dataset gives (B, T, F); Conv1d TCN expects (B, F, T).
                batch_X = batch["x"].transpose(1, 2).to(device)
                probs = torch.softmax(model(batch_X), dim=1).cpu().numpy()

                n = len(probs)
                all_probs.append(probs)
                all_labels.append(batch["y"].numpy())
                all_ask_px.append(batch["ask_px_1"].numpy())
                all_bid_px.append(batch["bid_px_1"].numpy())
                all_session.extend([session_id] * n)

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_ask_px),
        np.concatenate(all_bid_px),
        np.asarray(all_session),
    )


def infer_tick_from_quotes(bid_px, ask_px):
    px = np.unique(np.concatenate([bid_px, ask_px]))
    px = px[np.isfinite(px)]
    px = np.sort(px)
    diffs = np.diff(px)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.min(diffs))


def choose_quotes(strategy, prob, bid, ask, tick_size, inventory, params):
    """
    Decide passive quotes only. No taker crossing here.

    baseline:
        Always quote best bid and best ask, subject to inventory limit.

    tcn_gated:
        Cancel risky side.
        - If model says Down strongly: cancel bid.
        - If model says Up strongly: cancel ask.

    tcn_skew:
        Maker-only quote skew.
        - Strong Up: bid one tick more aggressive if spread allows, cancel ask.
        - Strong Down: ask one tick more aggressive if spread allows, cancel bid.
        This keeps the strategy passive: bid < ask always.
    """
    p_down, p_stat, p_up = float(prob[DOWN]), float(prob[STAT]), float(prob[UP])
    edge = p_up - p_down
    spread = ask - bid

    prob_threshold = params.get("prob_threshold", 0.70)
    edge_threshold = params.get("edge_threshold", 0.20)
    max_spread_ticks = params.get("max_spread_ticks", 3.0)
    skew_ticks = params.get("skew_ticks", 1)
    allow_inside_skew = params.get("allow_inside_skew", True)

    quote_bid = bid
    quote_ask = ask

    if strategy == "baseline":
        pass

    elif strategy == "tcn_gated":
        if p_down >= prob_threshold and edge <= -edge_threshold:
            quote_bid = None
        if p_up >= prob_threshold and edge >= edge_threshold:
            quote_ask = None

    elif strategy == "tcn_skew":
        # Avoid quoting when spread is unusually wide. Wide spread often means unstable quotes / bad fill risk.
        if spread / tick_size > max_spread_ticks:
            quote_bid = None
            quote_ask = None
        elif p_up >= prob_threshold and edge >= edge_threshold:
            # Bullish: avoid selling too early; make bid more competitive if still passive.
            quote_ask = None
            if allow_inside_skew and spread >= 2 * tick_size:
                quote_bid = min(bid + skew_ticks * tick_size, ask - tick_size)
        elif p_down >= prob_threshold and edge <= -edge_threshold:
            # Bearish: avoid buying falling price; make ask more competitive if still passive.
            quote_bid = None
            if allow_inside_skew and spread >= 2 * tick_size:
                quote_ask = max(ask - skew_ticks * tick_size, bid + tick_size)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Inventory guardrails.
    if inventory >= MAX_INVENTORY:
        quote_bid = None
    if inventory <= -MAX_INVENTORY:
        quote_ask = None

    return quote_bid, quote_ask


def run_passive_mm_simulation(probs, ask_px, bid_px, strategy, params=None):
    """
    Conservative maker fill simulation based on next quote movement.

    Buy fill condition:
        We posted bid q_bid at time t, and next ask <= q_bid.
        This means market moved down through our bid.

    Sell fill condition:
        We posted ask q_ask at time t, and next bid >= q_ask.
        This means market moved up through our ask.

    PnL:
        Cash from trades + ending inventory marked to final mid.
    """
    params = params or {}
    tick_size = params.get("tick_size") or infer_tick_from_quotes(bid_px, ask_px)

    cash = 0.0
    inventory = 0
    maker_trades = 0
    inventory_path = []
    trade_pnls = []

    for i in range(len(probs) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            inventory_path.append(inventory)
            continue
        if ask <= bid:
            inventory_path.append(inventory)
            continue

        quote_bid, quote_ask = choose_quotes(
            strategy=strategy,
            prob=probs[i],
            bid=bid,
            ask=ask,
            tick_size=tick_size,
            inventory=inventory,
            params=params,
        )

        # Passive buy fill.
        if quote_bid is not None and next_ask <= quote_bid and inventory < MAX_INVENTORY:
            cash -= quote_bid * CONTRACT_MULTIPLIER
            cash -= MAKER_FEE_TWD
            inventory += 1
            maker_trades += 1
            trade_pnls.append(-MAKER_FEE_TWD)

        # Passive sell fill.
        if quote_ask is not None and next_bid >= quote_ask and inventory > -MAX_INVENTORY:
            cash += quote_ask * CONTRACT_MULTIPLIER
            cash -= MAKER_FEE_TWD
            inventory -= 1
            maker_trades += 1
            trade_pnls.append(-MAKER_FEE_TWD)

        inventory_path.append(inventory)

    final_mid = float((bid_px[-1] + ask_px[-1]) / 2.0)
    pnl = cash + inventory * final_mid * CONTRACT_MULTIPLIER
    inv_path = np.asarray(inventory_path) if inventory_path else np.array([0])

    return {
        "strategy": strategy,
        "pnl": pnl,
        "trades": maker_trades,
        "maker_trades": maker_trades,
        "taker_trades": 0,
        "ending_inventory": inventory,
        "max_abs_inventory": int(np.max(np.abs(inv_path))),
        "avg_inventory": float(np.mean(inv_path)),
    }


def diagnose_signals(probs, labels, ask_px, bid_px):
    preds = probs.argmax(axis=1)
    tick_size = infer_tick_from_quotes(bid_px, ask_px)
    spread_ticks = (ask_px - bid_px) / tick_size

    print("\n" + "=" * 72)
    print("Signal diagnostics")
    print("=" * 72)
    print(f"Samples: {len(labels):,}")
    print(f"Class distribution true  [Down, Stat, Up]: {np.bincount(labels, minlength=3).tolist()}")
    print(f"Class distribution pred  [Down, Stat, Up]: {np.bincount(preds, minlength=3).tolist()}")
    print(f"Average spread: {np.nanmean(spread_ticks):.3f} ticks | median spread: {np.nanmedian(spread_ticks):.3f} ticks")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    for thr in [0.55, 0.60, 0.70, 0.80, 0.90]:
        active = (probs[:, UP] >= thr) | (probs[:, DOWN] >= thr)
        if active.sum() == 0:
            print(f"prob >= {thr:.2f}: no active signals")
            continue
        direction_correct = ((probs[:, UP] >= thr) & (labels == UP)) | ((probs[:, DOWN] >= thr) & (labels == DOWN))
        stationary_rate = np.mean(labels[active] == STAT)
        print(
            f"prob >= {thr:.2f}: active={active.sum():,}, "
            f"direction_precision={direction_correct[active].mean():.4f}, "
            f"stationary_rate={stationary_rate:.4f}"
        )


def sweep_strategies(probs, ask_px, bid_px, output_dir: Path):
    tick_size = infer_tick_from_quotes(bid_px, ask_px)
    rows = []

    # Always include baseline.
    baseline = run_passive_mm_simulation(
        probs, ask_px, bid_px, strategy="baseline", params={"tick_size": tick_size}
    )
    rows.append({"prob_threshold": np.nan, "edge_threshold": np.nan, **baseline})

    for strategy in ["tcn_gated", "tcn_skew"]:
        for prob_thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            for edge_thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
                result = run_passive_mm_simulation(
                    probs,
                    ask_px,
                    bid_px,
                    strategy=strategy,
                    params={
                        "tick_size": tick_size,
                        "prob_threshold": prob_thr,
                        "edge_threshold": edge_thr,
                        "max_spread_ticks": 3.0,
                        "skew_ticks": 1,
                    },
                )
                rows.append({"prob_threshold": prob_thr, "edge_threshold": edge_thr, **result})

    df = pd.DataFrame(rows)
    df["improvement_vs_baseline"] = df["pnl"] - baseline["pnl"]
    df = df.sort_values("pnl", ascending=False).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "09_strategy_sweep_results.csv"
    df.to_csv(csv_path, index=False)

    return baseline, df, csv_path


def print_strategy_result(title, result, baseline_pnl=None):
    improvement = "" if baseline_pnl is None else f" | vs baseline: {result['pnl'] - baseline_pnl:,.2f} TWD"
    print(f"\n{title}")
    print(f"  - PnL: {result['pnl']:,.2f} TWD{improvement}")
    print(f"  - trades: {result['trades']:,} (maker={result['maker_trades']:,}, taker={result['taker_trades']:,})")
    print(f"  - ending inventory: {result['ending_inventory']}")
    print(f"  - max abs inventory: {result['max_abs_inventory']}")


def main():
    """
    Session-aware, maker-only market-making backtest.

    Main fix compared with aggressive alpha:
    - Do not directly cross spread using TCN predictions.
    - Use TCN as defensive gating / maker quote skew.
    - Sweep thresholds on held-out test session for diagnostics.
      For research discipline, choose thresholds on validation sessions first,
      then run test only once.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    output_dir = project_root / "new_era" / "backtest_outputs"

    data_dir = project_root / "data" / "NEW"
    target_symbols = ["TXFB6"]
    lob_frames, trade_frames = load_new_tw_data(data_dir, target_symbols)
    symbol = target_symbols[0]

    if symbol not in lob_frames:
        print(f"Error: no LOB data found for {symbol}")
        return
    if not weights_path.exists():
        print(f"Error: model weights not found: {weights_path}")
        return

    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(
        symbol,
        pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
    )

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

    model = HLSFriendlyTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(safe_torch_load(weights_path, device))
    model.eval()

    probs, labels, ask_px, bid_px, sessions = collect_predictions(model, test_datasets, device)
    preds = probs.argmax(axis=1)

    print("\n" + "=" * 72)
    print("Session-aware TCN market-making backtest — fixed maker-only strategies")
    print("=" * 72)
    print(f"Accuracy on test sessions: {accuracy_score(labels, preds):.4f}")
    print("Confusion Matrix [Down, Stationary, Up]:")
    print(confusion_matrix(labels, preds, labels=[0, 1, 2]))

    diagnose_signals(probs, labels, ask_px, bid_px)

    baseline, sweep_df, csv_path = sweep_strategies(probs, ask_px, bid_px, output_dir)

    print("\n" + "=" * 72)
    print("Strategy results")
    print("=" * 72)
    print_strategy_result("[1] Baseline passive maker", baseline)

    print("\nTop 10 threshold-sweep results:")
    display_cols = [
        "strategy",
        "prob_threshold",
        "edge_threshold",
        "pnl",
        "improvement_vs_baseline",
        "trades",
        "ending_inventory",
        "max_abs_inventory",
    ]
    print(sweep_df[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 72)
    print("Saved sweep results")
    print("=" * 72)
    print(csv_path)
    print("\nNote: For final research reporting, tune thresholds on validation sessions, then run test once.")


if __name__ == "__main__":
    main()
