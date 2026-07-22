"""
Walk-forward analysis for TCN market-making (maker-only) on TW event-level LOB sessions.

Goals
-----
1) Evaluate chronologically per session (walk-forward) instead of a single train/val/test split.
2) Reuse the same HLS-friendly TCN + feature pipeline as training/backtest scripts.
3) Export a tidy CSV report so you can paste results into the Week 11 report/PPT.

Important note
--------------
This script does *walk-forward evaluation* of a fixed pretrained model by default
(weights = new_era/005_tcn_weights.pth). That is the fastest way to get a
chronological robustness picture.

If you want a strict "re-train each fold" walk-forward, add it later as a second mode
because training per fold from raw CSV can take time and is easy to do wrong.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
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
)


# Reuse the data loader from 01_feature_importance.py (keeps parsing consistent with your pipeline).
spec = importlib.util.spec_from_file_location(
    "feat_imp",
    str(project_root / "new_era" / "01_feature_importance.py"),
)
feat_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_imp)
load_new_tw_data = feat_imp.load_new_tw_data


CONTRACT_MULTIPLIER = 200.0
MAKER_FEE_TWD = 15.0
MAX_INVENTORY = 3

DOWN, STAT, UP = 0, 1, 2


@dataclass(frozen=True)
class FoldResult:
    symbol: str
    fold: int
    train_sessions: str
    val_session: str
    test_session: str
    n_test_samples: int
    accuracy: float
    cm00: int
    cm01: int
    cm02: int
    cm10: int
    cm11: int
    cm12: int
    cm20: int
    cm21: int
    cm22: int
    true_down: int
    true_stat: int
    true_up: int
    pred_down: int
    pred_stat: int
    pred_up: int
    avg_spread_ticks: float
    median_spread_ticks: float
    baseline_pnl: float
    baseline_trades: int
    best_pnl: float
    best_improvement: float
    best_strategy: str
    best_prob_threshold: float
    best_edge_threshold: float
    best_trades: int


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_session_dataset(lob_df, trade_df, symbol: str, tick_size: float) -> WindowedLOBSequenceDataset:
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


def infer_tick_from_quotes(bid_px: np.ndarray, ask_px: np.ndarray) -> float:
    px = np.unique(np.concatenate([bid_px, ask_px]))
    px = px[np.isfinite(px)]
    px = np.sort(px)
    diffs = np.diff(px)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.min(diffs))


def choose_quotes_maker_only(
    *,
    strategy: str,
    prob: np.ndarray,
    bid: float,
    ask: float,
    tick_size: float,
    inventory: int,
    prob_threshold: float,
    edge_threshold: float,
    max_spread_ticks: float,
    skew_ticks: int,
) -> tuple[float | None, float | None]:
    p_down, p_stat, p_up = float(prob[DOWN]), float(prob[STAT]), float(prob[UP])
    edge = p_up - p_down
    spread = ask - bid

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
        if spread / tick_size > max_spread_ticks:
            quote_bid = None
            quote_ask = None
        elif p_up >= prob_threshold and edge >= edge_threshold:
            quote_ask = None
            if spread >= 2 * tick_size:
                quote_bid = min(bid + skew_ticks * tick_size, ask - tick_size)
        elif p_down >= prob_threshold and edge <= -edge_threshold:
            quote_bid = None
            if spread >= 2 * tick_size:
                quote_ask = max(ask - skew_ticks * tick_size, bid + tick_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if inventory >= MAX_INVENTORY:
        quote_bid = None
    if inventory <= -MAX_INVENTORY:
        quote_ask = None
    return quote_bid, quote_ask


def run_passive_mm_simulation(
    probs: np.ndarray,
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    *,
    strategy: str,
    prob_threshold: float = 0.70,
    edge_threshold: float = 0.20,
    max_spread_ticks: float = 3.0,
    skew_ticks: int = 1,
) -> dict[str, float | int]:
    tick_size = infer_tick_from_quotes(bid_px, ask_px)
    cash = 0.0
    inventory = 0
    maker_trades = 0
    max_abs_inventory = 0

    for i in range(len(probs) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            max_abs_inventory = max(max_abs_inventory, abs(inventory))
            continue
        if ask <= bid:
            max_abs_inventory = max(max_abs_inventory, abs(inventory))
            continue

        q_bid, q_ask = choose_quotes_maker_only(
            strategy=strategy,
            prob=probs[i],
            bid=bid,
            ask=ask,
            tick_size=tick_size,
            inventory=inventory,
            prob_threshold=prob_threshold,
            edge_threshold=edge_threshold,
            max_spread_ticks=max_spread_ticks,
            skew_ticks=skew_ticks,
        )

        if q_bid is not None and next_ask <= q_bid and inventory < MAX_INVENTORY:
            cash -= q_bid * CONTRACT_MULTIPLIER
            cash -= MAKER_FEE_TWD
            inventory += 1
            maker_trades += 1

        if q_ask is not None and next_bid >= q_ask and inventory > -MAX_INVENTORY:
            cash += q_ask * CONTRACT_MULTIPLIER
            cash -= MAKER_FEE_TWD
            inventory -= 1
            maker_trades += 1

        max_abs_inventory = max(max_abs_inventory, abs(inventory))

    final_mid = float((bid_px[-1] + ask_px[-1]) / 2.0)
    pnl = cash + inventory * final_mid * CONTRACT_MULTIPLIER
    return {
        "pnl": float(pnl),
        "trades": int(maker_trades),
        "ending_inventory": int(inventory),
        "max_abs_inventory": int(max_abs_inventory),
    }


def sweep_strategies_for_session(
    probs: np.ndarray,
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    *,
    prob_thresholds: list[float],
    edge_thresholds: list[float],
) -> tuple[dict, dict]:
    baseline = run_passive_mm_simulation(probs, ask_px, bid_px, strategy="baseline")
    best = {
        "strategy": "baseline",
        "prob_threshold": 0.0,
        "edge_threshold": 0.0,
        **baseline,
        "improvement": 0.0,
    }

    for strategy in ["tcn_gated", "tcn_skew"]:
        for pthr in prob_thresholds:
            for ethr in edge_thresholds:
                res = run_passive_mm_simulation(
                    probs,
                    ask_px,
                    bid_px,
                    strategy=strategy,
                    prob_threshold=pthr,
                    edge_threshold=ethr,
                )
                improvement = float(res["pnl"] - baseline["pnl"])
                if res["pnl"] > best["pnl"]:
                    best = {
                        "strategy": strategy,
                        "prob_threshold": float(pthr),
                        "edge_threshold": float(ethr),
                        **res,
                        "improvement": improvement,
                    }

    return baseline, best


def apply_best_to_test(
    probs: np.ndarray,
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    *,
    best: dict,
) -> dict:
    """Apply the chosen (val-tuned) best config to a test session."""
    if best["strategy"] == "baseline":
        return run_passive_mm_simulation(probs, ask_px, bid_px, strategy="baseline")
    return run_passive_mm_simulation(
        probs,
        ask_px,
        bid_px,
        strategy=str(best["strategy"]),
        prob_threshold=float(best["prob_threshold"]),
        edge_threshold=float(best["edge_threshold"]),
    )


def collect_predictions_for_session(model, dataset: WindowedLOBSequenceDataset, device):
    probs_list, y_list, ask_list, bid_list = [], [], [], []
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].transpose(1, 2).to(device)  # (B, F, T)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            probs_list.append(probs)
            y_list.append(batch["y"].numpy())
            ask_list.append(batch["ask_px_1"].numpy())
            bid_list.append(batch["bid_px_1"].numpy())
    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list)
    ask_px = np.concatenate(ask_list)
    bid_px = np.concatenate(bid_list)
    return probs, y, ask_px, bid_px


def session_spread_stats(ask_px: np.ndarray, bid_px: np.ndarray, tick_size: float) -> tuple[float, float]:
    spread_ticks = (ask_px - bid_px) / tick_size
    spread_ticks = spread_ticks[np.isfinite(spread_ticks)]
    if len(spread_ticks) == 0:
        return 0.0, 0.0
    return float(np.mean(spread_ticks)), float(np.median(spread_ticks))


def build_walk_forward_folds(session_ids: list[str], min_train_sessions: int) -> list[tuple[list[str], str, str]]:
    """
    Create folds: train = sessions[:i], val = sessions[i], test = sessions[i+1]
    for i from min_train_sessions to len(sessions)-2.
    """
    ordered = sorted(session_ids)
    folds = []
    for i in range(min_train_sessions, len(ordered) - 1):
        train = ordered[:i]
        val = ordered[i]
        test = ordered[i + 1]
        folds.append((train, val, test))
    return folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward evaluation for TCN maker-only market making.")
    p.add_argument("--symbols", default="TXFB6", help="Comma-separated symbols, e.g. TXFB6,MXFB6")
    p.add_argument("--min-train-sessions", type=int, default=2, help="Minimum sessions used as walk-forward warmup.")
    p.add_argument("--prob-thresholds", default="0.55,0.60,0.70,0.80,0.90")
    p.add_argument("--edge-thresholds", default="0.05,0.10,0.15,0.20,0.30,0.40")
    p.add_argument("--out-dir", default="", help="Output directory (default: new_era/backtest_outputs/walk_forward)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prob_thresholds = [float(x) for x in args.prob_thresholds.split(",") if x.strip()]
    edge_thresholds = [float(x) for x in args.edge_thresholds.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "new_era" / "backtest_outputs" / "walk_forward")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {project_root / 'data' / 'NEW'} for CSVs...")
    lob_frames, trade_frames = load_new_tw_data(project_root / "data" / "NEW", symbols)

    model = HLSFriendlyTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(safe_torch_load(weights_path, device))
    model.eval()

    all_rows: list[FoldResult] = []

    for symbol in symbols:
        if symbol not in lob_frames:
            print(f"Skip {symbol}: no data found.")
            continue

        symbol_lob = lob_frames[symbol]
        symbol_trades = trade_frames.get(
            symbol,
            pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
        )
        tick_size = infer_tick_size_from_lob(symbol_lob)
        session_ids = list(symbol_lob["session_id"].drop_duplicates())
        folds = build_walk_forward_folds(session_ids, min_train_sessions=args.min_train_sessions)
        print(f"\n{symbol}: sessions={sorted(session_ids)}")
        print(f"{symbol}: folds={len(folds)} (min_train_sessions={args.min_train_sessions})")

        for fold_idx, (train_sessions, val_session, test_session) in enumerate(folds, start=1):
            # In this v1 walk-forward, we do not re-train each fold. We evaluate the fixed pretrained model.
            val_lob = symbol_lob[symbol_lob["session_id"] == val_session]
            val_trd = symbol_trades[symbol_trades["session_id"] == val_session]
            test_lob = symbol_lob[symbol_lob["session_id"] == test_session]
            test_trd = symbol_trades[symbol_trades["session_id"] == test_session]
            if len(val_lob) < WINDOW_SIZE + HORIZON:
                print(f"  Fold {fold_idx}: skip (val too short) val={val_session} rows={len(val_lob)}")
                continue
            if len(test_lob) < WINDOW_SIZE + HORIZON:
                print(f"  Fold {fold_idx}: skip (test too short) test={test_session} rows={len(test_lob)}")
                continue

            val_ds = build_session_dataset(val_lob, val_trd, symbol, tick_size)
            test_ds = build_session_dataset(test_lob, test_trd, symbol, tick_size)

            # Tune thresholds on the validation session only (research discipline).
            val_probs, val_y_true, val_ask, val_bid = collect_predictions_for_session(model, val_ds, device)
            _, best_on_val = sweep_strategies_for_session(
                val_probs,
                val_ask,
                val_bid,
                prob_thresholds=prob_thresholds,
                edge_thresholds=edge_thresholds,
            )

            # Evaluate classification + run strategies on test session.
            probs, y_true, ask_px, bid_px = collect_predictions_for_session(model, test_ds, device)
            y_pred = probs.argmax(axis=1)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            true_dist = np.bincount(y_true, minlength=3)
            pred_dist = np.bincount(y_pred, minlength=3)
            acc = float(accuracy_score(y_true, y_pred))

            avg_spread, med_spread = session_spread_stats(ask_px, bid_px, tick_size)

            baseline = run_passive_mm_simulation(probs, ask_px, bid_px, strategy="baseline")
            best_test = apply_best_to_test(probs, ask_px, bid_px, best=best_on_val)

            print(
                f"  Fold {fold_idx}: test={test_session} N={len(y_true):,} "
                f"acc={acc:.4f} baseline_pnl={baseline['pnl']:,.0f} "
                f"best_pnl={best_test['pnl']:,.0f} ({best_on_val['strategy']} tuned on val={val_session})"
            )

            all_rows.append(
                FoldResult(
                    symbol=symbol,
                    fold=fold_idx,
                    train_sessions=",".join(train_sessions),
                    val_session=val_session,
                    test_session=test_session,
                    n_test_samples=int(len(y_true)),
                    accuracy=acc,
                    cm00=int(cm[0, 0]),
                    cm01=int(cm[0, 1]),
                    cm02=int(cm[0, 2]),
                    cm10=int(cm[1, 0]),
                    cm11=int(cm[1, 1]),
                    cm12=int(cm[1, 2]),
                    cm20=int(cm[2, 0]),
                    cm21=int(cm[2, 1]),
                    cm22=int(cm[2, 2]),
                    true_down=int(true_dist[0]),
                    true_stat=int(true_dist[1]),
                    true_up=int(true_dist[2]),
                    pred_down=int(pred_dist[0]),
                    pred_stat=int(pred_dist[1]),
                    pred_up=int(pred_dist[2]),
                    avg_spread_ticks=avg_spread,
                    median_spread_ticks=med_spread,
                    baseline_pnl=float(baseline["pnl"]),
                    baseline_trades=int(baseline["trades"]),
                    best_pnl=float(best_test["pnl"]),
                    best_improvement=float(best_test["pnl"] - baseline["pnl"]),
                    best_strategy=str(best_on_val["strategy"]),
                    best_prob_threshold=float(best_on_val["prob_threshold"]),
                    best_edge_threshold=float(best_on_val["edge_threshold"]),
                    best_trades=int(best_test["trades"]),
                )
            )

    if not all_rows:
        print("No folds executed. Check data availability / session count.")
        return

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    csv_path = out_dir / "10_walk_forward_mm_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved walk-forward results CSV: {csv_path}")

    # Also dump a quick markdown summary for reporting
    md_path = out_dir / "10_walk_forward_mm_results.md"
    lines = ["# Walk-Forward Market Making Results", ""]
    lines.append(f"- Model weights: {weights_path}")
    lines.append(f"- Features: TOP_8_FEATURES={TOP_8_FEATURES}")
    lines.append(f"- Z-score: per-session rolling window=2000, min_periods=100, clip [-10,10]")
    lines.append("")
    for symbol in df["symbol"].unique().tolist():
        sdf = df[df["symbol"] == symbol].copy().sort_values("fold")
        lines.append(f"## {symbol}")
        lines.append("")
        for _, row in sdf.iterrows():
            lines.append(
                f"- Fold {int(row['fold'])}: test={row['test_session']} N={int(row['n_test_samples']):,} "
                f"acc={row['accuracy']:.4f} baseline_pnl={row['baseline_pnl']:,.0f} "
                f"best={row['best_strategy']} pnl={row['best_pnl']:,.0f} (+{row['best_improvement']:,.0f}) "
                f"(pthr={row['best_prob_threshold']:.2f}, ethr={row['best_edge_threshold']:.2f})"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved walk-forward results Markdown: {md_path}")


if __name__ == "__main__":
    main()
