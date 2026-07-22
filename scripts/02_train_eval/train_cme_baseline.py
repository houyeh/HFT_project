"""Train and evaluate the TCN directly on CME Logic20 sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.tw_train_helpers import (
    build_tw_feature_frame,
    compute_class_weights,
    feature_columns_from_name,
    parse_time_to_ns,
)
from src.backtest.baseline import run_signal_backtest
from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating
from src.data.cme_logic20 import discover_logic20_files, load_cme_logic20_bundle
from src.models.tcn import build_default_model
from src.training.datasets import build_symbol_datasets
from src.training.metrics import classification_metrics
from src.training.trainer import TrainerConfig, predict_model, select_device, train_with_early_stopping


DEFAULT_SYMBOLS = ["ESM6", "CLK6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or fine-tune a CME Logic20 baseline.")
    parser.add_argument("--cme-root", default="data/raw/CME")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--threshold-ticks", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=500)
    parser.add_argument("--feature-set", default="execution_aware")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--spread-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--gating-prob-threshold", type=float, default=0.45)
    parser.add_argument("--gating-margin-threshold", type=float, default=0.0)
    parser.add_argument("--gating-spread-penalty", type=float, default=0.0)
    parser.add_argument("--gating-volatility-penalty", type=float, default=0.0)
    parser.add_argument("--disable-probability-gating", action="store_true")
    parser.add_argument("--disable-class-weighting", action="store_true")
    parser.add_argument("--event-spread-scale", type=float, default=1.0)
    parser.add_argument("--event-max-holding-steps", type=int, default=10)
    parser.add_argument("--event-entry-mode", default="market")
    parser.add_argument("--event-passive-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--event-max-passive-wait-steps", type=int, default=3)
    parser.add_argument("--event-cooldown-steps", type=int, default=0)
    parser.add_argument("--event-size-mode", default="fixed")
    parser.add_argument("--event-max-position-size", type=int, default=1)
    parser.add_argument("--event-confidence-floor", type=float, default=0.334)
    parser.add_argument("--event-confidence-ceiling", type=float, default=0.360)
    parser.add_argument("--event-start-time", default="")
    parser.add_argument("--event-end-time", default="")
    parser.add_argument("--init-model-path", default="")
    parser.add_argument("--output-dir", default="artifacts/02_model_runs/cme_baseline")
    parser.add_argument("--per-symbol", action="store_true",
                        help="Train a separate model for each symbol instead of mixing them.")
    return parser.parse_args()


def discover_available_products(cme_root: str | Path) -> list[str]:
    products: set[str] = set()
    for csv_path in discover_logic20_files(cme_root):
        with Path(csv_path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
            header = handle.readline().strip().split(",")
        for column in header:
            if column.endswith("BQ0"):
                products.add(column[:-3])
    return sorted(products)


def main() -> None:
    args = parse_args()
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / timestamp_tag
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "pipeline_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    requested_symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    available_symbols = discover_available_products(args.cme_root)
    missing_symbols = [symbol for symbol in requested_symbols if symbol not in available_symbols]
    selected_symbols = [symbol for symbol in requested_symbols if symbol in available_symbols]
    if not selected_symbols:
        raise ValueError(f"No requested CME symbols found. Available symbols: {available_symbols}")

    bundle = load_cme_logic20_bundle(
        cme_root=args.cme_root,
        symbols=selected_symbols,
        depth=args.depth,
        dedupe_quotes=True,
    )
    bundle.summary.to_csv(output_root / "cme_file_summary.csv", index=False)

    feature_frames: dict[str, pd.DataFrame] = {}
    data_summary_rows = []
    for symbol in selected_symbols:
        lob_df = bundle.lob_frames[symbol]
        trade_df = bundle.trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
        tick_size = bundle.tick_sizes[symbol]
        feature_frames[symbol] = build_tw_feature_frame(
            lob_df=lob_df,
            trade_df=trade_df,
            symbol=symbol,
            tick_size=tick_size,
        )
        data_summary_rows.append(
            {
                "symbol": symbol,
                "rows": len(lob_df),
                "trades": int(len(trade_df)),
                "tick_size": tick_size,
                "timestamp_min": int(lob_df["timestamp"].min()),
                "timestamp_max": int(lob_df["timestamp"].max()),
            }
        )
    pd.DataFrame(data_summary_rows).sort_values("rows", ascending=False).to_csv(output_root / "data_summary.csv", index=False)

    # --- per-symbol mode: train each symbol independently ---
    if args.per_symbol:
        all_symbol_summaries = []
        for symbol in selected_symbols:
            print(f"\n{'='*60}")
            print(f"  PER-SYMBOL TRAINING: {symbol}")
            print(f"{'='*60}")
            sym_summary = _train_single_symbol_group(
                args=args,
                feature_frames={symbol: feature_frames[symbol]},
                tick_sizes={symbol: bundle.tick_sizes[symbol]},
                trade_tapes={symbol: bundle.trade_frames[symbol]} if symbol in bundle.trade_frames else {},
                output_dir=output_root / symbol,
                run_label=symbol,
            )
            all_symbol_summaries.append(sym_summary)

        with (output_root / "per_symbol_summaries.json").open("w", encoding="utf-8") as handle:
            json.dump(all_symbol_summaries, handle, indent=2)
        print("\n" + json.dumps(all_symbol_summaries, indent=2))
    else:
        # --- original mixed-symbol mode ---
        summary = _train_single_symbol_group(
            args=args,
            feature_frames=feature_frames,
            tick_sizes=bundle.tick_sizes,
            trade_tapes=bundle.trade_frames,
            output_dir=output_root,
            run_label="mixed",
        )
        print(json.dumps(summary, indent=2))


def _train_single_symbol_group(
    args: argparse.Namespace,
    feature_frames: dict[str, pd.DataFrame],
    tick_sizes: dict[str, float],
    trade_tapes: dict,
    output_dir: Path,
    run_label: str,
) -> dict:
    """Train and evaluate a model on one group of symbols (single or mixed)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_feature_cols = feature_columns_from_name(args.feature_set)
    splits = build_symbol_datasets(
        feature_frames=feature_frames,
        tick_sizes=tick_sizes,
        window_size=args.window_size,
        horizon=args.horizon,
        threshold_ticks=args.threshold_ticks,
        sample_stride=args.sample_stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        feature_cols=model_feature_cols,
    )
    pd.DataFrame.from_dict(splits.split_sizes, orient="index").reset_index().rename(columns={"index": "symbol"}).to_csv(
        output_dir / "split_sizes.csv",
        index=False,
    )

    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)

    model = build_default_model(num_inputs=len(model_feature_cols))
    if args.init_model_path:
        state_dict = torch.load(args.init_model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=select_device(),
    )

    class_weights = None
    if not args.disable_class_weighting:
        class_weights = compute_class_weights(splits.train)
        pd.DataFrame({"class": ["Down", "Stationary", "Up"], "weight": class_weights.tolist()}).to_csv(
            output_dir / "class_weights.csv",
            index=False,
        )

    model, history_df = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    val_predictions = predict_model(model, val_loader, config.device)
    test_predictions = predict_model(model, test_loader, config.device)
    if args.disable_probability_gating:
        for frame in (val_predictions, test_predictions):
            frame["gated_signal"] = frame["y_pred"]
            frame["gated_directional_prob"] = frame[["prob_up", "prob_down"]].max(axis=1)
            frame["gated_required_prob"] = 0.0
            frame["gated_margin"] = frame[["prob_up", "prob_down"]].max(axis=1) - frame["prob_stationary"]
            frame["trade_taken_by_gate"] = frame["gated_signal"] != 1
    else:
        val_predictions = apply_probability_gating(
            val_predictions,
            probability_threshold=args.gating_prob_threshold,
            margin_threshold=args.gating_margin_threshold,
            spread_penalty=args.gating_spread_penalty,
            volatility_penalty=args.gating_volatility_penalty,
        )
        test_predictions = apply_probability_gating(
            test_predictions,
            probability_threshold=args.gating_prob_threshold,
            margin_threshold=args.gating_margin_threshold,
            spread_penalty=args.gating_spread_penalty,
            volatility_penalty=args.gating_volatility_penalty,
        )

    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics = classification_metrics(
        test_predictions["y_true"].to_numpy(),
        test_predictions["y_pred"].to_numpy(),
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["true_down", "true_stationary", "true_up"],
        columns=["pred_down", "pred_stationary", "pred_up"],
    )
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")

    signal_trade_log, signal_summary, signal_by_symbol = run_signal_backtest(
        test_predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="gated_signal",
    )
    signal_trade_log.to_csv(output_dir / "signal_backtest_trade_log.csv", index=False)
    signal_by_symbol.to_csv(output_dir / "signal_backtest_by_symbol.csv", index=False)
    with (output_dir / "signal_backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(signal_summary, handle, indent=2)

    event_trade_log, event_summary, event_by_symbol = run_event_driven_backtest(
        test_predictions,
        spread_scale=args.event_spread_scale,
        max_holding_steps=args.event_max_holding_steps,
        close_on_stationary=True,
        signal_column="gated_signal",
        trade_tapes=trade_tapes,
        entry_mode=args.event_entry_mode,
        passive_confidence_threshold=args.event_passive_confidence_threshold,
        max_passive_wait_steps=args.event_max_passive_wait_steps,
        cooldown_steps=args.event_cooldown_steps,
        size_mode=args.event_size_mode,
        max_position_size=args.event_max_position_size,
        confidence_floor=args.event_confidence_floor,
        confidence_ceiling=args.event_confidence_ceiling,
        min_timestamp=parse_time_to_ns(args.event_start_time),
        max_timestamp=parse_time_to_ns(args.event_end_time),
    )
    event_trade_log.to_csv(output_dir / "event_backtest_trade_log.csv", index=False)
    event_by_symbol.to_csv(output_dir / "event_backtest_by_symbol.csv", index=False)
    with (output_dir / "event_backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(event_summary, handle, indent=2)

    symbols_in_group = list(feature_frames.keys())
    run_summary = {
        "run_label": run_label,
        "symbols": symbols_in_group,
        "output_dir": str(output_dir),
        "init_model_path": str(Path(args.init_model_path).resolve()) if args.init_model_path else "",
        "feature_set": args.feature_set,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "threshold_ticks": args.threshold_ticks,
        "sample_stride": args.sample_stride,
        "epochs_ran": int(len(history_df)),
        "device": config.device,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "signal_backtest_net_pnl_ticks": signal_summary["net_pnl_ticks"],
        "signal_backtest_sharpe_like": signal_summary["sharpe_like"],
        "event_backtest_net_pnl_ticks": event_summary["net_pnl_ticks"],
        "event_backtest_sharpe_like": event_summary["sharpe_like"],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    return run_summary


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
