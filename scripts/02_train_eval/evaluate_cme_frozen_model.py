"""Run the frozen Taiwan-trained model directly on CME Logic20 data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.tw_train_helpers import build_tw_feature_frame, feature_columns_from_name
from src.backtest.baseline import run_signal_backtest
from src.backtest.signals import apply_probability_gating
from src.data.cme_logic20 import discover_logic20_files, load_cme_logic20_bundle
from src.models.tcn import build_default_model
from src.training.datasets import WindowedLOBSequenceDataset
from src.training.metrics import classification_metrics
from src.training.trainer import predict_model, select_device


DEFAULT_SYMBOLS = ["ESM6", "CLK6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Taiwan-trained TCN on CME Logic20 data.")
    parser.add_argument("--cme-root", default="data/raw/CME")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--model-path", default="artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--threshold-ticks", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--feature-set", default="execution_aware")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--spread-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--probability-threshold", type=float, default=0.334)
    parser.add_argument("--margin-threshold", type=float, default=0.005)
    parser.add_argument("--output-dir", default="artifacts/02_model_runs/cme_eval")
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


def build_full_dataset(bundle, symbols: list[str], args: argparse.Namespace):
    feature_columns = feature_columns_from_name(args.feature_set)
    datasets = []
    feature_frames: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        lob_df = bundle.lob_frames[symbol]
        trade_df = bundle.trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
        feature_frame = build_tw_feature_frame(
            lob_df=lob_df,
            trade_df=trade_df,
            symbol=symbol,
            tick_size=bundle.tick_sizes[symbol],
        )
        feature_frames[symbol] = feature_frame
        datasets.append(
            WindowedLOBSequenceDataset(
                feature_df=feature_frame,
                symbol=symbol,
                tick_size=bundle.tick_sizes[symbol],
                window_size=args.window_size,
                horizon=args.horizon,
                threshold_ticks=args.threshold_ticks,
                sample_stride=args.sample_stride,
                feature_cols=feature_columns,
            )
        )
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return dataset, feature_frames


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    available_products = discover_available_products(args.cme_root)
    missing_symbols = [symbol for symbol in requested_symbols if symbol not in available_products]
    evaluated_symbols = [symbol for symbol in requested_symbols if symbol in available_products]
    if not evaluated_symbols:
        raise ValueError(f"No requested symbols were found under {args.cme_root}. Available products: {available_products}")

    bundle = load_cme_logic20_bundle(
        cme_root=args.cme_root,
        symbols=evaluated_symbols,
        depth=args.depth,
        dedupe_quotes=True,
    )
    bundle.summary.to_csv(output_dir / "cme_file_summary.csv", index=False)

    dataset, feature_frames = build_full_dataset(bundle=bundle, symbols=evaluated_symbols, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_default_model(num_inputs=len(feature_columns_from_name(args.feature_set)))
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    device = select_device()
    model.to(device)

    predictions = predict_model(model, dataloader, device=device)
    predictions.to_csv(output_dir / "predictions_raw.csv", index=False)

    raw_metrics = classification_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )
    raw_metrics_df = pd.DataFrame([raw_metrics])
    raw_metrics_df.to_csv(output_dir / "raw_metrics.csv", index=False)

    raw_trades, raw_signal_summary, raw_signal_by_symbol = run_signal_backtest(
        predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="y_pred",
    )
    raw_trades.to_csv(output_dir / "signal_backtest_raw.csv", index=False)
    raw_signal_by_symbol.to_csv(output_dir / "signal_backtest_raw_by_symbol.csv", index=False)

    gated_predictions = apply_probability_gating(
        predictions,
        probability_threshold=args.probability_threshold,
        margin_threshold=args.margin_threshold,
    )
    gated_predictions.to_csv(output_dir / "predictions_gated.csv", index=False)
    gated_metrics = classification_metrics(
        gated_predictions["y_true"].to_numpy(),
        gated_predictions["gated_signal"].to_numpy(),
    )
    pd.DataFrame([gated_metrics]).to_csv(output_dir / "gated_metrics.csv", index=False)
    gated_trades, gated_signal_summary, gated_signal_by_symbol = run_signal_backtest(
        gated_predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="gated_signal",
    )
    gated_trades.to_csv(output_dir / "signal_backtest_gated.csv", index=False)
    gated_signal_by_symbol.to_csv(output_dir / "signal_backtest_gated_by_symbol.csv", index=False)

    label_distribution = (
        predictions.groupby(["symbol", "y_true"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    label_distribution.to_csv(output_dir / "label_distribution.csv", index=False)

    prediction_distribution = (
        predictions.groupby(["symbol", "y_pred"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    prediction_distribution.to_csv(output_dir / "prediction_distribution.csv", index=False)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "cme_root": str(Path(args.cme_root).resolve()),
        "requested_symbols": requested_symbols,
        "evaluated_symbols": evaluated_symbols,
        "missing_symbols": missing_symbols,
        "feature_set": args.feature_set,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "threshold_ticks": args.threshold_ticks,
        "sample_stride": args.sample_stride,
        "num_predictions": int(len(predictions)),
        "tick_sizes": bundle.tick_sizes,
        "raw_metrics": raw_metrics,
        "raw_signal_summary": raw_signal_summary,
        "gated_metrics": gated_metrics,
        "gated_signal_summary": gated_signal_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
