"""Summaries for model hardware profile, PTQ behavior, and gating outcomes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.tw_train_helpers import (
    build_tw_feature_frame,
    feature_columns_from_name,
    parse_time_to_ns,
)
from src.backtest.baseline import run_signal_backtest
from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating
from src.data.tw_replay import infer_tick_size_from_lob, load_tw_replay_as_lob, load_tw_replay_trades
from src.models.hardware import profile_model_macs, quantize_model_fx_ptq
from src.models.tcn import build_default_model
from src.training.datasets import build_symbol_datasets
from src.training.metrics import classification_metrics
from src.training.trainer import predict_model


DEFAULT_SYMBOLS = ["TXFJ5", "MXFJ5", "CDFJ5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the current TCN and evaluate PTQ INT8 decay.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--trade-symbols", default="")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--quote-stride", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--feature-set", default="")
    parser.add_argument("--threshold-ticks", type=float, default=1.0)
    parser.add_argument("--threshold-mode", default="")
    parser.add_argument("--threshold-spread-multiplier", type=float, default=0.0)
    parser.add_argument("--threshold-volatility-multiplier", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--spread-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--gating-prob-threshold", type=float, default=-1.0)
    parser.add_argument("--gating-margin-threshold", type=float, default=-1.0)
    parser.add_argument("--gating-spread-penalty", type=float, default=0.0)
    parser.add_argument("--gating-volatility-penalty", type=float, default=0.0)
    parser.add_argument("--event-spread-scale", type=float, default=1.0)
    parser.add_argument("--event-max-holding-steps", type=int, default=10)
    parser.add_argument("--event-entry-mode", default="")
    parser.add_argument("--event-passive-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--event-max-passive-wait-steps", type=int, default=3)
    parser.add_argument("--event-cooldown-steps", type=int, default=-1)
    parser.add_argument("--event-size-mode", default="")
    parser.add_argument("--event-max-position-size", type=int, default=1)
    parser.add_argument("--event-confidence-floor", type=float, default=0.334)
    parser.add_argument("--event-confidence-ceiling", type=float, default=0.360)
    parser.add_argument("--event-start-time", default="")
    parser.add_argument("--event-end-time", default="")
    parser.add_argument("--ptq-calibration-batches", type=int, default=8)
    parser.add_argument("--output-subdir", default="hardware_quant")
    return parser.parse_args()


def _load_run_summary(run_dir: Path) -> dict:
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_symbols(args: argparse.Namespace, run_summary: dict) -> tuple[list[str], list[str], str, str, str, int, str]:
    symbols = (
        [item.strip() for item in args.symbols.split(",") if item.strip()]
        if args.symbols.strip()
        else run_summary.get("symbols", DEFAULT_SYMBOLS)
    )
    trade_symbols = (
        [item.strip() for item in args.trade_symbols.split(",") if item.strip()]
        if args.trade_symbols.strip()
        else run_summary.get("trade_symbols", symbols)
    )
    feature_set = args.feature_set or run_summary.get("feature_set", "base")
    threshold_mode = args.threshold_mode or run_summary.get("threshold_mode", "static")
    event_entry_mode = args.event_entry_mode or run_summary.get("event_entry_mode", "market")
    event_size_mode = args.event_size_mode or run_summary.get("event_size_mode", "fixed")
    event_cooldown_steps = (
        args.event_cooldown_steps
        if args.event_cooldown_steps >= 0
        else int(run_summary.get("event_cooldown_steps", 0))
    )
    return symbols, trade_symbols, feature_set, threshold_mode, event_entry_mode, event_size_mode, event_cooldown_steps


def _build_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> pd.DataFrame:
    return predict_model(model, dataloader, device="cpu")


def _apply_run_gating(
    predictions: pd.DataFrame,
    probability_gating: bool,
    gating_prob_threshold: float,
    gating_margin_threshold: float,
    gating_spread_penalty: float,
    gating_volatility_penalty: float,
) -> pd.DataFrame:
    gated = predictions.copy()
    if not probability_gating:
        gated["gated_signal"] = gated["y_pred"]
        gated["gated_directional_prob"] = gated[["prob_up", "prob_down"]].max(axis=1)
        gated["gated_required_prob"] = 0.0
        gated["gated_margin"] = gated[["prob_up", "prob_down"]].max(axis=1) - gated["prob_stationary"]
        gated["trade_taken_by_gate"] = gated["gated_signal"] != 1
        return gated

    return apply_probability_gating(
        gated,
        probability_threshold=gating_prob_threshold,
        margin_threshold=gating_margin_threshold,
        spread_penalty=gating_spread_penalty,
        volatility_penalty=gating_volatility_penalty,
    )


def _summarize_prediction_frame(predictions: pd.DataFrame) -> dict:
    metrics = classification_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_pred"].to_numpy(dtype=int),
    )
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _load_run_summary(run_dir)
    symbols, trade_symbols, feature_set, threshold_mode, event_entry_mode, event_size_mode, event_cooldown_steps = _resolve_symbols(
        args,
        run_summary,
    )

    gating_prob_threshold = (
        args.gating_prob_threshold
        if args.gating_prob_threshold >= 0
        else float(run_summary.get("gating_prob_threshold", 0.5))
    )
    gating_margin_threshold = (
        args.gating_margin_threshold
        if args.gating_margin_threshold >= 0
        else float(run_summary.get("gating_margin_threshold", 0.0))
    )
    probability_gating = bool(run_summary.get("probability_gating", True))

    lob_frames = load_tw_replay_as_lob(
        csv_path=args.replay_path,
        symbols=symbols,
        depth=args.depth,
        quote_stride=args.quote_stride,
        dedupe_quotes=True,
    )
    trade_tapes = load_tw_replay_trades(args.replay_path, symbols)

    tick_sizes: dict[str, float] = {}
    feature_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in lob_frames.items():
        tick_size = infer_tick_size_from_lob(frame, depth=args.depth)
        tick_sizes[symbol] = tick_size
        feature_frames[symbol] = build_tw_feature_frame(
            frame,
            trade_tapes.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"])),
            symbol,
            tick_size,
        )

    feature_cols = feature_columns_from_name(feature_set)
    splits = build_symbol_datasets(
        feature_frames=feature_frames,
        tick_sizes=tick_sizes,
        window_size=args.window_size,
        horizon=args.horizon,
        threshold_ticks=args.threshold_ticks,
        sample_stride=args.sample_stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        feature_cols=feature_cols,
        threshold_mode=threshold_mode,
        threshold_spread_multiplier=args.threshold_spread_multiplier,
        threshold_volatility_multiplier=args.threshold_volatility_multiplier,
    )

    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)

    float_model = build_default_model(num_inputs=len(feature_cols))
    float_model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    float_model.eval()

    first_batch = next(iter(val_loader))
    example_input = first_batch["x"][: min(len(first_batch["x"]), 8)].cpu()
    profile = profile_model_macs(float_model, example_input)
    pd.DataFrame(profile["module_rows"]).to_csv(output_dir / "tcn_module_profile.csv", index=False)
    hardware_profile = {
        "feature_set": feature_set,
        "window_size": args.window_size,
        "num_input_features": len(feature_cols),
        "total_params": int(profile["total_params"]),
        "trainable_params": int(profile["trainable_params"]),
        "total_macs_per_batch": int(profile["total_macs"]),
        "macs_per_sample": float(profile["total_macs"] / max(int(example_input.shape[0]), 1)),
        "weight_bytes_fp32": int(profile["weight_bytes_fp32"]),
        "weight_bytes_int8": int(profile["weight_bytes_int8"]),
        "compression_ratio_fp32_to_int8": float(profile["weight_bytes_fp32"] / max(profile["weight_bytes_int8"], 1)),
    }
    with (output_dir / "tcn_hardware_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(hardware_profile, handle, indent=2)

    calibration_batches = []
    for batch_index, batch in enumerate(val_loader):
        calibration_batches.append(batch["x"].cpu())
        if batch_index + 1 >= args.ptq_calibration_batches:
            break

    int8_model = quantize_model_fx_ptq(
        copy.deepcopy(float_model),
        example_input=example_input,
        calibration_batches=calibration_batches,
    )

    float_val_predictions = _apply_run_gating(
        _build_predictions(float_model, val_loader),
        probability_gating=probability_gating,
        gating_prob_threshold=gating_prob_threshold,
        gating_margin_threshold=gating_margin_threshold,
        gating_spread_penalty=args.gating_spread_penalty,
        gating_volatility_penalty=args.gating_volatility_penalty,
    )
    float_test_predictions = _apply_run_gating(
        _build_predictions(float_model, test_loader),
        probability_gating=probability_gating,
        gating_prob_threshold=gating_prob_threshold,
        gating_margin_threshold=gating_margin_threshold,
        gating_spread_penalty=args.gating_spread_penalty,
        gating_volatility_penalty=args.gating_volatility_penalty,
    )
    int8_val_predictions = _apply_run_gating(
        _build_predictions(int8_model, val_loader),
        probability_gating=probability_gating,
        gating_prob_threshold=gating_prob_threshold,
        gating_margin_threshold=gating_margin_threshold,
        gating_spread_penalty=args.gating_spread_penalty,
        gating_volatility_penalty=args.gating_volatility_penalty,
    )
    int8_test_predictions = _apply_run_gating(
        _build_predictions(int8_model, test_loader),
        probability_gating=probability_gating,
        gating_prob_threshold=gating_prob_threshold,
        gating_margin_threshold=gating_margin_threshold,
        gating_spread_penalty=args.gating_spread_penalty,
        gating_volatility_penalty=args.gating_volatility_penalty,
    )

    float_val_predictions.to_csv(output_dir / "float_val_predictions.csv", index=False)
    float_test_predictions.to_csv(output_dir / "float_test_predictions.csv", index=False)
    int8_val_predictions.to_csv(output_dir / "int8_val_predictions.csv", index=False)
    int8_test_predictions.to_csv(output_dir / "int8_test_predictions.csv", index=False)

    float_test_metrics = _summarize_prediction_frame(float_test_predictions)
    int8_test_metrics = _summarize_prediction_frame(int8_test_predictions)

    float_trade_predictions = float_test_predictions.loc[float_test_predictions["symbol"].isin(trade_symbols)].copy()
    int8_trade_predictions = int8_test_predictions.loc[int8_test_predictions["symbol"].isin(trade_symbols)].copy()

    _, float_signal_summary, _ = run_signal_backtest(
        float_trade_predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="gated_signal" if probability_gating else "y_pred",
    )
    _, int8_signal_summary, _ = run_signal_backtest(
        int8_trade_predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="gated_signal" if probability_gating else "y_pred",
    )

    float_event_log, float_event_summary, _ = run_event_driven_backtest(
        float_trade_predictions,
        spread_scale=args.event_spread_scale,
        max_holding_steps=args.event_max_holding_steps,
        signal_column="gated_signal" if probability_gating else "y_pred",
        entry_mode=event_entry_mode,
        passive_confidence_threshold=args.event_passive_confidence_threshold,
        max_passive_wait_steps=args.event_max_passive_wait_steps,
        cooldown_steps=event_cooldown_steps,
        size_mode=event_size_mode,
        max_position_size=args.event_max_position_size,
        confidence_floor=args.event_confidence_floor,
        confidence_ceiling=args.event_confidence_ceiling,
        min_timestamp=parse_time_to_ns(args.event_start_time),
        max_timestamp=parse_time_to_ns(args.event_end_time),
        trade_tapes=load_tw_replay_trades(args.replay_path, trade_symbols),
    )
    int8_event_log, int8_event_summary, _ = run_event_driven_backtest(
        int8_trade_predictions,
        spread_scale=args.event_spread_scale,
        max_holding_steps=args.event_max_holding_steps,
        signal_column="gated_signal" if probability_gating else "y_pred",
        entry_mode=event_entry_mode,
        passive_confidence_threshold=args.event_passive_confidence_threshold,
        max_passive_wait_steps=args.event_max_passive_wait_steps,
        cooldown_steps=event_cooldown_steps,
        size_mode=event_size_mode,
        max_position_size=args.event_max_position_size,
        confidence_floor=args.event_confidence_floor,
        confidence_ceiling=args.event_confidence_ceiling,
        min_timestamp=parse_time_to_ns(args.event_start_time),
        max_timestamp=parse_time_to_ns(args.event_end_time),
        trade_tapes=load_tw_replay_trades(args.replay_path, trade_symbols),
    )
    float_event_log.to_csv(output_dir / "float_event_trade_log.csv", index=False)
    int8_event_log.to_csv(output_dir / "int8_event_trade_log.csv", index=False)

    prediction_alignment = float_test_predictions.merge(
        int8_test_predictions[["symbol", "timestamp", "y_pred", "prob_down", "prob_stationary", "prob_up"]],
        on=["symbol", "timestamp"],
        suffixes=("_float", "_int8"),
    )
    prediction_agreement = float((prediction_alignment["y_pred_float"] == prediction_alignment["y_pred_int8"]).mean())
    mean_prob_shift = float(
        (
            (prediction_alignment["prob_down_float"] - prediction_alignment["prob_down_int8"]).abs()
            + (prediction_alignment["prob_stationary_float"] - prediction_alignment["prob_stationary_int8"]).abs()
            + (prediction_alignment["prob_up_float"] - prediction_alignment["prob_up_int8"]).abs()
        ).mean()
        / 3.0
    )

    ptq_summary = {
        "run_dir": str(run_dir),
        "feature_set": feature_set,
        "float_test_accuracy": float_test_metrics["accuracy"],
        "int8_test_accuracy": int8_test_metrics["accuracy"],
        "float_test_macro_f1": float_test_metrics["macro_f1"],
        "int8_test_macro_f1": int8_test_metrics["macro_f1"],
        "accuracy_delta": float(int8_test_metrics["accuracy"] - float_test_metrics["accuracy"]),
        "macro_f1_delta": float(int8_test_metrics["macro_f1"] - float_test_metrics["macro_f1"]),
        "prediction_agreement": prediction_agreement,
        "mean_probability_shift": mean_prob_shift,
        "float_signal_net_pnl_ticks": float(float_signal_summary["net_pnl_ticks"]),
        "int8_signal_net_pnl_ticks": float(int8_signal_summary["net_pnl_ticks"]),
        "signal_pnl_delta_ticks": float(int8_signal_summary["net_pnl_ticks"] - float_signal_summary["net_pnl_ticks"]),
        "float_event_net_pnl_ticks": float(float_event_summary["net_pnl_ticks"]),
        "int8_event_net_pnl_ticks": float(int8_event_summary["net_pnl_ticks"]),
        "event_pnl_delta_ticks": float(int8_event_summary["net_pnl_ticks"] - float_event_summary["net_pnl_ticks"]),
    }
    with (output_dir / "ptq_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(ptq_summary, handle, indent=2)

    print(json.dumps({"hardware_profile": hardware_profile, "ptq_summary": ptq_summary}, indent=2))


if __name__ == "__main__":
    main()
