"""台灣期交所歷史重播資料的主訓練腳本，用於建立 execution-aware (考慮執行情況) 的 TCN 基線模型。"""

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

# 匯入專案內部的自訂模組 (回測、特徵算子、模型架構與訓練工具)
from src.backtest.baseline import run_signal_backtest
from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating
from src.backtest.symbol_calibration import fit_symbol_rule_configuration, save_symbol_calibration_artifacts
from src.data.tw_replay import (
    infer_tick_size_from_lob,
    load_tw_replay_as_lob,
    load_tw_replay_trades,
    scan_replay_symbols,
)
from src.features.dimensionless_features import build_feature_dataframe
from src.features.execution_features import align_trade_features_to_quotes, compute_queue_imbalance
from src.models.tcn import build_default_model
from src.training.datasets import build_symbol_datasets
from src.training.metrics import classification_metrics
from src.training.trainer import TrainerConfig, predict_model, select_device, train_with_early_stopping

# 定義預設交易商品與多組特徵欄位組合 (基礎集、策略感知、執行感知等)
DEFAULT_SYMBOLS = ["TXFJ5", "MXFJ5", "CDFJ5"]
# 基礎微結構特徵
BASE_MODEL_FEATURE_COLS = [
    "normalized_spread",
    "micro_return",
    "obi",
    "weighted_mid_deviation",
    "deep_obi",
]
# 在base上加策略判斷常用特徵，如波動率、OBI變化、近深度狀態、以及深度斜率不平衡
STRATEGY_AWARE_FEATURE_COLS = BASE_MODEL_FEATURE_COLS + [
    "rolling_volatility",
    "obi_change",
    "near_depth_obi",
    "book_slope_imbalance",
]
#在base上加上執行層特徵，如佇列不平衡、成交對齊特徵，以及其他與執行相關的指標
EXECUTION_AWARE_FEATURE_COLS = BASE_MODEL_FEATURE_COLS + [
    "queue_imbalance",
    "queue_imbalance_change",
    "aggressor_imbalance_interval",
    "rolling_signed_flow_ratio",
    "trade_intensity_short",
    "net_cancel_pressure",
    "cancel_pressure_short",
]
#上面兩個的合併，包含基礎微結構特徵、策略感知特徵，以及執行感知特徵的完整組合
FULL_MICROSTRUCTURE_FEATURE_COLS = STRATEGY_AWARE_FEATURE_COLS + [
    "queue_imbalance",
    "queue_imbalance_change",
    "aggressor_imbalance_interval",
    "rolling_signed_flow_ratio",
    "trade_intensity_short",
    "net_cancel_pressure",
    "cancel_pressure_short",
]

# 集中定義全部 CLI 參數，回傳 `Namespace`
# 把整個 pipeline 做成可重現、可批次執行的命令列工具。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Taiwan replay FP32 baseline.")
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--trade-symbols", default="")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--quote-stride", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--feature-set", default="base")
    parser.add_argument("--threshold-ticks", type=float, default=1.0)
    parser.add_argument("--threshold-mode", default="static")
    parser.add_argument("--threshold-spread-multiplier", type=float, default=0.0)
    parser.add_argument("--threshold-volatility-multiplier", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--spread-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--gating-prob-threshold", type=float, default=0.5)
    parser.add_argument("--gating-margin-threshold", type=float, default=0.0)
    parser.add_argument("--gating-spread-penalty", type=float, default=0.0)
    parser.add_argument("--gating-volatility-penalty", type=float, default=0.0)
    parser.add_argument("--disable-probability-gating", action="store_true")
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
    parser.add_argument("--disable-class-weighting", action="store_true")
    parser.add_argument("--non-trade-symbol-weight", type=float, default=1.0)
    parser.add_argument("--train-symbol-weights", default="")
    parser.add_argument("--train-nonstationary-weight", type=float, default=1.0)
    parser.add_argument("--train-return-weight-alpha", type=float, default=0.0)
    parser.add_argument("--calibrate-symbol-rules-on-val", action="store_true")
    parser.add_argument("--calibration-prob-offsets", default="-0.002,0.0,0.002")
    parser.add_argument("--calibration-margin-thresholds", default="0.0,0.005")
    parser.add_argument("--calibration-end-times", default="12:00:00,12:15:00,12:30:00,13:00:00")
    parser.add_argument("--calibration-cooldowns", default="0,5")
    parser.add_argument("--calibration-vol-quantiles", default="0.80,0.90")
    parser.add_argument("--calibration-trade-quantiles", default="0.80,0.90")
    parser.add_argument("--calibration-cancel-quantiles", default="0.80,0.90")
    parser.add_argument("--calibration-liq-quantiles", default="0.20,0.25")
    parser.add_argument("--output-dir", default="artifacts/02_model_runs/tw_baseline")
    return parser.parse_args()

# 建立台灣期交所歷史資料的特徵 DataFrame，整合 LOB 快照與逐筆成交紀錄
def build_tw_feature_frame(
    lob_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    tick_size: float,
) -> pd.DataFrame:
    
    # 從無量綱（dimensionless）的基礎特徵集開始，
    # 接著疊加波動率、佇列狀態、深度斜率分布，以及對齊成交紀錄的特徵。
    # 這些特徵將用於後續對接 HLS (高階合成) 的策略實驗。
    base = build_feature_dataframe(lob_df, symbol=symbol, tick_size=tick_size)
    base["weighted_mid_deviation"] = (base["weighted_mid_price"] - base["mid_price"]) / tick_size
    rolling_vol = (
        base["micro_return"]
        .rolling(window=50, min_periods=5)
        .std()
        .bfill()
        .fillna(0.0)
        .astype(float)
    )

    # 主表
    feature_df = pd.DataFrame(
        {
            "timestamp": base["timestamp"],
            "mid_price": base["mid_price"],
            "normalized_spread": base["normalized_spread"],
            "micro_return": base["micro_return"],
            "obi": base["obi"],
            "weighted_mid_deviation": base["weighted_mid_deviation"],
            "deep_obi": base["deep_obi"],
            "rolling_volatility": rolling_vol,
            "bid_px_1": lob_df["bid_px_1"].astype(float).values,
            "ask_px_1": lob_df["ask_px_1"].astype(float).values,
            "bid_vol_1": lob_df["bid_vol_1"].astype(float).values,
            "ask_vol_1": lob_df["ask_vol_1"].astype(float).values,
        }
    )
    #額外計算一些特徵
    feature_df["obi_change"] = feature_df["obi"].diff().fillna(0.0)
    near_bid = lob_df["bid_vol_1"].astype(float) + lob_df["bid_vol_2"].astype(float)
    near_ask = lob_df["ask_vol_1"].astype(float) + lob_df["ask_vol_2"].astype(float)
    near_total = (near_bid + near_ask).replace(0, 1.0)
    feature_df["near_depth_obi"] = ((near_bid - near_ask) / near_total).astype(float)

    bid_depth_span = (lob_df["bid_px_1"].astype(float) - lob_df["bid_px_5"].astype(float)) / tick_size
    ask_depth_span = (lob_df["ask_px_5"].astype(float) - lob_df["ask_px_1"].astype(float)) / tick_size
    bid_depth_vol = (
        lob_df[[f"bid_vol_{level}" for level in range(1, 6)]].astype(float).sum(axis=1).replace(0, 1.0)
    )
    ask_depth_vol = (
        lob_df[[f"ask_vol_{level}" for level in range(1, 6)]].astype(float).sum(axis=1).replace(0, 1.0)
    )
    feature_df["book_slope_imbalance"] = ((bid_depth_span / bid_depth_vol) - (ask_depth_span / ask_depth_vol)).astype(float)
    feature_df["queue_imbalance"] = compute_queue_imbalance(lob_df).astype(float)
    feature_df["queue_imbalance_change"] = feature_df["queue_imbalance"].diff().fillna(0.0)

    # 用這個函式將成交特徵對其到quote時間軸
    trade_features = align_trade_features_to_quotes(
        quotes=lob_df,
        trades=trade_df,
        tick_size=tick_size,
        rolling_window=20,
    )
    for column in trade_features.columns:
        feature_df[column] = trade_features[column].astype(float).values

    # 回傳可訓練的特徵表
    return feature_df


def compute_class_weights(dataset) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float32)
    for index in range(len(dataset)):
        label = int(dataset[index]["y"])
        counts[label] += 1

    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = counts.sum() / (counts.numel() * counts)
    return weights


def feature_columns_from_name(feature_set: str) -> list[str]:
    if feature_set == "base":
        return BASE_MODEL_FEATURE_COLS
    if feature_set == "strategy_aware":
        return STRATEGY_AWARE_FEATURE_COLS
    if feature_set == "execution_aware":
        return EXECUTION_AWARE_FEATURE_COLS
    if feature_set == "full_microstructure":
        return FULL_MICROSTRUCTURE_FEATURE_COLS
    raise ValueError(f"Unknown feature set: {feature_set}")


def parse_time_to_ns(text: str) -> int | None:
    if not text:
        return None
    hour, minute, second = [int(part) for part in text.split(":")]
    return (
        hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + second * 1_000_000_000
    )

# 解析字串格式的商品權重設定，用於 loss 計算時對特定商品加權
def parse_symbol_weight_map(raw: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if not raw.strip():
        return mapping
    for item in raw.split(","):
        if not item.strip():
            continue
        symbol, value = item.split(":")
        mapping[symbol.strip()] = float(value)
    return mapping


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_text_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]

# 主程式進入點：負責資料載入、特徵計算、模型訓練、預測輸出及回測評估
def main() -> None:
    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pipeline_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    # 決定商品以及權重
    selected_symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    trade_symbols = (
        [symbol.strip() for symbol in args.trade_symbols.split(",") if symbol.strip()]
        if args.trade_symbols.strip()
        else list(selected_symbols)
    )
    symbol_weight_map = parse_symbol_weight_map(args.train_symbol_weights)
    for symbol in selected_symbols:
        if symbol not in trade_symbols and symbol not in symbol_weight_map:
            symbol_weight_map[symbol] = args.non_trade_symbol_weight
        if symbol in trade_symbols and symbol not in symbol_weight_map:
            symbol_weight_map[symbol] = 1.0
    symbol_scan = scan_replay_symbols(args.replay_path)
    symbol_scan.to_csv(output_dir / "symbol_scan.csv", index=False)

    # 從台灣重播檔案中載入委託簿快照與逐筆成交紀錄，
    # 然後把每個商品轉換為可被滑動視窗(windowed)切割以供訓練的特徵 DataFrame。
    print("Loading Taiwan replay data...")
    lob_frames = load_tw_replay_as_lob(
        csv_path=args.replay_path,
        symbols=selected_symbols,
        depth=args.depth,
        quote_stride=args.quote_stride,
        dedupe_quotes=True,
    )
    if not lob_frames:
        raise ValueError("No replay snapshots were loaded. Check symbols or replay path.")
    trade_tapes = load_tw_replay_trades(args.replay_path, selected_symbols)

    tick_sizes = {}
    feature_frames = {}
    data_summary = []

    for symbol, frame in lob_frames.items():
        tick_size = infer_tick_size_from_lob(frame, depth=args.depth)
        tick_sizes[symbol] = tick_size
        symbol_trades = trade_tapes.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))
        feature_frames[symbol] = build_tw_feature_frame(frame, symbol_trades, symbol, tick_size)
        data_summary.append(
            {
                "symbol": symbol,
                "rows": len(frame),
                "tick_size": tick_size,
                "timestamp_min": int(frame["timestamp"].min()),
                "timestamp_max": int(frame["timestamp"].max()),
            }
        )

    data_summary_df = pd.DataFrame(data_summary).sort_values("rows", ascending=False)
    data_summary_df.to_csv(output_dir / "data_summary.csv", index=False)
    model_feature_cols = feature_columns_from_name(args.feature_set)

    # 按照時間順序切分資料，確保 validation/test 在時間軸上嚴格處於訓練集之後。
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
        threshold_mode=args.threshold_mode,
        threshold_spread_multiplier=args.threshold_spread_multiplier,
        threshold_volatility_multiplier=args.threshold_volatility_multiplier,
        symbol_weights=symbol_weight_map,
        nonstationary_weight=args.train_nonstationary_weight,
        return_weight_alpha=args.train_return_weight_alpha,
    )
    pd.DataFrame.from_dict(splits.split_sizes, orient="index").reset_index().rename(
        columns={"index": "symbol"}
    ).to_csv(output_dir / "split_sizes.csv", index=False)

    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)

    model = build_default_model(num_inputs=len(model_feature_cols))
    # 訓練核心（loss/optimizer/epoch）實作在 src/training/trainer.py：
    # - loss: CrossEntropyLoss（可搭配 class/sample 權重）
    # - optimizer: Adam
    # - epoch: 完整跑過 train_loader 一次，並搭配 early stopping
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
        pd.DataFrame(
            {
                "class": ["Down", "Stationary", "Up"],
                "weight": class_weights.tolist(),
            }
        ).to_csv(output_dir / "class_weights.csv", index=False)

    # 先訓練 FP32 參考模型；後續的腳本會將最佳模型凍結、量化，
    # 並匯出以進入 HLS (高階合成) 部署流程。
    print(f"Training on device: {config.device}")
    model, history_df = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    # 先儲存原始預測結果，接著可選擇性套用機率門檻過濾 (probability gating)，
    # 讓下游的回測可以比較「純分類器品質」與「具備可交易優勢的訊號」。
    print("Evaluating on test split...")
    val_predictions = predict_model(model, val_loader, config.device)
    predictions = predict_model(model, test_loader, config.device)
    if args.disable_probability_gating:
        predictions["gated_signal"] = predictions["y_pred"]
        predictions["gated_directional_prob"] = predictions[["prob_up", "prob_down"]].max(axis=1)
        predictions["gated_required_prob"] = 0.0
        predictions["gated_margin"] = predictions[["prob_up", "prob_down"]].max(axis=1) - predictions["prob_stationary"]
        predictions["trade_taken_by_gate"] = predictions["gated_signal"] != 1
        val_predictions["gated_signal"] = val_predictions["y_pred"]
        val_predictions["gated_directional_prob"] = val_predictions[["prob_up", "prob_down"]].max(axis=1)
        val_predictions["gated_required_prob"] = 0.0
        val_predictions["gated_margin"] = val_predictions[["prob_up", "prob_down"]].max(axis=1) - val_predictions["prob_stationary"]
        val_predictions["trade_taken_by_gate"] = val_predictions["gated_signal"] != 1
    else:
        val_predictions = apply_probability_gating(
            val_predictions,
            probability_threshold=args.gating_prob_threshold,
            margin_threshold=args.gating_margin_threshold,
            spread_penalty=args.gating_spread_penalty,
            volatility_penalty=args.gating_volatility_penalty,
            allowed_symbols=set(trade_symbols),
        )
        predictions = apply_probability_gating(
            predictions,
            probability_threshold=args.gating_prob_threshold,
            margin_threshold=args.gating_margin_threshold,
            spread_penalty=args.gating_spread_penalty,
            volatility_penalty=args.gating_volatility_penalty,
            allowed_symbols=set(trade_symbols),
        )
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics = classification_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["true_down", "true_stationary", "true_up"],
        columns=["pred_down", "pred_stationary", "pred_up"],
    )
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")

    # 本專案同時追蹤兩種回測結果：
    # 1. 向量化訊號回測 (vectorized signal backtest)
    # 2. 更貼近真實、具備市價/限價進場邏輯的事件驅動回測 (event-driven replay)
    trade_log, backtest_summary, backtest_by_symbol = run_signal_backtest(
        predictions,
        spread_cost_multiplier=args.spread_cost_multiplier,
        signal_column="gated_signal",
    )
    trade_log.to_csv(output_dir / "backtest_trade_log.csv", index=False)
    backtest_by_symbol.to_csv(output_dir / "backtest_by_symbol.csv", index=False)
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(backtest_summary, handle, indent=2)

    # 執行事件驅動回測 (Event-Driven Backtest)，模擬真實的市價/限價進場與部位持有時間限制
    event_trade_log, event_summary, event_by_symbol = run_event_driven_backtest(
        predictions,
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

    # 可選的驗證集校準 (validation-time calibration)。搜尋針對個別商品的門檻、
    # 交易時段結束時間點，以及市場狀態切換的防護機制 (regime guards)。
    calibrated_summary = None
    # 若啟用校準，則在驗證集上尋找最佳的個別商品交易門檻、時段過濾與市場狀態防護機制
    if args.calibrate_symbol_rules_on_val:
        best_by_symbol, calibration_config, val_calibrated, val_trade_log, val_summary, val_by_symbol = fit_symbol_rule_configuration(
            predictions=val_predictions,
            trade_tapes=trade_tapes,
            allowed_symbols=set(trade_symbols),
            base_prob_threshold=args.gating_prob_threshold,
            prob_threshold_offsets=parse_float_list(args.calibration_prob_offsets),
            margin_thresholds=parse_float_list(args.calibration_margin_thresholds),
            end_times=parse_text_list(args.calibration_end_times),
            cooldowns=parse_int_list(args.calibration_cooldowns),
            holding_steps=args.event_max_holding_steps,
            passive_wait=args.event_max_passive_wait_steps,
            vol_quantiles=parse_float_list(args.calibration_vol_quantiles),
            trade_quantiles=parse_float_list(args.calibration_trade_quantiles),
            cancel_quantiles=parse_float_list(args.calibration_cancel_quantiles),
            liq_quantiles=parse_float_list(args.calibration_liq_quantiles),
        )
        # 儲存驗證集上的校準結果與相關的分析報告
        save_symbol_calibration_artifacts(
            output_dir=output_dir,
            prefix="val",
            best_by_symbol=best_by_symbol,
            config=calibration_config,
            gated_predictions=val_calibrated,
            trade_log=val_trade_log,
            summary=val_summary,
            by_symbol=val_by_symbol,
        )
        # 根據驗證集找出的最佳校準參數，套用到測試集上進行預測訊號過濾 (Probability Gating)
        test_calibrated = apply_probability_gating(
            predictions,
            probability_threshold=args.gating_prob_threshold,
            margin_threshold=args.gating_margin_threshold,
            symbol_probability_thresholds=calibration_config["symbol_prob_thresholds"],
            symbol_margin_thresholds=calibration_config["symbol_margin_thresholds"],
            symbol_end_times=calibration_config["symbol_end_times_ns"],
            regime_volatility_thresholds=calibration_config["regime_volatility_thresholds"] or None,
            regime_trade_intensity_thresholds=calibration_config["regime_trade_intensity_thresholds"] or None,
            regime_min_visible_queue=calibration_config["regime_min_visible_queue"] or None,
            regime_max_cancel_pressure=calibration_config["regime_max_cancel_pressure"] or None,
            allowed_symbols=set(trade_symbols),
        )
        test_trade_log, calibrated_summary, test_by_symbol = run_event_driven_backtest(
            test_calibrated,
            signal_column="gated_signal",
            trade_tapes=trade_tapes,
            entry_mode="passive",
            max_holding_steps=args.event_max_holding_steps,
            max_passive_wait_steps=args.event_max_passive_wait_steps,
            cooldown_steps=0,
            symbol_cooldown_steps=calibration_config["symbol_cooldowns"],
        )
        save_symbol_calibration_artifacts(
            output_dir=output_dir,
            prefix="test_calibrated_from_val",
            best_by_symbol=best_by_symbol,
            config=calibration_config,
            gated_predictions=test_calibrated,
            trade_log=test_trade_log,
            summary=calibrated_summary,
            by_symbol=test_by_symbol,
        )

    # 彙整並記錄本次執行的所有設定參數與最終評估指標 (分類指標與回測績效)
    run_summary = {
        "symbols": selected_symbols,
        "device": config.device,
        "output_dir": str(output_dir),
        "trade_symbols": trade_symbols,
        "symbol_weight_map": symbol_weight_map,
        "history_rows": len(history_df),
        "class_weighting": not args.disable_class_weighting,
        "feature_set": args.feature_set,
        "threshold_mode": args.threshold_mode,
        "probability_gating": not args.disable_probability_gating,
        "gating_prob_threshold": args.gating_prob_threshold,
        "gating_margin_threshold": args.gating_margin_threshold,
        "event_entry_mode": args.event_entry_mode,
        "event_cooldown_steps": args.event_cooldown_steps,
        "event_size_mode": args.event_size_mode,
        "train_nonstationary_weight": args.train_nonstationary_weight,
        "train_return_weight_alpha": args.train_return_weight_alpha,
        "calibrate_symbol_rules_on_val": args.calibrate_symbol_rules_on_val,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "signal_backtest_net_pnl_ticks": backtest_summary["net_pnl_ticks"],
        "signal_backtest_sharpe_like": backtest_summary["sharpe_like"],
        "event_backtest_net_pnl_ticks": event_summary["net_pnl_ticks"],
        "event_backtest_sharpe_like": event_summary["sharpe_like"],
    }
    # 若有完成校準，一併將校準後的事件驅動回測績效存入總結報告
    if calibrated_summary is not None:
        run_summary["val_calibrated_test_event_backtest_net_pnl_ticks"] = calibrated_summary["net_pnl_ticks"]
        run_summary["val_calibrated_test_event_backtest_sharpe_like"] = calibrated_summary["sharpe_like"]
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    torch.manual_seed(42) # 固定隨機亂數種子以確保訓練結果可重現
    main()
