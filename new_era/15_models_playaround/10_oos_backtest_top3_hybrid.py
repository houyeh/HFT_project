from __future__ import annotations

"""
10 - OOS backtest (queue-aware): top-3 playaround models
========================================================

你要的三件事（對應 user 需求）：
1) 用「舊 replay」+「所有沒有用來訓練的資料」做回測
   - NEW1：只跑 val/test sessions（非訓練 sessions）
   - NEW2：全資料（完全 OOS）
   - TW replay：replay1_TW.csv（完全 OOS / domain shift）
2) 把回測接到更擬真的模擬器：`src.backtest.hybrid_market_maker.run_hybrid_market_maker_backtest`
   - Queue-aware fill：用真實 trade tape 消耗 queue_ahead
   - 本腳本另外補上「以 TWD 計價、含 maker/taker fee」的摘要

注意：
- Playaround 的原始策略參數（tcn_skew 的 pthr/ethr）是用簡化的 next-quote fill 選出來的；
  這裡改用 queue-aware simulator，屬於不同回測器，因此結果用來「驗證穩健性 / 找問題」，
  而不是宣稱數值可 1:1 對標。
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _ensure_project_root_on_path

_ensure_project_root_on_path()

from new_era.models_playaround_15.config import (  # noqa: E402
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSet,
    LabelConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    WindowConfig,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from new_era.models_playaround_15.paths import get_project_root  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_load_cached_row(
    out_dir: Path,
    cache_name: str,
    expected_context: dict[str, object] | None = None,
) -> dict[str, object] | None:
    cached = out_dir / cache_name
    if not cached.exists():
        return None
    try:
        payload = _read_json(cached)
    except Exception:
        return None
    if not (isinstance(payload, dict) and payload.get("experiment_id") and payload.get("dataset")):
        return None
    if expected_context:
        for key, expected_value in expected_context.items():
            if payload.get(key) != expected_value:
                return None
        return payload  # type: ignore[return-value]
    return payload  # type: ignore[return-value]


def _load_cfg_from_experiment_dir(exp_dir: Path) -> ExperimentConfig:
    cfg_dict = _read_json(exp_dir / "experiment_config.json")

    data = cfg_dict.get("data") or {}
    window = cfg_dict.get("window") or {}
    split = cfg_dict.get("split") or {}
    features = cfg_dict.get("features") or {}
    labels = cfg_dict.get("labels") or {}
    model = cfg_dict.get("model") or {}
    train = cfg_dict.get("train") or {}
    backtest = cfg_dict.get("backtest") or {}

    return ExperimentConfig(
        data=DataConfig(
            data_dir=Path(str(data.get("data_dir") or "")),
            symbols=tuple(data.get("symbols") or ("TXFB6",)),
            zscore_window=int(data.get("zscore_window") or 2000),
        ),
        window=WindowConfig(
            window_size=int(window.get("window_size") or 100),
            horizon=int(window.get("horizon") or 10),
            threshold_ticks=float(window.get("threshold_ticks") or 1.0),
            sample_stride_train=int(window.get("sample_stride_train") or 5),
        ),
        split=SplitConfig(
            train_session_frac=float(split.get("train_session_frac") or 0.70),
            val_session_frac=float(split.get("val_session_frac") or 0.15),
        ),
        features=FeatureSet(feature_count=int(features.get("feature_count") or 8)),
        labels=LabelConfig(
            label_type=str(labels.get("label_type") or "mid_move_3class"),  # type: ignore[arg-type]
            adverse_threshold_ticks=float(labels.get("adverse_threshold_ticks") or 1.0),
            taker_fee_twd=float(labels.get("taker_fee_twd") or 50.0),
            point_value_twd_per_point=float(labels.get("point_value_twd_per_point") or 200.0),
            exit_cost_ticks=float(labels.get("exit_cost_ticks") or 0.0),
            margin_ticks=float(labels.get("margin_ticks") or 0.0),
        ),
        model=ModelConfig(
            channel_list=tuple(int(x) for x in (model.get("channel_list") or (16, 16, 16))),
            kernel_size=int(model.get("kernel_size") or 3),
            dropout=float(model.get("dropout") or 0.1),
        ),
        train=TrainingConfig(
            epochs=int(train.get("epochs") or 30),
            batch_size=int(train.get("batch_size") or 1024),
            lr=float(train.get("lr") or 1e-3),
            seed=int(train.get("seed") or 42),
            device=str(train.get("device") or "auto"),
        ),
        backtest=BacktestConfig(
            symbol=str(backtest.get("symbol") or "TXFB6"),
            point_value_twd_per_point=float(backtest.get("point_value_twd_per_point") or 200.0),
            maker_fee_twd=float(backtest.get("maker_fee_twd") or 15.0),
            taker_fee_twd=float(backtest.get("taker_fee_twd") or 50.0),
            max_inventory=int(backtest.get("max_inventory") or 3),
            sample_stride=int(backtest.get("sample_stride") or 1),
            allow_active_alpha=bool(backtest.get("allow_active_alpha") or False),
        ),
    )


def _load_selected_params(exp_dir: Path) -> dict:
    path = exp_dir / "backtest_outputs" / "selected_strategy_config.json"
    if not path.exists():
        raise FileNotFoundError(f"selected_strategy_config.json not found: {path}")
    return _read_json(path)


def _split_val_test_sessions(session_ids: list[str], train_frac: float, val_frac: float) -> tuple[list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 0:
        return [], []
    if total == 1:
        return [], ordered
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    val_sessions = ordered[train_end:val_end]
    test_sessions = ordered[val_end:]
    return val_sessions, test_sessions


def _detect_lob_symbol(data_dir: Path, preferred_prefix: str = "TXF") -> str | None:
    counts: Counter[str] = Counter()
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda path: str(path.relative_to(data_dir))):
        with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 34:
                    continue
                symbol = row[3].strip()
                if not symbol:
                    continue
                try:
                    has_best_quotes = float(row[21]) > 0 and float(row[22]) > 0
                except ValueError:
                    continue
                if has_best_quotes:
                    counts[symbol] += 1

    if not counts:
        return None
    preferred = [(symbol, count) for symbol, count in counts.items() if symbol.startswith(preferred_prefix)]
    if preferred:
        preferred.sort(key=lambda item: item[1], reverse=True)
        return preferred[0][0]
    return counts.most_common(1)[0][0]


def _max_drawdown(path: np.ndarray) -> float:
    arr = np.asarray(path, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    return float(np.max(peaks - arr))


def _attach_regime_volatility(
    predictions: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    if "current_mid" not in predictions.columns:
        raise ValueError("predictions is missing 'current_mid' column required for regime filter")

    ordered = predictions.sort_values(["symbol", "timestamp"]).copy()
    min_periods = max(5, int(window) // 4)
    ordered["rolling_volatility"] = (
        ordered.groupby("symbol", sort=False)["current_mid"]
        .transform(lambda series: series.pct_change().rolling(int(window), min_periods=min_periods).std())
        .astype(np.float64)
    )
    return ordered


def _attach_macro_trend(
    predictions: pd.DataFrame,
    *,
    lookback_ticks: int,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    if int(lookback_ticks) <= 0:
        raise ValueError(f"lookback_ticks must be > 0, got {lookback_ticks}")
    if "current_mid" not in predictions.columns:
        raise ValueError("predictions is missing 'current_mid' column required for macro trend gate")

    ordered = predictions.sort_values(["symbol", "timestamp"]).copy()
    ordered["macro_trend"] = (
        ordered.groupby("symbol", sort=False)["current_mid"]
        .transform(lambda series: series.diff(int(lookback_ticks)))
        .astype(np.float64)
        .fillna(0.0)
    )
    return ordered


def _build_regime_volatility_thresholds(
    predictions: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[dict[str, float], float]:
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError(f"regime_volatility_quantile must be in (0, 1], got {quantile}")
    if "rolling_volatility" not in predictions.columns:
        return {}, 0.0

    thresholds: dict[str, float] = {}
    total_rows = 0
    active_rows = 0
    for symbol, symbol_df in predictions.groupby("symbol", sort=False):
        vol = pd.to_numeric(symbol_df["rolling_volatility"], errors="coerce")
        finite = vol[np.isfinite(vol)]
        if finite.empty:
            continue
        threshold = float(finite.quantile(float(quantile)))
        thresholds[str(symbol)] = threshold
        active_mask = (vol <= threshold).fillna(False)
        total_rows += int(active_mask.shape[0])
        active_rows += int(active_mask.sum())

    active_ratio = float(active_rows / total_rows) if total_rows > 0 else 0.0
    return thresholds, active_ratio


def _summarize_trade_log_twd(
    trade_log: pd.DataFrame,
    *,
    tick_size: float,
    point_value_twd_per_point: float,
    maker_fee_twd: float,
    taker_fee_twd: float,
) -> dict[str, float | int]:
    if trade_log.empty:
        return {
            "realized_trades": 0,
            "gross_pnl_twd": 0.0,
            "fees_twd": 0.0,
            "net_pnl_twd": 0.0,
            "win_rate": 0.0,
            "avg_holding_steps": 0.0,
            "max_drawdown_twd": 0.0,
        }

    per_trade_gross = trade_log["pnl_ticks"].to_numpy(dtype=float) * float(tick_size) * float(point_value_twd_per_point)
    size = trade_log["size"].to_numpy(dtype=float)
    exit_is_passive = trade_log["reason"].astype(str).str.startswith("passive_").to_numpy(dtype=bool)
    per_trade_fees = size * float(maker_fee_twd) + size * np.where(exit_is_passive, float(maker_fee_twd), float(taker_fee_twd))
    per_trade_net = per_trade_gross - per_trade_fees

    cum = np.cumsum(per_trade_net, dtype=np.float64)
    return {
        "realized_trades": int(len(trade_log)),
        "gross_pnl_twd": float(per_trade_gross.sum()),
        "fees_twd": float(per_trade_fees.sum()),
        "net_pnl_twd": float(per_trade_net.sum()),
        "win_rate": float((per_trade_net > 0).mean()),
        "avg_holding_steps": float(trade_log["holding_steps"].mean()),
        "max_drawdown_twd": float(_max_drawdown(cum)),
    }


def _predict_as_hybrid_frame(
    *,
    model,
    datasets: list[tuple[str, object]],
    device,
    symbol: str,
    num_classes: int,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    frames: list[pd.DataFrame] = []

    model.eval()
    with torch.no_grad():
        for session_id, dataset in datasets:
            loader = DataLoader(dataset, batch_size=2048, shuffle=False)
            for batch in loader:
                x = batch["x"].to(device)  # (B, T, F) for TCNClassifier
                logits = model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred = prob.argmax(axis=1).astype(np.int64)

                if num_classes == 3:
                    prob_down = prob[:, 0]
                    prob_stat = prob[:, 1]
                    prob_up = prob[:, 2]
                elif num_classes == 2:
                    # Not directional, but keep columns for API compatibility.
                    prob_down = prob[:, 0]
                    prob_stat = np.zeros_like(prob_down)
                    prob_up = prob[:, 1]
                else:
                    raise ValueError(f"Unsupported num_classes={num_classes}")

                n = int(len(y_pred))
                frames.append(
                    pd.DataFrame(
                        {
                            "dataset_session_id": [session_id] * n,
                            "symbol": [symbol] * n,
                            "timestamp": batch["timestamp"].cpu().numpy().astype(np.int64),
                            "y_true": batch["y"].cpu().numpy().astype(np.int64),
                            "y_pred": y_pred,
                            "prob_down": prob_down.astype(np.float64),
                            "prob_stationary": prob_stat.astype(np.float64),
                            "prob_up": prob_up.astype(np.float64),
                            "spread_ticks": batch["spread_ticks"].cpu().numpy().astype(np.float64),
                            "tick_size": batch["tick_size"].cpu().numpy().astype(np.float64),
                            "rolling_volatility": batch["rolling_volatility"].cpu().numpy().astype(np.float64),
                            "current_mid": batch["current_mid"].cpu().numpy().astype(np.float64),
                            "bid_px_1": batch["bid_px_1"].cpu().numpy().astype(np.float64),
                            "ask_px_1": batch["ask_px_1"].cpu().numpy().astype(np.float64),
                            "bid_vol_1": batch["bid_vol_1"].cpu().numpy().astype(np.float64),
                            "ask_vol_1": batch["ask_vol_1"].cpu().numpy().astype(np.float64),
                            "queue_imbalance": batch["queue_imbalance"].cpu().numpy().astype(np.float64),
                            "queue_imbalance_change": batch["queue_imbalance_change"].cpu().numpy().astype(np.float64),
                            "trade_intensity_short": batch["trade_intensity_short"].cpu().numpy().astype(np.float64),
                            "aggressor_imbalance_interval": batch[
                                "aggressor_imbalance_interval"
                            ].cpu().numpy().astype(np.float64),
                            "bid_cancel_proxy": batch["bid_cancel_proxy"].cpu().numpy().astype(np.float64),
                            "ask_cancel_proxy": batch["ask_cancel_proxy"].cpu().numpy().astype(np.float64),
                            "net_cancel_pressure": batch["net_cancel_pressure"].cpu().numpy().astype(np.float64),
                            "cancel_pressure_short": batch["cancel_pressure_short"].cpu().numpy().astype(np.float64),
                        }
                    )
                )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_datasets_from_new_lob(
    *,
    cfg: ExperimentConfig,
    symbol: str,
    lob: pd.DataFrame,
    trades: pd.DataFrame,
    session_ids: list[str] | None,
    sample_stride: int,
) -> tuple[list[tuple[str, object]], float]:
    from src.data.tw_replay import infer_tick_size_from_lob
    from src.training.datasets import WindowedLOBSequenceDataset
    from src.utils.tw_train_helpers import build_tw_feature_frame
    from new_era.tcn_pipeline_utils import apply_rolling_zscore

    tick_size = float(infer_tick_size_from_lob(lob))
    datasets: list[tuple[str, object]] = []

    if session_ids is None:
        groups = [(None, lob)]
    else:
        groups = [(sid, lob[lob["session_id"] == sid]) for sid in session_ids]

    for sid, lob_df in groups:
        if lob_df.empty:
            continue
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
            continue

        if "session_id" in lob_df.columns and sid is not None and not trades.empty:
            trd = trades[trades["session_id"] == sid]
        else:
            trd = trades

        feature_frame = build_tw_feature_frame(
            lob_df.reset_index(drop=True),
            trd.reset_index(drop=True),
            symbol,
            tick_size,
        ).dropna()
        zscore_frame = apply_rolling_zscore(feature_frame, cfg.features.selected_features, window=cfg.data.zscore_window)

        ds = WindowedLOBSequenceDataset(
            feature_df=zscore_frame,
            symbol=symbol,
            tick_size=tick_size,
            window_size=cfg.window.window_size,
            horizon=cfg.window.horizon,
            threshold_ticks=cfg.window.threshold_ticks,
            sample_stride=sample_stride,
            feature_cols=cfg.features.z_features,
            label_type=cfg.labels.label_type,
            adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
            taker_fee_twd=cfg.labels.taker_fee_twd,
            point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
            exit_cost_ticks=cfg.labels.exit_cost_ticks,
            margin_ticks=cfg.labels.margin_ticks,
        )

        session_key = sid if sid is not None else "legacy_full_stream"
        datasets.append((session_key, ds))

    return datasets, tick_size


def _build_datasets_from_tw_replay(
    *,
    cfg: ExperimentConfig,
    replay_path: Path,
    symbol: str,
    depth: int,
    quote_stride: int,
    max_quotes: int,
    sample_stride: int,
) -> tuple[list[tuple[str, object]], dict[str, pd.DataFrame], float]:
    from src.data.tw_replay import infer_tick_size_from_lob, load_tw_replay_as_lob, load_tw_replay_trades
    from src.training.datasets import WindowedLOBSequenceDataset
    from src.utils.tw_train_helpers import build_tw_feature_frame
    from new_era.tcn_pipeline_utils import apply_rolling_zscore

    lob_map = load_tw_replay_as_lob(
        csv_path=replay_path,
        symbols=[symbol],
        depth=depth,
        quote_stride=quote_stride,
        dedupe_quotes=True,
    )
    if symbol not in lob_map or lob_map[symbol].empty:
        raise RuntimeError(f"TW replay: no LOB rows for symbol={symbol}")

    lob_df = lob_map[symbol]
    if max_quotes > 0:
        lob_df = lob_df.iloc[: int(max_quotes)].copy()

    trade_map = load_tw_replay_trades(replay_path, [symbol])
    trade_df = trade_map.get(symbol, pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"]))

    tick_size = float(infer_tick_size_from_lob(lob_df, depth=depth))
    feature_frame = build_tw_feature_frame(lob_df, trade_df, symbol, tick_size).dropna()
    zscore_frame = apply_rolling_zscore(feature_frame, cfg.features.selected_features, window=cfg.data.zscore_window)
    ds = WindowedLOBSequenceDataset(
        feature_df=zscore_frame,
        symbol=symbol,
        tick_size=tick_size,
        window_size=cfg.window.window_size,
        horizon=cfg.window.horizon,
        threshold_ticks=cfg.window.threshold_ticks,
        sample_stride=sample_stride,
        feature_cols=cfg.features.z_features,
        label_type=cfg.labels.label_type,
        adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
        taker_fee_twd=cfg.labels.taker_fee_twd,
        point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
        exit_cost_ticks=cfg.labels.exit_cost_ticks,
        margin_ticks=cfg.labels.margin_ticks,
    )
    return [("legacy_full_stream", ds)], trade_map, tick_size


def _run_hybrid_backtest_pair(
    *,
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame] | None,
    tick_size: float,
    cfg: ExperimentConfig,
    selected: dict,
    out_dir: Path,
    experiment_id: str,
    dataset_name: str,
    eval_sample_stride: int,
    regime_volatility_quantile: float | None,
    regime_volatility_window: int,
    cancel_threshold: float = 0.30,
    macro_trend_threshold: float = 5.0,
    macro_trend_lookback_ticks: int = 2000,
    inventory_stop_loss_ticks: float = 0.0,
) -> dict[str, object]:
    expected_context: dict[str, object] = {
        "eval_sample_stride": int(eval_sample_stride),
        "regime_volatility_quantile": (
            None if regime_volatility_quantile is None else float(regime_volatility_quantile)
        ),
        "regime_volatility_window": (
            None if regime_volatility_quantile is None else int(regime_volatility_window)
        ),
        "cancel_threshold": float(cancel_threshold),
        "macro_trend_threshold": float(macro_trend_threshold),
        "macro_trend_lookback_ticks": int(macro_trend_lookback_ticks),
        "inventory_stop_loss_ticks": float(inventory_stop_loss_ticks),
    }
    cached = _maybe_load_cached_row(
        out_dir,
        "summary_hybrid.json",
        expected_context=expected_context,
    )
    if cached is not None:
        print(f"[hybrid] cache hit: {experiment_id} / {dataset_name}")
        return cached

    from src.backtest.hybrid_market_maker import run_hybrid_market_maker_backtest

    if predictions.empty:
        raise RuntimeError(f"Empty predictions for {experiment_id} / {dataset_name}")

    edge_threshold = float(selected.get("edge_threshold") or 0.05)
    quote_step_ticks = float(selected.get("skew_ticks") or 1.0)
    predictions_for_backtest = _attach_macro_trend(
        predictions,
        lookback_ticks=int(macro_trend_lookback_ticks),
    )
    regime_vol_thresholds: dict[str, float] | None = None
    regime_active_ratio: float | None = None

    if regime_volatility_quantile is not None:
        predictions_for_backtest = _attach_regime_volatility(
            predictions_for_backtest,
            window=int(regime_volatility_window),
        )
        regime_vol_thresholds, regime_active_ratio = _build_regime_volatility_thresholds(
            predictions_for_backtest,
            quantile=float(regime_volatility_quantile),
        )
        if regime_vol_thresholds:
            min_thr = min(regime_vol_thresholds.values())
            max_thr = max(regime_vol_thresholds.values())
            print(
                "[hybrid] regime filter "
                f"{experiment_id} / {dataset_name}: "
                f"q={float(regime_volatility_quantile):.3f}, "
                f"window={int(regime_volatility_window)}, "
                f"active_ratio={float(regime_active_ratio or 0.0):.3f}, "
                f"vol_thr_range=[{min_thr:.8f}, {max_thr:.8f}]"
            )
        else:
            print(f"[hybrid] regime filter skipped: no finite rolling_volatility for {experiment_id} / {dataset_name}")

    # Baseline: no alpha (prob_edge=0), quote both sides at best.
    base_trade_log, base_summary_ticks, _, _ = run_hybrid_market_maker_backtest(
        predictions=predictions_for_backtest,
        trade_tapes=trade_tapes,
        max_inventory=int(cfg.backtest.max_inventory),
        score_threshold=1e9,
        one_sided_threshold=1e9,
        macro_trend_threshold=float("inf"),
        quote_step_ticks=quote_step_ticks,
        max_holding_steps=20,
        adverse_score_threshold=1e9,
        stationary_probability_cutoff=1.0,
        alpha_coefficients={"prob_edge": 0.0},
    )
    base_summary_twd = _summarize_trade_log_twd(
        base_trade_log,
        tick_size=tick_size,
        point_value_twd_per_point=float(cfg.backtest.point_value_twd_per_point),
        maker_fee_twd=float(cfg.backtest.maker_fee_twd),
        taker_fee_twd=float(cfg.backtest.taker_fee_twd),
    )

    # Model: directional alpha from prob_edge.
    model_trade_log, model_summary_ticks, _, _ = run_hybrid_market_maker_backtest(
        predictions=predictions_for_backtest,
        trade_tapes=trade_tapes,
        max_inventory=int(cfg.backtest.max_inventory),
        score_threshold=edge_threshold,
        one_sided_threshold=float(cancel_threshold),
        macro_trend_threshold=float(macro_trend_threshold),
        inventory_stop_loss_ticks=float(inventory_stop_loss_ticks),
        quote_step_ticks=quote_step_ticks,
        max_holding_steps=20,
        adverse_score_threshold=1e9,
        stationary_probability_cutoff=1.0,
        alpha_coefficients={"prob_edge": 1.0},
        regime_volatility_thresholds=regime_vol_thresholds if regime_vol_thresholds else None,
    )
    model_summary_twd = _summarize_trade_log_twd(
        model_trade_log,
        tick_size=tick_size,
        point_value_twd_per_point=float(cfg.backtest.point_value_twd_per_point),
        maker_fee_twd=float(cfg.backtest.maker_fee_twd),
        taker_fee_twd=float(cfg.backtest.taker_fee_twd),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base_trade_log.to_csv(out_dir / "baseline_hybrid_trade_log.csv", index=False, encoding="utf-8-sig")
    model_trade_log.to_csv(out_dir / "model_hybrid_trade_log.csv", index=False, encoding="utf-8-sig")
    _write_json(out_dir / "baseline_hybrid_summary_ticks.json", base_summary_ticks)
    _write_json(out_dir / "baseline_hybrid_summary_twd.json", base_summary_twd)
    _write_json(out_dir / "model_hybrid_summary_ticks.json", model_summary_ticks)
    _write_json(out_dir / "model_hybrid_summary_twd.json", model_summary_twd)
    _write_json(out_dir / "selected_strategy_config.json", selected)

    row: dict[str, object] = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "data_symbol": ",".join(sorted(str(x) for x in predictions_for_backtest["symbol"].dropna().unique().tolist())),
        "eval_sample_stride": int(eval_sample_stride),
        "regime_volatility_quantile": (
            None if regime_volatility_quantile is None else float(regime_volatility_quantile)
        ),
        "regime_volatility_window": (
            None if regime_volatility_quantile is None else int(regime_volatility_window)
        ),
        "regime_active_ratio": regime_active_ratio,
        "label_type": cfg.labels.label_type,
        "feature_count": cfg.features.feature_count,
        "tick_size": float(tick_size),
        "edge_threshold": float(edge_threshold),
        "cancel_threshold": float(cancel_threshold),
        "macro_trend_threshold": float(macro_trend_threshold),
        "macro_trend_lookback_ticks": int(macro_trend_lookback_ticks),
        "inventory_stop_loss_ticks": float(inventory_stop_loss_ticks),
        "quote_step_ticks": float(quote_step_ticks),
        "baseline_net_pnl_twd": float(base_summary_twd["net_pnl_twd"]),
        "model_net_pnl_twd": float(model_summary_twd["net_pnl_twd"]),
        "improvement_twd": float(model_summary_twd["net_pnl_twd"]) - float(base_summary_twd["net_pnl_twd"]),
        "baseline_realized_trades": int(base_summary_twd["realized_trades"]),
        "model_realized_trades": int(model_summary_twd["realized_trades"]),
        "baseline_max_drawdown_twd": float(base_summary_twd["max_drawdown_twd"]),
        "model_max_drawdown_twd": float(model_summary_twd["max_drawdown_twd"]),
        "baseline_fills": int(base_summary_ticks.get("fills", 0)),
        "model_fills": int(model_summary_ticks.get("fills", 0)),
    }
    _write_json(out_dir / "summary_hybrid.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--experiment-ids",
        type=str,
        default="",
        help=(
            "Comma-separated experiment_id list to run exactly. "
            "When set, this overrides --k auto-selection."
        ),
    )
    parser.add_argument("--results-root", type=str, default="new_era/15_models_playaround/015_models_playaround_result")
    parser.add_argument("--out-dir", type=str, default="new_era/backtest_outputs/oos_top3_hybrid")
    parser.add_argument("--symbol", type=str, default="TXFB6")
    parser.add_argument("--new1-dir", type=str, default="data/NEW1")
    parser.add_argument("--new2-dir", type=str, default="data/NEW2")
    parser.add_argument("--replay-path", type=str, default="data/raw/TW/replay1_TW.csv")
    parser.add_argument("--replay-symbol", type=str, default="TXFJ5")
    parser.add_argument("--replay-depth", type=int, default=5)
    parser.add_argument("--replay-quote-stride", type=int, default=5)
    parser.add_argument("--replay-max-quotes", type=int, default=0)
    parser.add_argument(
        "--eval-sample-stride",
        type=int,
        default=1,
        help="Downsample windows for faster hybrid replay (1 = most realistic but slowest).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="new1,new2,replay",
        help="Comma-separated datasets to run: new1,new2,replay. Use new2 for a focused NEW2/NEW3 OOS run.",
    )
    parser.add_argument(
        "--regime-volatility-quantile",
        type=float,
        default=None,
        help=(
            "If set, enable model-only regime gate: quote only when rolling_volatility <= "
            "per-symbol quantile (e.g. 0.6)."
        ),
    )
    parser.add_argument(
        "--regime-volatility-window",
        type=int,
        default=200,
        help="Rolling window used to compute volatility from current_mid for regime gating.",
    )
    parser.add_argument(
        "--cancel-threshold",
        type=float,
        default=0.30,
        help="one_sided_threshold for cancel-on-signal: alpha_score >= this cancels the risky side.",
    )
    parser.add_argument(
        "--macro-trend-threshold",
        type=float,
        default=5.0,
        help="Disable counter-trend quoting when |macro_trend| >= threshold (in ticks).",
    )
    parser.add_argument(
        "--macro-trend-lookback-ticks",
        type=int,
        default=2000,
        help="Lookback ticks for macro_trend = current_mid.diff(lookback_ticks).",
    )
    parser.add_argument(
        "--inventory-stop-loss-ticks",
        type=float,
        default=0.0,
        help="Hard stop loss in ticks for inventory liquidation (0 disables stop loss).",
    )
    args = parser.parse_args()
    if args.regime_volatility_quantile is not None and not (0.0 < float(args.regime_volatility_quantile) <= 1.0):
        raise ValueError("--regime-volatility-quantile must be in (0, 1].")
    if int(args.regime_volatility_window) <= 1:
        raise ValueError("--regime-volatility-window must be > 1.")
    if float(args.macro_trend_threshold) < 0:
        raise ValueError("--macro-trend-threshold must be >= 0.")
    if int(args.macro_trend_lookback_ticks) <= 0:
        raise ValueError("--macro-trend-lookback-ticks must be > 0.")
    if float(args.inventory_stop_loss_ticks) < 0:
        raise ValueError("--inventory-stop-loss-ticks must be >= 0.")

    project_root = get_project_root()
    results_root = (project_root / args.results_root).resolve()
    out_root = (project_root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = results_root / "experiment_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"experiment_summary.csv not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    for c in ("pnl_improvement_flatten", "pnl_improvement"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Hybrid simulator needs directional probabilities; filter to mid-move label types.
    if "label_type" in df.columns:
        df = df[df["label_type"].isin({"mid_move_3class", "smoothed_mid_move_3class"})].copy()

    requested_experiment_ids = [part.strip() for part in str(args.experiment_ids).split(",") if part.strip()]
    if requested_experiment_ids:
        by_id = df.set_index("experiment_id", drop=False)
        missing_ids = [exp_id for exp_id in requested_experiment_ids if exp_id not in by_id.index]
        if missing_ids:
            raise ValueError(
                "Requested --experiment-ids not found in experiment_summary.csv: "
                + ", ".join(missing_ids)
            )
        top = pd.concat([by_id.loc[[exp_id]] for exp_id in requested_experiment_ids], ignore_index=True)
    else:
        df = df.sort_values(["pnl_improvement_flatten", "pnl_improvement"], ascending=[False, False])
        top = df.head(int(args.k)).copy()
    if top.empty:
        raise RuntimeError("No eligible experiments found in experiment_summary.csv")

    requested_datasets = {part.strip().lower() for part in str(args.datasets).split(",") if part.strip()}
    if "all" in requested_datasets:
        requested_datasets = {"new1", "new2", "replay"}
    valid_datasets = {"new1", "new2", "replay"}
    invalid_datasets = requested_datasets - valid_datasets
    if invalid_datasets:
        raise ValueError(f"Unsupported --datasets entries: {sorted(invalid_datasets)}")
    if not requested_datasets:
        raise ValueError("--datasets must include at least one of: new1,new2,replay")

    import torch

    from src.models.tcn import TCNClassifier

    symbol = str(args.symbol)
    new1_dir = (project_root / args.new1_dir).resolve()
    new2_dir = (project_root / args.new2_dir).resolve()
    replay_path = (project_root / args.replay_path).resolve()
    replay_symbol = str(args.replay_symbol)

    if "new1" in requested_datasets and not new1_dir.exists():
        raise FileNotFoundError(f"NEW1 dir not found: {new1_dir}")
    if "new2" in requested_datasets and not new2_dir.exists():
        raise FileNotFoundError(f"NEW2 dir not found: {new2_dir}")
    if "replay" in requested_datasets and not replay_path.exists():
        raise FileNotFoundError(f"TW replay not found: {replay_path}")

    # Load requested OOS data once (shared across experiments).
    new1_lob = None
    new1_trades = pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])
    if "new1" in requested_datasets:
        new1_lob_frames, new1_trade_frames = load_new_tw_data(new1_dir, [symbol])
        new1_lob = new1_lob_frames.get(symbol)
        new1_trades = new1_trade_frames.get(
            symbol,
            pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
        )
        if new1_lob is None or new1_lob.empty:
            raise RuntimeError(f"No LOB rows found for symbol={symbol} in {new1_dir}")

    new2_symbol = symbol
    new2_lob = None
    new2_trades = pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])
    if "new2" in requested_datasets:
        new2_lob_frames, new2_trade_frames = load_new_tw_data(new2_dir, [new2_symbol])
        new2_lob = new2_lob_frames.get(new2_symbol)
        new2_trades = new2_trade_frames.get(
            new2_symbol,
            pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
        )
        if new2_lob is None or new2_lob.empty:
            detected_symbol = _detect_lob_symbol(new2_dir)
            if detected_symbol is None:
                raise RuntimeError(f"No LOB rows found in {new2_dir}")
            print(f"[OOS] {new2_dir.name}: symbol={symbol} not found; using detected symbol={detected_symbol}")
            new2_symbol = detected_symbol
            new2_lob_frames, new2_trade_frames = load_new_tw_data(new2_dir, [new2_symbol])
            new2_lob = new2_lob_frames.get(new2_symbol)
            new2_trades = new2_trade_frames.get(
                new2_symbol,
                pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
            )
            if new2_lob is None or new2_lob.empty:
                raise RuntimeError(f"No LOB rows found for detected symbol={new2_symbol} in {new2_dir}")
    new2_dataset_name = f"{new2_dir.name}_full"

    rows: list[dict[str, object]] = []
    for exp_id in top["experiment_id"].tolist():
        exp_dir = results_root / str(exp_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

        cfg = _load_cfg_from_experiment_dir(exp_dir)
        if cfg.labels.label_type not in ("mid_move_3class", "smoothed_mid_move_3class"):
            print(f"[skip] {exp_dir.name}: label_type={cfg.labels.label_type} not directional")
            continue

        selected = _load_selected_params(exp_dir)
        weights_path = exp_dir / "checkpoints" / "tcn_weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TCNClassifier(
            num_inputs=cfg.num_inputs,
            num_classes=cfg.num_classes,
            channel_list=list(cfg.model.channel_list),
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        # ----------------------------------------
        # NEW1 non-training sessions (val+test)
        # ----------------------------------------
        if "new1" in requested_datasets:
            assert new1_lob is not None
            new1_sessions = sorted(new1_lob["session_id"].drop_duplicates().tolist())
            val_sessions, test_sessions = _split_val_test_sessions(
                new1_sessions, train_frac=cfg.split.train_session_frac, val_frac=cfg.split.val_session_frac
            )
            nontrain_sessions = val_sessions + test_sessions
            print(f"[OOS] {exp_dir.name} NEW1 nontrain sessions: {nontrain_sessions}")
            new1_nontrain_ds, new1_tick = _build_datasets_from_new_lob(
                cfg=replace(cfg, data=replace(cfg.data, data_dir=new1_dir)),
                symbol=symbol,
                lob=new1_lob,
                trades=new1_trades,
                session_ids=nontrain_sessions,
                sample_stride=int(args.eval_sample_stride),
            )
            new1_pred = _predict_as_hybrid_frame(
                model=model,
                datasets=new1_nontrain_ds,
                device=device,
                symbol=symbol,
                num_classes=cfg.num_classes,
            )
            out_dir = out_root / exp_dir.name / "NEW1_nontrain"
            rows.append(
                _run_hybrid_backtest_pair(
                    predictions=new1_pred,
                    trade_tapes={symbol: new1_trades},
                    tick_size=new1_tick,
                    cfg=cfg,
                    selected=selected,
                    out_dir=out_dir,
                    experiment_id=exp_dir.name,
                    dataset_name="NEW1_nontrain",
                    eval_sample_stride=int(args.eval_sample_stride),
                    regime_volatility_quantile=(
                        None
                        if args.regime_volatility_quantile is None
                        else float(args.regime_volatility_quantile)
                    ),
                    regime_volatility_window=int(args.regime_volatility_window),
                    cancel_threshold=float(args.cancel_threshold),
                    macro_trend_threshold=float(args.macro_trend_threshold),
                    macro_trend_lookback_ticks=int(args.macro_trend_lookback_ticks),
                    inventory_stop_loss_ticks=float(args.inventory_stop_loss_ticks),
                )
            )

        # ----------------------------------------
        # NEW2 (full OOS)
        # ----------------------------------------
        if "new2" in requested_datasets:
            assert new2_lob is not None
            print(f"[OOS] {exp_dir.name} {new2_dir.name} full symbol={new2_symbol}")
            new2_ds, new2_tick = _build_datasets_from_new_lob(
                cfg=replace(cfg, data=replace(cfg.data, data_dir=new2_dir)),
                symbol=new2_symbol,
                lob=new2_lob,
                trades=new2_trades,
                session_ids=sorted(new2_lob["session_id"].drop_duplicates().tolist()),
                sample_stride=int(args.eval_sample_stride),
            )
            new2_pred = _predict_as_hybrid_frame(
                model=model,
                datasets=new2_ds,
                device=device,
                symbol=new2_symbol,
                num_classes=cfg.num_classes,
            )
            out_dir = out_root / exp_dir.name / new2_dataset_name
            rows.append(
                _run_hybrid_backtest_pair(
                    predictions=new2_pred,
                    trade_tapes={new2_symbol: new2_trades},
                    tick_size=new2_tick,
                    cfg=cfg,
                    selected=selected,
                    out_dir=out_dir,
                    experiment_id=exp_dir.name,
                    dataset_name=new2_dataset_name,
                    eval_sample_stride=int(args.eval_sample_stride),
                    regime_volatility_quantile=(
                        None
                        if args.regime_volatility_quantile is None
                        else float(args.regime_volatility_quantile)
                    ),
                    regime_volatility_window=int(args.regime_volatility_window),
                    cancel_threshold=float(args.cancel_threshold),
                    macro_trend_threshold=float(args.macro_trend_threshold),
                    macro_trend_lookback_ticks=int(args.macro_trend_lookback_ticks),
                    inventory_stop_loss_ticks=float(args.inventory_stop_loss_ticks),
                )
            )

        # ----------------------------------------
        # TW replay (domain shift, full stream)
        # ----------------------------------------
        if "replay" in requested_datasets:
            print(f"[OOS] {exp_dir.name} TW replay symbol={replay_symbol}")
            tw_ds, tw_trade_map, tw_tick = _build_datasets_from_tw_replay(
                cfg=cfg,
                replay_path=replay_path,
                symbol=replay_symbol,
                depth=int(args.replay_depth),
                quote_stride=int(args.replay_quote_stride),
                max_quotes=int(args.replay_max_quotes),
                sample_stride=int(args.eval_sample_stride),
            )
            tw_pred = _predict_as_hybrid_frame(
                model=model,
                datasets=tw_ds,
                device=device,
                symbol=replay_symbol,
                num_classes=cfg.num_classes,
            )
            out_dir = out_root / exp_dir.name / "TW_replay"
            rows.append(
                _run_hybrid_backtest_pair(
                    predictions=tw_pred,
                    trade_tapes=tw_trade_map,
                    tick_size=tw_tick,
                    cfg=cfg,
                    selected=selected,
                    out_dir=out_dir,
                    experiment_id=exp_dir.name,
                    dataset_name="TW_replay",
                    eval_sample_stride=int(args.eval_sample_stride),
                    regime_volatility_quantile=(
                        None
                        if args.regime_volatility_quantile is None
                        else float(args.regime_volatility_quantile)
                    ),
                    regime_volatility_window=int(args.regime_volatility_window),
                    cancel_threshold=float(args.cancel_threshold),
                    macro_trend_threshold=float(args.macro_trend_threshold),
                    macro_trend_lookback_ticks=int(args.macro_trend_lookback_ticks),
                    inventory_stop_loss_ticks=float(args.inventory_stop_loss_ticks),
                )
            )

    out_csv = out_root / "top3_oos_hybrid_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    _write_json(
        out_root / "top_selected_experiments.json",
        {
            "top_k": int(args.k),
            "requested_experiment_ids": requested_experiment_ids,
            "experiments": top.to_dict(orient="records"),
        },
    )
    print(f"[OOS hybrid] wrote -> {out_csv}")


if __name__ == "__main__":
    main()
