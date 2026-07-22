from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import apply_rolling_zscore, run_market_making_simulation, split_ordered_session_ids
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from src.training.datasets import WindowedLOBSequenceDataset
from src.utils.tw_train_helpers import build_tw_feature_frame


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _build_session_dataset(cfg: ExperimentConfig, lob_df: pd.DataFrame, trade_df: pd.DataFrame, symbol: str, tick_size: float) -> WindowedLOBSequenceDataset:
    feature_frame = build_tw_feature_frame(
        lob_df.reset_index(drop=True),
        trade_df.reset_index(drop=True),
        symbol,
        tick_size,
    ).dropna()
    zscore_frame = apply_rolling_zscore(feature_frame, cfg.features.selected_features, window=cfg.data.zscore_window)
    return WindowedLOBSequenceDataset(
        feature_df=zscore_frame,
        symbol=symbol,
        tick_size=tick_size,
        window_size=cfg.window.window_size,
        horizon=cfg.window.horizon,
        threshold_ticks=cfg.window.threshold_ticks,
        sample_stride=cfg.backtest.sample_stride,
        feature_cols=cfg.features.z_features,
        label_type=cfg.labels.label_type,
        adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
        taker_fee_twd=cfg.labels.taker_fee_twd,
        point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
        exit_cost_ticks=cfg.labels.exit_cost_ticks,
        margin_ticks=cfg.labels.margin_ticks,
    )


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    datasets: list[tuple[str, WindowedLOBSequenceDataset]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs: list[np.ndarray] = []
    all_ask_px: list[np.ndarray] = []
    all_bid_px: list[np.ndarray] = []

    for session_id, dataset in datasets:
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        for batch in loader:
            batch_x = batch["x"].transpose(1, 2).to(device)
            probs = torch.softmax(model(batch_x), dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_ask_px.append(batch["ask_px_1"].numpy())
            all_bid_px.append(batch["bid_px_1"].numpy())

    if not all_probs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_ask_px, axis=0), np.concatenate(all_bid_px, axis=0)


def backtest_basic(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)
    paths.ensure_dirs()
    save_experiment_config(paths, cfg)

    if cfg.num_classes != 3:
        raise NotImplementedError(
            "backtest_basic currently supports only 3-class direction models "
            "(mid_move_3class / smoothed_mid_move_3class)."
        )

    if not paths.model_weights_pth.exists():
        raise FileNotFoundError(f"Model weights not found: {paths.model_weights_pth}")

    data_dir = cfg.data.data_dir
    lob_frames, trade_frames = load_new_tw_data(data_dir, [cfg.backtest.symbol])
    if cfg.backtest.symbol not in lob_frames:
        raise RuntimeError(f"No LOB data found for {cfg.backtest.symbol} in {data_dir}")

    symbol_lob = lob_frames[cfg.backtest.symbol]
    symbol_trades = trade_frames.get(
        cfg.backtest.symbol,
        pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
    )
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(
        session_ids,
        train_frac=cfg.split.train_session_frac,
        val_frac=cfg.split.val_session_frac,
    )

    tick_size = infer_tick_size_from_lob(symbol_lob)
    datasets: list[tuple[str, WindowedLOBSequenceDataset]] = []
    for session_id in test_sessions:
        lob_df = symbol_lob[symbol_lob["session_id"] == session_id]
        trade_df = symbol_trades[symbol_trades["session_id"] == session_id]
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
            continue
        datasets.append((session_id, _build_session_dataset(cfg, lob_df, trade_df, cfg.backtest.symbol, tick_size)))

    if not datasets:
        raise RuntimeError("No test sessions contained enough rows for the configured window/horizon.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    model.load_state_dict(torch.load(paths.model_weights_pth, map_location=device, weights_only=True))
    model.eval()

    probs, ask_px, bid_px = _collect_predictions(model, datasets, device=device)
    baseline = run_market_making_simulation(
        probs,
        ask_px,
        bid_px,
        strategy="baseline",
        point_value=int(cfg.backtest.point_value_twd_per_point),
        maker_fee=int(cfg.backtest.maker_fee_twd),
        taker_fee=int(cfg.backtest.taker_fee_twd),
    )
    gated = run_market_making_simulation(
        probs,
        ask_px,
        bid_px,
        strategy="tcn_gated",
        point_value=int(cfg.backtest.point_value_twd_per_point),
        maker_fee=int(cfg.backtest.maker_fee_twd),
        taker_fee=int(cfg.backtest.taker_fee_twd),
    )

    results = {
        "symbol": cfg.backtest.symbol,
        "sessions": {"train": train_sessions, "val": val_sessions, "test": test_sessions},
        "device": str(device),
        "baseline": baseline,
        "tcn_gated": gated,
        "improvement_vs_baseline": float(gated["pnl"] - baseline["pnl"]),
    }

    write_json(paths.backtest_dir / "backtest_basic_results.json", results)
    return results


def build_default_config() -> ExperimentConfig:
    project_root = get_project_root()
    return ExperimentConfig(
        data=DataConfig(
            data_dir=project_root / "data" / "NEW",
            symbols=("TXFB6",),
            zscore_window=2000,
        )
    )


def main() -> None:
    cfg = build_default_config()
    paths = build_experiment_paths(cfg)
    print(f"[backtest] weights: {paths.model_weights_pth}")
    results = backtest_basic(cfg, paths)
    print(f"[backtest] baseline_pnl={results['baseline']['pnl']:.2f} gated_pnl={results['tcn_gated']['pnl']:.2f}")


if __name__ == "__main__":
    main()
