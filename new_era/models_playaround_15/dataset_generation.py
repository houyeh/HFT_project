from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import apply_rolling_zscore, safe_session_key
from src.data.tw_replay import infer_tick_size_from_lob
from src.training.datasets import WindowedLOBSequenceDataset
from src.utils.tw_train_helpers import build_tw_feature_frame


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def generate_dataset_npz(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)

    data_dir = cfg.data.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    print(f"[dataset] loading raw CSVs from: {data_dir} (symbols={list(cfg.data.symbols)})")
    lob_frames, trade_frames = load_new_tw_data(data_dir, list(cfg.data.symbols))
    for symbol in cfg.data.symbols:
        df = lob_frames.get(symbol)
        if df is None:
            print(f"[dataset] {symbol}: no LOB rows loaded")
        else:
            print(f"[dataset] {symbol}: rows={len(df):,} sessions={df['session_id'].nunique()}")
    output_data: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {
        "symbols": list(cfg.data.symbols),
        "selected_features": cfg.features.selected_features,
        "z_features": cfg.features.z_features,
        "label_type": cfg.labels.label_type,
        "window_size": cfg.window.window_size,
        "horizon": cfg.window.horizon,
        "threshold_ticks": cfg.window.threshold_ticks,
        "adverse_threshold_ticks": cfg.labels.adverse_threshold_ticks,
        "taker_fee_twd": cfg.labels.taker_fee_twd,
        "point_value_twd_per_point": cfg.labels.point_value_twd_per_point,
        "exit_cost_ticks": cfg.labels.exit_cost_ticks,
        "margin_ticks": cfg.labels.margin_ticks,
        "sample_stride_train": cfg.window.sample_stride_train,
        "zscore_window": cfg.data.zscore_window,
        "sessions": [],
        "class_distribution_by_session": {},
    }

    debug_stats: dict[str, object] = {
        "symbols_requested": list(cfg.data.symbols),
        "symbols_loaded": sorted(list(lob_frames.keys())),
        "sessions_skipped_too_short": 0,
        "sessions_skipped_no_samples": 0,
        "sessions_processed": 0,
    }

    for symbol in cfg.data.symbols:
        if symbol not in lob_frames:
            continue

        symbol_lob = lob_frames[symbol]
        symbol_trades = trade_frames.get(
            symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])
        )

        for session_id, lob_df in symbol_lob.groupby("session_id", sort=True):
            print(f"[dataset] {symbol} session={session_id} rows={len(lob_df):,}")
            if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
                debug_stats["sessions_skipped_too_short"] = int(debug_stats["sessions_skipped_too_short"]) + 1
                continue

            trade_df = symbol_trades[symbol_trades["session_id"] == session_id]
            tick_size = infer_tick_size_from_lob(lob_df)

            feature_frame = build_tw_feature_frame(
                lob_df.reset_index(drop=True),
                trade_df.reset_index(drop=True),
                symbol,
                tick_size,
            ).dropna()

            zscore_frame = apply_rolling_zscore(
                feature_frame,
                cfg.features.selected_features,
                window=cfg.data.zscore_window,
            )

            dataset = WindowedLOBSequenceDataset(
                feature_df=zscore_frame,
                symbol=symbol,
                tick_size=tick_size,
                window_size=cfg.window.window_size,
                horizon=cfg.window.horizon,
                threshold_ticks=cfg.window.threshold_ticks,
                sample_stride=cfg.window.sample_stride_train,
                feature_cols=cfg.features.z_features,
                label_type=cfg.labels.label_type,
                adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
                taker_fee_twd=cfg.labels.taker_fee_twd,
                point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
                exit_cost_ticks=cfg.labels.exit_cost_ticks,
                margin_ticks=cfg.labels.margin_ticks,
            )

            x_list: list[np.ndarray] = []
            y_list: list[int] = []
            for i in range(len(dataset)):
                item = dataset[i]
                x_list.append(item["x"].numpy())
                y_list.append(int(item["y"]))

            if not x_list:
                debug_stats["sessions_skipped_no_samples"] = int(debug_stats["sessions_skipped_no_samples"]) + 1
                continue

            session_key = f"{symbol}__{safe_session_key(session_id)}"
            output_data[f"{session_key}_X"] = np.stack(x_list)
            output_data[f"{session_key}_y"] = np.asarray(y_list, dtype=np.int64)

            manifest["sessions"].append(session_key)
            manifest["class_distribution_by_session"][session_key] = dict(Counter(y_list))
            debug_stats["sessions_processed"] = int(debug_stats["sessions_processed"]) + 1

    paths.ensure_dirs()
    save_experiment_config(paths, cfg)
    write_json(paths.root_dir / "dataset_manifest.json", manifest)
    write_json(paths.root_dir / "dataset_debug_stats.json", debug_stats)

    if not output_data:
        raise RuntimeError(
            "No sequences were generated; dataset is empty. "
            "Check `dataset_manifest.json` and `dataset_debug_stats.json` for details. "
            "Common causes: symbol mismatch, timestamp parsing issues, or feature/label generation producing no valid windows."
        )

    np.savez_compressed(paths.dataset_npz, **output_data)
    return manifest


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
    print(f"[dataset] writing to: {paths.root_dir}")
    manifest = generate_dataset_npz(cfg, paths)
    print(f"[dataset] sessions: {len(manifest['sessions'])}")


if __name__ == "__main__":
    main()
