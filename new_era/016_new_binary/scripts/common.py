from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.tcn_pipeline_utils import apply_rolling_zscore
from src.data.tw_replay import infer_tick_size_from_lob
from src.utils.tw_train_helpers import build_tw_feature_frame


ROOT_DIR = PROJECT_ROOT / "new_era" / "016_new_binary"
SCRIPTS_DIR = ROOT_DIR / "scripts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DATA_AUDIT_DIR = OUTPUTS_DIR / "data_audit"
LABELS_DIR = OUTPUTS_DIR / "labels"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"
COMPARISON_DIR = OUTPUTS_DIR / "comparison"
LOGS_DIR = OUTPUTS_DIR / "logs"

NEW_ALL_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
NEW3_DIR = PROJECT_ROOT / "data" / "NEW3"
NEW_SYMBOL = "TXFB6"
NEW3_SYMBOL = "TXFA6"

SELECTED_FEATURES: list[str] = [
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
]
Z_FEATURES: list[str] = [f"{name}_z" for name in SELECTED_FEATURES]

WINDOW_SIZE = 100
DEFAULT_ZSCORE_WINDOW = 2000
TRAIN_SAMPLE_STRIDE = 5
EVAL_SAMPLE_STRIDE = 1
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def ensure_output_dirs() -> None:
    for path in [
        ROOT_DIR,
        SCRIPTS_DIR,
        OUTPUTS_DIR,
        DATA_AUDIT_DIR,
        LABELS_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        BACKTESTS_DIR,
        COMPARISON_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def add_venv_site_packages_to_syspath() -> Path:
    site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    site_text = str(site_packages)
    if site_text not in sys.path:
        sys.path.insert(0, site_text)
    return site_packages


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def load_symbol_data(data_dir: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"LOB symbol not found: {symbol} in {data_dir}")
    lob_df = lob_frames[symbol].copy()
    trade_df = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()
    return lob_df, trade_df


def build_feature_frame_for_session(
    lob_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    tick_size: float,
    zscore_window: int = DEFAULT_ZSCORE_WINDOW,
) -> pd.DataFrame:
    raw = build_tw_feature_frame(
        lob_df.reset_index(drop=True),
        trade_df.reset_index(drop=True),
        symbol=symbol,
        tick_size=float(tick_size),
    ).dropna()
    zdf = apply_rolling_zscore(raw, SELECTED_FEATURES, window=int(zscore_window))
    return zdf


def detect_fill_proxy_masks(
    bid_px_1: np.ndarray,
    ask_px_1: np.ndarray,
    bid_vol_1: np.ndarray,
    ask_vol_1: np.ndarray,
    fill_window: int,
    min_size_decrease: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(bid_px_1))
    bid_proxy = np.zeros(n, dtype=bool)
    ask_proxy = np.zeros(n, dtype=bool)
    if n <= fill_window:
        return bid_proxy, ask_proxy

    for i in range(0, n - fill_window):
        cbid = float(bid_px_1[i])
        cask = float(ask_px_1[i])
        cbid_vol = float(bid_vol_1[i])
        cask_vol = float(ask_vol_1[i])
        j0 = i + 1
        j1 = i + fill_window + 1
        bid_future = bid_px_1[j0:j1]
        ask_future = ask_px_1[j0:j1]
        bid_vol_future = bid_vol_1[j0:j1]
        ask_vol_future = ask_vol_1[j0:j1]

        # Proxy C (price level removed) is primary per task.
        if np.any(bid_future < cbid):
            bid_proxy[i] = True
        if np.any(ask_future > cask):
            ask_proxy[i] = True

        # Proxy B and D are computed elsewhere for auditing; this function remains primary.
        _ = cbid_vol + cask_vol + float(np.sum(bid_vol_future)) + float(np.sum(ask_vol_future))

    return bid_proxy, ask_proxy


def make_window_tensor(
    feature_matrix: np.ndarray,
    label_indices: np.ndarray,
    window_size: int,
) -> np.ndarray:
    n = int(len(label_indices))
    f = int(feature_matrix.shape[1])
    out = np.zeros((n, f, window_size), dtype=np.float32)
    for k, idx in enumerate(label_indices):
        end = int(idx) + 1
        start = end - int(window_size)
        out[k] = feature_matrix[start:end].T.astype(np.float32, copy=False)
    return out


def compute_markouts(
    bid_px_1: np.ndarray,
    ask_px_1: np.ndarray,
    mid: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(mid)
    bid_markout = np.full(n, np.nan, dtype=np.float64)
    ask_markout = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return bid_markout, ask_markout
    future_mid = mid[horizon:]
    bid_markout[:-horizon] = future_mid - bid_px_1[:-horizon]
    ask_markout[:-horizon] = ask_px_1[:-horizon] - future_mid
    return bid_markout, ask_markout


@dataclass
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    label_indices: np.ndarray
    session_ids: np.ndarray
    ask_px_1: np.ndarray
    bid_px_1: np.ndarray


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
