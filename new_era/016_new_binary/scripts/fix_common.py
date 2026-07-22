from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project and local venv packages importable even when using system python.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import torch

from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.tcn_pipeline_utils import apply_rolling_zscore
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from src.utils.tw_train_helpers import build_tw_feature_frame


ROOT_DIR = PROJECT_ROOT / "new_era" / "016_new_binary"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_FIX_DIR = ROOT_DIR / "outputs_fix"

DIAG_DIR = OUTPUTS_FIX_DIR / "diagnostics"
DATASETS_DIR = OUTPUTS_FIX_DIR / "datasets"
MODELS_DIR = OUTPUTS_FIX_DIR / "models"
BACKTESTS_DIR = OUTPUTS_FIX_DIR / "backtests"
COMPARISON_DIR = OUTPUTS_FIX_DIR / "comparison"
FIGURES_DIR = OUTPUTS_FIX_DIR / "figures"
REPORTS_DIR = OUTPUTS_FIX_DIR / "reports"

NEW_ALL_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
NEW3_DIR = PROJECT_ROOT / "data" / "NEW3"
NEW_SYMBOL = "TXFB6"
NEW3_SYMBOL = "TXFA6"

WINDOW_SIZE = 100
HORIZON = 10
Z_WINDOW = 2000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

SELECTED_FEATURES = [
    "obi",
    "queue_imbalance",
    "weighted_mid_deviation",
    "trade_intensity_short",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "cancel_pressure_short",
]
Z_FEATURES = [f"{name}_z" for name in SELECTED_FEATURES]

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
ASYM_PAIRS = [
    (0.10, 0.20),
    (0.20, 0.10),
    (0.20, 0.30),
    (0.30, 0.20),
    (0.30, 0.50),
    (0.50, 0.30),
]


def ensure_fix_dirs() -> None:
    for path in [
        OUTPUTS_FIX_DIR,
        DIAG_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        BACKTESTS_DIR,
        COMPARISON_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total <= 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def _load_symbol(data_dir: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"Missing symbol={symbol} in {data_dir}")
    lob = lob_frames[symbol].copy()
    trade = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()
    return lob, trade


def _session_feature_frame(
    lob_session: pd.DataFrame,
    trade_session: pd.DataFrame,
    symbol: str,
    tick_size: float,
) -> pd.DataFrame:
    fdf = build_tw_feature_frame(
        lob_session.reset_index(drop=True),
        trade_session.reset_index(drop=True),
        symbol=symbol,
        tick_size=tick_size,
    ).dropna()
    zdf = apply_rolling_zscore(fdf, SELECTED_FEATURES, window=Z_WINDOW)
    return zdf.reset_index(drop=True)


def build_processed_frame(
    data_dir: Path,
    symbol: str,
    session_ids: list[str] | None,
) -> tuple[pd.DataFrame, float]:
    lob, trade = _load_symbol(data_dir, symbol)
    session_filter = set(session_ids) if session_ids is not None else None
    frames: list[pd.DataFrame] = []
    ticks: list[float] = []
    for sid, lob_s in lob.groupby("session_id", sort=True):
        if session_filter is not None and sid not in session_filter:
            continue
        trd_s = trade[trade["session_id"] == sid]
        tick = float(infer_tick_size_from_lob(lob_s))
        zdf = _session_feature_frame(lob_s, trd_s, symbol=symbol, tick_size=tick)
        if zdf.empty:
            continue
        zdf["session_id"] = sid
        frames.append(zdf)
        ticks.append(tick)
    if not frames:
        return pd.DataFrame(), float("nan")
    out = pd.concat(frames, ignore_index=True)
    return out, float(np.median(np.asarray(ticks, dtype=np.float64)))


def load_new_split_frames() -> dict[str, tuple[pd.DataFrame, float]]:
    split_meta_path = OUTPUTS_DIR / "datasets" / "new_session_split.json"
    if split_meta_path.exists():
        split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
        train_s = split_meta.get("train_sessions", [])
        val_s = split_meta.get("val_sessions", [])
        test_s = split_meta.get("test_sessions", [])
    else:
        lob, _ = _load_symbol(NEW_ALL_DIR, NEW_SYMBOL)
        all_s = sorted(lob["session_id"].unique().tolist())
        train_s, val_s, test_s = split_ordered_session_ids(all_s)

    train_frame, tick_train = build_processed_frame(NEW_ALL_DIR, NEW_SYMBOL, train_s)
    val_frame, tick_val = build_processed_frame(NEW_ALL_DIR, NEW_SYMBOL, val_s)
    test_frame, tick_test = build_processed_frame(NEW_ALL_DIR, NEW_SYMBOL, test_s)
    new3_frame, tick_new3 = build_processed_frame(NEW3_DIR, NEW3_SYMBOL, None)
    return {
        "NEW_train": (train_frame, tick_train),
        "NEW_val": (val_frame, tick_val),
        "NEW_test": (test_frame, tick_test),
        "NEW3_full": (new3_frame, tick_new3),
    }


def load_model(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    model = HLSFriendlyTCNClassifier(
        num_inputs=int(cfg["feature_count"]),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_all_event_probs(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    n = len(frame)
    out = np.zeros(n, dtype=np.float32)
    if n <= WINDOW_SIZE + HORIZON:
        return out
    feat = frame[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
    label_indices = np.arange(WINDOW_SIZE - 1, n - HORIZON, dtype=np.int64)
    for i0 in range(0, len(label_indices), batch_size):
        idx = label_indices[i0 : i0 + batch_size]
        x = np.zeros((len(idx), len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
        for k, ii in enumerate(idx):
            st = int(ii) - WINDOW_SIZE + 1
            ed = int(ii) + 1
            x[k] = feat[st:ed].T
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        out[idx] = p
    return out


def import_simulator():
    path = PROJECT_ROOT / "new_era" / "15_models_playaround" / "run_diagnosis_1.py"
    parent = path.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("run_diagnosis_1_local_fix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate_with_metrics


def import_cfg():
    common_path = PROJECT_ROOT / "new_era" / "15_models_playaround"
    if str(common_path) not in sys.path:
        sys.path.insert(0, str(common_path))
    import _common  # type: ignore

    return _common.build_cfg()


def run_backtest_from_probs(
    *,
    frame: pd.DataFrame,
    tick_size: float,
    bid_probs: np.ndarray,
    ask_probs: np.ndarray,
    thr_bid: float,
    thr_ask: float,
    cfg,
    simulate_with_metrics,
) -> dict[str, float]:
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)
    pull_bid = bid_probs >= float(thr_bid)
    pull_ask = ask_probs >= float(thr_ask)

    res = simulate_with_metrics(
        cfg,
        prob=None,
        ask_px=ask,
        bid_px=bid,
        tick_size=float(tick_size),
        threshold=1.1,
        force_pull_bid_mask=pull_bid,
        force_pull_ask_mask=pull_ask,
    )

    active_event_rate = float(((~pull_bid) | (~pull_ask)).mean()) if pull_bid.size else float("nan")
    bid_signal_rate = float(pull_bid.mean()) if pull_bid.size else float("nan")
    ask_signal_rate = float(pull_ask.mean()) if pull_ask.size else float("nan")
    return {
        "net_pnl": float(res["net_pnl"]),
        "trade_count": int(res["trade_count"]),
        "max_drawdown": float(res["max_drawdown"]),
        "net_cancel_value": float(res["net_cancel_value"]),
        "avg_pnl_per_fill": float(res["avg_pnl_per_fill"]),
        "active_event_rate": active_event_rate,
        "bid_signal_rate": bid_signal_rate,
        "ask_signal_rate": ask_signal_rate,
        "gross_pnl": float(res["gross_pnl"]),
        "fee": float(res["fee"]),
        "realized_bid_pnl": float(res["realized_bid_pnl"]),
        "realized_ask_pnl": float(res["realized_ask_pnl"]),
    }


def matched_random_baseline(
    *,
    frame: pd.DataFrame,
    tick_size: float,
    signal_rate_bid: float,
    signal_rate_ask: float,
    cfg,
    simulate_with_metrics,
    seeds: int = 20,
) -> dict[str, float]:
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)
    pnls: list[float] = []
    for seed in range(int(seeds)):
        rng = np.random.default_rng(seed)
        pull_bid = rng.random(len(bid)) < float(signal_rate_bid)
        pull_ask = rng.random(len(ask)) < float(signal_rate_ask)
        res = simulate_with_metrics(
            cfg,
            prob=None,
            ask_px=ask,
            bid_px=bid,
            tick_size=float(tick_size),
            threshold=1.1,
            force_pull_bid_mask=pull_bid,
            force_pull_ask_mask=pull_ask,
            seed=seed,
        )
        pnls.append(float(res["net_pnl"]))
    arr = np.asarray(pnls, dtype=np.float64)
    return {
        "random_mean": float(np.nanmean(arr)),
        "random_std": float(np.nanstd(arr)),
        "random_mean_plus_1std": float(np.nanmean(arr) + np.nanstd(arr)),
    }

