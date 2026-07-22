from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_era.models_playaround_15.backtest_val_sweep import _build_session_dataset, _max_drawdown
from new_era.models_playaround_15.config import (
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    LabelConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    WindowConfig,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.models_playaround_15.features import FeatureSet
from new_era.tcn_pipeline_utils import split_ordered_session_ids
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier

TASK_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = TASK_ROOT / "outputs"
FEATURES_DIR = OUTPUT_ROOT / "features"
POLICIES_DIR = OUTPUT_ROOT / "policies"
BACKTESTS_DIR = OUTPUT_ROOT / "backtests"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = OUTPUT_ROOT / "logs"

MODEL_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
)
MODEL_DIAG_DIR = MODEL_DIR / "eval_outputs" / "diagnosis"
MODEL_DIAG_V3_DIR = MODEL_DIR / "eval_outputs" / "diagnosis_v3"

NEW_ALL_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
NEW3_DIR = PROJECT_ROOT / "data" / "NEW3"
NEW_SYMBOL = "TXFB6"
NEW3_SYMBOL = "TXFA6"

MODEL_THRESHOLDS = [0.55, 0.60, 0.70]
ROLLING_WINDOWS = [50, 100, 200, 500, 1000]
EWMA_ALPHAS = [0.03, 0.05, 0.10]

REQUIRED_FEATURE_COLUMNS = [
    "dataset",
    "session_id",
    "event_idx",
    "timestamp",
    "mid",
    "spread_ticks",
    "rolling_return_ticks_50",
    "rolling_return_ticks_100",
    "rolling_return_ticks_200",
    "rolling_return_ticks_500",
    "rolling_return_ticks_1000",
    "rolling_vol_ticks_50",
    "rolling_vol_ticks_100",
    "rolling_vol_ticks_200",
    "rolling_vol_ticks_500",
    "rolling_vol_ticks_1000",
    "rolling_trend_score_50",
    "rolling_trend_score_100",
    "rolling_trend_score_200",
    "rolling_trend_score_500",
    "rolling_trend_score_1000",
    "rolling_spread_mean_100",
    "rolling_spread_mean_500",
    "imbalance",
    "rolling_imbalance_mean_100",
    "microprice_minus_mid_ticks",
    "ewma_velocity_alpha_0p03",
    "ewma_velocity_alpha_0p05",
    "ewma_velocity_alpha_0p10",
    "ewma_vol_alpha_0p03",
    "ewma_vol_alpha_0p05",
    "ewma_vol_alpha_0p10",
]


@dataclass
class DatasetBundle:
    dataset: str
    symbol: str
    tick_size: float
    session_ids: list[str]
    split_map: dict[str, str]
    event_df: pd.DataFrame


def ensure_output_dirs() -> None:
    for path in [
        OUTPUT_ROOT,
        FEATURES_DIR,
        POLICIES_DIR,
        BACKTESTS_DIR,
        COMPARISON_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_logger(log_name: str) -> logging.Logger:
    ensure_output_dirs()
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(LOGS_DIR / f"{log_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def alpha_name(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def load_experiment_cfg(model_dir: Path = MODEL_DIR) -> ExperimentConfig:
    payload = json.loads((model_dir / "experiment_config.json").read_text(encoding="utf-8"))
    return ExperimentConfig(
        data=DataConfig(
            data_dir=Path(payload["data"]["data_dir"]),
            symbols=tuple(payload["data"]["symbols"]),
            zscore_window=int(payload["data"]["zscore_window"]),
        ),
        window=WindowConfig(
            window_size=int(payload["window"]["window_size"]),
            horizon=int(payload["window"]["horizon"]),
            threshold_ticks=float(payload["window"]["threshold_ticks"]),
            sample_stride_train=int(payload["window"]["sample_stride_train"]),
        ),
        split=SplitConfig(
            train_session_frac=float(payload["split"]["train_session_frac"]),
            val_session_frac=float(payload["split"]["val_session_frac"]),
        ),
        features=FeatureSet(feature_count=int(payload["features"]["feature_count"])),
        labels=LabelConfig(
            label_type=str(payload["labels"]["label_type"]),
            adverse_threshold_ticks=float(payload["labels"]["adverse_threshold_ticks"]),
            taker_fee_twd=float(payload["labels"]["taker_fee_twd"]),
            point_value_twd_per_point=float(payload["labels"]["point_value_twd_per_point"]),
            exit_cost_ticks=float(payload["labels"]["exit_cost_ticks"]),
            margin_ticks=float(payload["labels"]["margin_ticks"]),
        ),
        model=ModelConfig(
            channel_list=tuple(int(x) for x in payload["model"]["channel_list"]),
            kernel_size=int(payload["model"]["kernel_size"]),
            dropout=float(payload["model"]["dropout"]),
        ),
        train=TrainingConfig(
            epochs=int(payload["train"]["epochs"]),
            batch_size=int(payload["train"]["batch_size"]),
            lr=float(payload["train"]["lr"]),
            seed=int(payload["train"]["seed"]),
            device=str(payload["train"]["device"]),
        ),
        backtest=BacktestConfig(
            symbol=str(payload["backtest"]["symbol"]),
            point_value_twd_per_point=float(payload["backtest"]["point_value_twd_per_point"]),
            maker_fee_twd=float(payload["backtest"]["maker_fee_twd"]),
            taker_fee_twd=float(payload["backtest"]["taker_fee_twd"]),
            max_inventory=int(payload["backtest"]["max_inventory"]),
            sample_stride=int(payload["backtest"]["sample_stride"]),
            allow_active_alpha=bool(payload["backtest"]["allow_active_alpha"]),
        ),
    )


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    if cfg.train.device == "cpu":
        return torch.device("cpu")
    if cfg.train.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(cfg: ExperimentConfig, device: torch.device, model_dir: Path = MODEL_DIR) -> torch.nn.Module:
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    ckpt_path = model_dir / "checkpoints" / "tcn_weights.pth"
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def detect_symbol_from_data_dir(data_dir: Path, preferred_symbol: str) -> str:
    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir)))
    symbol_counts: dict[str, int] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            next(handle, None)
            for row in handle:
                parts = [item.strip() for item in row.split(",")]
                if len(parts) < 34:
                    continue
                symbol = parts[3]
                if not symbol:
                    continue
                try:
                    best_bid = float(parts[21])
                    best_ask = float(parts[22])
                except ValueError:
                    continue
                if best_bid <= 0 or best_ask <= 0:
                    continue
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    if preferred_symbol in symbol_counts:
        return preferred_symbol
    if not symbol_counts:
        raise RuntimeError(f"No valid symbol detected under {data_dir}")
    return sorted(symbol_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def build_split_map(session_ids: list[str], cfg: ExperimentConfig) -> dict[str, str]:
    ordered_sessions = sorted(session_ids)
    train_s, val_s, test_s = split_ordered_session_ids(
        ordered_sessions,
        train_frac=cfg.split.train_session_frac,
        val_frac=cfg.split.val_session_frac,
    )
    split_map: dict[str, str] = {}
    for sid in train_s:
        split_map[sid] = "train"
    for sid in val_s:
        split_map[sid] = "val"
    for sid in test_s:
        split_map[sid] = "test"
    for sid in ordered_sessions:
        split_map.setdefault(sid, "test")
    return split_map


def build_session_datasets(
    cfg: ExperimentConfig,
    symbol_lob: pd.DataFrame,
    symbol_trades: pd.DataFrame,
    symbol: str,
    tick_size: float,
    session_ids: list[str],
) -> list[tuple[str, Any]]:
    datasets: list[tuple[str, Any]] = []
    min_required = cfg.window.window_size + cfg.window.horizon
    for sid in session_ids:
        lob_df = symbol_lob[symbol_lob["session_id"] == sid]
        trd_df = symbol_trades[symbol_trades["session_id"] == sid]
        if len(lob_df) < min_required:
            continue
        datasets.append(
            (
                sid,
                _build_session_dataset(
                    cfg,
                    lob_df,
                    trd_df,
                    symbol,
                    tick_size,
                    sample_stride=cfg.backtest.sample_stride,
                ),
            )
        )
    return datasets


@torch.no_grad()
def collect_predictions_extended(
    model: torch.nn.Module,
    datasets: list[tuple[str, Any]],
    device: torch.device,
    num_classes: int,
) -> dict[str, np.ndarray]:
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    ask_all: list[np.ndarray] = []
    bid_all: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []
    sess_all: list[np.ndarray] = []
    mid_all: list[np.ndarray] = []
    spread_all: list[np.ndarray] = []
    bid_vol_all: list[np.ndarray] = []
    ask_vol_all: list[np.ndarray] = []
    qimb_all: list[np.ndarray] = []
    rvol_all: list[np.ndarray] = []

    for session_id, dataset in datasets:
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        for batch in loader:
            batch_x = batch["x"].transpose(1, 2).to(device)
            probs = torch.softmax(model(batch_x), dim=1).detach().cpu().numpy()
            n = int(probs.shape[0])
            probs_all.append(probs)
            y_all.append(batch["y"].cpu().numpy())
            ask = batch["ask_px_1"].cpu().numpy()
            bid = batch["bid_px_1"].cpu().numpy()
            ask_all.append(ask)
            bid_all.append(bid)
            ts_all.append(batch["timestamp"].cpu().numpy() if "timestamp" in batch else np.arange(n))
            sess_all.append(np.array([session_id] * n, dtype=object))
            mid_all.append(batch["current_mid"].cpu().numpy() if "current_mid" in batch else (ask + bid) / 2.0)
            spread_all.append(batch["spread_ticks"].cpu().numpy() if "spread_ticks" in batch else (ask - bid))
            bid_vol_all.append(batch["bid_vol_1"].cpu().numpy() if "bid_vol_1" in batch else np.full(n, np.nan))
            ask_vol_all.append(batch["ask_vol_1"].cpu().numpy() if "ask_vol_1" in batch else np.full(n, np.nan))
            qimb_all.append(batch["queue_imbalance"].cpu().numpy() if "queue_imbalance" in batch else np.full(n, np.nan))
            rvol_all.append(batch["rolling_volatility"].cpu().numpy() if "rolling_volatility" in batch else np.full(n, np.nan))

    if not probs_all:
        empty_f = np.array([], dtype=np.float64)
        return {
            "probs": np.zeros((0, num_classes), dtype=np.float64),
            "y": np.zeros((0,), dtype=np.int64),
            "ask": empty_f,
            "bid": empty_f,
            "timestamp": np.array([], dtype=np.int64),
            "session_id": np.array([], dtype=object),
            "current_mid": empty_f,
            "spread_ticks_raw": empty_f,
            "bid_vol_1": empty_f,
            "ask_vol_1": empty_f,
            "queue_imbalance": empty_f,
            "rolling_volatility_raw": empty_f,
        }

    return {
        "probs": np.concatenate(probs_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "ask": np.concatenate(ask_all, axis=0),
        "bid": np.concatenate(bid_all, axis=0),
        "timestamp": np.concatenate(ts_all, axis=0).astype(np.int64),
        "session_id": np.concatenate(sess_all, axis=0),
        "current_mid": np.concatenate(mid_all, axis=0),
        "spread_ticks_raw": np.concatenate(spread_all, axis=0),
        "bid_vol_1": np.concatenate(bid_vol_all, axis=0),
        "ask_vol_1": np.concatenate(ask_vol_all, axis=0),
        "queue_imbalance": np.concatenate(qimb_all, axis=0),
        "rolling_volatility_raw": np.concatenate(rvol_all, axis=0),
    }


def build_event_frame(
    pred_pack: dict[str, np.ndarray],
    *,
    dataset: str,
    tick_size: float,
    split_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if pred_pack["probs"].size == 0:
        return pd.DataFrame(
            columns=[
                "dataset",
                "session_id",
                "split",
                "event_idx",
                "timestamp",
                "ask_px_1",
                "bid_px_1",
                "mid",
                "spread_ticks",
                "prob_down",
                "prob_stationary",
                "prob_up",
                "bid_vol_1",
                "ask_vol_1",
                "queue_imbalance",
                "rolling_volatility_raw",
                "tick_size",
            ]
        )

    df = pd.DataFrame(
        {
            "session_id": pred_pack["session_id"],
            "timestamp": pred_pack["timestamp"],
            "ask_px_1": pred_pack["ask"],
            "bid_px_1": pred_pack["bid"],
            "prob_down": pred_pack["probs"][:, 0],
            "prob_stationary": pred_pack["probs"][:, 1],
            "prob_up": pred_pack["probs"][:, 2],
            "bid_vol_1": pred_pack["bid_vol_1"],
            "ask_vol_1": pred_pack["ask_vol_1"],
            "queue_imbalance": pred_pack["queue_imbalance"],
            "rolling_volatility_raw": pred_pack["rolling_volatility_raw"],
        }
    )
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    df["dataset"] = dataset
    if split_map is None:
        df["split"] = "full"
    else:
        df["split"] = df["session_id"].map(split_map).fillna("unknown")
    df["event_idx"] = df.groupby("session_id").cumcount().astype(np.int64)
    df["mid"] = (df["ask_px_1"] + df["bid_px_1"]) / 2.0
    safe_tick = max(float(tick_size), 1e-9)
    df["spread_ticks"] = (df["ask_px_1"] - df["bid_px_1"]) / safe_tick
    df["tick_size"] = float(tick_size)
    return df


def build_regime_features(
    df: pd.DataFrame,
    *,
    tick_size: float,
    windows: list[int] | None = None,
    ewma_alphas: list[float] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = list(ROLLING_WINDOWS)
    if ewma_alphas is None:
        ewma_alphas = list(EWMA_ALPHAS)

    out = df.copy()
    safe_tick = max(float(tick_size), 1e-9)

    if out.empty:
        for col in REQUIRED_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    grouped = out.groupby("session_id", sort=False)
    out["mid_diff_ticks_1"] = grouped["mid"].diff(1) / safe_tick

    for w in windows:
        ret_col = f"rolling_return_ticks_{w}"
        abs_ret_col = f"rolling_abs_return_ticks_{w}"
        vol_col = f"rolling_vol_ticks_{w}"
        trend_col = f"rolling_trend_score_{w}"

        out[ret_col] = grouped["mid"].diff(w) / safe_tick
        out[abs_ret_col] = out[ret_col].abs()
        out[vol_col] = out.groupby("session_id", sort=False)["mid_diff_ticks_1"].transform(
            lambda s, win=w: s.rolling(win, min_periods=win).std()
        )
        out[trend_col] = out[abs_ret_col] / np.maximum(out[vol_col].to_numpy(dtype=np.float64), 1e-6)

    out["rolling_spread_mean_100"] = grouped["spread_ticks"].transform(
        lambda s: s.rolling(100, min_periods=1).mean()
    )
    out["rolling_spread_mean_500"] = grouped["spread_ticks"].transform(
        lambda s: s.rolling(500, min_periods=1).mean()
    )
    out["spread_ge_1tick"] = (out["spread_ticks"] >= 1.0).astype(float)
    out["spread_ge_2tick"] = (out["spread_ticks"] >= 2.0).astype(float)

    if {"bid_vol_1", "ask_vol_1"}.issubset(out.columns):
        denom = np.maximum(out["bid_vol_1"].to_numpy(dtype=np.float64) + out["ask_vol_1"].to_numpy(dtype=np.float64), 1e-9)
        out["imbalance"] = (
            out["bid_vol_1"].to_numpy(dtype=np.float64) - out["ask_vol_1"].to_numpy(dtype=np.float64)
        ) / denom
        out["rolling_imbalance_mean_100"] = out.groupby("session_id", sort=False)["imbalance"].transform(
            lambda s: s.rolling(100, min_periods=1).mean()
        )
        microprice = (
            out["ask_px_1"].to_numpy(dtype=np.float64) * out["bid_vol_1"].to_numpy(dtype=np.float64)
            + out["bid_px_1"].to_numpy(dtype=np.float64) * out["ask_vol_1"].to_numpy(dtype=np.float64)
        ) / denom
        out["microprice_minus_mid_ticks"] = (microprice - out["mid"].to_numpy(dtype=np.float64)) / safe_tick
    else:
        out["imbalance"] = np.nan
        out["rolling_imbalance_mean_100"] = np.nan
        out["microprice_minus_mid_ticks"] = np.nan

    for alpha in ewma_alphas:
        suffix = alpha_name(alpha)
        ewma_mid_col = f"ewma_mid_alpha_{suffix}"
        vel_col = f"ewma_velocity_alpha_{suffix}"
        residual_col = f"ewma_residual_alpha_{suffix}"
        abs_ret_col = f"ewma_abs_return_alpha_{suffix}"
        vol_col = f"ewma_vol_alpha_{suffix}"

        out[ewma_mid_col] = grouped["mid"].transform(lambda s, a=alpha: s.ewm(alpha=a, adjust=False).mean())
        out[vel_col] = out.groupby("session_id", sort=False)[ewma_mid_col].diff(1) / safe_tick
        out[residual_col] = (out["mid"] - out[ewma_mid_col]) / safe_tick
        out[abs_ret_col] = grouped["mid_diff_ticks_1"].transform(
            lambda s, a=alpha: s.abs().ewm(alpha=a, adjust=False).mean()
        )
        out[vol_col] = grouped["mid_diff_ticks_1"].transform(
            lambda s, a=alpha: s.ewm(alpha=a, adjust=False).std(bias=False)
        )

    for col in REQUIRED_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def ordered_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset",
        "session_id",
        "split",
        "event_idx",
        "timestamp",
        "tick_size",
        "ask_px_1",
        "bid_px_1",
        "mid",
        "spread_ticks",
        "spread_ge_1tick",
        "spread_ge_2tick",
        "prob_down",
        "prob_stationary",
        "prob_up",
        "bid_vol_1",
        "ask_vol_1",
        "queue_imbalance",
        "rolling_volatility_raw",
    ]
    preferred.extend(REQUIRED_FEATURE_COLUMNS)
    seen: set[str] = set()
    ordered: list[str] = []
    for col in preferred:
        if col in df.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in df.columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def serialize_params(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def extract_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = df[["prob_down", "prob_stationary", "prob_up"]].to_numpy(dtype=np.float64)
    ask = df["ask_px_1"].to_numpy(dtype=np.float64)
    bid = df["bid_px_1"].to_numpy(dtype=np.float64)
    return probs, ask, bid


def safe_quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def simulate_with_metrics(
    cfg: ExperimentConfig,
    prob: np.ndarray | None,
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    tick_size: float,
    threshold: float,
    random_cancel_rate: float = 0.0,
    prediction_delay_events: int = 0,
    cancel_delay_events: int = 0,
    force_pull_bid_mask: np.ndarray | None = None,
    force_pull_ask_mask: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, float | int]:
    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    max_inv = int(cfg.backtest.max_inventory)
    horizon = int(cfg.window.horizon)

    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0

    signal_count = 0
    total_opportunities = 0

    missed_fill_count = 0
    missed_profitable_fill_count = 0
    missed_profitable_pnl = 0.0
    missed_toxic_fill_count = 0
    missed_toxic_pnl_avoided = 0.0

    pnl_path: list[float] = []
    effective_pull_bid_state = False
    effective_pull_ask_state = False
    pending_bid_delay = 0
    pending_ask_delay = 0
    delayed_fill_count = 0
    delayed_bid_fill_count = 0
    delayed_ask_fill_count = 0
    delayed_toxic_fill_count = 0
    delayed_toxic_loss = 0.0

    open_trades: list[float] = []
    realized_bid_pnl = 0.0
    realized_ask_pnl = 0.0
    bid_fill_count = 0
    ask_fill_count = 0
    closed_trade_pnls: list[float] = []

    mid = (ask_px + bid_px) / 2.0
    rng = np.random.default_rng(seed)

    def process_fill(fill_px: float, side: str) -> None:
        nonlocal inventory, gross_cash, fees_paid, maker_trades
        nonlocal realized_bid_pnl, realized_ask_pnl, bid_fill_count, ask_fill_count

        gross_cash += fill_px * point_value if side == "ask" else -fill_px * point_value
        fees_paid += maker_fee
        maker_trades += 1

        if side == "bid":
            bid_fill_count += 1
            if inventory < 0:
                open_px = open_trades.pop(0)
                pnl_closed = (open_px - fill_px) * point_value - 2 * maker_fee
                realized_ask_pnl += pnl_closed
                closed_trade_pnls.append(pnl_closed)
            else:
                open_trades.append(fill_px)
            inventory += 1
        else:
            ask_fill_count += 1
            if inventory > 0:
                open_px = open_trades.pop(0)
                pnl_closed = (fill_px - open_px) * point_value - 2 * maker_fee
                realized_bid_pnl += pnl_closed
                closed_trade_pnls.append(pnl_closed)
            else:
                open_trades.append(fill_px)
            inventory -= 1

    n_events = min(len(ask_px), len(bid_px))
    if prob is not None:
        n_events = min(n_events, len(prob))
    if force_pull_bid_mask is not None:
        n_events = min(n_events, len(force_pull_bid_mask))
    if force_pull_ask_mask is not None:
        n_events = min(n_events, len(force_pull_ask_mask))
    if active_mask is not None:
        n_events = min(n_events, len(active_mask))

    for i in range(max(0, n_events - 1)):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])
        mid_now = float(mid[i])

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue
        if ask <= bid:
            pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)
            continue

        score_idx = i - int(prediction_delay_events)
        if prob is not None and score_idx >= 0:
            p_down = float(prob[score_idx, 0])
            p_up = float(prob[score_idx, 2])
        else:
            p_down = 0.0
            p_up = 0.0

        if random_cancel_rate > 0:
            raw_pull_bid = bool(rng.random() < random_cancel_rate)
            raw_pull_ask = bool(rng.random() < random_cancel_rate)
        else:
            raw_pull_bid = p_down >= threshold if prob is not None else False
            raw_pull_ask = p_up >= threshold if prob is not None else False

        if cancel_delay_events <= 0:
            effective_pull_bid = bool(raw_pull_bid)
            effective_pull_ask = bool(raw_pull_ask)
            effective_pull_bid_state = effective_pull_bid
            effective_pull_ask_state = effective_pull_ask
            pending_bid_delay = 0
            pending_ask_delay = 0
        else:
            if raw_pull_bid:
                if effective_pull_bid_state:
                    pass
                elif pending_bid_delay <= 0:
                    pending_bid_delay = int(cancel_delay_events)
                else:
                    pending_bid_delay -= 1
                    if pending_bid_delay <= 0:
                        effective_pull_bid_state = True
            else:
                effective_pull_bid_state = False
                pending_bid_delay = 0

            if raw_pull_ask:
                if effective_pull_ask_state:
                    pass
                elif pending_ask_delay <= 0:
                    pending_ask_delay = int(cancel_delay_events)
                else:
                    pending_ask_delay -= 1
                    if pending_ask_delay <= 0:
                        effective_pull_ask_state = True
            else:
                effective_pull_ask_state = False
                pending_ask_delay = 0

            effective_pull_bid = bool(effective_pull_bid_state)
            effective_pull_ask = bool(effective_pull_ask_state)

        if active_mask is not None and not bool(active_mask[i]):
            effective_pull_bid = True
            effective_pull_ask = True
        if force_pull_bid_mask is not None and bool(force_pull_bid_mask[i]):
            effective_pull_bid = True
        if force_pull_ask_mask is not None and bool(force_pull_ask_mask[i]):
            effective_pull_ask = True

        if effective_pull_bid:
            signal_count += 1
        if effective_pull_ask:
            signal_count += 1
        total_opportunities += 2

        baseline_bid_fill = next_ask <= bid
        baseline_ask_fill = next_bid >= ask
        fut_mid = float(mid[i + horizon]) if i + horizon < len(mid) else float(mid[-1])

        if baseline_bid_fill and effective_pull_bid:
            missed_fill_count += 1
            markout = (fut_mid - bid) * point_value
            if markout > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += -markout
        elif baseline_bid_fill and raw_pull_bid and not effective_pull_bid:
            delayed_fill_count += 1
            delayed_bid_fill_count += 1
            delayed_markout = (fut_mid - bid) * point_value
            if delayed_markout < 0:
                delayed_toxic_fill_count += 1
                delayed_toxic_loss += -delayed_markout

        if baseline_ask_fill and effective_pull_ask:
            missed_fill_count += 1
            markout = (ask - fut_mid) * point_value
            if markout > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += -markout
        elif baseline_ask_fill and raw_pull_ask and not effective_pull_ask:
            delayed_fill_count += 1
            delayed_ask_fill_count += 1
            delayed_markout = (ask - fut_mid) * point_value
            if delayed_markout < 0:
                delayed_toxic_fill_count += 1
                delayed_toxic_loss += -delayed_markout

        quote_bid = bid if not effective_pull_bid and inventory < max_inv else None
        quote_ask = ask if not effective_pull_ask and inventory > -max_inv else None

        if quote_bid is not None and next_ask <= quote_bid:
            process_fill(quote_bid, "bid")
        if quote_ask is not None and next_bid >= quote_ask:
            process_fill(quote_ask, "ask")

        pnl_path.append(gross_cash - fees_paid + inventory * mid_now * point_value)

    final_mid = float(mid[-1]) if len(mid) else 0.0
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid

    pnl_arr = np.asarray(pnl_path, dtype=np.float64)
    mdd = _max_drawdown(pnl_arr)
    closed_arr = np.asarray(closed_trade_pnls, dtype=np.float64)
    median_pnl_per_fill = float(np.median(closed_arr)) if closed_arr.size > 0 else 0.0
    win_rate = float((closed_arr > 0).mean()) if closed_arr.size > 0 else 0.0

    return {
        "threshold": float(threshold),
        "signal_rate": signal_count / total_opportunities if total_opportunities else 0.0,
        "cancel_count": int(signal_count),
        "fill_rate": maker_trades / total_opportunities if total_opportunities else 0.0,
        "trade_count": int(maker_trades),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "fee": float(fees_paid),
        "avg_pnl_per_fill": float(net_pnl / maker_trades) if maker_trades else 0.0,
        "median_pnl_per_fill": median_pnl_per_fill,
        "win_rate": win_rate,
        "toxic_fill_count": int(missed_toxic_fill_count),
        "toxic_fill_avoided_count": int(missed_toxic_fill_count),
        "toxic_fill_avoided": int(missed_toxic_fill_count),
        "missed_fill_count": int(missed_fill_count),
        "missed_profitable_fill_count": int(missed_profitable_fill_count),
        "missed_profitable_pnl": float(missed_profitable_pnl),
        "missed_toxic_fill_count": int(missed_toxic_fill_count),
        "missed_toxic_pnl_avoided": float(missed_toxic_pnl_avoided),
        "net_cancel_value": float(missed_toxic_pnl_avoided - missed_profitable_pnl),
        "max_drawdown": float(mdd),
        "bid_fill_count": int(bid_fill_count),
        "ask_fill_count": int(ask_fill_count),
        "realized_bid_pnl": float(realized_bid_pnl),
        "realized_ask_pnl": float(realized_ask_pnl),
        "prediction_delay_events": int(prediction_delay_events),
        "cancel_delay_events": int(cancel_delay_events),
        "delayed_fill_count": int(delayed_fill_count),
        "delayed_bid_fill_count": int(delayed_bid_fill_count),
        "delayed_ask_fill_count": int(delayed_ask_fill_count),
        "delayed_toxic_fill_count": int(delayed_toxic_fill_count),
        "delayed_toxic_loss": float(delayed_toxic_loss),
    }


def evaluate_policy(
    cfg: ExperimentConfig,
    *,
    probs: np.ndarray,
    ask: np.ndarray,
    bid: np.ndarray,
    tick_size: float,
    threshold: float,
    bid_active_mask: np.ndarray,
    ask_active_mask: np.ndarray,
) -> dict[str, float | int]:
    bid_active = np.asarray(bid_active_mask, dtype=bool)
    ask_active = np.asarray(ask_active_mask, dtype=bool)
    n = min(len(probs), len(ask), len(bid), len(bid_active), len(ask_active))
    probs = probs[:n]
    ask = ask[:n]
    bid = bid[:n]
    bid_active = bid_active[:n]
    ask_active = ask_active[:n]
    force_bid = ~bid_active
    force_ask = ~ask_active

    baseline = simulate_with_metrics(
        cfg,
        None,
        ask,
        bid,
        tick_size,
        threshold=1.1,
        force_pull_bid_mask=force_bid,
        force_pull_ask_mask=force_ask,
    )
    model = simulate_with_metrics(
        cfg,
        probs,
        ask,
        bid,
        tick_size,
        threshold=threshold,
        force_pull_bid_mask=force_bid,
        force_pull_ask_mask=force_ask,
    )

    side_active_rate = float((bid_active.astype(np.float64) + ask_active.astype(np.float64)).mean() / 2.0)
    event_active_rate = float((bid_active | ask_active).mean())
    return {
        **model,
        "baseline_net_pnl": float(baseline["net_pnl"]),
        "improvement": float(model["net_pnl"] - baseline["net_pnl"]),
        "active_event_rate": side_active_rate,
        "quote_event_rate": event_active_rate,
    }
