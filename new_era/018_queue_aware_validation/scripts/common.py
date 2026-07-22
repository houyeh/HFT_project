from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob

TASK_ROOT = PROJECT_ROOT / "new_era" / "018_queue_aware_validation"
SCRIPTS_DIR = TASK_ROOT / "scripts"
OUTPUT_ROOT = TASK_ROOT / "outputs"
AUDIT_DIR = OUTPUT_ROOT / "audit"
BACKTEST_DIR = OUTPUT_ROOT / "backtests"
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

FEATURE_NEW = (
    PROJECT_ROOT / "new_era" / "017_regime_auto_gate" / "outputs" / "features" / "regime_features_NEW.csv"
)
FEATURE_NEW3 = (
    PROJECT_ROOT / "new_era" / "017_regime_auto_gate" / "outputs" / "features" / "regime_features_NEW3.csv"
)

DATA_NEW = PROJECT_ROOT / "data" / "NEW_ALL"
DATA_NEW3 = PROJECT_ROOT / "data" / "NEW3"
SYMBOL_NEW = "TXFB6"
SYMBOL_NEW3 = "TXFA6"

POINT_VALUE = 200.0
MAKER_FEE = 15.0
TAKER_FEE = 50.0
MAX_INVENTORY = 3
HORIZON = 5

RANDOM_REFERENCE = {
    "NEW_055_mean_plus_1std": 42362.1 + 38111.08740366771,
    "NEW_070_mean_plus_1std": 53209.4 + 19569.30332536138,
}


@dataclass(frozen=True)
class PolicySpec:
    dataset: str
    policy_name: str
    model_threshold: float
    gate_type: str
    gate_params: dict[str, Any]
    notes: str


def ensure_dirs() -> None:
    for path in [
        TASK_ROOT,
        OUTPUT_ROOT,
        AUDIT_DIR,
        BACKTEST_DIR,
        COMPARISON_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def detect_symbol_from_data_dir(data_dir: Path, preferred_symbol: str) -> str:
    csv_paths = sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir)))
    counts: dict[str, int] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            next(handle, None)
            for row in handle:
                parts = [x.strip() for x in row.split(",")]
                if len(parts) < 34:
                    continue
                symbol = parts[3]
                if not symbol:
                    continue
                try:
                    bid = float(parts[21])
                    ask = float(parts[22])
                except ValueError:
                    continue
                if bid <= 0 or ask <= 0:
                    continue
                counts[symbol] = counts.get(symbol, 0) + 1
    if preferred_symbol in counts:
        return preferred_symbol
    if not counts:
        raise RuntimeError(f"No valid symbol found under {data_dir}")
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def load_feature_frame(dataset: str) -> pd.DataFrame:
    path = FEATURE_NEW if dataset == "NEW" else FEATURE_NEW3
    if not path.exists():
        raise FileNotFoundError(
            f"Required feature file missing: {path}. Run new_era/017_regime_auto_gate/scripts/build_regime_features.py first."
        )
    return pd.read_csv(path, low_memory=False, dtype={"session_id": "string", "split": "string"})


def load_trade_frame(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    data_dir = DATA_NEW if dataset == "NEW" else DATA_NEW3
    preferred = SYMBOL_NEW if dataset == "NEW" else SYMBOL_NEW3
    symbol = detect_symbol_from_data_dir(data_dir, preferred)
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    lob_df = lob_frames[symbol]
    trade_df = trade_frames.get(
        symbol,
        pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
    )
    tick_size = float(infer_tick_size_from_lob(lob_df))
    return lob_df, trade_df, tick_size, symbol


def build_split_map_for_new(lob_df: pd.DataFrame) -> dict[str, str]:
    sessions = list(lob_df["session_id"].drop_duplicates())
    ordered = sorted(str(s) for s in sessions)
    total = len(ordered)
    if total <= 1:
        train_s, val_s, test_s = ordered, [], []
    else:
        train_end = max(1, int(total * 0.70))
        val_end = max(train_end + 1, int(total * (0.70 + 0.15)))
        val_end = min(val_end, total - 1)
        train_s = ordered[:train_end]
        val_s = ordered[train_end:val_end]
        test_s = ordered[val_end:]
    out: dict[str, str] = {}
    for sid in train_s:
        out[str(sid)] = "train"
    for sid in val_s:
        out[str(sid)] = "val"
    for sid in test_s:
        out[str(sid)] = "test"
    return out


def load_static_gate_params() -> dict[tuple[str, str], dict[str, Any]]:
    static_path = (
        PROJECT_ROOT / "new_era" / "017_regime_auto_gate" / "outputs" / "policies" / "static_gate_summary.csv"
    )
    if not static_path.exists():
        raise FileNotFoundError(
            f"{static_path} missing. Run new_era/017_regime_auto_gate/scripts/evaluate_static_gates.py first."
        )
    df = pd.read_csv(static_path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = (str(row["dataset"]), str(row["policy_name"]))
        try:
            params = json.loads(str(row["gate_params"]))
        except json.JSONDecodeError:
            params = {}
        out[key] = {
            "gate_type": str(row["gate_type"]),
            "gate_params": params,
            "model_threshold": float(row["model_threshold"]) if pd.notna(row["model_threshold"]) else np.nan,
            "net_pnl": float(row["net_pnl"]) if pd.notna(row["net_pnl"]) else np.nan,
            "active_event_rate": float(row["active_event_rate"]) if pd.notna(row["active_event_rate"]) else np.nan,
        }
    return out


def build_policy_specs(static_map: dict[tuple[str, str], dict[str, Any]]) -> list[PolicySpec]:
    specs: list[PolicySpec] = []

    def add(dataset: str, policy_name: str, thr: float, gate_type: str, params: dict[str, Any], notes: str) -> None:
        specs.append(
            PolicySpec(
                dataset=dataset,
                policy_name=policy_name,
                model_threshold=float(thr),
                gate_type=gate_type,
                gate_params=params,
                notes=notes,
            )
        )

    new_spread = static_map[("NEW", "always_spread_filter")]
    new_combined = static_map[("NEW", "always_combined_filter")]
    new3_combined = static_map[("NEW3", "always_combined_filter")]

    add("NEW", "no_gate_tcn_055", 0.55, "no_gate", {}, "Policy A")
    add("NEW", "spread_filter_tcn_055", 0.55, "spread_filter", new_spread["gate_params"], "Policy B")
    add("NEW", "combined_filter_tcn_055", 0.55, "combined_filter", new_combined["gate_params"], "Policy C")
    add("NEW3", "combined_filter_tcn_055", 0.55, "combined_filter", new3_combined["gate_params"], "Policy D")
    add("NEW", "no_gate_tcn_070", 0.70, "no_gate", {}, "Policy E")
    add("NEW", "val_selected_tcn_070", 0.70, "no_gate", {}, "Policy F; val-selected threshold from prior run.")
    add("NEW3", "no_gate_tcn_055", 0.55, "no_gate", {}, "Policy A on NEW3")
    return specs


def build_gate_masks(df: pd.DataFrame, spec: PolicySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    active = np.ones(n, dtype=bool)
    force_bid = np.zeros(n, dtype=bool)
    force_ask = np.zeros(n, dtype=bool)
    spread = df["spread_ticks"].to_numpy(dtype=np.float64)

    if spec.gate_type == "spread_filter":
        spread_min = float(spec.gate_params.get("spread_ticks_min", 1.0))
        active = np.isfinite(spread) & (spread >= spread_min)
    elif spec.gate_type == "combined_filter":
        spread_min = float(spec.gate_params.get("spread_ticks_min", 2.0))
        vol_thr = float(spec.gate_params.get("vol_threshold", np.nan))
        trend_ticks = float(spec.gate_params.get("trend_ticks", 20.0))
        ret = df["rolling_return_ticks_200"].to_numpy(dtype=np.float64)
        vol = df["rolling_vol_ticks_200"].to_numpy(dtype=np.float64)
        active = np.isfinite(spread) & np.isfinite(vol) & (spread >= spread_min) & (vol <= vol_thr)
        force_bid = np.isfinite(ret) & (ret <= -trend_ticks)
        force_ask = np.isfinite(ret) & (ret >= trend_ticks)
    return active, force_bid, force_ask


def compute_future_trade_volume(df: pd.DataFrame, trade_df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or trade_df.empty:
        return out

    grouped_events = df.groupby("session_id", sort=False)
    trade_grouped = trade_df.groupby("session_id", sort=False)
    for sid, idx in grouped_events.indices.items():
        indices = np.asarray(idx, dtype=np.int64)
        ts = df.iloc[indices]["timestamp"].to_numpy(dtype=np.int64)
        if sid not in trade_grouped.indices:
            continue
        tidx = np.asarray(trade_grouped.indices[sid], dtype=np.int64)
        t = trade_df.iloc[tidx]
        t_ts = t["timestamp"].to_numpy(dtype=np.int64)
        t_vol = t["trade_volume"].to_numpy(dtype=np.float64)
        map_idx = np.searchsorted(ts, t_ts, side="right") - 1
        good = (map_idx >= 0) & (map_idx < len(indices))
        if not np.any(good):
            continue
        np.add.at(out, indices[map_idx[good]], t_vol[good])
    return out


def future_sum(volume: np.ndarray, window: int) -> np.ndarray:
    n = len(volume)
    csum = np.concatenate([[0.0], np.cumsum(volume, dtype=np.float64)])
    out = np.zeros(n, dtype=np.float64)
    if window <= 0:
        return out
    end = np.minimum(np.arange(n) + window + 1, n)
    start = np.arange(n) + 1
    valid = start < n
    out[valid] = csum[end[valid]] - csum[start[valid]]
    return out
