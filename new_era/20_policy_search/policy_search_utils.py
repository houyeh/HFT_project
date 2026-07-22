from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "20_policy_search"
ROBUSTNESS_DIR = ROOT / "robustness"
SEARCH_DIR = ROOT / "search"
EVALUATION_DIR = ROOT / "evaluation"
CACHE_DIR = ROOT / "cache"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V18_DIR = PROJECT_ROOT / "new_era" / "18_queue_aware"
V15_MODEL_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1"
)
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
for path in [PROJECT_ROOT, V16_DIR, V18_DIR, VENV_SITE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round2_audit_utils import add_market_buckets, load_test_frame, load_val_frame
from queue_stress_core import MAKER_FEE, MASK_SEED, MAX_INV, POINT_VALUE, StressConfig, build_action_table


F8_Z_FEATURES = [
    "obi_z",
    "queue_imbalance_z",
    "weighted_mid_deviation_z",
    "trade_intensity_short_z",
    "deep_obi_z",
    "queue_imbalance_change_z",
    "normalized_spread_z",
    "cancel_pressure_short_z",
]


@dataclass(frozen=True)
class EvalResult:
    net_pnl: float
    trade_count: int
    max_drawdown: float
    cancel_count: int
    bid_cancel_count: int
    ask_cancel_count: int
    bid_cancel_rate: float
    ask_cancel_rate: float
    any_cancel_rate: float


def ensure_dirs() -> None:
    for path in [ROOT, ROBUSTNESS_DIR, SEARCH_DIR, EVALUATION_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_labels_with_event_id() -> pd.DataFrame:
    labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    return labels


def _merge_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in F8_Z_FEATURES + ["queue_imbalance", "trade_intensity_short", "cancel_pressure_short"] if c not in frame.columns]
    if not missing:
        return frame
    labels = load_labels_with_event_id()
    extra_cols = [
        "event_id",
        "obi",
        "queue_imbalance",
        "weighted_mid_deviation",
        "trade_intensity_short",
        "deep_obi",
        "queue_imbalance_change",
        "normalized_spread",
        "cancel_pressure_short",
        *F8_Z_FEATURES,
    ]
    keep = [c for c in extra_cols if c in labels.columns]
    return frame.merge(labels[keep], on="event_id", how="left")


def threshold_for_target_rate(scores: np.ndarray, target_rate: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    sorted_scores = np.sort(scores)[::-1]
    kth = max(0, min(len(sorted_scores) - 1, int(np.ceil(float(target_rate) * len(sorted_scores))) - 1))
    return float(sorted_scores[kth])


def _load_015_model_objects():
    import torch

    from src.models.tcn import HLSFriendlyTCNClassifier

    selected = json.loads((V15_MODEL_DIR / "backtest_outputs" / "selected_strategy_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((V15_MODEL_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=list(cfg["model"]["channel_list"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        seq_length=int(cfg["window"]["window_size"]),
    ).to(device)
    model.load_state_dict(torch.load(V15_MODEL_DIR / "checkpoints" / "tcn_weights.pth", map_location=device, weights_only=True))
    model.eval()
    return model, selected, cfg, device


def ensure_015_val_predictions(batch_size: int = 4096) -> pd.DataFrame:
    out_path = CACHE_DIR / "015_val_predictions.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    ensure_dirs()
    import torch

    model, selected, cfg, device = _load_015_model_objects()
    frame = _merge_features(load_val_frame()).sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
    window_size = int(cfg["window"]["window_size"])
    out_frames = []
    for session_id, group in frame.groupby("session_id", sort=True):
        g = group.sort_values("local_row_id").reset_index(drop=True).copy()
        features = g[F8_Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
        probs = np.zeros((len(g), 3), dtype=np.float32)
        available = np.zeros(len(g), dtype=bool)
        if len(g) >= window_size:
            views = np.lib.stride_tricks.sliding_window_view(features, window_shape=window_size, axis=0)
            positions = np.arange(window_size - 1, len(g), dtype=np.int64)
            with torch.no_grad():
                for start in range(0, len(positions), int(batch_size)):
                    pos = positions[start : start + int(batch_size)]
                    window_start = pos - window_size + 1
                    x = torch.from_numpy(views[window_start].copy()).to(device)
                    logits = model(x)
                    probs[pos] = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
                    available[pos] = True

        g["prob_down_015"] = probs[:, 0]
        g["prob_stationary_015"] = probs[:, 1]
        g["prob_up_015"] = probs[:, 2]
        g["prediction_available_015"] = available
        edge = g["prob_up_015"].to_numpy(dtype=float) - g["prob_down_015"].to_numpy(dtype=float)
        spread = g["spread_ticks"].to_numpy(dtype=float)
        pthr = float(selected["prob_threshold"])
        ethr = float(selected["edge_threshold"])
        max_spread = float(selected["max_spread_ticks"])
        wide = spread > max_spread
        g["pull_bid_015_common"] = available & (wide | ((g["prob_down_015"].to_numpy(dtype=float) >= pthr) & (edge <= -ethr)))
        g["pull_ask_015_common"] = available & (wide | ((g["prob_up_015"].to_numpy(dtype=float) >= pthr) & (edge >= ethr)))
        g["score_bid_015"] = g["prob_down_015"]
        g["score_ask_015"] = g["prob_up_015"]
        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)
    out.to_parquet(out_path, index=False)
    return out


def load_policy_frame(split: str) -> pd.DataFrame:
    split = split.lower()
    if split == "val":
        frame = _merge_features(load_val_frame()).sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
        pred015 = ensure_015_val_predictions()
        cols015 = [
            "event_id",
            "prob_down_015",
            "prob_up_015",
            "prediction_available_015",
            "pull_bid_015_common",
            "pull_ask_015_common",
            "score_bid_015",
            "score_ask_015",
        ]
        frame = frame.merge(pred015[cols015], on="event_id", how="left")
    elif split == "test":
        frame = _merge_features(load_test_frame()).sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
        pred015 = pd.read_parquet(V18_DIR / "015_common_audit" / "015_common_predictions.parquet")
        cols015 = [
            "event_id",
            "prob_down_015",
            "prob_up_015",
            "prediction_available_015",
            "pull_bid_015_common",
            "pull_ask_015_common",
            "score_bid_015",
            "score_ask_015",
        ]
        frame = frame.merge(pred015[cols015], on="event_id", how="left")
    else:
        raise ValueError(f"Unknown split: {split}")

    fill_bool = {
        "prediction_available_015": False,
        "pull_bid_015_common": False,
        "pull_ask_015_common": False,
    }
    for col, val in fill_bool.items():
        frame[col] = frame[col].fillna(val).astype(bool)
    for col in ["prob_down_015", "prob_up_015", "score_bid_015", "score_ask_015"]:
        frame[col] = frame[col].fillna(0.0).astype(float)
    return add_market_buckets(frame).sort_values(["session_id", "local_row_id"]).reset_index(drop=True)


def build_actions_with_masks(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    actions = build_action_table(frame)
    seed = MASK_SEED + 11 if split == "val" else MASK_SEED
    rng = np.random.default_rng(seed)
    actions["neutral_u"] = rng.random(len(actions), dtype=np.float64)
    actions["good_u"] = rng.random(len(actions), dtype=np.float64)
    return actions


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(np.max(peak - arr))


def simulate_policy_fast(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    pull_bid: np.ndarray,
    pull_ask: np.ndarray,
    stress: StressConfig | None = None,
) -> EvalResult:
    stress = stress or StressConfig()
    pull_bid = np.asarray(pull_bid, dtype=bool)
    pull_ask = np.asarray(pull_ask, dtype=bool)
    if len(pull_bid) != len(frame) or len(pull_ask) != len(frame):
        raise ValueError("Pull-mask length mismatch.")

    total_pnl = 0.0
    total_trades = 0
    max_dd = 0.0
    neutral_cut = float(stress.neutral_fill_haircut)
    good_cut = float(stress.good_fill_haircut)
    extra_mult = max(float(stress.toxic_multiplier) - 1.0, 0.0)
    mid_by_index = frame["mid_price"].to_dict()

    for session_id, group in actions.groupby("session_id", sort=True):
        g = group.sort_values(["row_index", "side"]).reset_index(drop=True)
        session_frame = frame[frame["session_id"].astype(str) == str(session_id)].sort_values("local_row_id")
        final_mid = float(session_frame["mid_price"].iloc[-1])
        cash = 0.0
        fees = 0.0
        inventory = 0
        trades = 0
        equity: list[float] = []
        for row in g.itertuples(index=False):
            idx = int(row.row_index)
            side = str(row.side)
            canceled = pull_bid[idx] if side == "bid" else pull_ask[idx]
            if canceled:
                continue
            if float(row.neutral_u) > neutral_cut:
                continue
            future_ticks = float(row.future_mid_return_H_ticks)
            favorable = (side == "bid" and future_ticks > 0) or (side == "ask" and future_ticks < 0)
            if favorable and float(row.good_u) > good_cut:
                continue
            if side == "bid":
                if inventory >= MAX_INV:
                    continue
                cash -= float(row.price) * POINT_VALUE
                adverse = max(0.0, -future_ticks)
                inventory += 1
            else:
                if inventory <= -MAX_INV:
                    continue
                cash += float(row.price) * POINT_VALUE
                adverse = max(0.0, future_ticks)
                inventory -= 1
            if extra_mult:
                cash -= extra_mult * adverse * POINT_VALUE
            fees += MAKER_FEE
            trades += 1
            equity.append(cash - fees + inventory * float(mid_by_index[idx]) * POINT_VALUE)
        final_equity = cash - fees + inventory * final_mid * POINT_VALUE
        if equity:
            equity[-1] = final_equity
        total_pnl += final_equity
        total_trades += trades
        max_dd = max(max_dd, max_drawdown(equity))

    bid_count = int(pull_bid.sum())
    ask_count = int(pull_ask.sum())
    return EvalResult(
        net_pnl=float(total_pnl),
        trade_count=int(total_trades),
        max_drawdown=float(max_dd),
        cancel_count=int(bid_count + ask_count),
        bid_cancel_count=bid_count,
        ask_cancel_count=ask_count,
        bid_cancel_rate=float(bid_count / max(len(frame), 1)),
        ask_cancel_rate=float(ask_count / max(len(frame), 1)),
        any_cancel_rate=float(np.mean(pull_bid | pull_ask)) if len(frame) else 0.0,
    )


def policy_masks_from_spec(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spread = frame["spread_ticks"].to_numpy(dtype=float)
    wide = spread > float(policy.get("max_spread_ticks", np.inf))
    kind = policy["policy_type"]

    if kind == "spread_toxic":
        pull_bid = wide | (frame["bid_prob_toxic"].to_numpy(dtype=float) >= float(policy["bid_threshold"]))
        pull_ask = wide | (frame["ask_prob_toxic"].to_numpy(dtype=float) >= float(policy["ask_threshold"]))
    elif kind == "ensemble":
        mode = policy["ensemble_mode"]
        bid016 = frame["bid_prob_toxic"].to_numpy(dtype=float) >= float(policy["bid_threshold"])
        ask016 = frame["ask_prob_toxic"].to_numpy(dtype=float) >= float(policy["ask_threshold"])
        bid015 = frame["pull_bid_015_common"].to_numpy(dtype=bool)
        ask015 = frame["pull_ask_015_common"].to_numpy(dtype=bool)
        if mode == "or":
            pull_bid = wide | bid016 | bid015
            pull_ask = wide | ask016 | ask015
        elif mode == "and":
            pull_bid = wide | (bid016 & bid015)
            pull_ask = wide | (ask016 & ask015)
        elif mode == "weighted":
            w = float(policy["weight_016"])
            bid_score = w * frame["bid_prob_toxic"].to_numpy(dtype=float) + (1.0 - w) * frame["score_bid_015"].to_numpy(dtype=float)
            ask_score = w * frame["ask_prob_toxic"].to_numpy(dtype=float) + (1.0 - w) * frame["score_ask_015"].to_numpy(dtype=float)
            pull_bid = wide | (bid_score >= float(policy["bid_weighted_threshold"]))
            pull_ask = wide | (ask_score >= float(policy["ask_weighted_threshold"]))
        else:
            raise ValueError(f"Unknown ensemble_mode: {mode}")
    elif kind == "regime_threshold":
        regime_col = policy["regime_col"]
        pull_bid = wide.copy()
        pull_ask = wide.copy()
        for regime, thresholds in policy["regime_thresholds"].items():
            mask = frame[regime_col].astype(str).to_numpy() == str(regime)
            pull_bid |= mask & (frame["bid_prob_toxic"].to_numpy(dtype=float) >= float(thresholds["bid_threshold"]))
            pull_ask |= mask & (frame["ask_prob_toxic"].to_numpy(dtype=float) >= float(thresholds["ask_threshold"]))
    elif kind == "depth_proxy_diagnostic":
        risk = np.maximum(frame["bid_prob_toxic"].to_numpy(dtype=float), frame["ask_prob_toxic"].to_numpy(dtype=float))
        pull_bid = wide | (risk >= float(policy["risk_threshold"]))
        pull_ask = wide | (risk >= float(policy["risk_threshold"]))
    elif kind == "simple_rule":
        pull_bid, pull_ask = simple_rule_masks(frame, policy)
    else:
        raise ValueError(f"Unknown policy_type: {kind}")
    return np.asarray(pull_bid, dtype=bool), np.asarray(pull_ask, dtype=bool)


def simple_rule_masks(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spread = frame["spread_ticks"].to_numpy(dtype=float)
    wide = spread > float(policy.get("max_spread_ticks", 3.0))
    pull_bid = wide.copy()
    pull_ask = wide.copy()
    rule = policy["rule"]
    if rule == "spread_filter_only":
        return pull_bid, pull_ask
    if rule == "spread_filter_random_cancel":
        rng = np.random.default_rng(int(policy.get("seed", 20260525)))
        rate = float(policy.get("random_cancel_rate", 0.10))
        pull_bid |= rng.random(len(frame)) < rate
        pull_ask |= rng.random(len(frame)) < rate
    elif rule == "spread_filter_queue_imbalance":
        q = frame["queue_imbalance_z"].to_numpy(dtype=float)
        threshold = float(policy.get("queue_threshold", 1.0))
        pull_bid |= q < -threshold
        pull_ask |= q > threshold
    elif rule == "spread_filter_event_rate":
        rate = frame["event_rate_hz"].to_numpy(dtype=float)
        threshold = float(policy.get("event_rate_threshold", np.nanquantile(rate, 0.75)))
        pull_bid |= rate >= threshold
        pull_ask |= rate >= threshold
    else:
        raise ValueError(f"Unknown simple rule: {rule}")
    return np.asarray(pull_bid, dtype=bool), np.asarray(pull_ask, dtype=bool)


def eval_policy_row(frame: pd.DataFrame, actions: pd.DataFrame, policy: dict[str, Any], *, split: str, stress: StressConfig | None = None) -> dict[str, Any]:
    pull_bid, pull_ask = policy_masks_from_spec(frame, policy)
    result = simulate_policy_fast(frame, actions, pull_bid, pull_ask, stress)
    return {
        **policy,
        "split": split,
        "stress_name": policy.get("stress_name", "base"),
        "net_pnl": result.net_pnl,
        "trade_count": result.trade_count,
        "max_drawdown": result.max_drawdown,
        "cancel_count": result.cancel_count,
        "bid_cancel_count": result.bid_cancel_count,
        "ask_cancel_count": result.ask_cancel_count,
        "bid_cancel_rate": result.bid_cancel_rate,
        "ask_cancel_rate": result.ask_cancel_rate,
        "any_cancel_rate": result.any_cancel_rate,
    }


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def load_training_module():
    path = V16_DIR / "04_train_binary_model.py"
    spec = importlib.util.spec_from_file_location("train_binary_model_016_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_binary_model_016_v2"] = module
    spec.loader.exec_module(module)
    return module


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
