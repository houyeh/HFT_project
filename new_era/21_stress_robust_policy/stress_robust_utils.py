from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "21_stress_robust_policy"
DIAGNOSTICS_DIR = ROOT / "diagnostics"
ORACLE_DIR = ROOT / "oracle"
SEARCH_DIR = ROOT / "search"
EVALUATION_DIR = ROOT / "evaluation"
HLS_DIR = ROOT / "hls"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V18_DIR = PROJECT_ROOT / "new_era" / "18_queue_aware"
V20_DIR = PROJECT_ROOT / "new_era" / "20_policy_search"
for path in [V16_DIR, V18_DIR, V20_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from policy_search_utils import (  # noqa: E402
    MAKER_FEE,
    MASK_SEED,
    MAX_INV,
    POINT_VALUE,
    StressConfig,
    build_actions_with_masks,
    json_sanitize,
    load_policy_frame,
    policy_masks_from_spec,
    threshold_for_target_rate,
)


STRESSES = {
    "base": StressConfig(neutral_fill_haircut=1.0, toxic_multiplier=1.0, cancel_delay_ticks=0),
    "mild": StressConfig(neutral_fill_haircut=0.75, toxic_multiplier=1.25, cancel_delay_ticks=0),
    "moderate": StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=0),
}


@dataclass(frozen=True)
class DecompResult:
    net_pnl: float
    spread_capture: float
    adverse_loss: float
    fees: float
    trade_count: int
    toxic_fill_count: int
    non_toxic_fill_count: int
    max_drawdown: float
    bid_cancel_count: int
    ask_cancel_count: int
    any_cancel_rate: float


def ensure_dirs() -> None:
    for path in [ROOT, DIAGNOSTICS_DIR, ORACLE_DIR, SEARCH_DIR, EVALUATION_DIR, HLS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_selected_020_policy() -> dict[str, Any]:
    selected = json.loads((V20_DIR / "search" / "selected_policy.json").read_text(encoding="utf-8"))
    return selected["selected_main_policy"]["policy"]


def load_016_thresholds() -> dict[str, float]:
    selected = json.loads((V16_DIR / "outputs" / "selected_thresholds.json").read_text(encoding="utf-8"))
    return {
        "bid": float(selected["selected_thresholds"]["bid"]),
        "ask": float(selected["selected_thresholds"]["ask"]),
    }


def load_frames_actions(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_policy_frame(split)
    actions = build_actions_with_masks(frame, split=split)
    return frame, augment_actions(frame, actions)


def augment_actions(frame: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    lookup_cols = [
        "mid_price",
        "spread_ticks",
        "bid_prob_toxic",
        "ask_prob_toxic",
        "score_bid_015",
        "score_ask_015",
        "pull_bid_015_common",
        "pull_ask_015_common",
        "event_rate_bucket",
        "coarse_event_rate_bucket",
        "spread_bucket",
        "coarse_spread_bucket",
    ]
    lookup = frame[lookup_cols].copy()
    for col in lookup_cols:
        out[col] = lookup.loc[out["row_index"].to_numpy(dtype=np.int64), col].to_numpy()
    side_bid = out["side"].astype(str).to_numpy() == "bid"
    fut = out["future_mid_return_H_ticks"].to_numpy(dtype=float)
    out["side_score_016"] = np.where(side_bid, out["bid_prob_toxic"].astype(float), out["ask_prob_toxic"].astype(float))
    out["side_score_015"] = np.where(side_bid, out["score_bid_015"].astype(float), out["score_ask_015"].astype(float))
    out["side_toxic"] = np.where(side_bid, fut < 0, fut > 0)
    out["half_spread_value"] = (out["spread_ticks"].astype(float) * 0.5 * POINT_VALUE).astype(float)
    final_mid = frame.sort_values("local_row_id").groupby("session_id", sort=True)["mid_price"].last().astype(float)
    out["final_mid"] = out["session_id"].astype(str).map({str(k): float(v) for k, v in final_mid.items()})
    return out


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(np.max(peak - arr))


def simulate_decomp(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    pull_bid: np.ndarray,
    pull_ask: np.ndarray,
    stress: StressConfig,
) -> DecompResult:
    pull_bid = np.asarray(pull_bid, dtype=bool)
    pull_ask = np.asarray(pull_ask, dtype=bool)
    neutral_cut = float(stress.neutral_fill_haircut)
    good_cut = float(stress.good_fill_haircut)
    extra_mult = max(float(stress.toxic_multiplier) - 1.0, 0.0)

    total_pnl = 0.0
    total_spread = 0.0
    total_fees = 0.0
    total_trades = 0
    toxic_fills = 0
    non_toxic_fills = 0
    max_dd = 0.0

    for session_id, group in actions.groupby("session_id", sort=True):
        g = group.sort_values(["row_index", "side"]).reset_index(drop=True)
        final_mid = float(g["final_mid"].iloc[0]) if len(g) else 0.0
        cash = 0.0
        fees = 0.0
        inventory = 0
        equity: list[float] = []
        for row in g.itertuples(index=False):
            idx = int(row.row_index)
            side = str(row.side)
            canceled = bool(pull_bid[idx]) if side == "bid" else bool(pull_ask[idx])
            if canceled or float(row.neutral_u) > neutral_cut:
                continue
            future_ticks = float(row.future_mid_return_H_ticks)
            favorable = (side == "bid" and future_ticks > 0) or (side == "ask" and future_ticks < 0)
            if favorable and float(row.good_u) > good_cut:
                continue
            if side == "bid":
                if inventory >= MAX_INV:
                    continue
                cash -= float(row.price) * POINT_VALUE
                adverse_ticks = max(0.0, -future_ticks)
                inventory += 1
            else:
                if inventory <= -MAX_INV:
                    continue
                cash += float(row.price) * POINT_VALUE
                adverse_ticks = max(0.0, future_ticks)
                inventory -= 1
            if extra_mult:
                cash -= extra_mult * adverse_ticks * POINT_VALUE
            fees += MAKER_FEE
            total_spread += float(row.half_spread_value)
            total_trades += 1
            if bool(row.side_toxic):
                toxic_fills += 1
            else:
                non_toxic_fills += 1
            equity.append(cash - fees + inventory * float(row.mid_price) * POINT_VALUE)

        final_equity = cash - fees + inventory * final_mid * POINT_VALUE
        if equity:
            equity[-1] = final_equity
        total_pnl += final_equity
        total_fees += fees
        max_dd = max(max_dd, max_drawdown(equity))

    adverse_loss = float(total_spread - total_fees - total_pnl)
    return DecompResult(
        net_pnl=float(total_pnl),
        spread_capture=float(total_spread),
        adverse_loss=adverse_loss,
        fees=float(total_fees),
        trade_count=int(total_trades),
        toxic_fill_count=int(toxic_fills),
        non_toxic_fill_count=int(non_toxic_fills),
        max_drawdown=float(max_dd),
        bid_cancel_count=int(np.sum(pull_bid)),
        ask_cancel_count=int(np.sum(pull_ask)),
        any_cancel_rate=float(np.mean(pull_bid | pull_ask)) if len(frame) else 0.0,
    )


def baseline_policy_masks(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    if name == "no_cancel":
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if name == "015_common":
        return frame["pull_bid_015_common"].to_numpy(dtype=bool), frame["pull_ask_015_common"].to_numpy(dtype=bool)
    if name == "016_v2":
        thr = load_016_thresholds()
        return (
            frame["bid_prob_toxic"].to_numpy(dtype=float) >= thr["bid"],
            frame["ask_prob_toxic"].to_numpy(dtype=float) >= thr["ask"],
        )
    if name == "020_selected_policy":
        return policy_masks_from_spec(frame, load_selected_020_policy())
    raise ValueError(f"Unknown baseline policy: {name}")


def apply_policy(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kind = policy["policy_type"]
    n = len(frame)
    if kind in {"spread_toxic", "ensemble", "regime_threshold", "depth_proxy_diagnostic", "simple_rule"}:
        return policy_masks_from_spec(frame, policy)
    if kind == "ev_score":
        spread_value = frame["spread_ticks"].to_numpy(dtype=float) * 0.5 * POINT_VALUE
        bid_ev = spread_value - float(policy["lambda_toxic"]) * frame["bid_prob_toxic"].to_numpy(dtype=float) * POINT_VALUE - MAKER_FEE
        ask_ev = spread_value - float(policy["lambda_toxic"]) * frame["ask_prob_toxic"].to_numpy(dtype=float) * POINT_VALUE - MAKER_FEE
        wide = frame["spread_ticks"].to_numpy(dtype=float) > float(policy["max_spread_ticks"])
        return wide | (bid_ev < float(policy["ev_threshold"])), wide | (ask_ev < float(policy["ev_threshold"]))
    if kind == "risk_lut":
        spread_bucket = frame["coarse_spread_bucket"].astype(str).to_numpy()
        bid_bucket = risk_bucket(frame["bid_prob_toxic"].to_numpy(dtype=float), policy["risk_thresholds"])
        ask_bucket = risk_bucket(frame["ask_prob_toxic"].to_numpy(dtype=float), policy["risk_thresholds"])
        lut = policy["cancel_lut"]
        pull_bid = np.zeros(n, dtype=bool)
        pull_ask = np.zeros(n, dtype=bool)
        for i in range(n):
            pull_bid[i] = bool(lut.get(f"bid|{spread_bucket[i]}|{bid_bucket[i]}", False))
            pull_ask[i] = bool(lut.get(f"ask|{spread_bucket[i]}|{ask_bucket[i]}", False))
        return pull_bid, pull_ask
    if kind == "disagreement_ensemble":
        bid016 = frame["bid_prob_toxic"].to_numpy(dtype=float) >= float(policy["bid_threshold"])
        ask016 = frame["ask_prob_toxic"].to_numpy(dtype=float) >= float(policy["ask_threshold"])
        bid015 = frame["pull_bid_015_common"].to_numpy(dtype=bool)
        ask015 = frame["pull_ask_015_common"].to_numpy(dtype=bool)
        spread = frame["spread_ticks"].to_numpy(dtype=float)
        wide = spread > float(policy["max_spread_ticks"])
        mode = policy["agreement_mode"]
        if mode == "both_or_wide":
            return wide | (bid016 & bid015), wide | (ask016 & ask015)
        if mode == "016_unless_015_disagrees":
            return wide | (bid016 & (bid015 | (spread <= float(policy["safe_spread_ticks"])))), wide | (
                ask016 & (ask015 | (spread <= float(policy["safe_spread_ticks"])))
            )
        if mode == "either_high":
            return wide | bid016 | bid015, wide | ask016 | ask015
        raise ValueError(f"Unknown disagreement mode: {mode}")
    if kind == "regime_safe":
        pull_bid = np.zeros(n, dtype=bool)
        pull_ask = np.zeros(n, dtype=bool)
        regime_col = policy["regime_col"]
        regimes = frame[regime_col].astype(str).to_numpy()
        for regime, thresholds in policy["regime_thresholds"].items():
            mask = regimes == str(regime)
            pull_bid |= mask & (frame["bid_prob_toxic"].to_numpy(dtype=float) >= float(thresholds["bid_threshold"]))
            pull_ask |= mask & (frame["ask_prob_toxic"].to_numpy(dtype=float) >= float(thresholds["ask_threshold"]))
        return pull_bid, pull_ask
    raise ValueError(f"Unknown policy_type: {kind}")


def risk_bucket(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.full(len(scores), "q0", dtype=object)
    for i, thr in enumerate(sorted(float(x) for x in thresholds), start=1):
        out[scores >= thr] = f"q{i}"
    return out


def quote_action_value(action: pd.Series, stress: StressConfig) -> float:
    if float(action["neutral_u"]) > float(stress.neutral_fill_haircut):
        return 0.0
    side = str(action["side"])
    fut = float(action["future_mid_return_H_ticks"])
    favorable = (side == "bid" and fut > 0) or (side == "ask" and fut < 0)
    if favorable and float(action["good_u"]) > float(stress.good_fill_haircut):
        return 0.0
    terminal_value = (float(action["final_mid"]) - float(action["price"])) if side == "bid" else (float(action["price"]) - float(action["final_mid"]))
    side_markout_h = fut if side == "bid" else -fut
    adverse = max(0.0, -side_markout_h)
    return terminal_value * POINT_VALUE - MAKER_FEE - max(float(stress.toxic_multiplier) - 1.0, 0.0) * adverse * POINT_VALUE


def oracle_masks(frame: pd.DataFrame, actions: pd.DataFrame, oracle_type: str, stress: StressConfig) -> tuple[np.ndarray, np.ndarray]:
    pull_bid = np.zeros(len(frame), dtype=bool)
    pull_ask = np.zeros(len(frame), dtype=bool)
    for row in actions.itertuples(index=False):
        idx = int(row.row_index)
        side = str(row.side)
        cancel = False
        if oracle_type == "perfect_toxic_cancel":
            cancel = bool(row.side_toxic)
        elif oracle_type == "perfect_quote_pnl_filter":
            cancel = quote_action_value(pd.Series(row._asdict()), stress) <= 0.0
        else:
            raise ValueError(f"Unknown oracle: {oracle_type}")
        if cancel:
            if side == "bid":
                pull_bid[idx] = True
            else:
                pull_ask[idx] = True
    return pull_bid, pull_ask


def random_v3_mean_for_policy(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    pull_bid: np.ndarray,
    pull_ask: np.ndarray,
    stress: StressConfig,
    *,
    trials: int = 128,
    seed: int = 20260526,
) -> float:
    rng = np.random.default_rng(seed)
    grouped = frame.groupby(["session_id", "spread_bucket"], observed=False, sort=True)
    pnls = []
    for _ in range(int(trials)):
        rb = np.zeros(len(frame), dtype=bool)
        ra = np.zeros(len(frame), dtype=bool)
        for _, idx in grouped.groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k_bid = int(np.sum(pull_bid[idx_arr]))
            k_ask = int(np.sum(pull_ask[idx_arr]))
            if k_bid:
                rb[rng.choice(idx_arr, size=min(k_bid, len(idx_arr)), replace=False)] = True
            if k_ask:
                ra[rng.choice(idx_arr, size=min(k_ask, len(idx_arr)), replace=False)] = True
        pnls.append(simulate_decomp(frame, actions, rb, ra, stress).net_pnl)
    return float(np.mean(pnls))


def policy_id_from_dict(policy: dict[str, Any]) -> str:
    text = json.dumps(json_sanitize(policy), sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def decomp_to_row(policy_name: str, split: str, stress_name: str, result: DecompResult) -> dict[str, Any]:
    return {
        "policy": policy_name,
        "split": split,
        "stress_name": stress_name,
        "net_pnl": result.net_pnl,
        "spread_capture": result.spread_capture,
        "adverse_loss": result.adverse_loss,
        "fees": result.fees,
        "reconstructed_pnl": result.spread_capture - result.adverse_loss - result.fees,
        "reconstruction_error": (result.spread_capture - result.adverse_loss - result.fees) - result.net_pnl,
        "trade_count": result.trade_count,
        "toxic_fill_count": result.toxic_fill_count,
        "non_toxic_fill_count": result.non_toxic_fill_count,
        "max_drawdown": result.max_drawdown,
        "bid_cancel_count": result.bid_cancel_count,
        "ask_cancel_count": result.ask_cancel_count,
        "any_cancel_rate": result.any_cancel_rate,
    }
