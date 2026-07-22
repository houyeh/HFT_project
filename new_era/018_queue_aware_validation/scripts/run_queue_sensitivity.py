from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    AUDIT_DIR,
    BACKTEST_DIR,
    COMPARISON_DIR,
    FIGURES_DIR,
    HORIZON,
    LOGS_DIR,
    MAKER_FEE,
    MAX_INVENTORY,
    MODEL_DIAG_DIR,
    MODEL_DIR,
    POINT_VALUE,
    PROJECT_ROOT,
    RANDOM_REFERENCE,
    REPORTS_DIR,
    OUTPUT_ROOT,
    PolicySpec,
    build_gate_masks,
    build_policy_specs,
    compute_future_trade_volume,
    ensure_dirs,
    future_sum,
    get_logger,
    load_feature_frame,
    load_static_gate_params,
    load_trade_frame,
)


@dataclass(frozen=True)
class DatasetPack:
    dataset: str
    df: pd.DataFrame
    trade_volume_by_event: np.ndarray
    price_tick: float


@dataclass(frozen=True)
class FillScenario:
    fill_model: str
    scenario_name: str
    fill_probability: float | None = None
    p_good: float | None = None
    p_toxic: float | None = None
    queue_fraction: float | None = None
    fill_window: int | None = None
    cancel_delay_events: int = 0
    seeds: tuple[int, ...] = ()


def _load_original_simulator(logger) -> Callable[..., dict[str, Any]]:
    module_path = PROJECT_ROOT / "new_era" / "15_models_playaround" / "run_diagnosis_1.py"
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("run_diagnosis_1_dynamic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded simulate_with_metrics from %s", module_path)
    return module.simulate_with_metrics


def _build_cfg_stub() -> Any:
    return SimpleNamespace(
        backtest=SimpleNamespace(
            maker_fee_twd=float(MAKER_FEE),
            taker_fee_twd=50.0,
            max_inventory=int(MAX_INVENTORY),
            point_value_twd_per_point=float(POINT_VALUE),
            allow_active_alpha=False,
        ),
        window=SimpleNamespace(horizon=int(HORIZON)),
    )


def _to_np(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "prob": df[["prob_down", "prob_stationary", "prob_up"]].to_numpy(dtype=np.float64),
        "ask": df["ask_px_1"].to_numpy(dtype=np.float64),
        "bid": df["bid_px_1"].to_numpy(dtype=np.float64),
        "mid": df["mid"].to_numpy(dtype=np.float64),
        "spread": df["spread_ticks"].to_numpy(dtype=np.float64),
        "bid_vol_1": df["bid_vol_1"].to_numpy(dtype=np.float64),
        "ask_vol_1": df["ask_vol_1"].to_numpy(dtype=np.float64),
    }


def _forward_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= 1 or window <= 0:
        return out
    dq: deque[int] = deque()
    right = 1
    for i in range(n):
        start = i + 1
        end = min(i + window, n - 1)
        while right <= end:
            val = arr[right]
            while dq and arr[dq[-1]] >= val:
                dq.pop()
            dq.append(right)
            right += 1
        while dq and dq[0] < start:
            dq.popleft()
        if start <= end and dq:
            out[i] = arr[dq[0]]
    return out


def _forward_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= 1 or window <= 0:
        return out
    dq: deque[int] = deque()
    right = 1
    for i in range(n):
        start = i + 1
        end = min(i + window, n - 1)
        while right <= end:
            val = arr[right]
            while dq and arr[dq[-1]] <= val:
                dq.pop()
            dq.append(right)
            right += 1
        while dq and dq[0] < start:
            dq.popleft()
        if start <= end and dq:
            out[i] = arr[dq[0]]
    return out


def _build_queue_cache(arrs: dict[str, np.ndarray], trade_volume: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    cache: dict[int, dict[str, np.ndarray]] = {}
    for w in [1, 3, 5, 10]:
        cache[w] = {
            "min_future_ask": _forward_min(arrs["ask"], w),
            "max_future_bid": _forward_max(arrs["bid"], w),
            "min_future_bid": _forward_min(arrs["bid"], w),
            "max_future_ask": _forward_max(arrs["ask"], w),
            "future_trade_vol": future_sum(trade_volume, w),
        }
    return cache


def _build_effective_pulls(
    prob: np.ndarray,
    threshold: float,
    active: np.ndarray,
    force_bid: np.ndarray,
    force_ask: np.ndarray,
    cancel_delay_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(prob)
    eff_bid = np.zeros(n, dtype=bool)
    eff_ask = np.zeros(n, dtype=bool)

    state_bid = False
    state_ask = False
    pending_bid = 0
    pending_ask = 0

    for i in range(max(0, n - 1)):
        p_down = float(prob[i, 0])
        p_up = float(prob[i, 2])
        raw_bid = p_down >= threshold
        raw_ask = p_up >= threshold

        if cancel_delay_events <= 0:
            pull_bid = raw_bid
            pull_ask = raw_ask
            state_bid = pull_bid
            state_ask = pull_ask
            pending_bid = 0
            pending_ask = 0
        else:
            if raw_bid:
                if state_bid:
                    pass
                elif pending_bid <= 0:
                    pending_bid = int(cancel_delay_events)
                else:
                    pending_bid -= 1
                    if pending_bid <= 0:
                        state_bid = True
            else:
                state_bid = False
                pending_bid = 0

            if raw_ask:
                if state_ask:
                    pass
                elif pending_ask <= 0:
                    pending_ask = int(cancel_delay_events)
                else:
                    pending_ask -= 1
                    if pending_ask <= 0:
                        state_ask = True
            else:
                state_ask = False
                pending_ask = 0

            pull_bid = bool(state_bid)
            pull_ask = bool(state_ask)

        if not bool(active[i]):
            pull_bid = True
            pull_ask = True
        if bool(force_bid[i]):
            pull_bid = True
        if bool(force_ask[i]):
            pull_ask = True

        eff_bid[i] = bool(pull_bid)
        eff_ask[i] = bool(pull_ask)
    return eff_bid, eff_ask


def _queue_condition(
    i: int,
    side: str,
    queue_fraction: float,
    fill_window: int,
    arrs: dict[str, np.ndarray],
    queue_cache: dict[int, dict[str, np.ndarray]],
) -> bool:
    qc = queue_cache[fill_window]
    if side == "bid":
        touch = bool(qc["min_future_ask"][i] <= arrs["bid"][i]) if np.isfinite(qc["min_future_ask"][i]) else False
        level_removed = bool(qc["min_future_bid"][i] < arrs["bid"][i]) if np.isfinite(qc["min_future_bid"][i]) else False
        queue_ahead = max(0.0, arrs["bid_vol_1"][i]) * queue_fraction
    else:
        touch = bool(qc["max_future_bid"][i] >= arrs["ask"][i]) if np.isfinite(qc["max_future_bid"][i]) else False
        level_removed = bool(qc["max_future_ask"][i] > arrs["ask"][i]) if np.isfinite(qc["max_future_ask"][i]) else False
        queue_ahead = max(0.0, arrs["ask_vol_1"][i]) * queue_fraction
    trade_proxy = float(qc["future_trade_vol"][i]) >= queue_ahead
    return bool(touch and (level_removed or trade_proxy))


def _simulate_proxy(
    *,
    arrs: dict[str, np.ndarray],
    eff_pull_bid: np.ndarray,
    eff_pull_ask: np.ndarray,
    scenario: FillScenario,
    seed: int,
    active_event_rate: float,
    queue_cache: dict[int, dict[str, np.ndarray]] | None,
) -> dict[str, float | int]:
    ask = arrs["ask"]
    bid = arrs["bid"]
    mid = arrs["mid"]
    n = len(ask)
    if n <= 1:
        return {
            "trade_count": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fee": 0.0,
            "fill_rate": 0.0,
            "signal_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_pnl_per_fill": 0.0,
            "realized_bid_pnl": 0.0,
            "realized_ask_pnl": 0.0,
            "toxic_fill_count": 0,
            "profitable_fill_count": 0,
            "net_cancel_value": 0.0,
            "active_event_rate": active_event_rate,
        }

    rng = np.random.default_rng(seed)
    queue_cache = queue_cache or {}

    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0

    signal_count = 0
    total_opportunities = 0
    toxic_fill_count = 0
    profitable_fill_count = 0
    missed_profitable_pnl = 0.0
    missed_toxic_avoided = 0.0

    realized_bid_pnl = 0.0
    realized_ask_pnl = 0.0
    open_prices: list[float] = []
    pnl_path: list[float] = []

    for i in range(n - 1):
        b = float(bid[i])
        a = float(ask[i])
        nb = float(bid[i + 1])
        na = float(ask[i + 1])
        if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(nb) and np.isfinite(na)) or a <= b:
            pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * POINT_VALUE)
            continue

        pull_b = bool(eff_pull_bid[i])
        pull_a = bool(eff_pull_ask[i])
        if pull_b:
            signal_count += 1
        if pull_a:
            signal_count += 1
        total_opportunities += 2

        quote_bid = (not pull_b) and (inventory < MAX_INVENTORY)
        quote_ask = (not pull_a) and (inventory > -MAX_INVENTORY)

        fut_mid = float(mid[min(i + HORIZON, n - 1)])

        bid_touch = quote_bid and (na <= b)
        ask_touch = quote_ask and (nb >= a)

        def decide_fill(side: str, touched: bool, markout_pnl: float) -> bool:
            if not touched:
                return False
            if scenario.fill_model == "uniform_fill_haircut":
                p = float(scenario.fill_probability if scenario.fill_probability is not None else 1.0)
                return bool(rng.random() < p)
            if scenario.fill_model == "good_fill_haircut":
                p_good = float(scenario.p_good if scenario.p_good is not None else 1.0)
                p_toxic = float(scenario.p_toxic if scenario.p_toxic is not None else 1.0)
                p = p_good if markout_pnl > 0 else p_toxic
                return bool(rng.random() < p)
            if scenario.fill_model == "queue_ahead_best_level":
                return _queue_condition(
                    i,
                    side,
                    float(scenario.queue_fraction),
                    int(scenario.fill_window),
                    arrs,
                    queue_cache,
                )
            if scenario.fill_model == "delayed_cancel_plus_queue":
                # Composite scenario: queue condition + optional good-fill haircut.
                queue_ok = _queue_condition(
                    i,
                    side,
                    float(scenario.queue_fraction),
                    int(scenario.fill_window),
                    arrs,
                    queue_cache,
                )
                if not queue_ok:
                    return False
                if scenario.p_good is None or scenario.p_toxic is None:
                    return True
                p = float(scenario.p_good if markout_pnl > 0 else scenario.p_toxic)
                return bool(rng.random() < p)
            return touched

        bid_markout = (fut_mid - b) * POINT_VALUE
        ask_markout = (a - fut_mid) * POINT_VALUE
        bid_exec = decide_fill("bid", bid_touch, bid_markout)
        ask_exec = decide_fill("ask", ask_touch, ask_markout)

        if bid_touch and not bid_exec:
            if bid_markout > 0:
                missed_profitable_pnl += bid_markout
            else:
                missed_toxic_avoided += -bid_markout
        if ask_touch and not ask_exec:
            if ask_markout > 0:
                missed_profitable_pnl += ask_markout
            else:
                missed_toxic_avoided += -ask_markout

        if bid_exec:
            gross_cash -= b * POINT_VALUE
            fees_paid += MAKER_FEE
            maker_trades += 1
            if bid_markout > 0:
                profitable_fill_count += 1
            else:
                toxic_fill_count += 1
            if inventory < 0:
                open_px = open_prices.pop(0)
                realized_ask_pnl += (open_px - b) * POINT_VALUE - 2.0 * MAKER_FEE
            else:
                open_prices.append(b)
            inventory += 1

        if ask_exec:
            gross_cash += a * POINT_VALUE
            fees_paid += MAKER_FEE
            maker_trades += 1
            if ask_markout > 0:
                profitable_fill_count += 1
            else:
                toxic_fill_count += 1
            if inventory > 0:
                open_px = open_prices.pop(0)
                realized_bid_pnl += (a - open_px) * POINT_VALUE - 2.0 * MAKER_FEE
            else:
                open_prices.append(a)
            inventory -= 1

        pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * POINT_VALUE)

    final_mid = float(mid[-1])
    gross_pnl = gross_cash + inventory * final_mid * POINT_VALUE
    net_pnl = gross_pnl - fees_paid
    pnl_arr = np.asarray(pnl_path, dtype=np.float64)
    if pnl_arr.size == 0:
        mdd = 0.0
    else:
        peaks = np.maximum.accumulate(pnl_arr)
        mdd = float(np.max(peaks - pnl_arr))

    total_ops = max(1, total_opportunities)
    return {
        "trade_count": int(maker_trades),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "fee": float(fees_paid),
        "fill_rate": float(maker_trades / total_ops),
        "signal_rate": float(signal_count / total_ops),
        "max_drawdown": float(mdd),
        "avg_pnl_per_fill": float(net_pnl / maker_trades) if maker_trades else 0.0,
        "realized_bid_pnl": float(realized_bid_pnl),
        "realized_ask_pnl": float(realized_ask_pnl),
        "toxic_fill_count": int(toxic_fill_count),
        "profitable_fill_count": int(profitable_fill_count),
        "net_cancel_value": float(missed_toxic_avoided - missed_profitable_pnl),
        "active_event_rate": float(active_event_rate),
    }


def _scenario_rows(
    *,
    dataset_pack: DatasetPack,
    spec: PolicySpec,
    scenario: FillScenario,
    eff_pull_bid: np.ndarray,
    eff_pull_ask: np.ndarray,
    base_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arrs = _to_np(dataset_pack.df)
    queue_cache: dict[int, dict[str, np.ndarray]] | None = None
    if scenario.fill_model in {"queue_ahead_best_level", "delayed_cancel_plus_queue"}:
        queue_cache = _build_queue_cache(arrs, dataset_pack.trade_volume_by_event)

    seeds = scenario.seeds if scenario.seeds else (0,)
    for seed in seeds:
        if scenario.fill_model == "original_fill":
            if base_metrics is None:
                raise RuntimeError("base_metrics required for original_fill rows")
            metrics = dict(base_metrics)
        else:
            metrics = _simulate_proxy(
                arrs=arrs,
                eff_pull_bid=eff_pull_bid,
                eff_pull_ask=eff_pull_ask,
                scenario=scenario,
                seed=int(seed),
                active_event_rate=float(base_metrics["active_event_rate"]) if base_metrics else 1.0,
                queue_cache=queue_cache,
            )
        metrics.setdefault("profitable_fill_count", 0)
        metrics.setdefault("toxic_fill_count", 0)
        metrics.setdefault("net_cancel_value", 0.0)
        metrics.setdefault("realized_bid_pnl", 0.0)
        metrics.setdefault("realized_ask_pnl", 0.0)
        metrics.setdefault("active_event_rate", float(base_metrics["active_event_rate"]) if base_metrics else 1.0)
        rows.append(
            {
                "dataset": dataset_pack.dataset,
                "policy_name": spec.policy_name,
                "model_threshold": float(spec.model_threshold),
                "fill_model": scenario.fill_model,
                "scenario_name": scenario.scenario_name,
                "fill_probability": scenario.fill_probability,
                "p_good": scenario.p_good,
                "p_toxic": scenario.p_toxic,
                "queue_fraction": scenario.queue_fraction,
                "fill_window": scenario.fill_window,
                "cancel_delay_events": int(scenario.cancel_delay_events),
                "seed": int(seed),
                "net_pnl": float(metrics["net_pnl"]),
                "gross_pnl": float(metrics["gross_pnl"]),
                "fee": float(metrics["fee"]),
                "trade_count": int(metrics["trade_count"]),
                "fill_rate": float(metrics["fill_rate"]),
                "active_event_rate": float(metrics["active_event_rate"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "avg_pnl_per_fill": float(metrics["avg_pnl_per_fill"]),
                "realized_bid_pnl": float(metrics["realized_bid_pnl"]),
                "realized_ask_pnl": float(metrics["realized_ask_pnl"]),
                "toxic_fill_count": int(metrics["toxic_fill_count"]),
                "profitable_fill_count": int(metrics["profitable_fill_count"]),
                "net_cancel_value": float(metrics["net_cancel_value"]),
                "notes": spec.notes,
            }
        )
    return rows


def _aggregate(summary_df: pd.DataFrame, random_ref_055: float) -> pd.DataFrame:
    grp_cols = ["dataset", "policy_name", "fill_model", "scenario_name", "model_threshold"]
    rows: list[dict[str, Any]] = []
    for key, g in summary_df.groupby(grp_cols, dropna=False):
        dataset, policy_name, fill_model, scenario_name, thr = key
        pnl = g["net_pnl"].to_numpy(dtype=np.float64)
        trade = g["trade_count"].to_numpy(dtype=np.float64)
        mdd = g["max_drawdown"].to_numpy(dtype=np.float64)
        ncv = g["net_cancel_value"].to_numpy(dtype=np.float64)
        mean_pnl = float(np.mean(pnl))
        row = {
            "dataset": dataset,
            "policy_name": policy_name,
            "fill_model": fill_model,
            "scenario_name": scenario_name,
            "model_threshold": float(thr),
            "net_pnl_mean": mean_pnl,
            "net_pnl_std": float(np.std(pnl, ddof=0)),
            "net_pnl_p10": float(np.quantile(pnl, 0.10)),
            "net_pnl_p50": float(np.quantile(pnl, 0.50)),
            "net_pnl_p90": float(np.quantile(pnl, 0.90)),
            "trade_count_mean": float(np.mean(trade)),
            "trade_count_std": float(np.std(trade, ddof=0)),
            "max_drawdown_mean": float(np.mean(mdd)),
            "net_cancel_value_mean": float(np.mean(ncv)),
            "survives_positive_pnl": bool(mean_pnl > 0),
            "beats_no_model_original": bool(mean_pnl > 56100.0) if dataset == "NEW" else bool(mean_pnl > -557165.0),
            "beats_random_reference": bool(mean_pnl > random_ref_055) if dataset == "NEW" else False,
            "notes": "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_figures(summary_df: pd.DataFrame) -> None:
    # 1) net_pnl vs fill probability
    d1 = summary_df[
        (summary_df["fill_model"] == "uniform_fill_haircut")
        & (summary_df["dataset"].isin(["NEW", "NEW3"]))
        & (summary_df["policy_name"].isin(["no_gate_tcn_055", "spread_filter_tcn_055", "combined_filter_tcn_055"]))
    ].copy()
    if not d1.empty:
        g = (
            d1.groupby(["dataset", "policy_name", "fill_probability"], dropna=False)["net_pnl"]
            .mean()
            .reset_index()
            .sort_values("fill_probability")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for (dataset, policy), sub in g.groupby(["dataset", "policy_name"]):
            ax.plot(sub["fill_probability"], sub["net_pnl"], marker="o", label=f"{dataset}:{policy}")
        ax.set_xlabel("Fill Probability p")
        ax.set_ylabel("Net PnL")
        ax.set_title("Net PnL vs Fill Probability")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "net_pnl_vs_fill_probability.png", dpi=160)
        plt.close(fig)

    # 2) good vs toxic haircut
    d2 = summary_df[
        (summary_df["fill_model"].isin(["original_fill", "good_fill_haircut"]))
        & (summary_df["policy_name"].isin(["spread_filter_tcn_055", "combined_filter_tcn_055"]))
    ].copy()
    if not d2.empty:
        agg = d2.groupby(["dataset", "policy_name", "scenario_name"], dropna=False)["net_pnl"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        xlabels = sorted(agg["scenario_name"].unique().tolist())
        x = np.arange(len(xlabels))
        width = 0.12
        keys = list(agg.groupby(["dataset", "policy_name"]).groups.keys())
        for k_idx, (dataset, policy) in enumerate(keys):
            sub = agg[(agg["dataset"] == dataset) & (agg["policy_name"] == policy)]
            vals = [float(sub[sub["scenario_name"] == lbl]["net_pnl"].iloc[0]) if (sub["scenario_name"] == lbl).any() else np.nan for lbl in xlabels]
            ax.bar(x + (k_idx - len(keys) / 2) * width, vals, width=width, label=f"{dataset}:{policy}")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha="right")
        ax.set_ylabel("Net PnL")
        ax.set_title("Original vs Good/Toxic Fill Haircut")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "good_vs_toxic_fill_haircut.png", dpi=160)
        plt.close(fig)

    # 3) queue fraction sensitivity
    d3 = summary_df[summary_df["fill_model"] == "queue_ahead_best_level"].copy()
    if not d3.empty:
        g = d3.groupby(["dataset", "policy_name", "queue_fraction"], dropna=False)["net_pnl"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        for (dataset, policy), sub in g.groupby(["dataset", "policy_name"]):
            ax.plot(sub["queue_fraction"], sub["net_pnl"], marker="o", label=f"{dataset}:{policy}")
        ax.set_xlabel("Queue Fraction")
        ax.set_ylabel("Net PnL")
        ax.set_title("Queue Fraction Sensitivity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "queue_fraction_sensitivity.png", dpi=160)
        plt.close(fig)

    # 4) cancel delay under queue
    d4 = summary_df[summary_df["fill_model"] == "delayed_cancel_plus_queue"].copy()
    if not d4.empty:
        g = d4.groupby(["policy_name", "scenario_name", "cancel_delay_events"], dropna=False)["net_pnl"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        for (policy, scenario_name), sub in g.groupby(["policy_name", "scenario_name"]):
            ax.plot(sub["cancel_delay_events"], sub["net_pnl"], marker="o", label=f"{policy}:{scenario_name}")
        ax.set_xlabel("Cancel Delay Events")
        ax.set_ylabel("Net PnL")
        ax.set_title("Cancel Delay Under Queue-Aware Assumptions")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cancel_delay_under_queue.png", dpi=160)
        plt.close(fig)

    # 5) trade_count vs pnl
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df["trade_count"], summary_df["net_pnl"], alpha=0.5, s=18)
    ax.set_xlabel("Trade Count")
    ax.set_ylabel("Net PnL")
    ax.set_title("Trade Count vs Net PnL Under Queue Scenarios")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trade_count_vs_pnl_queue.png", dpi=160)
    plt.close(fig)


def _write_verdict_table(summary_df: pd.DataFrame, static_map: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchor = summary_df[
        (
            (summary_df["fill_model"] == "uniform_fill_haircut")
            & (summary_df["fill_probability"] == 0.50)
        )
        | (
            (summary_df["fill_model"] == "good_fill_haircut")
            & (summary_df["scenario_name"] == "good_fill_medium")
        )
        | (
            (summary_df["fill_model"] == "queue_ahead_best_level")
            & (summary_df["queue_fraction"] == 0.50)
            & (summary_df["fill_window"] == 5)
        )
    ].copy()

    for (dataset, policy_name), g in summary_df.groupby(["dataset", "policy_name"], dropna=False):
        base = g[g["fill_model"] == "original_fill"]
        if base.empty:
            continue
        original = float(base["net_pnl"].mean())
        orig_trade = float(base["trade_count"].mean())
        cons = anchor[(anchor["dataset"] == dataset) & (anchor["policy_name"] == policy_name)]
        if cons.empty:
            cons_mean = np.nan
            cons_p10 = np.nan
            trade_mean = np.nan
        else:
            vals = cons["net_pnl"].to_numpy(dtype=np.float64)
            cons_mean = float(np.mean(vals))
            cons_p10 = float(np.quantile(vals, 0.10))
            trade_mean = float(cons["trade_count"].mean())

        if dataset == "NEW":
            ref_random = RANDOM_REFERENCE["NEW_055_mean_plus_1std"] if "070" not in policy_name else RANDOM_REFERENCE["NEW_070_mean_plus_1std"]
            still_beats_random = bool(np.isfinite(cons_mean) and cons_mean > ref_random)
            rule_baseline = 101455.0 if "spread_filter" in policy_name else 84980.0
            still_beats_rule = bool(np.isfinite(cons_mean) and cons_mean >= rule_baseline)
        else:
            still_beats_random = False
            rule_baseline = 72520.0 if "combined_filter" in policy_name else -321405.0
            still_beats_rule = bool(np.isfinite(cons_mean) and cons_mean >= rule_baseline)

        still_positive = bool(np.isfinite(cons_mean) and cons_mean > 0)
        if not np.isfinite(cons_mean):
            verdict = "inconclusive_due_to_missing_trade_side"
        elif still_positive and still_beats_rule:
            verdict = "robust_under_queue_sensitivity"
        elif still_positive:
            verdict = "fragile_but_positive"
        else:
            verdict = "not_robust"

        deg = (
            float((original - cons_mean) / max(abs(original), 1e-9))
            if np.isfinite(cons_mean)
            else np.nan
        )
        rows.append(
            {
                "dataset": dataset,
                "policy_name": policy_name,
                "original_net_pnl": original,
                "conservative_net_pnl_mean": cons_mean,
                "conservative_net_pnl_p10": cons_p10,
                "trade_count_original": orig_trade,
                "trade_count_conservative_mean": trade_mean,
                "pnl_degradation_pct": deg,
                "still_positive": still_positive,
                "still_beats_random": still_beats_random,
                "still_beats_rule_baseline": still_beats_rule,
                "verdict": verdict,
            }
        )

    out = pd.DataFrame(rows).sort_values(["dataset", "policy_name"]).reset_index(drop=True)
    out.to_csv(COMPARISON_DIR / "queue_aware_verdict_table.csv", index=False)
    return out


def _write_reports(
    summary_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    verdict_df: pd.DataFrame,
    static_map: dict[tuple[str, str], dict[str, Any]],
) -> None:
    verdict_priority = {"robust_under_queue_sensitivity": "A", "fragile_but_positive": "B", "not_robust": "C", "inconclusive_due_to_missing_trade_side": "D"}
    if verdict_df.empty:
        letter = "D"
        reason = "No conservative scenario rows available."
    else:
        counts = verdict_df["verdict"].value_counts().to_dict()
        if counts.get("not_robust", 0) > 0:
            letter = "C"
            reason = "At least one key policy loses edge under conservative queue assumptions."
        elif counts.get("fragile_but_positive", 0) > 0:
            letter = "B"
            reason = "PnL remains positive in some scenarios but degrades materially."
        elif counts.get("robust_under_queue_sensitivity", 0) > 0:
            letter = "A"
            reason = "Key policies remain positive and competitive under conservative scenarios."
        else:
            letter = "D"
            reason = "Inconclusive because queue reconstruction is approximate without aggressor side."

    # Final report
    lines: list[str] = []
    lines.append("# 1. Executive Summary")
    lines.append(f"Queue-aware sensitivity verdict: **{letter}** ({reason})")
    lines.append("")
    lines.append("# 2. Why Queue Matters")
    lines.append(
        "Current maker-only backtest is useful for first-order signal comparison, but price touch does not guarantee our queue position fills."
    )
    lines.append("")
    lines.append("# 3. Available Data and Limitations")
    lines.append("- LOB depth: available (L1-L5 with volume).")
    lines.append("- Trade tape: available (timestamp, price, volume).")
    lines.append("- aggressor_side: not available.")
    lines.append("- True exchange queue replay: not possible from this data.")
    lines.append("")
    lines.append("# 4. Sensitivity Scenarios")
    lines.append("- original_fill (reference)")
    lines.append("- uniform_fill_haircut p in {0.75, 0.50, 0.25}")
    lines.append("- good_fill_haircut (mild / medium / severe)")
    lines.append("- queue_ahead_best_level (queue_fraction {0.25,0.50,1.00}, fill_window {1,3,5,10})")
    lines.append("- delayed_cancel_plus_queue (cancel_delay in {0,1,2,3,5,10})")
    lines.append("")
    lines.append("# 5. NEW Results")
    for policy in ["no_gate_tcn_055", "spread_filter_tcn_055", "combined_filter_tcn_055", "no_gate_tcn_070", "val_selected_tcn_070"]:
        sub = verdict_df[(verdict_df["dataset"] == "NEW") & (verdict_df["policy_name"] == policy)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"- {policy}: original={r['original_net_pnl']:.2f}, conservative_mean={r['conservative_net_pnl_mean']:.2f}, "
            f"p10={r['conservative_net_pnl_p10']:.2f}, verdict={r['verdict']}."
        )
    lines.append("")
    lines.append("# 6. NEW3 Results")
    for policy in ["combined_filter_tcn_055", "no_gate_tcn_055"]:
        sub = verdict_df[(verdict_df["dataset"] == "NEW3") & (verdict_df["policy_name"] == policy)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"- {policy}: original={r['original_net_pnl']:.2f}, conservative_mean={r['conservative_net_pnl_mean']:.2f}, "
            f"p10={r['conservative_net_pnl_p10']:.2f}, verdict={r['verdict']}."
        )
    lines.append("")
    lines.append("# 7. Latency Under Queue Assumptions")
    dlat = summary_df[summary_df["fill_model"] == "delayed_cancel_plus_queue"]
    if dlat.empty:
        lines.append("No delayed-cancel scenarios available.")
    else:
        pivot = (
            dlat.groupby(["policy_name", "scenario_name", "cancel_delay_events"], dropna=False)["net_pnl"]
            .mean()
            .reset_index()
        )
        for (policy, scenario_name), sub in pivot.groupby(["policy_name", "scenario_name"]):
            s0 = sub[sub["cancel_delay_events"] == 0]["net_pnl"]
            s10 = sub[sub["cancel_delay_events"] == 10]["net_pnl"]
            if not s0.empty and not s10.empty:
                delta = float(s10.iloc[0] - s0.iloc[0])
                lines.append(f"- {policy} / {scenario_name}: delay 10 vs 0 delta = {delta:.2f}.")
    lines.append("")
    lines.append("# 8. Verdict")
    lines.append(f"{letter}. {reason}")
    lines.append("")
    lines.append("# 9. Meeting-ready Claims")
    lines.append("- Current backtest is first-order signal comparison, not full queue replay.")
    lines.append("- Queue-aware sensitivity is necessary before production profitability claims.")
    lines.append(f"- Under conservative assumptions, the result is verdict {letter}.")
    lines.append("")
    lines.append("# 10. Next Steps")
    if letter in {"A", "B"}:
        lines.append("- Integrate queue sensitivity benchmarks into the final scorecard.")
        lines.append("- Evaluate cancel-latency / FPGA value under conservative fill models.")
    else:
        lines.append("- Do not claim production trading profitability from current assumptions.")
        lines.append("- Prioritize control-system reliability, risk management, and queue data enrichment.")
    (REPORTS_DIR / "queue_aware_validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    # slide summary
    slide = []
    slide.append("# Queue-aware Validation: Does the Edge Survive Conservative Fill Assumptions?")
    slide.append("")
    slide.append("## Why Queue Matters")
    slide.append("- Price touch at best level does not guarantee our fill because queue position is unknown.")
    slide.append("- Missing aggressor-side prevents exact exchange queue replay.")
    slide.append("")
    slide.append("## Tested Conservative Assumptions")
    slide.append("- Uniform fill haircut (0.75 / 0.50 / 0.25)")
    slide.append("- Good-fill haircut vs toxic-fill preference (mild/medium/severe)")
    slide.append("- Queue-ahead proxy with best-level volume and future trade-volume windows")
    slide.append("- Cancel-delay sensitivity under queue-aware conditions")
    slide.append("")
    slide.append("## Key Original vs Conservative Snapshot")
    for _, r in verdict_df.iterrows():
        slide.append(
            f"- {r['dataset']} {r['policy_name']}: original {r['original_net_pnl']:.1f} -> conservative mean {r['conservative_net_pnl_mean']:.1f} (p10 {r['conservative_net_pnl_p10']:.1f})"
        )
    slide.append("")
    slide.append("## Final Verdict")
    slide.append(f"- Verdict {letter}: {reason}")
    slide.append("- FPGA claim impact: latency still matters, but economic edge must be judged under conservative fill stress.")
    (REPORTS_DIR / "queue_aware_slide_summary.md").write_text("\n".join(slide), encoding="utf-8")

    # final checkpoint
    chk: list[str] = []
    chk.append("# Final Checkpoint")
    chk.append("")
    chk.append("1. Original fill reproduction:")
    for dataset, policy in [("NEW", "no_gate_tcn_055"), ("NEW", "spread_filter_tcn_055"), ("NEW3", "combined_filter_tcn_055")]:
        sub = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["policy_name"] == policy)
            & (summary_df["fill_model"] == "original_fill")
        ]
        if not sub.empty:
            r = sub.iloc[0]
            chk.append(f"- {dataset} {policy}: net_pnl={r['net_pnl']:.2f}, trade_count={int(r['trade_count'])}")
    chk.append("")
    chk.append("2. Conservative fill result:")
    for dataset, policy in [("NEW", "spread_filter_tcn_055"), ("NEW3", "combined_filter_tcn_055")]:
        u50 = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["policy_name"] == policy)
            & (summary_df["fill_model"] == "uniform_fill_haircut")
            & (summary_df["fill_probability"] == 0.50)
        ]["net_pnl"]
        gm = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["policy_name"] == policy)
            & (summary_df["fill_model"] == "good_fill_haircut")
            & (summary_df["scenario_name"] == "good_fill_medium")
        ]["net_pnl"]
        q50 = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["policy_name"] == policy)
            & (summary_df["fill_model"] == "queue_ahead_best_level")
            & (summary_df["queue_fraction"] == 0.50)
            & (summary_df["fill_window"] == 5)
        ]["net_pnl"]
        chk.append(
            f"- {dataset} {policy}: uniform50_mean={u50.mean() if len(u50) else float('nan'):.2f}, "
            f"good_medium_mean={gm.mean() if len(gm) else float('nan'):.2f}, "
            f"queue_q0.5_w5={q50.mean() if len(q50) else float('nan'):.2f}"
        )
    chk.append("")
    chk.append("3. Survival check:")
    for dataset, policy in [("NEW", "spread_filter_tcn_055"), ("NEW3", "combined_filter_tcn_055")]:
        row = verdict_df[(verdict_df["dataset"] == dataset) & (verdict_df["policy_name"] == policy)]
        if not row.empty:
            r = row.iloc[0]
            chk.append(
                f"- {dataset} {policy}: still_positive={bool(r['still_positive'])}, still_beats_random={bool(r['still_beats_random'])}, "
                f"still_beats_rule_baseline={bool(r['still_beats_rule_baseline'])}"
            )
    chk.append("")
    chk.append("4. Latency under queue:")
    dlat = summary_df[
        (summary_df["fill_model"] == "delayed_cancel_plus_queue")
        & (summary_df["policy_name"].isin(["spread_filter_tcn_055", "combined_filter_tcn_055"]))
    ]
    if dlat.empty:
        chk.append("- No delayed-cancel data.")
    else:
        for (dataset, policy, scenario_name), sub in dlat.groupby(["dataset", "policy_name", "scenario_name"]):
            s0 = sub[sub["cancel_delay_events"] == 0]["net_pnl"].mean()
            s10 = sub[sub["cancel_delay_events"] == 10]["net_pnl"].mean()
            chk.append(
                f"- {dataset} {policy} {scenario_name}: delay10-delay0 = {float(s10 - s0):.2f}"
            )
    chk.append("")
    chk.append("5. Final verdict:")
    chk.append(f"- {letter}")
    chk.append(f"- reason: {reason}")
    (REPORTS_DIR / "final_checkpoint.md").write_text("\n".join(chk), encoding="utf-8")


def _write_manifest() -> None:
    entries = [
        ("queue_data_audit.md", AUDIT_DIR / "queue_data_audit.md", "Queue field availability audit."),
        ("queue_sensitivity_summary.csv", BACKTEST_DIR / "queue_sensitivity_summary.csv", "Per-seed scenario results."),
        ("queue_sensitivity_aggregate.csv", COMPARISON_DIR / "queue_sensitivity_aggregate.csv", "Aggregated scenario stats."),
        ("queue_aware_verdict_table.csv", COMPARISON_DIR / "queue_aware_verdict_table.csv", "Original vs conservative verdict table."),
        ("net_pnl_vs_fill_probability.png", FIGURES_DIR / "net_pnl_vs_fill_probability.png", "Net PnL vs fill probability."),
        ("good_vs_toxic_fill_haircut.png", FIGURES_DIR / "good_vs_toxic_fill_haircut.png", "Adverse-selection haircut comparison."),
        ("queue_fraction_sensitivity.png", FIGURES_DIR / "queue_fraction_sensitivity.png", "Queue fraction sensitivity."),
        ("cancel_delay_under_queue.png", FIGURES_DIR / "cancel_delay_under_queue.png", "Cancel delay under queue assumptions."),
        ("trade_count_vs_pnl_queue.png", FIGURES_DIR / "trade_count_vs_pnl_queue.png", "Trade count vs PnL scatter."),
        ("queue_aware_validation_report.md", REPORTS_DIR / "queue_aware_validation_report.md", "Main queue-aware validation report."),
        ("queue_aware_slide_summary.md", REPORTS_DIR / "queue_aware_slide_summary.md", "One-page slide summary."),
        ("final_checkpoint.md", REPORTS_DIR / "final_checkpoint.md", "Final checkpoint summary."),
    ]
    rows = []
    for file_name, path, desc in entries:
        rows.append({"file": file_name, "path": str(path), "exists": bool(path.exists()), "description": desc})
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "output_manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    logger = get_logger("run_queue_sensitivity")
    sim_original = _load_original_simulator(logger)
    cfg_stub = _build_cfg_stub()

    static_map = load_static_gate_params()
    specs = build_policy_specs(static_map)

    packs: dict[str, DatasetPack] = {}
    for ds in ["NEW", "NEW3"]:
        feat = load_feature_frame(ds)
        if ds == "NEW":
            feat = feat[feat["split"] == "test"].copy()
        else:
            feat = feat.copy()
        _, trade_df, tick_size, symbol = load_trade_frame(ds)
        feat["session_id"] = feat["session_id"].astype("string")
        trade_df = trade_df.copy()
        trade_df["session_id"] = trade_df["session_id"].astype("string")
        trade_vol = compute_future_trade_volume(feat, trade_df)
        packs[ds] = DatasetPack(dataset=ds, df=feat.reset_index(drop=True), trade_volume_by_event=trade_vol, price_tick=tick_size)
        logger.info("Loaded %s pack rows=%s symbol=%s tick=%.6f", ds, len(feat), symbol, tick_size)

    core_policies = {
        ("NEW", "no_gate_tcn_055"),
        ("NEW", "spread_filter_tcn_055"),
        ("NEW", "combined_filter_tcn_055"),
        ("NEW3", "combined_filter_tcn_055"),
    }
    extra_anchor = {
        ("NEW", "no_gate_tcn_070"),
        ("NEW", "val_selected_tcn_070"),
        ("NEW3", "no_gate_tcn_055"),
    }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        pack = packs[spec.dataset]
        active, force_bid, force_ask = build_gate_masks(pack.df, spec)
        arrs = _to_np(pack.df)
        eff_bid0, eff_ask0 = _build_effective_pulls(
            arrs["prob"], spec.model_threshold, active, force_bid, force_ask, cancel_delay_events=0
        )

        # Model 0 original_fill
        base_metrics = sim_original(
            cfg_stub,
            arrs["prob"],
            arrs["ask"],
            arrs["bid"],
            pack.price_tick,
            threshold=float(spec.model_threshold),
            force_pull_bid_mask=force_bid,
            force_pull_ask_mask=force_ask,
            active_mask=active,
            cancel_delay_events=0,
        )
        base_metrics["active_event_rate"] = float(np.mean(active))
        rows.extend(
            _scenario_rows(
                dataset_pack=pack,
                spec=spec,
                scenario=FillScenario(fill_model="original_fill", scenario_name="original_fill"),
                eff_pull_bid=eff_bid0,
                eff_pull_ask=eff_ask0,
                base_metrics=base_metrics,
            )
        )

        key = (spec.dataset, spec.policy_name)
        run_full = key in core_policies
        run_anchor_only = key in extra_anchor
        if not (run_full or run_anchor_only):
            continue

        # Anchor conservative scenarios for every evaluated policy
        for scenario in [
            FillScenario(
                fill_model="uniform_fill_haircut",
                scenario_name="uniform_p50",
                fill_probability=0.50,
                seeds=tuple(range(20)),
            ),
            FillScenario(
                fill_model="good_fill_haircut",
                scenario_name="good_fill_medium",
                p_good=0.25,
                p_toxic=0.75,
                seeds=tuple(range(20)),
            ),
            FillScenario(
                fill_model="queue_ahead_best_level",
                scenario_name="queue_q50_w5",
                queue_fraction=0.50,
                fill_window=5,
                seeds=(0,),
            ),
        ]:
            rows.extend(
                _scenario_rows(
                    dataset_pack=pack,
                    spec=spec,
                    scenario=scenario,
                    eff_pull_bid=eff_bid0,
                    eff_pull_ask=eff_ask0,
                    base_metrics=base_metrics,
                )
            )

        if not run_full:
            continue

        # Model 1 full sweep
        for p in [0.75, 0.50, 0.25]:
            rows.extend(
                _scenario_rows(
                    dataset_pack=pack,
                    spec=spec,
                    scenario=FillScenario(
                        fill_model="uniform_fill_haircut",
                        scenario_name=f"uniform_p{str(p).replace('.', 'p')}",
                        fill_probability=float(p),
                        seeds=tuple(range(20)),
                    ),
                    eff_pull_bid=eff_bid0,
                    eff_pull_ask=eff_ask0,
                    base_metrics=base_metrics,
                )
            )

        # Model 2 full sweep
        for name, p_good, p_toxic in [
            ("good_fill_mild", 0.50, 0.75),
            ("good_fill_medium", 0.25, 0.75),
            ("good_fill_severe", 0.25, 1.00),
        ]:
            rows.extend(
                _scenario_rows(
                    dataset_pack=pack,
                    spec=spec,
                    scenario=FillScenario(
                        fill_model="good_fill_haircut",
                        scenario_name=name,
                        p_good=float(p_good),
                        p_toxic=float(p_toxic),
                        seeds=tuple(range(20)),
                    ),
                    eff_pull_bid=eff_bid0,
                    eff_pull_ask=eff_ask0,
                    base_metrics=base_metrics,
                )
            )

        # Model 3 full sweep
        for qf in [0.25, 0.50, 1.00]:
            for fw in [1, 3, 5, 10]:
                rows.extend(
                    _scenario_rows(
                        dataset_pack=pack,
                        spec=spec,
                        scenario=FillScenario(
                            fill_model="queue_ahead_best_level",
                            scenario_name=f"queue_q{str(qf).replace('.', 'p')}_w{fw}",
                            queue_fraction=float(qf),
                            fill_window=int(fw),
                            seeds=(0,),
                        ),
                        eff_pull_bid=eff_bid0,
                        eff_pull_ask=eff_ask0,
                        base_metrics=base_metrics,
                    )
                )

        # Model 4 delay sweep (focused)
        for delay in [0, 1, 2, 3, 5, 10]:
            eff_bid_d, eff_ask_d = _build_effective_pulls(
                arrs["prob"], spec.model_threshold, active, force_bid, force_ask, cancel_delay_events=delay
            )
            for scen in [
                FillScenario(
                    fill_model="delayed_cancel_plus_queue",
                    scenario_name="delay_original_like",
                    queue_fraction=0.0,
                    fill_window=1,
                    cancel_delay_events=int(delay),
                    seeds=(0,),
                ),
                FillScenario(
                    fill_model="delayed_cancel_plus_queue",
                    scenario_name="delay_good_medium",
                    p_good=0.25,
                    p_toxic=0.75,
                    queue_fraction=0.50,
                    fill_window=5,
                    cancel_delay_events=int(delay),
                    seeds=tuple(range(10)),
                ),
                FillScenario(
                    fill_model="delayed_cancel_plus_queue",
                    scenario_name="delay_queue_q50",
                    queue_fraction=0.50,
                    fill_window=5,
                    cancel_delay_events=int(delay),
                    seeds=(0,),
                ),
            ]:
                rows.extend(
                    _scenario_rows(
                        dataset_pack=pack,
                        spec=spec,
                        scenario=scen,
                        eff_pull_bid=eff_bid_d,
                        eff_pull_ask=eff_ask_d,
                        base_metrics=base_metrics,
                    )
                )

        logger.info("Completed scenarios for %s / %s", spec.dataset, spec.policy_name)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(BACKTEST_DIR / "queue_sensitivity_summary.csv", index=False)
    logger.info("Saved queue sensitivity summary with %s rows", len(summary_df))

    agg_df = _aggregate(summary_df, random_ref_055=RANDOM_REFERENCE["NEW_055_mean_plus_1std"])
    agg_df.to_csv(COMPARISON_DIR / "queue_sensitivity_aggregate.csv", index=False)
    logger.info("Saved aggregate table with %s rows", len(agg_df))

    verdict_df = _write_verdict_table(summary_df, static_map)
    logger.info("Saved verdict table with %s rows", len(verdict_df))

    _plot_figures(summary_df)
    _write_reports(summary_df, agg_df, verdict_df, static_map)
    _write_manifest()
    logger.info("Queue-aware sensitivity run complete.")


if __name__ == "__main__":
    main()
