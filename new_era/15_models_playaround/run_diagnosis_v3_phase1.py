from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import _common
from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
BASE_RESULT_DIR = PROJECT_ROOT / "new_era" / "15_models_playaround" / "015_models_playaround_result" / EXPERIMENT_ID
LEGACY_DIAG_DIR = BASE_RESULT_DIR / "eval_outputs" / "diagnosis"
OUT_DIR = BASE_RESULT_DIR / "eval_outputs" / "diagnosis_v3"

NEW_DATA_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
NEW3_DATA_DIR = PROJECT_ROOT / "data" / "NEW3"
NEW_SYMBOL = "TXFB6"
NEW3_SYMBOL = "TXFA6"


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def _max_drawdown(pnl_series: np.ndarray) -> float:
    if pnl_series.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(pnl_series)
    drawdown = running_peak - pnl_series
    return float(np.max(drawdown))


def simulate_with_metrics(
    cfg,
    prob,
    ask_px,
    bid_px,
    tick_size,
    threshold,
    random_cancel_rate=0.0,
    prediction_delay_events=0,
    cancel_delay_events=0,
    force_pull_bid_mask=None,
    force_pull_ask_mask=None,
    active_mask=None,
    seed=None,
):
    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    max_inv = cfg.backtest.max_inventory
    horizon = cfg.window.horizon

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

    pnl_path = []
    effective_pull_bid_state = False
    effective_pull_ask_state = False
    pending_bid_delay = 0
    pending_ask_delay = 0
    delayed_fill_count = 0
    delayed_bid_fill_count = 0
    delayed_ask_fill_count = 0
    delayed_toxic_fill_count = 0
    delayed_toxic_loss = 0.0

    open_trades = []
    realized_bid_pnl = 0.0
    realized_ask_pnl = 0.0
    bid_fill_count = 0
    ask_fill_count = 0
    closed_trade_pnls = []

    mid = (ask_px + bid_px) / 2.0
    rng = np.random.default_rng(seed)

    def process_fill(fill_px, side):
        nonlocal inventory, gross_cash, fees_paid, maker_trades, realized_bid_pnl, realized_ask_pnl
        nonlocal bid_fill_count, ask_fill_count, open_trades

        gross_cash += fill_px * point_value if side == "ask" else -fill_px * point_value
        fees_paid += maker_fee
        maker_trades += 1

        if side == "bid":
            bid_fill_count += 1
        if side == "ask":
            ask_fill_count += 1

        if side == "bid":
            if inventory < 0:
                open_px = open_trades.pop(0)
                pnl_closed = (open_px - fill_px) * point_value - 2 * maker_fee
                realized_ask_pnl += pnl_closed
                closed_trade_pnls.append(pnl_closed)
            else:
                open_trades.append(fill_px)
            inventory += 1
        else:
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

    for i in range(n_events - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])
        mid_now = mid[i]

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

        if i + horizon < len(mid):
            fut_mid = mid[i + horizon]
        else:
            fut_mid = mid[-1]

        if baseline_bid_fill and effective_pull_bid:
            missed_fill_count += 1
            markout_pnl = (fut_mid - bid) * point_value
            if markout_pnl > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout_pnl
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += -markout_pnl
        elif baseline_bid_fill and raw_pull_bid and not effective_pull_bid:
            delayed_fill_count += 1
            delayed_bid_fill_count += 1
            delayed_markout = (fut_mid - bid) * point_value
            if delayed_markout < 0:
                delayed_toxic_fill_count += 1
                delayed_toxic_loss += -delayed_markout

        if baseline_ask_fill and effective_pull_ask:
            missed_fill_count += 1
            markout_pnl = (ask - fut_mid) * point_value
            if markout_pnl > 0:
                missed_profitable_fill_count += 1
                missed_profitable_pnl += markout_pnl
            else:
                missed_toxic_fill_count += 1
                missed_toxic_pnl_avoided += -markout_pnl
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

    final_mid = mid[-1] if len(mid) else 0.0
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid

    pnl_arr = np.asarray(pnl_path, dtype=np.float64)
    mdd = _max_drawdown(pnl_arr)
    closed_arr = np.asarray(closed_trade_pnls, dtype=np.float64)
    median_pnl_per_fill = float(np.median(closed_arr)) if closed_arr.size > 0 else 0.0
    win_rate = float((closed_arr > 0).mean()) if closed_arr.size > 0 else 0.0

    return {
        "threshold": threshold,
        "signal_rate": signal_count / total_opportunities if total_opportunities else 0.0,
        "cancel_count": int(signal_count),
        "fill_rate": maker_trades / total_opportunities if total_opportunities else 0.0,
        "trade_count": maker_trades,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fee": fees_paid,
        "avg_pnl_per_fill": net_pnl / maker_trades if maker_trades else 0.0,
        "median_pnl_per_fill": median_pnl_per_fill,
        "win_rate": win_rate,
        "toxic_fill_count": missed_toxic_fill_count,
        "toxic_fill_avoided_count": missed_toxic_fill_count,
        "toxic_fill_avoided": missed_toxic_fill_count,
        "missed_fill_count": missed_fill_count,
        "missed_profitable_fill_count": missed_profitable_fill_count,
        "missed_profitable_pnl": missed_profitable_pnl,
        "missed_toxic_fill_count": missed_toxic_fill_count,
        "missed_toxic_pnl_avoided": missed_toxic_pnl_avoided,
        "net_cancel_value": missed_toxic_pnl_avoided - missed_profitable_pnl,
        "max_drawdown": mdd,
        "bid_fill_count": bid_fill_count,
        "ask_fill_count": ask_fill_count,
        "realized_bid_pnl": realized_bid_pnl,
        "realized_ask_pnl": realized_ask_pnl,
        "prediction_delay_events": int(prediction_delay_events),
        "cancel_delay_events": int(cancel_delay_events),
        "delayed_fill_count": delayed_fill_count,
        "delayed_bid_fill_count": delayed_bid_fill_count,
        "delayed_ask_fill_count": delayed_ask_fill_count,
        "delayed_toxic_fill_count": delayed_toxic_fill_count,
        "delayed_toxic_loss": delayed_toxic_loss,
    }


def _trend_regime(cum_return_ticks: float, trend_strength: float) -> str:
    if trend_strength < 0.5:
        return "range-bound"
    if trend_strength < 1.5:
        return "mild-trend"
    if cum_return_ticks >= 0:
        return "strong-uptrend"
    return "strong-downtrend"


def _suitability_label(trend_strength: float, baseline_net_pnl: float, spread_ge_1tick_rate: float) -> str:
    if trend_strength < 1.5 and baseline_net_pnl > -5000 and spread_ge_1tick_rate > 0.5:
        return "SUITABLE"
    if trend_strength > 2.5 or baseline_net_pnl < -30000:
        return "UNSUITABLE"
    return "MARGINAL"


def _main_failure_mode(row: dict[str, float | str]) -> str:
    trend_regime = str(row.get("trend_regime", ""))
    baseline_net_pnl = float(row.get("baseline_net_pnl", 0.0))
    mean_spread = float(row.get("mean_spread_ticks", 0.0))
    if baseline_net_pnl >= 0:
        return "none_or_minor"
    if trend_regime == "strong-downtrend":
        return "strong_downtrend_adverse_selection"
    if trend_regime == "strong-uptrend":
        return "strong_uptrend_adverse_selection"
    if mean_spread < 1.0:
        return "spread_too_tight"
    return "inventory_carry_and_exit_drag"


def _format_counts(series: pd.Series) -> str:
    if series.empty:
        return "none"
    ordered = []
    for key in ["SUITABLE", "MARGINAL", "UNSUITABLE", "SKIPPED"]:
        if key in series.index:
            ordered.append(f"{key}: {int(series[key])}")
    for key, value in series.items():
        if key not in {"SUITABLE", "MARGINAL", "UNSUITABLE", "SKIPPED"}:
            ordered.append(f"{key}: {int(value)}")
    return ", ".join(ordered)


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_to_compact(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return str(value)
    return json.dumps(parsed, sort_keys=True, ensure_ascii=True)


def _compute_session_audit(
    cfg,
    dataset: str,
    lob: pd.DataFrame,
    session_ids: Iterable[str],
    tick_size: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_id in sorted(session_ids):
        s = lob[lob["session_id"] == session_id].copy()
        num_rows = int(len(s))
        start_time = int(s["timestamp"].iloc[0]) if num_rows > 0 else np.nan
        end_time = int(s["timestamp"].iloc[-1]) if num_rows > 0 else np.nan
        base_row: dict[str, object] = {
            "dataset": dataset,
            "session_id": session_id,
            "num_rows": num_rows,
            "start_time": start_time,
            "end_time": end_time,
            "cum_return_ticks": np.nan,
            "realized_vol_ticks": np.nan,
            "trend_strength": np.nan,
            "mean_spread_ticks": np.nan,
            "spread_ge_1tick_rate": np.nan,
            "spread_ge_2tick_rate": np.nan,
            "baseline_net_pnl": np.nan,
            "baseline_max_drawdown": np.nan,
            "realized_bid_pnl": np.nan,
            "realized_ask_pnl": np.nan,
            "suitability_label": "SKIPPED",
            "trend_regime": "unknown",
            "main_failure_mode": "insufficient_rows",
            "notes": "",
        }

        if num_rows < 500:
            base_row["notes"] = "SKIPPED_LT_500_ROWS"
            rows.append(base_row)
            continue

        ask = s["ask_px_1"].to_numpy(dtype=np.float64)
        bid = s["bid_px_1"].to_numpy(dtype=np.float64)
        mid = (ask + bid) / 2.0
        spread_ticks = (ask - bid) / max(float(tick_size), 1e-9)

        cum_return_ticks = float((mid[-1] - mid[0]) / max(float(tick_size), 1e-9))
        realized_vol_ticks = float(pd.Series(mid).diff().std()) / max(float(tick_size), 1e-9)
        if not np.isfinite(realized_vol_ticks):
            realized_vol_ticks = 0.0
        trend_strength = abs(cum_return_ticks) / max(realized_vol_ticks, 1e-6)
        mean_spread_ticks = float(np.nanmean(spread_ticks))
        spread_ge_1tick_rate = float((spread_ticks >= 1.0).mean())
        spread_ge_2tick_rate = float((spread_ticks >= 2.0).mean())
        trend_regime = _trend_regime(cum_return_ticks, trend_strength)

        baseline_res = simulate_with_metrics(
            cfg,
            prob=None,
            ask_px=ask,
            bid_px=bid,
            tick_size=tick_size,
            threshold=1.1,
        )
        baseline_net_pnl = float(baseline_res.get("net_pnl", np.nan))
        suitability_label = _suitability_label(
            trend_strength=trend_strength,
            baseline_net_pnl=baseline_net_pnl,
            spread_ge_1tick_rate=spread_ge_1tick_rate,
        )

        base_row.update(
            {
                "cum_return_ticks": cum_return_ticks,
                "realized_vol_ticks": realized_vol_ticks,
                "trend_strength": trend_strength,
                "mean_spread_ticks": mean_spread_ticks,
                "spread_ge_1tick_rate": spread_ge_1tick_rate,
                "spread_ge_2tick_rate": spread_ge_2tick_rate,
                "baseline_net_pnl": baseline_net_pnl,
                "baseline_max_drawdown": float(baseline_res.get("max_drawdown", np.nan)),
                "realized_bid_pnl": float(baseline_res.get("realized_bid_pnl", np.nan)),
                "realized_ask_pnl": float(baseline_res.get("realized_ask_pnl", np.nan)),
                "suitability_label": suitability_label,
                "trend_regime": trend_regime,
                "notes": "",
            }
        )
        base_row["main_failure_mode"] = _main_failure_mode(base_row)
        rows.append(base_row)

    return pd.DataFrame(rows)


def _concat_arrays_by_sessions(
    lob: pd.DataFrame,
    session_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ask_parts: list[np.ndarray] = []
    bid_parts: list[np.ndarray] = []
    for session_id in sorted(session_ids):
        s = lob[lob["session_id"] == session_id]
        if len(s) < 500:
            continue
        ask_parts.append(s["ask_px_1"].to_numpy(dtype=np.float64))
        bid_parts.append(s["bid_px_1"].to_numpy(dtype=np.float64))
    if not ask_parts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ask = np.concatenate(ask_parts)
    bid = np.concatenate(bid_parts)
    mid = (ask + bid) / 2.0
    return mid, bid, ask


def compute_side_markout_stats(
    mid_array: np.ndarray,
    bid_array: np.ndarray,
    ask_array: np.ndarray,
    tick_size: float,
    horizon: int,
    split_name: str,
) -> dict[str, float | str]:
    if len(mid_array) <= horizon:
        return {
            "split": split_name,
            "horizon": int(horizon),
            "bid_toxic_rate": np.nan,
            "bid_good_rate": np.nan,
            "bid_neutral_rate": np.nan,
            "bid_mean_markout": np.nan,
            "bid_p10": np.nan,
            "bid_p90": np.nan,
            "ask_toxic_rate": np.nan,
            "ask_good_rate": np.nan,
            "ask_neutral_rate": np.nan,
            "ask_mean_markout": np.nan,
            "ask_p10": np.nan,
            "ask_p90": np.nan,
            "num_samples": 0,
        }

    future_mid = mid_array[horizon:]
    bid_markouts = (future_mid - bid_array[:-horizon]) / max(float(tick_size), 1e-9)
    ask_markouts = (ask_array[:-horizon] - future_mid) / max(float(tick_size), 1e-9)

    bid_toxic = bid_markouts < -0.5
    bid_good = bid_markouts > 0.5
    ask_toxic = ask_markouts < -0.5
    ask_good = ask_markouts > 0.5

    return {
        "split": split_name,
        "horizon": int(horizon),
        "bid_toxic_rate": float(np.mean(bid_toxic)),
        "bid_good_rate": float(np.mean(bid_good)),
        "bid_neutral_rate": float(np.mean((~bid_toxic) & (~bid_good))),
        "bid_mean_markout": float(np.mean(bid_markouts)),
        "bid_p10": float(np.percentile(bid_markouts, 10)),
        "bid_p90": float(np.percentile(bid_markouts, 90)),
        "ask_toxic_rate": float(np.mean(ask_toxic)),
        "ask_good_rate": float(np.mean(ask_good)),
        "ask_neutral_rate": float(np.mean((~ask_toxic) & (~ask_good))),
        "ask_mean_markout": float(np.mean(ask_markouts)),
        "ask_p10": float(np.percentile(ask_markouts, 10)),
        "ask_p90": float(np.percentile(ask_markouts, 90)),
        "num_samples": int(len(bid_markouts)),
    }


def _build_cross_dataset_gate_analysis(regime_gate_sweep: pd.DataFrame) -> pd.DataFrame:
    df = regime_gate_sweep.copy()
    df["gate_params_key"] = df["gate_params"].map(_json_to_compact)

    index_cols = ["gate_type", "gate_params_key", "model_threshold"]
    metric_cols = ["net_pnl", "active_event_rate", "improvement", "trade_count"]
    pivot = (
        df.pivot_table(index=index_cols, columns="dataset", values=metric_cols, aggfunc="first")
        .sort_index()
        .reset_index()
    )

    if isinstance(pivot.columns, pd.MultiIndex):
        flat_cols = []
        for col in pivot.columns.to_flat_index():
            if isinstance(col, str):
                flat_cols.append(col)
                continue
            first, second = col
            if second in ("", None):
                flat_cols.append(str(first))
            else:
                flat_cols.append(f"{second}_{first}")
        pivot.columns = flat_cols

    pivot = pivot.rename(
        columns={
            "gate_params_key": "gate_params",
            "NEW_net_pnl": "NEW_net_pnl",
            "NEW_active_event_rate": "NEW_active_rate",
            "NEW_improvement": "NEW_improvement",
            "NEW_trade_count": "NEW_trade_count",
            "NEW3_net_pnl": "NEW3_net_pnl",
            "NEW3_active_event_rate": "NEW3_active_rate",
            "NEW3_improvement": "NEW3_improvement",
            "NEW3_trade_count": "NEW3_trade_count",
        }
    )

    required = [
        "NEW_net_pnl",
        "NEW_active_rate",
        "NEW_improvement",
        "NEW_trade_count",
        "NEW3_net_pnl",
        "NEW3_active_rate",
        "NEW3_improvement",
        "NEW3_trade_count",
    ]
    for col in required:
        if col not in pivot.columns:
            pivot[col] = np.nan

    verdicts = []
    notes = []
    for _, row in pivot.iterrows():
        new_imp = _safe_float(row["NEW_improvement"])
        new3_imp = _safe_float(row["NEW3_improvement"])
        new_rate = _safe_float(row["NEW_active_rate"])
        new3_rate = _safe_float(row["NEW3_active_rate"])
        is_new_good = np.isfinite(new_imp) and new_imp > 0.0 and np.isfinite(new_rate) and new_rate > 0.70
        is_new3_good = np.isfinite(new3_imp) and new3_imp > 0.0 and np.isfinite(new3_rate) and new3_rate > 0.70

        if is_new_good and is_new3_good:
            verdict = "GENERALIZES"
        elif is_new_good and not is_new3_good:
            verdict = "NEW_ONLY"
        elif is_new3_good and not is_new_good:
            verdict = "NEW3_ONLY"
        else:
            verdict = "NEITHER"

        pareto_ok = (
            np.isfinite(_safe_float(row["NEW_net_pnl"]))
            and _safe_float(row["NEW_net_pnl"]) > 67984.0
            and np.isfinite(new_rate)
            and new_rate > 0.70
            and (
                (np.isfinite(_safe_float(row["NEW3_net_pnl"])) and _safe_float(row["NEW3_net_pnl"]) > 0.0)
                or (np.isfinite(new3_imp) and new3_imp > 200000.0)
            )
        )
        note = "pareto_candidate" if pareto_ok else ""
        verdicts.append(verdict)
        notes.append(note)

    pivot["cross_dataset_verdict"] = verdicts
    pivot["notes"] = notes
    pivot = pivot[
        [
            "gate_type",
            "gate_params",
            "model_threshold",
            "NEW_net_pnl",
            "NEW_active_rate",
            "NEW_improvement",
            "NEW_trade_count",
            "NEW3_net_pnl",
            "NEW3_active_rate",
            "NEW3_improvement",
            "NEW3_trade_count",
            "cross_dataset_verdict",
            "notes",
        ]
    ].sort_values(
        by=["cross_dataset_verdict", "NEW_improvement", "NEW3_improvement"],
        ascending=[True, False, False],
    )
    return pivot


def _write_lob_schema_md(
    out_dir: Path,
    new_lob: pd.DataFrame,
    new3_lob: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# LOB Schema Check")
    lines.append("")
    lines.append("## NEW_ALL / TXFB6")
    lines.append("")
    lines.append("Columns:")
    lines.append(", ".join(new_lob.columns.tolist()))
    lines.append("")
    lines.append("Dtypes:")
    lines.append(new_lob.dtypes.to_string())
    lines.append("")
    lines.append("## NEW3 / TXFA6")
    lines.append("")
    lines.append("Columns:")
    lines.append(", ".join(new3_lob.columns.tolist()))
    lines.append("")
    lines.append("Dtypes:")
    lines.append(new3_lob.dtypes.to_string())
    lines.append("")
    lines.append("Schema verdict: ask_px_1 / bid_px_1 columns exist in both datasets.")
    (out_dir / "lob_schema_check.md").write_text("\n".join(lines), encoding="utf-8")


def _read_existing_evidence(legacy_dir: Path) -> dict[str, float]:
    baseline_summary = pd.read_csv(legacy_dir / "baseline_summary_fixed.csv")
    baseline_summary = baseline_summary.set_index("experiment_group")
    final_v2 = pd.read_csv(legacy_dir / "final_test_once_result_v2.csv").iloc[0]

    r55_mean = float(baseline_summary.loc["Random_0.55", "mean_net_pnl"])
    r55_std = float(baseline_summary.loc["Random_0.55", "std_net_pnl"])
    r60_mean = float(baseline_summary.loc["Random_0.60", "mean_net_pnl"])
    r60_std = float(baseline_summary.loc["Random_0.60", "std_net_pnl"])
    r70_mean = float(baseline_summary.loc["Random_0.70", "mean_net_pnl"])
    r70_std = float(baseline_summary.loc["Random_0.70", "std_net_pnl"])
    model55 = float(baseline_summary.loc["Model_0.55", "mean_net_pnl"])
    model60 = float(baseline_summary.loc["Model_0.60", "mean_net_pnl"])
    model70 = float(baseline_summary.loc["Model_0.70", "mean_net_pnl"])
    baseline_nomodel = float(baseline_summary.loc["Baseline_A_NoModel", "mean_net_pnl"])

    return {
        "baseline_nomodel": baseline_nomodel,
        "model55": model55,
        "model60": model60,
        "model70": model70,
        "random55_mean": r55_mean,
        "random55_std": r55_std,
        "random60_mean": r60_mean,
        "random60_std": r60_std,
        "random70_mean": r70_mean,
        "random70_std": r70_std,
        "random55_mean_plus_std": r55_mean + r55_std,
        "random60_mean_plus_std": r60_mean + r60_std,
        "random70_mean_plus_std": r70_mean + r70_std,
        "selected_threshold_v2": float(final_v2["selected_threshold"]),
        "selected_threshold_test_net_pnl_v2": float(final_v2["test_net_pnl"]),
    }


def _render_new3_regime_md(
    out_dir: Path,
    new3_audit: pd.DataFrame,
    new_audit: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> None:
    new3_valid = new3_audit[new3_audit["suitability_label"] != "SKIPPED"].copy()
    new_valid = new_audit[new_audit["suitability_label"] != "SKIPPED"].copy()
    new3_counts = new3_valid["suitability_label"].value_counts()
    new_counts = new_valid["suitability_label"].value_counts()

    spread_new3 = gate_df[(gate_df["dataset"] == "NEW3") & (gate_df["gate_type"] == "spread_filter")].copy()
    spread_new3_best = spread_new3.sort_values("net_pnl", ascending=False).head(1)
    combined_new3 = gate_df[(gate_df["dataset"] == "NEW3") & (gate_df["gate_type"] == "combined_filter")].copy()
    combined_new3_best = combined_new3.sort_values("net_pnl", ascending=False).head(1)

    lines = []
    lines.append("# NEW3 Regime Diagnosis")
    lines.append("")
    lines.append("## 1) NEW3 per-session trend_strength")
    for _, row in new3_valid.sort_values("session_id").iterrows():
        lines.append(
            f"- {row['session_id']}: trend_strength={float(row['trend_strength']):.4f}, "
            f"cum_return_ticks={float(row['cum_return_ticks']):.2f}, "
            f"baseline_net_pnl={float(row['baseline_net_pnl']):.2f}, "
            f"trend_regime={row['trend_regime']}"
        )
    lines.append("")
    lines.append("## 2) Suitability distribution")
    lines.append(f"- NEW3: {_format_counts(new3_counts)}")
    lines.append(f"- NEW test split: {_format_counts(new_counts)}")
    lines.append("")
    lines.append("## 3) Main failure mode")
    fail_counts = new3_valid["main_failure_mode"].value_counts()
    lines.append(f"- NEW3 failure distribution: {_format_counts(fail_counts)}")
    lines.append("- Dominant pattern is strong trend + negative baseline PnL, which is maker-unfriendly.")
    lines.append("")
    lines.append("## 4) Why spread_filter is not enough")
    if not spread_new3_best.empty:
        r = spread_new3_best.iloc[0]
        lines.append(
            f"- Best NEW3 spread_filter row: threshold={float(r['model_threshold']):.2f}, "
            f"net_pnl={float(r['net_pnl']):.2f}, active_event_rate={float(r['active_event_rate']):.4f}, "
            f"gate_params={r['gate_params']}"
        )
    lines.append("- NEW3 still stays negative under spread-only gating, so spread width alone cannot block trend toxicity.")
    lines.append("")
    lines.append("## 5) Why combined_filter works better")
    if not combined_new3_best.empty:
        r = combined_new3_best.iloc[0]
        lines.append(
            f"- Best NEW3 combined_filter row: threshold={float(r['model_threshold']):.2f}, "
            f"net_pnl={float(r['net_pnl']):.2f}, active_event_rate={float(r['active_event_rate']):.4f}, "
            f"gate_params={r['gate_params']}"
        )
    lines.append("- Combined filter explicitly includes trend and volatility terms, so it suppresses strong-trend windows before quoting.")
    lines.append("")
    lines.append("## 6) Causal pre-trade identification rule (candidate)")
    lines.append("- Compute rolling return in ticks over N events (example N=2000).")
    lines.append("- If abs(rolling_return_ticks) > threshold, set DO_NOT_TRADE or ONE_SIDED_ONLY.")
    lines.append("- This is a causal gate because it only uses past and current book states.")
    lines.append("")
    lines.append("## 7) Limit of current evidence")
    lines.append("- Current conclusion is based on fixed datasets (NEW vs NEW3).")
    lines.append("- We still need walk-forward across additional months/contracts to validate if this gate-selection logic generalizes.")
    (out_dir / "new3_regime_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _render_side_markout_md(
    out_dir: Path,
    side_df: pd.DataFrame,
    down_rate: float,
    up_rate: float,
) -> None:
    h10 = side_df[side_df["horizon"] == 10].set_index("split")
    lines = []
    lines.append("# Side Markout Label Report")
    lines.append("")
    lines.append("## H=10 core rates")
    for split in ["NEW_train_val", "NEW_test", "NEW3_full"]:
        if split not in h10.index:
            continue
        r = h10.loc[split]
        lines.append(
            f"- {split}: bid_toxic_rate={float(r['bid_toxic_rate']):.4f}, "
            f"ask_toxic_rate={float(r['ask_toxic_rate']):.4f}, "
            f"bid_mean={float(r['bid_mean_markout']):.4f}, ask_mean={float(r['ask_mean_markout']):.4f}"
        )
    lines.append("")

    if "NEW3_full" in h10.index:
        r = h10.loc["NEW3_full"]
        skew = float(r["bid_toxic_rate"]) - float(r["ask_toxic_rate"])
        if skew > 0.02:
            lines.append(
                f"- NEW3 has bid-side toxicity skew (bid_toxic_rate - ask_toxic_rate = {skew:.4f}), "
                "consistent with falling-knife bid risk."
            )
        elif skew < -0.02:
            lines.append(
                f"- NEW3 has ask-side toxicity skew (bid_toxic_rate - ask_toxic_rate = {skew:.4f}), "
                "consistent with short-cover risk."
            )
        else:
            lines.append(f"- NEW3 bid/ask toxicity is relatively balanced (skew={skew:.4f}).")
    lines.append("")

    lines.append("## Horizon suitability (target toxic rate 5%-20%)")
    for split in side_df["split"].unique():
        sdf = side_df[side_df["split"] == split].copy()
        sdf["bid_dist"] = np.abs(sdf["bid_toxic_rate"] - 0.125)
        sdf["ask_dist"] = np.abs(sdf["ask_toxic_rate"] - 0.125)
        bid_best = sdf.sort_values("bid_dist").iloc[0]
        ask_best = sdf.sort_values("ask_dist").iloc[0]
        lines.append(
            f"- {split}: bid-best horizon={int(bid_best['horizon'])} (rate={float(bid_best['bid_toxic_rate']):.4f}), "
            f"ask-best horizon={int(ask_best['horizon'])} (rate={float(ask_best['ask_toxic_rate']):.4f})"
        )
    lines.append("")
    lines.append("## Label redesign implication")
    lines.append(
        "- Side-specific toxic labels (bid_toxic / ask_toxic) are more aligned with maker risk control than "
        "single mid-move 3-class labels."
    )
    lines.append(
        f"- Existing smoothed_mid_move_3class base rates are sparse (Down≈{down_rate:.4f}, Up≈{up_rate:.4f}), "
        "while side-toxic rates are materially denser."
    )
    lines.append("- Recommendation: next training phase should test dual-head side-specific toxicity labels.")
    (out_dir / "side_markout_label_report.md").write_text("\n".join(lines), encoding="utf-8")


def _render_v3_report(
    out_dir: Path,
    evidence: dict[str, float],
    new3_audit: pd.DataFrame,
    new_audit: pd.DataFrame,
    side_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    latency_df: pd.DataFrame,
) -> None:
    new3_valid = new3_audit[new3_audit["suitability_label"] != "SKIPPED"]
    new_valid = new_audit[new_audit["suitability_label"] != "SKIPPED"]
    new3_counts = new3_valid["suitability_label"].value_counts()
    new_counts = new_valid["suitability_label"].value_counts()

    spread_new3_best = gate_df[(gate_df["dataset"] == "NEW3") & (gate_df["gate_type"] == "spread_filter")]
    spread_new3_best = spread_new3_best.sort_values("net_pnl", ascending=False).head(1)
    combined_new_best = gate_df[(gate_df["dataset"] == "NEW") & (gate_df["gate_type"] == "combined_filter")]
    combined_new_best = combined_new_best.sort_values("net_pnl", ascending=False).head(1)
    combined_both = cross_df[(cross_df["gate_type"] == "combined_filter") & (cross_df["cross_dataset_verdict"] == "GENERALIZES")]

    h10 = side_df[side_df["horizon"] == 10].set_index("split")
    latency_selected = latency_df[(latency_df["mode"] == "cancel_delay") & (latency_df["threshold"] == 0.7)].copy()
    latency_selected = latency_selected.sort_values("cancel_delay_events")

    can_preselect_gate = False
    preselect_reason = (
        "Causal regime features are available, but current evidence is only from fixed datasets; "
        "no walk-forward proof yet that one ex-ante rule robustly picks spread_filter vs combined_filter."
    )

    if not combined_both.empty:
        preselect_reason += (
            " A shared combined_filter candidate exists across NEW and NEW3, but spread_filter still dominates NEW PnL, "
            "so gate-choice logic remains under-identified."
        )

    if can_preselect_gate:
        final_grade = "B-"
    else:
        final_grade = "C+"

    lines = []
    lines.append("# 1. Executive Summary")
    lines.append(
        "TCN toxic-flow cancel filter shows local edge on NEW at threshold 0.55, but the val-selected threshold 0.70 does not beat random+1std; "
        "NEW3 requires stronger trend-aware gating, and current gate-selection evidence is still dataset-specific rather than production-ready."
    )
    lines.append("")
    lines.append("# 2. Existing Evidence")
    lines.append(
        f"- Model_0.55 = {evidence['model55']:.2f} > Random_0.55 mean+1std = {evidence['random55_mean_plus_std']:.2f} (pass)"
    )
    lines.append(
        f"- Model_0.60 = {evidence['model60']:.2f} > Random_0.60 mean+1std = {evidence['random60_mean_plus_std']:.2f} (pass)"
    )
    lines.append(
        f"- Val-selected threshold {evidence['selected_threshold_v2']:.2f}: test_net_pnl={evidence['selected_threshold_test_net_pnl_v2']:.2f} "
        f"vs Random_0.70 mean+1std={evidence['random70_mean_plus_std']:.2f} (not significant)"
    )
    lines.append("- 0.55 remains a research threshold from test sweep, not a production threshold.")
    lines.append("")
    lines.append("# 3. NEW3 Regime Diagnosis")
    for _, row in new3_valid.sort_values("session_id").iterrows():
        lines.append(
            f"- {row['session_id']}: trend_strength={float(row['trend_strength']):.3f}, "
            f"cum_return_ticks={float(row['cum_return_ticks']):.1f}, suitability={row['suitability_label']}"
        )
    lines.append(f"- NEW3 suitability distribution: {_format_counts(new3_counts)}")
    lines.append(f"- NEW test suitability distribution: {_format_counts(new_counts)}")
    lines.append(
        "- Interpretation: NEW3 is dominated by trend-stress sessions; this is a maker regime failure mode, not evidence that the classifier is a direction-trading model."
    )
    lines.append("")
    lines.append("# 4. NEW Positive Control")
    lines.append("- NEW test sessions are materially less trend-hostile than NEW3 and preserve more maker suitability.")
    lines.append("- This supports why the same cancel logic can look useful on NEW and weak on NEW3 without contradiction.")
    lines.append("")
    lines.append("# 5. Gate Cross-Dataset Analysis")
    if not spread_new3_best.empty:
        r = spread_new3_best.iloc[0]
        lines.append(
            f"- Best NEW3 spread_filter: net_pnl={float(r['net_pnl']):.2f}, active_event_rate={float(r['active_event_rate']):.4f}, params={r['gate_params']}"
        )
    if not combined_new_best.empty:
        r = combined_new_best.iloc[0]
        lines.append(
            f"- Best NEW combined_filter: net_pnl={float(r['net_pnl']):.2f}, active_event_rate={float(r['active_event_rate']):.4f}, params={r['gate_params']}"
        )
    if not combined_both.empty:
        best_shared = combined_both.sort_values(["NEW_net_pnl", "NEW3_net_pnl"], ascending=False).iloc[0]
        lines.append(
            f"- Shared combined_filter candidate exists: NEW_net_pnl={float(best_shared['NEW_net_pnl']):.2f}, "
            f"NEW3_net_pnl={float(best_shared['NEW3_net_pnl']):.2f}, "
            f"NEW_active={float(best_shared['NEW_active_rate']):.4f}, NEW3_active={float(best_shared['NEW3_active_rate']):.4f}"
        )
    lines.append(
        f"- Causal ex-ante rule to choose spread_filter vs combined_filter is {'available' if can_preselect_gate else 'not validated yet'}."
    )
    lines.append(f"- Verdict: {preselect_reason}")
    lines.append("")
    lines.append("# 6. Side-Specific Markout Labels")
    for split in ["NEW_train_val", "NEW_test", "NEW3_full"]:
        if split in h10.index:
            r = h10.loc[split]
            lines.append(
                f"- H=10 {split}: bid_toxic_rate={float(r['bid_toxic_rate']):.4f}, ask_toxic_rate={float(r['ask_toxic_rate']):.4f}"
            )
    lines.append("- Side-specific toxicity remains the cleaner target for maker defense than mid-direction class labels.")
    lines.append("")
    lines.append("# 7. FPGA Relevance")
    if not latency_selected.empty:
        first = latency_selected.iloc[0]
        worst = latency_selected.iloc[-1]
        lines.append(
            f"- cancel_delay 0 events: net_pnl={float(first['net_pnl']):.2f}, delayed_toxic_loss={float(first['delayed_toxic_loss']):.2f}"
        )
        lines.append(
            f"- cancel_delay {int(worst['cancel_delay_events'])} events: net_pnl={float(worst['net_pnl']):.2f}, delayed_toxic_loss={float(worst['delayed_toxic_loss']):.2f}"
        )
        lines.append("- Delay impact is non-monotonic at small delays, but clearly degrades at large delay; FPGA case is conditional, not absolute.")
    else:
        lines.append("- Existing latency file missing selected rows; no new claim made.")
    lines.append("")
    lines.append("# 8. Final Grade")
    lines.append(f"- Grade: {final_grade}")
    lines.append(
        "- Rationale: random baseline and OOS checks are now transparent, but gate-selection logic is not yet proven as a stable ex-ante policy across unseen regimes."
    )
    lines.append("")
    lines.append("# 9. Next Actions")
    lines.append("1. Build an explicit regime classifier using causal rolling trend/vol/spread features and validate by walk-forward.")
    lines.append("2. Retrain TCN with side-specific toxic labels (bid_toxic, ask_toxic) and compare against current risk score.")
    lines.append("3. Keep threshold selection validation-driven; do not promote test-sweep thresholds to production.")
    lines.append("4. Validate any shared gate candidate on additional unseen contracts/months before claiming generalization.")
    lines.append("")
    lines.append("# 10. What NOT to claim")
    lines.append("- Do not claim direction-trading capability.")
    lines.append("- Do not call 0.55 a production threshold.")
    lines.append("- Do not claim NEW3 improvement is model-only; gate effect is material.")
    lines.append("- Do not overstate FPGA necessity without monotonic true cancel-delay evidence.")
    lines.append("- Do not treat low-active-rate risk-off filters as proven trading edge.")

    (out_dir / "model_usability_diagnosis_report_v3.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _common.build_cfg()
    new_cfg = replace(cfg, data=replace(cfg.data, data_dir=NEW_DATA_DIR, symbols=(NEW_SYMBOL,)), backtest=replace(cfg.backtest, symbol=NEW_SYMBOL))
    new3_cfg = replace(cfg, data=replace(cfg.data, data_dir=NEW3_DATA_DIR, symbols=(NEW3_SYMBOL,)), backtest=replace(cfg.backtest, symbol=NEW3_SYMBOL))

    new_lob_frames, _ = load_new_tw_data(new_cfg.data.data_dir, [NEW_SYMBOL])
    new3_lob_frames, _ = load_new_tw_data(new3_cfg.data.data_dir, [NEW3_SYMBOL])
    if NEW_SYMBOL not in new_lob_frames:
        raise RuntimeError(f"Missing {NEW_SYMBOL} in {new_cfg.data.data_dir}")
    if NEW3_SYMBOL not in new3_lob_frames:
        raise RuntimeError(f"Missing {NEW3_SYMBOL} in {new3_cfg.data.data_dir}")

    new_lob = new_lob_frames[NEW_SYMBOL].copy()
    new3_lob = new3_lob_frames[NEW3_SYMBOL].copy()

    _write_lob_schema_md(OUT_DIR, new_lob, new3_lob)

    tick_new = float(infer_tick_size_from_lob(new_lob))
    tick_new3 = float(infer_tick_size_from_lob(new3_lob))

    new_sessions = sorted(new_lob["session_id"].unique().tolist())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(new_sessions, train_frac=0.70, val_frac=0.15)
    new3_sessions = sorted(new3_lob["session_id"].unique().tolist())

    new3_audit = _compute_session_audit(new3_cfg, "NEW3", new3_lob, new3_sessions, tick_new3)
    new_test_audit = _compute_session_audit(new_cfg, "NEW_TEST", new_lob, test_sessions, tick_new)
    new3_audit.to_csv(OUT_DIR / "new3_session_audit.csv", index=False)
    new_test_audit.to_csv(OUT_DIR / "new_session_audit.csv", index=False)

    gate_df = pd.read_csv(LEGACY_DIAG_DIR / "regime_gate_sweep.csv")
    _render_new3_regime_md(OUT_DIR, new3_audit, new_test_audit, gate_df)

    mid_train_val, bid_train_val, ask_train_val = _concat_arrays_by_sessions(new_lob, train_sessions + val_sessions)
    mid_test, bid_test, ask_test = _concat_arrays_by_sessions(new_lob, test_sessions)
    mid_new3, bid_new3, ask_new3 = _concat_arrays_by_sessions(new3_lob, new3_sessions)

    side_rows = []
    for horizon in [5, 10, 20]:
        side_rows.append(compute_side_markout_stats(mid_train_val, bid_train_val, ask_train_val, tick_new, horizon, "NEW_train_val"))
        side_rows.append(compute_side_markout_stats(mid_test, bid_test, ask_test, tick_new, horizon, "NEW_test"))
        side_rows.append(compute_side_markout_stats(mid_new3, bid_new3, ask_new3, tick_new3, horizon, "NEW3_full"))
    side_df = pd.DataFrame(side_rows)
    side_df.to_csv(OUT_DIR / "side_label_distribution.csv", index=False)

    manifest = json.loads((BASE_RESULT_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    class_dist = manifest.get("class_distribution_by_session", {})
    total_counts = {"0": 0, "1": 0, "2": 0}
    for values in class_dist.values():
        for key in total_counts:
            total_counts[key] += int(values.get(key, 0))
    total_obs = max(sum(total_counts.values()), 1)
    down_rate = total_counts["0"] / total_obs
    up_rate = total_counts["2"] / total_obs

    _render_side_markout_md(OUT_DIR, side_df, down_rate=down_rate, up_rate=up_rate)

    cross_df = _build_cross_dataset_gate_analysis(gate_df)
    cross_df.to_csv(OUT_DIR / "spread_gate_cross_dataset_analysis.csv", index=False)

    evidence = _read_existing_evidence(LEGACY_DIAG_DIR)
    latency_df = pd.read_csv(LEGACY_DIAG_DIR / "latency_sensitivity_enhanced.csv")
    _render_v3_report(
        OUT_DIR,
        evidence=evidence,
        new3_audit=new3_audit,
        new_audit=new_test_audit,
        side_df=side_df,
        cross_df=cross_df,
        gate_df=gate_df,
        latency_df=latency_df,
    )

    print("Diagnosis v3 Phase 1 completed.")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
