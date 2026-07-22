from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "38_sequential_proxy_inventory_backtest"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

V16 = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V27 = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"
V31 = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"

for path in [PROJECT_ROOT, V27, PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_027 import build_new4_frame  # noqa: E402


POINT_VALUE = 200.0
MAKER_FEE = 15.0
HORIZONS: list[int | str] = [1, 5, 10, 20, 50, 100, 200, "session_final"]
MODERATE_NEUTRAL_FILL_HAIRCUT = 0.5
MODERATE_TOXIC_MULTIPLIER = 1.5


@dataclass(frozen=True)
class MidSeries:
    local_rows: np.ndarray
    mids: np.ndarray
    pos_by_local: dict[int, int]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(equity) - equity))


def current_mid_proxy(actions: pd.DataFrame) -> pd.Series:
    mid = actions["mid_price"].astype(float) if "mid_price" in actions.columns else pd.Series(np.nan, index=actions.index)
    tick = actions["tick_size"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    spread = actions["spread_ticks"].astype(float)
    price = actions["price"].astype(float)
    side = actions["side"].astype(str)
    reconstructed = np.where(side == "bid", price + 0.5 * spread * tick, price - 0.5 * spread * tick)
    return mid.where(np.isfinite(mid), reconstructed).astype(float)


def load_selected_actions() -> pd.DataFrame:
    actions = pd.read_parquet(V31 / "data" / "full_action_universe.parquet")
    actions = actions[actions["phase031_split"].astype(str).isin(["test", "new4"])].copy()
    actions = actions[actions["selected_by_baseline"].astype(bool)].copy()
    actions["eval_split"] = actions["phase031_split"].astype(str).replace({"new4": "NEW4"})
    actions["side"] = actions["side"].astype(str)
    actions["local_row_id"] = actions["local_row_id"].astype(int)
    actions["timestamp"] = actions["timestamp"].astype(np.int64)
    actions["price"] = actions["price"].astype(float)
    actions["spread_ticks"] = actions["spread_ticks"].astype(float)
    actions["current_mid_proxy"] = current_mid_proxy(actions)
    actions["event_rate_bucket"] = actions["event_rate_bucket"].astype(str)
    return actions.sort_values(["eval_split", "session_id", "local_row_id", "action_id"]).reset_index(drop=True)


def make_mid_series(frame: pd.DataFrame) -> dict[tuple[str, str], MidSeries]:
    out: dict[tuple[str, str], MidSeries] = {}
    for (split, session_id), group in frame.groupby(["eval_split", "session_id"], sort=True):
        g = group.sort_values("local_row_id")
        rows = g["local_row_id"].astype(int).to_numpy()
        mids = g["mid_price"].astype(float).to_numpy()
        out[(str(split), str(session_id))] = MidSeries(
            local_rows=rows,
            mids=mids,
            pos_by_local={int(row): pos for pos, row in enumerate(rows)},
        )
    return out


def load_mid_lookup() -> dict[tuple[str, str], MidSeries]:
    hist_cols = ["split", "session_id", "local_row_id", "mid_price"]
    hist = pd.read_parquet(V16 / "outputs" / "binary_labels_v2.parquet", columns=hist_cols)
    hist = hist[hist["split"].astype(str) == "test"].copy()
    hist["eval_split"] = "test"
    hist = hist[["eval_split", "session_id", "local_row_id", "mid_price"]]

    new4_frame, _, _ = build_new4_frame()
    new4 = new4_frame[["session_id", "local_row_id", "mid_price"]].copy()
    new4["eval_split"] = "NEW4"
    frame = pd.concat([hist, new4[["eval_split", "session_id", "local_row_id", "mid_price"]]], ignore_index=True)
    return make_mid_series(frame)


def close_info(action: pd.Series, mid_lookup: dict[tuple[str, str], MidSeries], horizon: int | str) -> tuple[float, int, int, float]:
    key = (str(action["eval_split"]), str(action["session_id"]))
    series = mid_lookup[key]
    entry_local = int(action["local_row_id"])
    pos = series.pos_by_local.get(entry_local)
    if pos is None:
        # Fallback should not be needed, but keeps the audit explicit if source rows drift.
        pos = int(np.searchsorted(series.local_rows, entry_local, side="left"))
        pos = min(max(pos, 0), len(series.local_rows) - 1)
    if horizon == "session_final":
        close_pos = len(series.local_rows) - 1
    else:
        close_pos = min(pos + int(horizon), len(series.local_rows) - 1)
    close_mid = float(series.mids[close_pos])
    close_local = int(series.local_rows[close_pos])
    holding_events = int(close_pos - pos)
    session_progress = float(pos / max(len(series.local_rows) - 1, 1))
    return close_mid, close_local, holding_events, session_progress


def side_markout(side: str, exit_mid: float, quote_price: float) -> float:
    if side == "bid":
        return (float(exit_mid) - float(quote_price)) * POINT_VALUE
    return (float(quote_price) - float(exit_mid)) * POINT_VALUE


def trade_pnl(side: str, exit_mid: float, quote_price: float) -> float:
    return side_markout(side, exit_mid, quote_price) - MAKER_FEE


def simulate_session_for_horizon(session_actions: pd.DataFrame, mid_lookup: dict[tuple[str, str], MidSeries], horizon: int | str) -> tuple[dict[str, object], pd.DataFrame]:
    split = str(session_actions["eval_split"].iloc[0])
    session_id = str(session_actions["session_id"].iloc[0])
    events: list[dict[str, object]] = []
    trade_pnls: list[float] = []
    holdings: list[int] = []
    for action in session_actions.itertuples(index=False):
        row = pd.Series(action._asdict())
        close_mid, close_local, holding_events, _ = close_info(row, mid_lookup, horizon)
        pnl = trade_pnl(str(row["side"]), close_mid, float(row["price"]))
        trade_pnls.append(float(pnl))
        holdings.append(int(holding_events))
        events.append(
            {
                "event_row": int(row["local_row_id"]),
                "event_order": 1,
                "event_type": "fill",
                "side": str(row["side"]),
                "price": float(row["price"]),
                "mark_mid": float(row["current_mid_proxy"]),
                "action_id": int(row["action_id"]),
                "close_mid": close_mid,
                "trade_pnl": pnl,
            }
        )
        events.append(
            {
                "event_row": int(close_local),
                "event_order": 0,
                "event_type": "close",
                "side": str(row["side"]),
                "price": float(row["price"]),
                "mark_mid": close_mid,
                "action_id": int(row["action_id"]),
                "close_mid": close_mid,
                "trade_pnl": pnl,
            }
        )

    event_df = pd.DataFrame(events).sort_values(["event_row", "event_order", "action_id"]).reset_index(drop=True)
    cash = 0.0
    inventory = 0
    max_abs_inventory = 0
    equity_rows: list[dict[str, object]] = []
    open_lots: dict[int, dict[str, object]] = {}
    for seq, event in enumerate(event_df.itertuples(index=False)):
        mark_mid = float(event.mark_mid)
        action_id = int(event.action_id)
        side = str(event.side)
        if event.event_type == "fill":
            if side == "bid":
                cash -= float(event.price) * POINT_VALUE
                inventory += 1
            else:
                cash += float(event.price) * POINT_VALUE
                inventory -= 1
            cash -= MAKER_FEE
            open_lots[action_id] = {"side": side, "close_mid": float(event.close_mid)}
        else:
            lot = open_lots.pop(action_id, None)
            if lot is not None:
                if lot["side"] == "bid":
                    cash += float(lot["close_mid"]) * POINT_VALUE
                    inventory -= 1
                else:
                    cash -= float(lot["close_mid"]) * POINT_VALUE
                    inventory += 1
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        equity = cash + inventory * mark_mid * POINT_VALUE
        equity_rows.append(
            {
                "split": split,
                "session_id": session_id,
                "horizon": str(horizon),
                "event_seq": seq,
                "event_row": int(event.event_row),
                "cash": float(cash),
                "inventory": int(inventory),
                "equity": float(equity),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trade_arr = np.asarray(trade_pnls, dtype=float)
    summary = {
        "split": split,
        "session_id": session_id,
        "horizon": str(horizon),
        "horizon_sort": 9999 if horizon == "session_final" else int(horizon),
        "fill_count": int(len(trade_arr)),
        "sequential_proxy_pnl": float(trade_arr.sum()),
        "max_drawdown": max_drawdown(equity_df["equity"].astype(float).to_numpy()) if len(equity_df) else 0.0,
        "max_abs_inventory": int(max_abs_inventory),
        "average_holding_events": float(np.mean(holdings)) if holdings else 0.0,
        "win_count": int((trade_arr > 0).sum()),
        "win_rate": float((trade_arr > 0).mean()) if len(trade_arr) else np.nan,
        "average_trade_pnl": float(trade_arr.mean()) if len(trade_arr) else np.nan,
    }
    return summary, equity_df


def aggregate_multi_horizon(session_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, horizon), group in session_summary.groupby(["split", "horizon"], sort=False):
        fill_count = int(group["fill_count"].sum())
        pnl = float(group["sequential_proxy_pnl"].sum())
        rows.append(
            {
                "split": str(split),
                "horizon": str(horizon),
                "horizon_sort": int(group["horizon_sort"].iloc[0]),
                "fill_count": fill_count,
                "sequential_proxy_pnl": pnl,
                "max_drawdown": float(group["max_drawdown"].max()),
                "max_abs_inventory": int(group["max_abs_inventory"].max()),
                "average_holding_events": float(np.average(group["average_holding_events"], weights=group["fill_count"])) if fill_count else 0.0,
                "win_rate": float(group["win_count"].sum() / fill_count) if fill_count else np.nan,
                "average_trade_pnl": float(pnl / fill_count) if fill_count else np.nan,
                "scope": "split",
            }
        )
    for horizon, group in session_summary.groupby("horizon", sort=False):
        fill_count = int(group["fill_count"].sum())
        pnl = float(group["sequential_proxy_pnl"].sum())
        rows.append(
            {
                "split": "ALL",
                "horizon": str(horizon),
                "horizon_sort": int(group["horizon_sort"].iloc[0]),
                "fill_count": fill_count,
                "sequential_proxy_pnl": pnl,
                "max_drawdown": float(group["max_drawdown"].max()),
                "max_abs_inventory": int(group["max_abs_inventory"].max()),
                "average_holding_events": float(np.average(group["average_holding_events"], weights=group["fill_count"])) if fill_count else 0.0,
                "win_rate": float(group["win_count"].sum() / fill_count) if fill_count else np.nan,
                "average_trade_pnl": float(pnl / fill_count) if fill_count else np.nan,
                "scope": "combined",
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "split", "horizon_sort"]).reset_index(drop=True)


def run_multi_horizon(actions: pd.DataFrame, mid_lookup: dict[tuple[str, str], MidSeries]) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    equity_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        for _, session_actions in actions.groupby(["eval_split", "session_id"], sort=True):
            summary, equity = simulate_session_for_horizon(session_actions, mid_lookup, horizon)
            summaries.append(summary)
            equity_parts.append(equity)
    session_summary = pd.DataFrame(summaries)
    session_summary.to_csv(RESULTS / "multi_horizon_sequential_by_session.csv", index=False)
    pd.concat(equity_parts, ignore_index=True).to_csv(RESULTS / "multi_horizon_equity_curves.csv", index=False)
    summary = aggregate_multi_horizon(session_summary)
    summary.to_csv(RESULTS / "multi_horizon_sequential_summary.csv", index=False)
    return summary


def add_gap_decomposition(actions: pd.DataFrame, mid_lookup: dict[tuple[str, str], MidSeries]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for action in actions.itertuples(index=False):
        row = pd.Series(action._asdict())
        close_mid, close_local, holding_events, session_progress = close_info(row, mid_lookup, 10)
        side = str(row["side"])
        price = float(row["price"])
        terminal_gross = side_markout(side, float(row["final_mid"]), price)
        h10_gross = side_markout(side, close_mid, price)
        sequential_h10_pnl = h10_gross - MAKER_FEE
        future_ticks = float(row["future_mid_return_H_ticks"])
        side_markout_h_ticks = future_ticks if side == "bid" else -future_ticks
        raw_adverse_penalty = max(MODERATE_TOXIC_MULTIPLIER - 1.0, 0.0) * max(0.0, -side_markout_h_ticks) * POINT_VALUE
        kept_by_moderate_haircut = float(row["neutral_u"]) <= MODERATE_NEUTRAL_FILL_HAIRCUT
        adverse_penalty = raw_adverse_penalty if kept_by_moderate_haircut else 0.0
        original_fee_applied = MAKER_FEE if kept_by_moderate_haircut else 0.0
        quote_ev_moderate = float(row["quote_ev_moderate"])
        time_bucket = pd.cut(
            [session_progress],
            bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        )[0]
        future_bucket = pd.cut(
            [future_ticks],
            bins=[-np.inf, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, np.inf],
            labels=["<=-5", "(-5,-2]", "(-2,-1]", "(-1,0]", "(0,1]", "(1,2]", "(2,5]", ">5"],
        )[0]
        rows.append(
            {
                "split": str(row["eval_split"]),
                "session_id": str(row["session_id"]),
                "action_id": int(row["action_id"]),
                "side": side,
                "local_row_id": int(row["local_row_id"]),
                "close_local_row_id_H10": int(close_local),
                "holding_events_H10": int(holding_events),
                "spread_ticks": float(row["spread_ticks"]),
                "spread_bucket": str(row["spread_bucket"]),
                "event_rate_bucket": str(row["event_rate_bucket"]),
                "event_rate_past_50": float(row["event_rate_past_50"]),
                "time_in_session_bucket": str(time_bucket),
                "future_mid_return_H_ticks": future_ticks,
                "future_mid_return_H_ticks_bucket": str(future_bucket),
                "quote_ev_moderate": quote_ev_moderate,
                "terminal_markout_component": float(terminal_gross),
                "H10_markout_component": float(h10_gross),
                "sequential_H10_pnl": float(sequential_h10_pnl),
                "fee": float(MAKER_FEE),
                "original_quote_ev_fee_applied": float(original_fee_applied),
                "raw_adverse_penalty_before_fill_haircut": float(raw_adverse_penalty),
                "adverse_penalty": float(adverse_penalty),
                "kept_by_moderate_haircut": bool(kept_by_moderate_haircut),
                "terminal_minus_H10_component_for_bridge": float((terminal_gross if kept_by_moderate_haircut else 0.0) - h10_gross),
                "fee_haircut_component_for_bridge": float(MAKER_FEE - original_fee_applied),
                "adverse_penalty_component_for_bridge": float(-adverse_penalty),
                "difference": float(quote_ev_moderate - sequential_h10_pnl),
            }
        )
    out = pd.DataFrame(rows)
    bridge = (
        out["sequential_H10_pnl"]
        + out["terminal_minus_H10_component_for_bridge"]
        + out["fee_haircut_component_for_bridge"]
        + out["adverse_penalty_component_for_bridge"]
        - out["quote_ev_moderate"]
    )
    out["bridge_residual"] = bridge.astype(float)
    return out


def aggregate_gap(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, observed=False)
        .agg(
            quote_count=("action_id", "count"),
            quote_ev_moderate=("quote_ev_moderate", "sum"),
            terminal_markout_component=("terminal_markout_component", "sum"),
            H10_markout_component=("H10_markout_component", "sum"),
            sequential_H10_pnl=("sequential_H10_pnl", "sum"),
            fee=("fee", "sum"),
            original_quote_ev_fee_applied=("original_quote_ev_fee_applied", "sum"),
            adverse_penalty=("adverse_penalty", "sum"),
            raw_adverse_penalty_before_fill_haircut=("raw_adverse_penalty_before_fill_haircut", "sum"),
            difference=("difference", "sum"),
            kept_by_moderate_haircut_rate=("kept_by_moderate_haircut", "mean"),
            bridge_residual_abs_max=("bridge_residual", lambda s: float(np.abs(s).max()) if len(s) else 0.0),
        )
        .reset_index()
    )
    return out


def write_gap_outputs(gap: pd.DataFrame) -> dict[str, float]:
    gap.to_csv(RESULTS / "quote_vs_sequential_gap_detail.csv", index=False)
    by_session = aggregate_gap(gap, ["split", "session_id"])
    by_session.to_csv(RESULTS / "quote_vs_sequential_gap_by_session.csv", index=False)

    bucket_frames: list[pd.DataFrame] = []
    bucket_specs = [
        ("spread_ticks", ["split", "spread_ticks"]),
        ("event_rate_bucket", ["split", "event_rate_bucket"]),
        ("time_in_session_bucket", ["split", "time_in_session_bucket"]),
        ("future_mid_return_H_ticks_bucket", ["split", "future_mid_return_H_ticks_bucket"]),
    ]
    for bucket_type, group_cols in bucket_specs:
        agg = aggregate_gap(gap, group_cols)
        value_col = group_cols[-1]
        agg = agg.rename(columns={value_col: "bucket_value"})
        agg.insert(1, "bucket_type", bucket_type)
        bucket_frames.append(agg)
    by_bucket = pd.concat(bucket_frames, ignore_index=True)
    by_bucket.to_csv(RESULTS / "quote_vs_sequential_gap_by_bucket.csv", index=False)

    totals = {
        "sequential_H10_pnl": float(gap["sequential_H10_pnl"].sum()),
        "terminal_minus_H10_bridge": float(gap["terminal_minus_H10_component_for_bridge"].sum()),
        "fee_haircut_bridge": float(gap["fee_haircut_component_for_bridge"].sum()),
        "adverse_penalty_bridge": float(gap["adverse_penalty_component_for_bridge"].sum()),
        "quote_ev_moderate": float(gap["quote_ev_moderate"].sum()),
        "difference": float(gap["difference"].sum()),
        "terminal_markout_component": float(gap["terminal_markout_component"].sum()),
        "H10_markout_component": float(gap["H10_markout_component"].sum()),
        "adverse_penalty": float(gap["adverse_penalty"].sum()),
        "fee": float(gap["fee"].sum()),
    }
    return totals


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343b",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "font.size": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def plot_multi_horizon(summary: pd.DataFrame) -> None:
    set_style()
    split_summary = summary[summary["scope"] == "split"].copy()
    order = [str(h) for h in HORIZONS]
    colors = {"test": "#276ef1", "NEW4": "#2f9e44"}
    for metric, filename, ylabel, title in [
        ("sequential_proxy_pnl", "multi_horizon_pnl_curve.png", "Sequential proxy PnL", "Multi-Horizon Sequential Proxy PnL"),
        ("max_drawdown", "multi_horizon_drawdown_curve.png", "Max drawdown", "Multi-Horizon Sequential Proxy Drawdown"),
    ]:
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        for split, group in split_summary.groupby("split", sort=True):
            group = group.copy()
            group["horizon"] = pd.Categorical(group["horizon"], categories=order, ordered=True)
            group = group.sort_values("horizon")
            ax.plot(group["horizon"].astype(str), group[metric], marker="o", linewidth=2, label=split, color=colors.get(str(split), None))
        if metric == "sequential_proxy_pnl":
            ax.axhline(0, color="#20242b", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Close horizon")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_gap_waterfall(totals: dict[str, float]) -> None:
    set_style()
    labels = [
        "Sequential H10",
        "Kept terminal\nvs all H10",
        "Fee reversal\nfor haircut",
        "Adverse\npenalty",
        "Quote EV\nmoderate",
    ]
    start = totals["sequential_H10_pnl"]
    components = [
        totals["terminal_minus_H10_bridge"],
        totals["fee_haircut_bridge"],
        totals["adverse_penalty_bridge"],
    ]
    end = totals["quote_ev_moderate"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    running = start
    ax.bar(x[0], start, color="#5b677a")
    for i, comp in enumerate(components, start=1):
        bottom = running if comp >= 0 else running + comp
        ax.bar(x[i], abs(comp), bottom=bottom, color="#2f9e44" if comp >= 0 else "#c43c35")
        ax.plot([x[i - 1] + 0.35, x[i] - 0.35], [running, running], color="#8a94a6", linewidth=1)
        running += comp
    ax.bar(x[-1], end, color="#7a4cc2")
    ax.axhline(0, color="#20242b", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PnL units")
    ax.set_title("Gap Waterfall: Sequential H10 Proxy to Quote EV Moderate")
    for xi, value in zip([0, 1, 2, 3, 4], [start, *components, end]):
        y = value if xi in {0, 4} else (running if False else value)
        ax.text(xi, value if xi in {0, 4} else (start + sum(components[:xi])), f"{value:,.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "gap_waterfall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def first_positive_horizon(summary: pd.DataFrame, split: str) -> str:
    sub = summary[(summary["scope"] == "split") & (summary["split"] == split)].sort_values("horizon_sort")
    positive = sub[sub["sequential_proxy_pnl"] > 0]
    if positive.empty:
        return "none"
    return str(positive.iloc[0]["horizon"])


def write_horizon_report(summary: pd.DataFrame) -> None:
    test_break = first_positive_horizon(summary, "test")
    new4_break = first_positive_horizon(summary, "NEW4")
    display = summary[summary["scope"] == "split"].copy()
    for col in ["sequential_proxy_pnl", "max_drawdown", "average_trade_pnl"]:
        display[col] = display[col].map(lambda v: f"{float(v):,.2f}")
    display["win_rate"] = display["win_rate"].map(lambda v: f"{float(v):.2%}")
    lines = [
        "# Horizon Break-Even Report",
        "",
        "This is a diagnostic evaluation of the frozen `bid_only_spread_ge_1` baseline. No horizon is selected for deployment and no strategy parameter is tuned.",
        "",
        f"- First positive test horizon: `{test_break}`.",
        f"- First positive NEW4 horizon: `{new4_break}`.",
        "",
        "## Multi-Horizon Summary",
        "",
        display[
            [
                "split",
                "horizon",
                "fill_count",
                "sequential_proxy_pnl",
                "max_drawdown",
                "max_abs_inventory",
                "average_holding_events",
                "win_rate",
                "average_trade_pnl",
            ]
        ].to_markdown(index=False, disable_numparse=True),
        "",
        "## Interpretation",
        "",
        "Short horizons should be read as a closer proxy for market-making execution. The `session_final` row is a long-horizon directional markout diagnostic, not an executable short-horizon market-making claim.",
    ]
    if test_break == "session_final" or new4_break == "session_final":
        lines.append("At least one split only turns positive at `session_final`, which means the original positive quote-level EV is not well supported as a short-horizon market-making edge.")
    elif test_break == "none" or new4_break == "none":
        lines.append("At least one split never turns positive over the tested horizons, reinforcing that the quote-level EV should not be read as deployable trading PnL.")
    else:
        lines.append("Both splits become positive before `session_final`; this would still be only a proxy result and requires stricter execution modeling before any trading interpretation.")
    write_markdown(ROOT / "horizon_break_even_report.md", lines)


def write_gap_report(totals: dict[str, float]) -> None:
    diff = totals["difference"]
    terminal_bridge = totals["terminal_minus_H10_bridge"]
    terminal_share = terminal_bridge / diff if diff else np.nan
    answer = (
        "Yes. The bridge is dominated by replacing short-horizon H10 markout with session-final terminal markout on fills kept by the moderate haircut."
        if diff and abs(terminal_share) > 0.5
        else "The gap is not dominated by a single terminal-drift bridge component; inspect the bucket files for localized effects."
    )
    lines = [
        "# Gap Explanation Report",
        "",
        "This report decomposes why `quote_ev_moderate` is positive while the H10 sequential proxy is negative.",
        "",
        "## Aggregate Bridge",
        "",
        "| Component | PnL units |",
        "| --- | ---: |",
        f"| Sequential H10 proxy PnL | {totals['sequential_H10_pnl']:,.0f} |",
        f"| Add: kept terminal markout vs all H10 markout | {totals['terminal_minus_H10_bridge']:,.0f} |",
        f"| Add: fee reversal for haircut-removed fills | {totals['fee_haircut_bridge']:,.0f} |",
        f"| Add: adverse penalty bridge | {totals['adverse_penalty_bridge']:,.0f} |",
        f"| Result: quote_ev_moderate | {totals['quote_ev_moderate']:,.0f} |",
        "",
        "## Required Answer",
        "",
        f"Is the original positive quote_ev mainly coming from session-final directional drift rather than short-horizon market-making edge? {answer}",
        "",
        f"The terminal-vs-H10 bridge is `{terminal_bridge:,.0f}` against a total H10-to-quote-EV difference of `{diff:,.0f}`, a share of `{terminal_share:.2%}`.",
        "",
        "Interpretation boundary: this does not prove the controller has no use. It proves the large positive quote-level EV is not the same object as short-horizon executable market-making PnL.",
        "",
        "## Outputs",
        "",
        "- `results/quote_vs_sequential_gap_detail.csv`",
        "- `results/quote_vs_sequential_gap_by_session.csv`",
        "- `results/quote_vs_sequential_gap_by_bucket.csv`",
        "- `figures/gap_waterfall.png`",
    ]
    write_markdown(ROOT / "gap_explanation_report.md", lines)


def main() -> None:
    ensure_dirs()
    actions = load_selected_actions()
    mid_lookup = load_mid_lookup()
    summary = run_multi_horizon(actions, mid_lookup)
    gap = add_gap_decomposition(actions, mid_lookup)
    totals = write_gap_outputs(gap)
    plot_multi_horizon(summary)
    plot_gap_waterfall(totals)
    write_horizon_report(summary)
    write_gap_report(totals)
    print("Multi-horizon summary:")
    print(summary[summary["scope"] == "split"].to_string(index=False))
    print("Gap totals:")
    print(pd.Series(totals).to_string())


if __name__ == "__main__":
    main()
