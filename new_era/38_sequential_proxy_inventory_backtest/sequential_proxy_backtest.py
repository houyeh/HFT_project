from __future__ import annotations

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

FULL_ACTION_UNIVERSE = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit" / "data" / "full_action_universe.parquet"

POINT_VALUE = 200.0
MAKER_FEE = 15.0
HORIZON_EVENTS = 10


@dataclass
class FillLot:
    entry_row: int
    close_row: int
    entry_price: float
    close_mid: float
    fee: float
    side: str


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def load_baseline_actions() -> pd.DataFrame:
    actions = pd.read_parquet(FULL_ACTION_UNIVERSE)
    actions = actions[actions["phase031_split"].astype(str).isin(["test", "new4"])].copy()
    actions = actions[actions["selected_by_baseline"].astype(bool)].copy()
    actions["eval_split"] = actions["phase031_split"].astype(str).replace({"new4": "NEW4"})
    actions["side"] = actions["side"].astype(str)
    actions["local_row_id"] = actions["local_row_id"].astype(int)
    actions["timestamp"] = actions["timestamp"].astype(np.int64)
    actions["price"] = actions["price"].astype(float)
    actions["spread_ticks"] = actions["spread_ticks"].astype(float)
    actions["tick_size_effective"] = actions["tick_size"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    actions["current_mid_proxy"] = current_mid_proxy(actions)
    actions["future_mid_H_proxy"] = actions["current_mid_proxy"] + actions["future_mid_return_H_ticks"].astype(float) * actions["tick_size_effective"]
    actions["close_row_id"] = actions["local_row_id"].astype(int) + HORIZON_EVENTS
    return actions.sort_values(["eval_split", "session_id", "local_row_id", "action_id"]).reset_index(drop=True)


def current_mid_proxy(actions: pd.DataFrame) -> pd.Series:
    mid = actions["mid_price"].astype(float) if "mid_price" in actions.columns else pd.Series(np.nan, index=actions.index)
    tick = actions["tick_size"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    spread = actions["spread_ticks"].astype(float)
    price = actions["price"].astype(float)
    side = actions["side"].astype(str)
    reconstructed = np.where(side == "bid", price + 0.5 * spread * tick, price - 0.5 * spread * tick)
    return mid.where(np.isfinite(mid), reconstructed).astype(float)


def simulate_session(session_actions: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    session_actions = session_actions.sort_values(["local_row_id", "action_id"]).reset_index(drop=True)
    split = str(session_actions["eval_split"].iloc[0])
    session_id = str(session_actions["session_id"].iloc[0])

    events: list[dict[str, object]] = []
    for action in session_actions.itertuples(index=False):
        row = int(action.local_row_id)
        side = str(action.side)
        price = float(action.price)
        current_mid = float(action.current_mid_proxy)
        close_row = int(action.close_row_id)
        close_mid = float(action.future_mid_H_proxy)
        if not np.isfinite(close_mid):
            close_mid = float(action.final_mid)
        events.append(
            {
                "event_row": row,
                "event_order": 1,
                "event_type": "fill",
                "side": side,
                "price": price,
                "mark_mid": current_mid,
                "close_row": close_row,
                "close_mid": close_mid,
                "action_id": int(action.action_id),
            }
        )
        events.append(
            {
                "event_row": close_row,
                "event_order": 0,
                "event_type": "close_H_mid",
                "side": side,
                "price": price,
                "mark_mid": close_mid,
                "close_row": close_row,
                "close_mid": close_mid,
                "action_id": int(action.action_id),
            }
        )
    events_df = pd.DataFrame(events).sort_values(["event_row", "event_order", "action_id"]).reset_index(drop=True)

    cash = 0.0
    realized_pnl = 0.0
    inventory = 0
    max_abs_inventory = 0
    fills = 0
    fees_paid = 0.0
    open_lots: dict[int, FillLot] = {}
    curve_rows: list[dict[str, object]] = []

    for seq, event in enumerate(events_df.itertuples(index=False)):
        mark_mid = float(event.mark_mid)
        action_id = int(event.action_id)
        if event.event_type == "fill":
            # The frozen final controller is bid_only_spread_ge_1, so the expected side is bid.
            # The branch is kept explicit for auditability if a future action universe includes asks.
            side = str(event.side)
            if side == "bid":
                cash -= float(event.price) * POINT_VALUE
                inventory += 1
            else:
                cash += float(event.price) * POINT_VALUE
                inventory -= 1
            cash -= MAKER_FEE
            fees_paid += MAKER_FEE
            fills += 1
            open_lots[action_id] = FillLot(
                entry_row=int(event.event_row),
                close_row=int(event.close_row),
                entry_price=float(event.price),
                close_mid=float(event.close_mid),
                fee=MAKER_FEE,
                side=side,
            )
        else:
            lot = open_lots.pop(action_id, None)
            if lot is not None:
                if lot.side == "bid":
                    cash += lot.close_mid * POINT_VALUE
                    inventory -= 1
                    realized_pnl += (lot.close_mid - lot.entry_price) * POINT_VALUE - lot.fee
                else:
                    cash -= lot.close_mid * POINT_VALUE
                    inventory += 1
                    realized_pnl += (lot.entry_price - lot.close_mid) * POINT_VALUE - lot.fee

        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        unrealized_pnl = inventory * mark_mid * POINT_VALUE
        equity = cash + unrealized_pnl
        curve_rows.append(
            {
                "split": split,
                "session_id": session_id,
                "event_seq": seq,
                "event_row": int(event.event_row),
                "event_type": str(event.event_type),
                "action_id": action_id,
                "cash": float(cash),
                "inventory": int(inventory),
                "realized_pnl": float(realized_pnl),
                "unrealized_pnl": float(unrealized_pnl),
                "equity": float(equity),
                "mark_mid": mark_mid,
            }
        )

    curve = pd.DataFrame(curve_rows)
    if curve.empty:
        session_pnl = 0.0
        max_dd = 0.0
    else:
        session_pnl = float(curve["equity"].iloc[-1])
        max_dd = max_drawdown(curve["equity"].astype(float).to_numpy())

    quote_level_ev_moderate = float(session_actions["quote_ev_moderate"].astype(float).sum())
    quote_level_ev_base = float(session_actions["quote_value_base"].astype(float).sum()) if "quote_value_base" in session_actions.columns else np.nan
    summary = {
        "split": split,
        "session_id": session_id,
        "policy_id": "bid_only_spread_ge_1",
        "horizon_events": HORIZON_EVENTS,
        "fill_count": int(fills),
        "quote_level_ev_sum_moderate": quote_level_ev_moderate,
        "quote_level_ev_sum_base": quote_level_ev_base,
        "sequential_proxy_pnl_H_mid": session_pnl,
        "sequential_minus_quote_ev_moderate": float(session_pnl - quote_level_ev_moderate),
        "sequential_minus_quote_ev_base": float(session_pnl - quote_level_ev_base) if np.isfinite(quote_level_ev_base) else np.nan,
        "fees_paid": float(fees_paid),
        "final_realized_pnl": float(realized_pnl),
        "final_unrealized_pnl": float(curve["unrealized_pnl"].iloc[-1]) if not curve.empty else 0.0,
        "final_cash": float(cash),
        "final_inventory": int(inventory),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_dd),
        "mark_to_market_rule": "Each fill is closed at its H-event future mid proxy; no strategy tuning.",
    }
    return summary, curve


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
        }
    )


def plot_curves(curves: pd.DataFrame) -> None:
    set_style()
    plot_specs = [
        ("equity", "Equity Curve by Session", "equity_curve_by_session.png", "Proxy equity"),
        ("inventory", "Inventory Curve by Session", "inventory_curve_by_session.png", "Inventory"),
    ]
    for col, title, filename, ylabel in plot_specs:
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=False)
        axes = axes.ravel()
        groups = list(curves.groupby(["split", "session_id"], sort=True))
        for ax, ((split, session_id), group) in zip(axes, groups):
            ax.plot(group["event_seq"], group[col], color="#276ef1" if col == "equity" else "#7a4cc2", linewidth=1.6)
            ax.set_title(f"{split} {session_id}")
            ax.set_xlabel("Event sequence")
            ax.set_ylabel(ylabel)
        for ax in axes[len(groups) :]:
            ax.axis("off")
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=False)
    axes = axes.ravel()
    groups = list(curves.groupby(["split", "session_id"], sort=True))
    for ax, ((split, session_id), group) in zip(axes, groups):
        equity = group["equity"].astype(float)
        drawdown = np.maximum.accumulate(equity) - equity
        ax.plot(group["event_seq"], drawdown, color="#c43c35", linewidth=1.6)
        ax.set_title(f"{split} {session_id}")
        ax.set_xlabel("Event sequence")
        ax.set_ylabel("Drawdown")
    for ax in axes[len(groups) :]:
        ax.axis("off")
    fig.suptitle("Drawdown Curve by Session", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "drawdown_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    total = (
        summary.groupby("split", as_index=False)
        .agg(
            fill_count=("fill_count", "sum"),
            quote_level_ev_sum_moderate=("quote_level_ev_sum_moderate", "sum"),
            quote_level_ev_sum_base=("quote_level_ev_sum_base", "sum"),
            sequential_proxy_pnl_H_mid=("sequential_proxy_pnl_H_mid", "sum"),
            max_abs_inventory=("max_abs_inventory", "max"),
            max_drawdown=("max_drawdown", "max"),
        )
        .sort_values("split")
    )
    display = total.copy()
    for col in ["quote_level_ev_sum_moderate", "quote_level_ev_sum_base", "sequential_proxy_pnl_H_mid", "max_drawdown"]:
        display[col] = display[col].map(lambda v: f"{float(v):,.0f}")
    lines = [
        "# Sequential Proxy Inventory Backtest Report",
        "",
        "This is an evaluation framework upgrade only. It does not tune the strategy, change thresholds, or alter the final controller.",
        "",
        "## Method",
        "",
        "- Policy: frozen `bid_only_spread_ge_1`.",
        "- Universe: filled candidate actions from the Phase 031 full action universe for frozen `test` and external OOS `NEW4` only.",
        "- Chronological state variables: `cash`, `inventory`, `realized_pnl`, `unrealized_pnl`, and drawdown from proxy equity.",
        f"- Point value: `{POINT_VALUE}`.",
        f"- Maker fee per fill: `{MAKER_FEE}`.",
        f"- Close rule: each fill is closed at the H-event future mid proxy with `H={HORIZON_EVENTS}`.",
        "- This is still a proxy. It does not model true queue priority, exchange order acknowledgements, partial fills, latency races, or production inventory controls.",
        "",
        "## Split Summary",
        "",
        display.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "`quote_ev_moderate` and the sequential proxy are intentionally different measurements. `quote_ev_moderate` is a quote-level session-final markout with a queue haircut and adverse H-horizon stress penalty. The sequential proxy opens inventory on each fill and closes each lot at the H-event future mid. The comparison is useful for intuition, but neither number is real-world exchange PnL.",
        "",
        "Maximum inventory and drawdown should be read as proxy exposure diagnostics. They help explain whether quote opportunities cluster in time, but they are not a deployable risk model.",
        "",
        "## Outputs",
        "",
        "- `results/sequential_proxy_backtest_by_session.csv`",
        "- `results/sequential_proxy_equity_curve.csv`",
        "- `figures/equity_curve_by_session.png`",
        "- `figures/inventory_curve_by_session.png`",
        "- `figures/drawdown_curve.png`",
    ]
    write_markdown(ROOT / "SEQUENTIAL_PROXY_BACKTEST_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    actions = load_baseline_actions()
    summaries: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for _, session_actions in actions.groupby(["eval_split", "session_id"], sort=True):
        summary, curve = simulate_session(session_actions)
        summaries.append(summary)
        curves.append(curve)

    summary_df = pd.DataFrame(summaries).sort_values(["split", "session_id"]).reset_index(drop=True)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    summary_df.to_csv(RESULTS / "sequential_proxy_backtest_by_session.csv", index=False)
    curve_df.to_csv(RESULTS / "sequential_proxy_equity_curve.csv", index=False)
    plot_curves(curve_df)
    write_report(summary_df)
    print("Sequential proxy backtest by split:")
    print(
        summary_df.groupby("split")
        .agg(
            fill_count=("fill_count", "sum"),
            quote_level_ev_sum_moderate=("quote_level_ev_sum_moderate", "sum"),
            quote_level_ev_sum_base=("quote_level_ev_sum_base", "sum"),
            sequential_proxy_pnl_H_mid=("sequential_proxy_pnl_H_mid", "sum"),
            max_abs_inventory=("max_abs_inventory", "max"),
            max_drawdown=("max_drawdown", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
