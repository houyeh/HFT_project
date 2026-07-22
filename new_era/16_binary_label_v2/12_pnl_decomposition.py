from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
POINT_VALUE = 200.0
MAKER_FEE = 15.0
MAX_INV = 3


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def _simulate_session(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.sort_values("local_row_id").reset_index(drop=True)
    bid = frame["bid_px_1"].to_numpy(dtype=float)
    ask = frame["ask_px_1"].to_numpy(dtype=float)
    mid = frame["mid_price"].to_numpy(dtype=float)
    pull_bid = frame["pull_bid"].to_numpy(dtype=bool)
    pull_ask = frame["pull_ask"].to_numpy(dtype=bool)
    spread_bucket = frame["spread_bucket"].astype(str).to_numpy()
    rate_bucket = frame["event_rate_bucket"].astype(str).to_numpy()

    cash = 0.0
    fees = 0.0
    inventory = 0
    trades = 0
    prev_equity = 0.0
    rows = []
    equity_curve = []

    for i in range(max(len(frame) - 1, 0)):
        if not pull_bid[i] and inventory < MAX_INV and ask[i + 1] <= bid[i]:
            cash -= bid[i] * POINT_VALUE
            fees += MAKER_FEE
            inventory += 1
            trades += 1
        if not pull_ask[i] and inventory > -MAX_INV and bid[i + 1] >= ask[i]:
            cash += ask[i] * POINT_VALUE
            fees += MAKER_FEE
            inventory -= 1
            trades += 1
        equity = cash - fees + inventory * mid[i] * POINT_VALUE
        delta = equity - prev_equity
        prev_equity = equity
        equity_curve.append(equity)
        rows.append(
            {
                "session_id": str(frame["session_id"].iloc[0]),
                "event_pos": i,
                "equity": equity,
                "pnl_delta": delta,
                "spread_bucket": spread_bucket[i],
                "event_rate_bucket": rate_bucket[i],
            }
        )

    final_equity = cash - fees + inventory * mid[-1] * POINT_VALUE if len(frame) else 0.0
    if rows:
        rows[-1]["pnl_delta"] += final_equity - prev_equity
        rows[-1]["equity"] = final_equity
        equity_curve[-1] = final_equity

    equity_arr = np.asarray(equity_curve, dtype=float)
    summary = {
        "session_id": str(frame["session_id"].iloc[0]),
        "rows": int(len(frame)),
        "trade_count": int(trades),
        "ending_inventory": int(inventory),
        "net_pnl": float(final_equity),
        "max_drawdown": _max_drawdown(equity_arr),
    }
    return pd.DataFrame(rows), summary


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(OUTPUTS / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    preds = pd.read_parquet(OUTPUTS / "test_predictions_binary_cancel.parquet")
    test = preds.merge(
        labels[["event_id", "timestamp", "spread_ticks", "tick_size"]],
        on="event_id",
        how="left",
    )

    test["spread_bucket"] = pd.cut(
        test["spread_ticks"],
        bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
        labels=["<=1", "(1,2]", "(2,3]", ">3"],
    )
    dt = test.groupby("session_id")["timestamp"].diff()
    positive_dt = dt[dt > 0]
    fallback_dt = float(positive_dt.median()) if len(positive_dt) else 1.0
    dt_filled = dt.fillna(fallback_dt).clip(lower=1.0)
    test["event_rate_hz"] = 1_000_000_000.0 / dt_filled
    test["event_rate_bucket"] = pd.qcut(
        test["event_rate_hz"].rank(method="first"),
        q=4,
        labels=["q1_slow", "q2", "q3", "q4_fast"],
    )

    event_rows = []
    session_rows = []
    cum_offset = 0.0
    curve_rows = []
    for _, group in test.groupby("session_id", sort=True):
        events, summary = _simulate_session(group)
        session_rows.append(summary)
        if not events.empty:
            events["global_equity"] = events["equity"] + cum_offset
            curve_rows.append(events[["session_id", "event_pos", "global_equity"]])
            cum_offset += float(summary["net_pnl"])
            event_rows.append(events)

    event_df = pd.concat(event_rows, ignore_index=True)
    pd.DataFrame(session_rows).to_csv(results_dir / "pnl_by_session.csv", index=False)
    event_df.groupby("spread_bucket", observed=False)["pnl_delta"].sum().reset_index(name="net_pnl").to_csv(
        results_dir / "pnl_by_spread_bucket.csv", index=False
    )
    event_df.groupby("event_rate_bucket", observed=False)["pnl_delta"].sum().reset_index(name="net_pnl").to_csv(
        results_dir / "pnl_by_event_rate_bucket.csv", index=False
    )

    curve = pd.concat(curve_rows, ignore_index=True)
    equity = curve["global_equity"].to_numpy(dtype=float)
    drawdown = np.maximum.accumulate(equity) - equity

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(equity)
    ax.set_title("Model Test Equity Curve")
    ax.set_xlabel("Event")
    ax.set_ylabel("Net PnL")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(drawdown, color="firebrick")
    ax.set_title("Model Test Drawdown Curve")
    ax.set_xlabel("Event")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "drawdown_curve.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PnL decomposition artifacts.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"PnL decomposition written to {args.results_dir}")
