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


def _simulate_actions(actions: list[tuple[int, int, float]], bid_cancel: np.ndarray, ask_cancel: np.ndarray, final_mid: float) -> float:
    inventory = 0
    cash = 0.0
    fees = 0.0
    for _, side, px in actions:
        if side == 0:
            if bid_cancel[_]:
                continue
            if inventory >= MAX_INV:
                continue
            cash -= px * POINT_VALUE
            fees += MAKER_FEE
            inventory += 1
        else:
            if ask_cancel[_]:
                continue
            if inventory <= -MAX_INV:
                continue
            cash += px * POINT_VALUE
            fees += MAKER_FEE
            inventory -= 1
    return float(cash + inventory * final_mid * POINT_VALUE - fees)


def _prepare_session(frame: pd.DataFrame) -> dict:
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    pull_bid = frame["pull_bid"].to_numpy(dtype=bool)
    pull_ask = frame["pull_ask"].to_numpy(dtype=bool)
    actions: list[tuple[int, int, float]] = []
    bid_action_ids = []
    ask_action_ids = []
    for i in range(len(frame) - 1):
        if ask[i + 1] <= bid[i]:
            bid_action_ids.append(len(actions))
            actions.append((len(bid_action_ids) - 1, 0, float(bid[i])))
        if bid[i + 1] >= ask[i]:
            ask_action_ids.append(len(actions))
            actions.append((len(ask_action_ids) - 1, 1, float(ask[i])))
    return {
        "session_id": str(frame["session_id"].iloc[0]),
        "n": int(len(frame)),
        "actions": actions,
        "bid_fill_events": int(len(bid_action_ids)),
        "ask_fill_events": int(len(ask_action_ids)),
        "model_bid_cancel_count": int(pull_bid.sum()),
        "model_ask_cancel_count": int(pull_ask.sum()),
        "final_mid": float(mid[-1]),
    }


def _trial_session_pnl(session: dict, rng: np.random.Generator) -> float:
    n = session["n"]
    m_bid = session["bid_fill_events"]
    m_ask = session["ask_fill_events"]
    k_bid = session["model_bid_cancel_count"]
    k_ask = session["model_ask_cancel_count"]
    hit_bid = int(rng.hypergeometric(m_bid, max(n - m_bid, 0), min(k_bid, n))) if m_bid else 0
    hit_ask = int(rng.hypergeometric(m_ask, max(n - m_ask, 0), min(k_ask, n))) if m_ask else 0
    bid_cancel = np.zeros(m_bid, dtype=bool)
    ask_cancel = np.zeros(m_ask, dtype=bool)
    if hit_bid:
        bid_cancel[rng.choice(m_bid, size=hit_bid, replace=False)] = True
    if hit_ask:
        ask_cancel[rng.choice(m_ask, size=hit_ask, replace=False)] = True
    return _simulate_actions(session["actions"], bid_cancel, ask_cancel, session["final_mid"])


def run(trials: int, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(OUTPUTS / "test_predictions_binary_cancel.parquet")
    test_summary = pd.read_csv(OUTPUTS / "test_backtest_summary.csv").iloc[0]
    model_pnl = float(test_summary["net_pnl"])
    sessions = [_prepare_session(g.sort_values("local_row_id").reset_index(drop=True)) for _, g in preds.groupby("session_id", sort=True)]

    rng = np.random.default_rng(20260523)
    pnls = np.zeros(int(trials), dtype=np.float64)
    for t in range(int(trials)):
        pnls[t] = sum(_trial_session_pnl(s, rng) for s in sessions)

    p_value = float((1 + np.sum(pnls >= model_pnl)) / (len(pnls) + 1))
    dist = pd.DataFrame(
        [
            {
                "trials": int(trials),
                "model_net_pnl": model_pnl,
                "mean": float(np.mean(pnls)),
                "std": float(np.std(pnls, ddof=0)),
                "p90": float(np.percentile(pnls, 90)),
                "p95": float(np.percentile(pnls, 95)),
                "p99": float(np.percentile(pnls, 99)),
                "empirical_p_value_random_ge_model": p_value,
            }
        ]
    )
    dist.to_csv(results_dir / "random_baseline_distribution.csv", index=False)

    audit_rows = []
    for s in sessions:
        audit_rows.append(
            {
                "session_id": s["session_id"],
                "rows": s["n"],
                "model_bid_cancel_count": s["model_bid_cancel_count"],
                "model_ask_cancel_count": s["model_ask_cancel_count"],
                "random_bid_cancel_count_per_trial": s["model_bid_cancel_count"],
                "random_ask_cancel_count_per_trial": s["model_ask_cancel_count"],
                "bid_ask_cancel_ratio": s["model_bid_cancel_count"] / max(s["model_ask_cancel_count"], 1),
                "bid_fill_events": s["bid_fill_events"],
                "ask_fill_events": s["ask_fill_events"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(results_dir / "random_matching_audit.csv", index=False)
    pd.DataFrame({"trial": np.arange(len(pnls)), "net_pnl": pnls}).to_csv(results_dir / "random_baseline_trials.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pnls, bins=80, alpha=0.75)
    ax.axvline(model_pnl, color="red", linewidth=2, label=f"Model PnL={model_pnl:,.0f}")
    ax.set_title("5000-Trial Matched Random Cancel Baseline")
    ax.set_xlabel("Net PnL")
    ax.set_ylabel("Trials")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "random_pnl_histogram.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robust matched random baseline.")
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.trials < 5000:
        raise ValueError("Advisor audit requires at least 5000 trials.")
    run(args.trials, args.results_dir)
    print(f"Robust random baseline written to {args.results_dir}")
