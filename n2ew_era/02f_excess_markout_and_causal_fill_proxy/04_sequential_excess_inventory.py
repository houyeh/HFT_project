from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from common_phase02f import (
    EVAL_SPLITS,
    FIGURES,
    FEE_MULTIPLIERS,
    FILL_MODELS,
    INVENTORY_CAPS,
    MAKER_FEE,
    RESULTS,
    SELECTED_POLICY_ID,
    compute_train_fill_norms,
    deterministic_fill_mask,
    ensure_dirs,
    iter_excess_event_frames,
    policy_decision,
    policy_set,
)


HORIZON = 20


@njit(cache=True)
def simulate_scheduled_excess_numba(decision: np.ndarray, fills: np.ndarray, base_pnl: np.ndarray, horizon: int, inventory_cap: int, fee_multiplier: float, maker_fee: float) -> tuple[float, int, int, int, float, int, int]:
    n = len(decision)
    close_side_sum = np.zeros(n, dtype=np.int64)
    close_pnl_sum = np.zeros(n, dtype=np.float64)
    inventory = 0
    max_abs_inventory = 0
    forced_no_quote = 0
    fill_count = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    fee_adjust = maker_fee * (fee_multiplier - 1.0)

    for t in range(n):
        if close_side_sum[t] != 0 or close_pnl_sum[t] != 0.0:
            inventory -= close_side_sum[t]
            equity += close_pnl_sum[t]
            total_pnl += close_pnl_sum[t]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_drawdown:
                max_drawdown = dd

        if not fills[t]:
            continue
        side_sign = 1 if decision[t] == 1 else -1
        next_inventory = inventory + side_sign
        if abs(next_inventory) > inventory_cap:
            forced_no_quote += 1
            continue
        trade_pnl = base_pnl[t] - fee_adjust
        inventory = next_inventory
        if abs(inventory) > max_abs_inventory:
            max_abs_inventory = abs(inventory)
        close_idx = t + horizon
        if close_idx >= n:
            close_idx = n - 1
        close_side_sum[close_idx] += side_sign
        close_pnl_sum[close_idx] += trade_pnl
        fill_count += 1
        if trade_pnl > 0.0:
            win_count += 1
        elif trade_pnl < 0.0:
            loss_count += 1

    return total_pnl, fill_count, forced_no_quote, max_abs_inventory, max_drawdown, win_count, loss_count


def base_excess_pnl(frame: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    bid = frame[f"bid_excess_pnl_H{HORIZON}"].to_numpy(dtype=float)
    ask = frame[f"ask_excess_pnl_H{HORIZON}"].to_numpy(dtype=float)
    out = np.zeros(len(frame), dtype=float)
    bid_mask = decision == 1
    ask_mask = decision == 2
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    return np.nan_to_num(out, nan=0.0)


def zero_rows(split: str, policy_id: str) -> list[dict]:
    rows = []
    for fill_model in FILL_MODELS:
        for cap in INVENTORY_CAPS:
            for fee in FEE_MULTIPLIERS:
                rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "horizon": HORIZON,
                        "fill_model": fill_model,
                        "inventory_cap": int(cap),
                        "fee_multiplier": float(fee),
                        "sequential_excess_pnl": 0.0,
                        "fill_count": 0,
                        "forced_no_quote_count": 0,
                        "max_abs_inventory": 0,
                        "max_drawdown": 0.0,
                        "win_rate": 0.0,
                        "loss_rate": 0.0,
                        "avg_trade_excess_pnl": 0.0,
                        "fill_uses_future_pnl": False,
                    }
                )
    return rows


def run_policy_split(split: str, policy: dict, norms) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    curve_rows = []
    for fill_model in FILL_MODELS:
        accum: dict[tuple[int, float], dict[str, float]] = {
            (cap, fee): {
                "pnl": 0.0,
                "fill_count": 0.0,
                "forced": 0.0,
                "max_inv": 0.0,
                "max_dd": 0.0,
                "wins": 0.0,
                "losses": 0.0,
            }
            for cap in INVENTORY_CAPS
            for fee in FEE_MULTIPLIERS
        }
        curve_equity = 0.0
        curve_point = 0
        for frame in iter_excess_event_frames(split, horizons=[HORIZON]):
            decision = policy_decision(policy, frame).astype(np.int8)
            fills, _ = deterministic_fill_mask(frame, decision, fill_model, norms)
            pnl = base_excess_pnl(frame, decision)
            for cap in INVENTORY_CAPS:
                for fee in FEE_MULTIPLIERS:
                    total, fills_n, forced, max_inv, max_dd, wins, losses = simulate_scheduled_excess_numba(
                        decision,
                        fills.astype(np.bool_),
                        pnl.astype(np.float64),
                        HORIZON,
                        int(cap),
                        float(fee),
                        float(MAKER_FEE),
                    )
                    state = accum[(cap, fee)]
                    state["pnl"] += float(total)
                    state["fill_count"] += float(fills_n)
                    state["forced"] += float(forced)
                    state["max_inv"] = max(state["max_inv"], float(max_inv))
                    state["max_dd"] = max(state["max_dd"], float(max_dd))
                    state["wins"] += float(wins)
                    state["losses"] += float(losses)
            if fill_model == "conservative_l2_stress":
                total, _, _, _, _, _, _ = simulate_scheduled_excess_numba(
                    decision,
                    fills.astype(np.bool_),
                    pnl.astype(np.float64),
                    HORIZON,
                    1,
                    1.0,
                    float(MAKER_FEE),
                )
                curve_equity += float(total)
                curve_rows.append(
                    {
                        "split": split,
                        "policy_id": policy["policy_id"],
                        "fill_model": fill_model,
                        "point": curve_point,
                        "equity": curve_equity,
                    }
                )
                curve_point += 1
        for (cap, fee), state in accum.items():
            fill_count = int(state["fill_count"])
            denom = max(fill_count, 1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy["policy_id"],
                    "horizon": HORIZON,
                    "fill_model": fill_model,
                    "inventory_cap": int(cap),
                    "fee_multiplier": float(fee),
                    "sequential_excess_pnl": float(state["pnl"]),
                    "fill_count": fill_count,
                    "forced_no_quote_count": int(state["forced"]),
                    "max_abs_inventory": int(state["max_inv"]),
                    "max_drawdown": float(state["max_dd"]),
                    "win_rate": float(state["wins"] / denom) if fill_count else 0.0,
                    "loss_rate": float(state["losses"] / denom) if fill_count else 0.0,
                    "avg_trade_excess_pnl": float(state["pnl"] / denom) if fill_count else 0.0,
                    "fill_uses_future_pnl": False,
                }
            )
    return rows, pd.DataFrame(curve_rows)


def plot_curves(curves: pd.DataFrame) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (split, policy_id), group in curves.groupby(["split", "policy_id"]):
        ax.plot(group["point"], group["equity"], marker="o", linewidth=1.4, label=f"{split} {policy_id}")
    ax.axhline(0.0, color="#20242b", linewidth=1.0)
    ax.set_title("Sequential Excess Equity by Session: conservative_l2_stress, cap=1, fee=1, H20")
    ax.set_xlabel("Session part index")
    ax.set_ylabel("Cumulative excess PnL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sequential_excess_equity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    norms = compute_train_fill_norms()
    policies = policy_set()
    rows = []
    curves = []
    for split in EVAL_SPLITS:
        rows.extend(zero_rows(split, "no_quote"))
        for policy_id in ["old_bid_only_spread_ge_1", SELECTED_POLICY_ID]:
            policy_rows, curve = run_policy_split(split, policies[policy_id], norms)
            rows.extend(policy_rows)
            if not curve.empty:
                curves.append(curve)
            print(f"sequential excess inventory complete split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sequential_excess_inventory_backtest.csv", index=False)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if not curve_df.empty:
        curve_df.to_csv(RESULTS / "sequential_excess_equity_curve_sample.csv", index=False)
        plot_curves(curve_df)
    focus = out[(out["fill_model"] == "conservative_l2_stress") & (out["inventory_cap"].astype(int) == 1) & (out["fee_multiplier"].astype(float) == 1.0)]
    print("Sequential excess inventory complete.")
    print(focus[["split", "policy_id", "sequential_excess_pnl", "fill_count", "win_rate", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()

