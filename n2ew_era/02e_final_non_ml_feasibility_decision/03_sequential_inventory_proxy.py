from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from common_phase02e import (
    ACTION_SLOT_ASK,
    ACTION_SLOT_BID,
    FROZEN_SPLITS,
    MAKER_FEE,
    RANDOM_SEED,
    POINT_VALUE,
    RESULTS,
    ROOT,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    delayed_decision,
    ensure_dirs,
    event_frame_from_part,
    part_paths,
    pnl_from_decision,
    policy_decision,
    policy_set,
    spread_bucket_values,
    write_markdown,
)


FILL_MODES = ["markout_fill_all", "conservative_adverse_fill", "random_fill_rate"]
INVENTORY_CAPS: list[int | None] = [1, 2, 5, None]
EXIT_RULES = ["fixed_horizon_H20", "session_end_mtm"]
RANDOM_TRIALS_FOR_95P_POLICY = 250
POSITIVE_FILL_KEEP_FOR_CONSERVATIVE = 0.5
RANDOM_FILL_RATE = 0.5


@njit(cache=True)
def simulate_arrays_numba(
    decision: np.ndarray,
    trade_pnl: np.ndarray,
    finite_trade: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    mid: np.ndarray,
    random_values: np.ndarray,
    fill_mode_code: int,
    inventory_cap: int,
    exit_rule_code: int,
    horizon: int,
    point_value: float,
    maker_fee: float,
) -> tuple[float, int, int, int, float, float, float]:
    n = len(decision)
    cash = 0.0
    inventory = 0
    max_abs_inventory = 0
    forced_no_quote_count = 0
    fill_count = 0
    win_count = 0
    trade_pnl_sum = 0.0
    peak = 0.0
    max_dd = 0.0
    close_side_sum = np.zeros(n, dtype=np.int64)
    final_mid = mid[n - 1] if n > 0 else 0.0

    for t in range(n):
        if exit_rule_code == 0 and close_side_sum[t] != 0:
            side_sum = close_side_sum[t]
            cash += side_sum * mid[t] * point_value
            inventory -= side_sum

        code = decision[t]
        selected = code != 0 and finite_trade[t]
        filled = False
        if selected:
            if fill_mode_code == 0:
                filled = True
            elif fill_mode_code == 1:
                filled = trade_pnl[t] <= 0.0 or random_values[t] < 0.5
            else:
                filled = random_values[t] < 0.5

        if filled:
            side_sign = 1 if code == 1 else -1
            next_inventory = inventory + side_sign
            if inventory_cap >= 0 and abs(next_inventory) > inventory_cap:
                forced_no_quote_count += 1
            else:
                if side_sign > 0:
                    cash -= best_bid[t] * point_value + maker_fee
                else:
                    cash += best_ask[t] * point_value - maker_fee
                inventory = next_inventory
                if abs(inventory) > max_abs_inventory:
                    max_abs_inventory = abs(inventory)
                fill_count += 1
                trade_pnl_sum += trade_pnl[t]
                if trade_pnl[t] > 0.0:
                    win_count += 1
                if exit_rule_code == 0:
                    close_idx = t + horizon
                    if close_idx >= n:
                        close_idx = n - 1
                    close_side_sum[close_idx] += side_sign

        equity = cash + inventory * mid[t] * point_value
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    if exit_rule_code == 1 and inventory != 0:
        cash += inventory * final_mid * point_value
        inventory = 0
        equity = cash
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    avg_trade_pnl = trade_pnl_sum / fill_count if fill_count > 0 else 0.0
    win_rate = win_count / fill_count if fill_count > 0 else 0.0
    return cash, fill_count, forced_no_quote_count, max_abs_inventory, max_dd, win_rate, avg_trade_pnl


def load_split_frames(split: str) -> list[pd.DataFrame]:
    return [event_frame_from_part(path, horizons=[SELECTED_HORIZON]) for path in part_paths(split)]


def random_95p_decisions(frames: list[pd.DataFrame], base_policy: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    group_info: list[tuple[int, str, np.ndarray, np.ndarray, int, int]] = []
    for frame_idx, frame in enumerate(frames):
        base_decision = policy_decision(base_policy, frame)
        buckets = spread_bucket_values(frame["spread_ticks"])
        session_id = str(frame["session_id"].iloc[0])
        for side_name, code, pnl_col in [
            ("bid", ACTION_SLOT_BID + 1, f"bid_pnl_H{SELECTED_HORIZON}"),
            ("ask", ACTION_SLOT_ASK + 1, f"ask_pnl_H{SELECTED_HORIZON}"),
        ]:
            values = np.nan_to_num(frame[pnl_col].to_numpy(dtype=float), nan=0.0)
            selected = base_decision == code
            for bucket in np.unique(buckets):
                idx = np.flatnonzero(buckets == bucket)
                k = int((selected & (buckets == bucket)).sum())
                if len(idx) == 0 or k == 0:
                    continue
                group_info.append((frame_idx, side_name, idx, values[idx], k, code))

    trial_pnls = []
    for trial in range(RANDOM_TRIALS_FOR_95P_POLICY):
        rng = np.random.default_rng(RANDOM_SEED + 7100 + trial + len(split))
        total = 0.0
        for _, _, idx, values, k, _ in group_info:
            if k >= len(idx):
                total += float(values.sum())
            else:
                chosen = rng.choice(len(idx), size=k, replace=False)
                total += float(values[chosen].sum())
        trial_pnls.append(total)
    q95 = float(np.quantile(trial_pnls, 0.95)) if trial_pnls else 0.0
    chosen_trial = int(np.argmin(np.abs(np.asarray(trial_pnls) - q95))) if trial_pnls else 0

    decisions = {str(frame["session_id"].iloc[0]): np.zeros(len(frame), dtype=np.int8) for frame in frames}
    rng = np.random.default_rng(RANDOM_SEED + 7100 + chosen_trial + len(split))
    for frame_idx, _, idx, _, k, code in group_info:
        session_id = str(frames[frame_idx]["session_id"].iloc[0])
        if k >= len(idx):
            selected_idx = idx
        else:
            selected_idx = idx[rng.choice(len(idx), size=k, replace=False)]
        decisions[session_id][selected_idx] = code
    return decisions


def fill_mask(
    selected: np.ndarray,
    trade_pnl: np.ndarray,
    fill_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if fill_mode == "markout_fill_all":
        return selected.copy()
    if fill_mode == "conservative_adverse_fill":
        out = selected & (trade_pnl <= 0.0)
        good = selected & (trade_pnl > 0.0)
        out[good] = rng.random(int(good.sum())) < POSITIVE_FILL_KEEP_FOR_CONSERVATIVE
        return out
    if fill_mode == "random_fill_rate":
        out = np.zeros_like(selected, dtype=bool)
        out[selected] = rng.random(int(selected.sum())) < RANDOM_FILL_RATE
        return out
    raise ValueError(f"Unknown fill mode: {fill_mode}")


def close_trade_pnl(frame: pd.DataFrame, decision: np.ndarray, exit_rule: str) -> np.ndarray:
    bid = decision == (ACTION_SLOT_BID + 1)
    ask = decision == (ACTION_SLOT_ASK + 1)
    out = np.zeros(len(frame), dtype=float)
    if exit_rule == "fixed_horizon_H20":
        out[:] = np.nan
        bid_pnl = frame[f"bid_pnl_H{SELECTED_HORIZON}"].to_numpy(dtype=float)
        ask_pnl = frame[f"ask_pnl_H{SELECTED_HORIZON}"].to_numpy(dtype=float)
        out[bid] = bid_pnl[bid]
        out[ask] = ask_pnl[ask]
        out[decision == 0] = 0.0
        return out
    final_mid = float(frame["mid_price"].iloc[-1])
    out[bid] = (final_mid - frame.loc[bid, "best_bid"].astype(float).to_numpy()) * POINT_VALUE - MAKER_FEE
    out[ask] = (frame.loc[ask, "best_ask"].astype(float).to_numpy() - final_mid) * POINT_VALUE - MAKER_FEE
    return out


def simulate_session(
    frame: pd.DataFrame,
    decision: np.ndarray,
    fill_mode: str,
    inventory_cap: int | None,
    exit_rule: str,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    decision = decision.copy()
    trade_pnl = close_trade_pnl(frame, decision, exit_rule)
    finite_trade = np.isfinite(trade_pnl)
    trade_pnl = np.nan_to_num(trade_pnl, nan=0.0)
    mid = frame["mid_price"].astype(float).to_numpy()
    best_bid = frame["best_bid"].astype(float).to_numpy()
    best_ask = frame["best_ask"].astype(float).to_numpy()
    random_values = rng.random(len(frame))
    fill_mode_code = {"markout_fill_all": 0, "conservative_adverse_fill": 1, "random_fill_rate": 2}[fill_mode]
    exit_rule_code = {"fixed_horizon_H20": 0, "session_end_mtm": 1}[exit_rule]
    cap_value = -1 if inventory_cap is None else int(inventory_cap)
    pnl, fill_count, forced_no_quote_count, max_abs_inventory, max_dd, win_rate, avg_trade_pnl = simulate_arrays_numba(
        decision.astype(np.int8),
        trade_pnl.astype(np.float64),
        finite_trade.astype(np.bool_),
        best_bid.astype(np.float64),
        best_ask.astype(np.float64),
        mid.astype(np.float64),
        random_values.astype(np.float64),
        int(fill_mode_code),
        int(cap_value),
        int(exit_rule_code),
        int(SELECTED_HORIZON),
        float(POINT_VALUE),
        float(MAKER_FEE),
    )
    return {
        "sequential_pnl": float(pnl),
        "fill_count": int(fill_count),
        "forced_no_quote_count": int(forced_no_quote_count),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_trade_pnl),
    }


def run_split(split: str) -> pd.DataFrame:
    frames = load_split_frames(split)
    policies = policy_set()
    random_map = random_95p_decisions(frames, policies[SELECTED_POLICY_ID], split)
    rows = []
    for policy_id in ["no_quote", "old_bid_only_spread_ge_1", SELECTED_POLICY_ID, "matched_random_95p"]:
        for fill_mode in FILL_MODES:
            for inventory_cap in INVENTORY_CAPS:
                for exit_rule in EXIT_RULES:
                    totals = defaultdict(float)
                    int_totals = defaultdict(int)
                    weighted_win = []
                    weighted_avg = []
                    for frame in frames:
                        session_id = str(frame["session_id"].iloc[0])
                        if policy_id == "matched_random_95p":
                            decision = random_map[session_id]
                        else:
                            decision = policy_decision(policies[policy_id], frame)
                        metrics = simulate_session(
                            frame,
                            decision,
                            fill_mode=fill_mode,
                            inventory_cap=inventory_cap,
                            exit_rule=exit_rule,
                            seed=RANDOM_SEED + len(session_id) + len(fill_mode),
                        )
                        totals["sequential_pnl"] += float(metrics["sequential_pnl"])
                        totals["max_drawdown"] = max(totals["max_drawdown"], float(metrics["max_drawdown"]))
                        int_totals["fill_count"] += int(metrics["fill_count"])
                        int_totals["forced_no_quote_count"] += int(metrics["forced_no_quote_count"])
                        int_totals["max_abs_inventory"] = max(int_totals["max_abs_inventory"], int(metrics["max_abs_inventory"]))
                        if int(metrics["fill_count"]) > 0:
                            weighted_win.append((float(metrics["win_rate"]), int(metrics["fill_count"])))
                            weighted_avg.append((float(metrics["avg_trade_pnl"]), int(metrics["fill_count"])))
                    fill_count = max(int_totals["fill_count"], 1)
                    rows.append(
                        {
                            "split": split,
                            "policy_id": policy_id,
                            "horizon": SELECTED_HORIZON,
                            "fill_mode": fill_mode,
                            "inventory_cap": "unlimited" if inventory_cap is None else int(inventory_cap),
                            "exit_rule": exit_rule,
                            "sequential_pnl": float(totals["sequential_pnl"]),
                            "fill_count": int(int_totals["fill_count"]),
                            "forced_no_quote_count": int(int_totals["forced_no_quote_count"]),
                            "max_abs_inventory": int(int_totals["max_abs_inventory"]),
                            "max_drawdown": float(totals["max_drawdown"]),
                            "win_rate": float(sum(w * n for w, n in weighted_win) / fill_count) if weighted_win else 0.0,
                            "avg_trade_pnl": float(sum(a * n for a, n in weighted_avg) / fill_count) if weighted_avg else 0.0,
                            "random_trials_for_95p_policy": RANDOM_TRIALS_FOR_95P_POLICY if policy_id == "matched_random_95p" else 0,
                        }
                    )
        print(f"sequential inventory simulated split={split} policy={policy_id}")
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    out = pd.concat([run_split(split) for split in FROZEN_SPLITS if part_paths(split)], ignore_index=True)
    out.to_csv(RESULTS / "sequential_inventory_backtest.csv", index=False)
    moderate_view = out[
        (out["fill_mode"] == "conservative_adverse_fill")
        & (out["inventory_cap"].astype(str) == "1")
        & (out["exit_rule"] == "fixed_horizon_H20")
    ].copy()
    lines = [
        "# Sequential Inventory Proxy Report",
        "",
        "This is a stricter chronological proxy simulator for frozen policies. It tracks cash, signed inventory, forced no-quote from inventory caps, and drawdown over fill/close events. It remains a proxy: it still lacks true queue priority, exchange acknowledgements, partial fills, and aggressor-side replay.",
        "",
        "Fill modes:",
        "",
        "- `markout_fill_all`: every selected quote fills; optimistic.",
        "- `conservative_adverse_fill`: losing fills always occur, winning fills are kept at probability 0.5.",
        "- `random_fill_rate`: all selected quotes fill at probability 0.5.",
        "",
        "Exit rules:",
        "",
        "- `fixed_horizon_H20`: close each fill at the H20 future mid.",
        "- `session_end_mtm`: mark all open inventory to the session-final mid.",
        "",
        "## Conservative H20 / Cap=1 View",
        "",
        moderate_view.to_markdown(index=False) if not moderate_view.empty else "No conservative view rows were produced.",
        "",
        "## Decision Use",
        "",
        "If policies collapse under conservative adverse fills and small inventory caps, they should be classified as signal discovery rather than final strategy candidates.",
    ]
    write_markdown(ROOT / "SEQUENTIAL_INVENTORY_PROXY_REPORT.md", lines)
    print("Sequential inventory proxy simulation complete.")
    print(moderate_view[["split", "policy_id", "sequential_pnl", "fill_count", "max_abs_inventory", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
