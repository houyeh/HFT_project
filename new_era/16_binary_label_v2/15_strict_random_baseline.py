from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from round2_audit_utils import MAKER_FEE, MAX_INV, POINT_VALUE, RESULTS, load_test_frame, max_drawdown


SIDES = ["bid", "ask"]
METHODS = ["random_v1", "random_v2", "random_v3", "random_v4"]


@dataclass
class RandomSpec:
    method: str
    side: str
    bucket_id: str
    required_model_cancels: int
    candidate_pool_size: int
    action_ids: np.ndarray
    replacement_used: bool
    fallback_level: str


def _build_actions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float]]:
    actions: list[dict[str, Any]] = []
    final_mid: dict[str, float] = {}
    for session_id, group in frame.groupby("session_id", sort=True):
        g = group.sort_values("local_row_id")
        final_mid[str(session_id)] = float(g["mid_price"].iloc[-1])
        bid = g["bid_px_1"].to_numpy(dtype=float)
        ask = g["ask_px_1"].to_numpy(dtype=float)
        idx = g.index.to_numpy()
        for pos in range(max(len(g) - 1, 0)):
            row = g.iloc[pos]
            if ask[pos + 1] <= bid[pos]:
                actions.append(_action_row(len(actions), "bid", str(session_id), int(idx[pos]), float(bid[pos]), row))
            if bid[pos + 1] >= ask[pos]:
                actions.append(_action_row(len(actions), "ask", str(session_id), int(idx[pos]), float(ask[pos]), row))
    action_df = pd.DataFrame(actions)
    by_session = {
        str(session_id): g.sort_values(["row_pos", "side"]).reset_index(drop=True)
        for session_id, g in action_df.groupby("session_id", sort=True)
    }
    return action_df, by_session, final_mid


def _action_row(action_id: int, side: str, session_id: str, row_pos: int, price: float, row: pd.Series) -> dict[str, Any]:
    return {
        "action_id": int(action_id),
        "side": side,
        "session_id": session_id,
        "row_pos": row_pos,
        "price": price,
        "spread_bucket": str(row["spread_bucket"]),
        "event_rate_bucket": str(row["event_rate_bucket"]),
        "coarse_spread_bucket": str(row["coarse_spread_bucket"]),
        "coarse_event_rate_bucket": str(row["coarse_event_rate_bucket"]),
    }


def _candidate_counts(frame: pd.DataFrame, side: str, keys: list[str]) -> pd.Series:
    if not keys:
        return pd.Series({"ALL": int(len(frame))})
    return frame.groupby(keys, observed=False).size()


def _model_counts(frame: pd.DataFrame, side: str, keys: list[str]) -> pd.Series:
    col = f"pull_{side}"
    if not keys:
        return pd.Series({"ALL": int(frame[col].sum())})
    return frame[frame[col]].groupby(keys, observed=False).size()


def _action_groups(action_df: pd.DataFrame, side: str, keys: list[str]) -> dict[tuple, np.ndarray]:
    side_actions = action_df[action_df["side"] == side]
    out: dict[tuple, np.ndarray] = {}
    if side_actions.empty:
        return out
    if not keys:
        return {("ALL",): side_actions["action_id"].to_numpy(dtype=np.int64)}
    for key, group in side_actions.groupby(keys, observed=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        out[key] = group["action_id"].to_numpy(dtype=np.int64)
    return out


def _make_spec(
    *,
    method: str,
    side: str,
    bucket_id: str,
    required: int,
    candidate_pool_size: int,
    action_ids: np.ndarray,
    fallback_level: str,
) -> RandomSpec:
    return RandomSpec(
        method=method,
        side=side,
        bucket_id=bucket_id,
        required_model_cancels=int(required),
        candidate_pool_size=int(candidate_pool_size),
        action_ids=np.asarray(action_ids, dtype=np.int64),
        replacement_used=bool(int(required) > int(candidate_pool_size)),
        fallback_level=fallback_level,
    )


def _lookup_count(counts: pd.Series, key: tuple) -> int:
    if getattr(counts.index, "nlevels", 1) == 1 and len(key) == 1 and key[0] in counts.index:
        return int(counts.loc[key[0]])
    if key in counts.index:
        return int(counts.loc[key])
    raise KeyError(key)


def _series_tuple_map(counts: pd.Series) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for key, value in counts.items():
        if not isinstance(key, tuple):
            key = (key,)
        out[key] = int(value)
    return out


def _health_row(spec: RandomSpec, fallback_reason: str) -> dict[str, Any]:
    k = int(spec.required_model_cancels)
    n = int(spec.candidate_pool_size)
    return {
        "method": spec.method,
        "side": spec.side,
        "bucket_id": spec.bucket_id,
        "required_model_cancels": k,
        "candidate_pool_size": n,
        "effective_unique_candidates": n,
        "replacement_used": bool(spec.replacement_used),
        "duplicate_rate": float(max(0, k - n) / max(k, 1)) if spec.replacement_used else 0.0,
        "overlap_with_model_rate": float(min(k, n) / max(n, 1)),
        "fallback_level": spec.fallback_level,
        "fallback_reason": fallback_reason,
    }


def _build_standard_specs(
    frame: pd.DataFrame,
    action_df: pd.DataFrame,
    *,
    method: str,
    key_cols: list[str],
) -> tuple[list[RandomSpec], list[dict[str, Any]]]:
    specs: list[RandomSpec] = []
    health: list[dict[str, Any]] = []
    for side in SIDES:
        counts = _candidate_counts(frame, side, key_cols)
        count_map = _series_tuple_map(counts)
        demands = _model_counts(frame, side, key_cols)
        fill_groups = _action_groups(action_df, side, key_cols)
        for key, required in demands.items():
            if not isinstance(key, tuple):
                key = (key,)
            n = int(count_map[key])
            action_ids = fill_groups.get(key, np.array([], dtype=np.int64))
            bucket_id = f"{method}|{side}|" + "|".join(map(str, key))
            spec = _make_spec(
                method=method,
                side=side,
                bucket_id=bucket_id,
                required=int(required),
                candidate_pool_size=n,
                action_ids=action_ids,
                fallback_level=f"{method}_exact",
            )
            specs.append(spec)
            health.append(_health_row(spec, "exact_match"))
    return specs, health


def _build_v4_specs(frame: pd.DataFrame, action_df: pd.DataFrame) -> tuple[list[RandomSpec], list[dict[str, Any]]]:
    exact_cols = ["session_id", "spread_bucket", "event_rate_bucket"]
    levels = [
        ("v4_exact", exact_cols, 3.0, 0.50),
        ("v4_coarse_eventrate", ["session_id", "spread_bucket", "coarse_event_rate_bucket"], 3.0, 0.50),
        (
            "v4_coarse_spread_eventrate",
            ["session_id", "coarse_spread_bucket", "coarse_event_rate_bucket"],
            3.0,
            0.50,
        ),
        ("v4_fallback_v3", ["session_id", "spread_bucket"], 1.0, 1.00),
    ]
    specs: list[RandomSpec] = []
    health: list[dict[str, Any]] = []

    for side in SIDES:
        exact_demand = _model_counts(frame, side, exact_cols)
        side_counts = {name: _candidate_counts(frame, side, cols) for name, cols, _, _ in levels}
        side_count_maps = {name: _series_tuple_map(counts) for name, counts in side_counts.items()}
        side_actions = {name: _action_groups(action_df, side, cols) for name, cols, _, _ in levels}

        for exact_key, required in exact_demand.items():
            if not isinstance(exact_key, tuple):
                exact_key = (exact_key,)
            exact_lookup = dict(zip(exact_cols, exact_key))
            chosen: tuple[str, list[str], tuple, int, np.ndarray, str] | None = None
            reasons = []
            for level_name, cols, min_multiple, max_overlap in levels:
                key = tuple(exact_lookup[c] if c in exact_lookup else _coarse_from_exact(c, exact_lookup) for c in cols)
                count_map = side_count_maps[level_name]
                if key not in count_map:
                    reasons.append(f"{level_name}:empty_candidate_pool")
                    continue
                n = int(count_map[key])
                overlap = int(required) / max(n, 1)
                sparse = n < int(required) * float(min_multiple)
                high_overlap = overlap > float(max_overlap)
                if sparse and level_name != "v4_fallback_v3":
                    reasons.append(f"{level_name}:candidate_pool_too_sparse")
                    continue
                if high_overlap and level_name != "v4_fallback_v3":
                    reasons.append(f"{level_name}:expected_overlap_too_high")
                    continue
                action_ids = side_actions[level_name].get(key, np.array([], dtype=np.int64))
                reason = "exact_match" if level_name == "v4_exact" else "hierarchical_fallback;" + ";".join(reasons)
                chosen = (level_name, cols, key, n, action_ids, reason)
                break

            if chosen is None:
                level_name, cols, _, _ = levels[-1]
                key = (exact_lookup["session_id"], exact_lookup["spread_bucket"])
                n = int(side_count_maps[level_name][key])
                action_ids = side_actions[level_name].get(key, np.array([], dtype=np.int64))
                chosen = (level_name, cols, key, n, action_ids, "forced_v3_fallback")

            level_name, _, key, n, action_ids, reason = chosen
            bucket_id = f"random_v4|{side}|exact=" + "|".join(map(str, exact_key)) + f"|effective={level_name}:" + "|".join(
                map(str, key)
            )
            spec = _make_spec(
                method="random_v4",
                side=side,
                bucket_id=bucket_id,
                required=int(required),
                candidate_pool_size=n,
                action_ids=action_ids,
                fallback_level=level_name,
            )
            specs.append(spec)
            health.append(_health_row(spec, reason))
    return specs, health


def _coarse_from_exact(col: str, exact_lookup: dict[str, Any]) -> str:
    if col == "coarse_event_rate_bucket":
        return "slow_half" if exact_lookup["event_rate_bucket"] in {"q1_slow", "q2"} else "fast_half"
    if col == "coarse_spread_bucket":
        return "tight_<=2" if exact_lookup["spread_bucket"] in {"<=1", "(1,2]"} else "wide_>2"
    raise KeyError(col)


def _sample_cancels(specs: list[RandomSpec], rng: np.random.Generator, num_actions: int) -> np.ndarray:
    canceled = np.zeros(num_actions, dtype=bool)
    for spec in specs:
        n = int(spec.candidate_pool_size)
        m = int(len(spec.action_ids))
        k = int(spec.required_model_cancels)
        if n <= 0 or m <= 0 or k <= 0:
            continue
        if spec.replacement_used:
            hit_count = int(rng.binomial(k, min(max(m / n, 0.0), 1.0)))
            if hit_count:
                chosen = rng.choice(spec.action_ids, size=hit_count, replace=True)
                canceled[np.unique(chosen)] = True
        else:
            draws = min(k, n)
            hit_count = int(rng.hypergeometric(m, max(n - m, 0), draws))
            if hit_count:
                chosen = rng.choice(spec.action_ids, size=min(hit_count, m), replace=False)
                canceled[chosen] = True
    return canceled


def _simulate_actions(
    frame: pd.DataFrame,
    actions_by_session: dict[str, pd.DataFrame],
    final_mid: dict[str, float],
    canceled: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    session_rows = []
    mid_by_row = frame["mid_price"].to_dict()
    for session_id, actions in actions_by_session.items():
        cash = 0.0
        fees = 0.0
        inventory = 0
        trades = 0
        equity_curve: list[float] = []
        for _, action in actions.iterrows():
            action_id = int(action["action_id"])
            if canceled[action_id]:
                continue
            if action["side"] == "bid":
                if inventory >= MAX_INV:
                    continue
                cash -= float(action["price"]) * POINT_VALUE
                inventory += 1
            else:
                if inventory <= -MAX_INV:
                    continue
                cash += float(action["price"]) * POINT_VALUE
                inventory -= 1
            fees += MAKER_FEE
            trades += 1
            equity_curve.append(cash - fees + inventory * float(mid_by_row[int(action["row_pos"])]) * POINT_VALUE)
        final_equity = cash - fees + inventory * float(final_mid[session_id]) * POINT_VALUE
        if equity_curve:
            equity_curve[-1] = final_equity
        equity = np.asarray(equity_curve, dtype=float)
        session_rows.append(
            {
                "session_id": session_id,
                "net_pnl": float(final_equity),
                "trade_count": int(trades),
                "ending_inventory": int(inventory),
                "max_drawdown": max_drawdown(equity),
            }
        )
    session_df = pd.DataFrame(session_rows)
    agg = {
        "net_pnl": float(session_df["net_pnl"].sum()),
        "trade_count": int(session_df["trade_count"].sum()),
        "max_drawdown": float(session_df["max_drawdown"].max()),
    }
    return agg, session_df


def _method_specs(frame: pd.DataFrame, action_df: pd.DataFrame) -> tuple[dict[str, list[RandomSpec]], pd.DataFrame]:
    methods: dict[str, list[RandomSpec]] = {}
    health_rows: list[dict[str, Any]] = []
    key_map = {
        "random_v1": [],
        "random_v2": ["session_id"],
        "random_v3": ["session_id", "spread_bucket"],
    }
    for method, cols in key_map.items():
        specs, health = _build_standard_specs(frame, action_df, method=method, key_cols=cols)
        methods[method] = specs
        health_rows.extend(health)
    specs, health = _build_v4_specs(frame, action_df)
    methods["random_v4"] = specs
    health_rows.extend(health)
    return methods, pd.DataFrame(health_rows)


def run(trials: int, results_dir: Path) -> None:
    if int(trials) < 5000:
        raise ValueError("Professor audit requires at least 5000 trials per random baseline.")
    results_dir.mkdir(parents=True, exist_ok=True)

    frame = load_test_frame()
    action_df, actions_by_session, final_mid = _build_actions(frame)
    methods, health = _method_specs(frame, action_df)

    test_summary = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "test_backtest_summary.csv").iloc[0]
    model_pnl = float(test_summary["net_pnl"])
    rng = np.random.default_rng(20260523)

    trial_rows: list[dict[str, Any]] = []
    session_trial_rows: list[dict[str, Any]] = []
    total_cancel_count = int(frame["pull_bid"].sum() + frame["pull_ask"].sum())
    for method in METHODS:
        specs = methods[method]
        for trial in range(int(trials)):
            canceled = _sample_cancels(specs, rng, len(action_df))
            agg, session_df = _simulate_actions(frame, actions_by_session, final_mid, canceled)
            trial_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "net_pnl": agg["net_pnl"],
                    "trade_count": agg["trade_count"],
                    "max_drawdown": agg["max_drawdown"],
                    "cancel_count": total_cancel_count,
                }
            )
            for _, row in session_df.iterrows():
                session_trial_rows.append(
                    {
                        "method": method,
                        "trial": trial,
                        "session_id": row["session_id"],
                        "net_pnl": row["net_pnl"],
                        "trade_count": row["trade_count"],
                        "max_drawdown": row["max_drawdown"],
                    }
                )
        print(f"{method}: {trials} trials complete")

    trials_df = pd.DataFrame(trial_rows)
    session_trials = pd.DataFrame(session_trial_rows)
    health.to_csv(results_dir / "random_matching_health_report.csv", index=False)
    trials_df.to_csv(results_dir / "strict_random_baseline_trials.csv", index=False)
    session_trials.to_csv(results_dir / "strict_random_baseline_session_trials.csv", index=False)

    v4_health = health[health["method"] == "random_v4"]
    v4_max_overlap = float(v4_health["overlap_with_model_rate"].max()) if not v4_health.empty else 0.0
    v4_diagnostic_only = bool(v4_max_overlap > 0.50)

    dist_rows = []
    for method, group in trials_df.groupby("method", sort=True):
        pnl = group["net_pnl"].to_numpy(dtype=float)
        diagnostic_only = bool(method == "random_v4" and v4_diagnostic_only)
        dist_rows.append(
            {
                "method": method,
                "trials": int(len(group)),
                "model_net_pnl": model_pnl,
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl, ddof=0)),
                "p90": float(np.percentile(pnl, 90)),
                "p95": float(np.percentile(pnl, 95)),
                "p99": float(np.percentile(pnl, 99)),
                "empirical_p_value_random_ge_model": float(np.mean(pnl >= model_pnl)),
                "mean_trade_count": float(group["trade_count"].mean()),
                "mean_max_drawdown": float(group["max_drawdown"].mean()),
                "cancel_count": total_cancel_count,
                "diagnostic_only": diagnostic_only,
                "diagnostic_reason": (
                    f"v4 max overlap_with_model_rate={v4_max_overlap:.4f} exceeds 0.50"
                    if diagnostic_only
                    else ""
                ),
            }
        )
    pd.DataFrame(dist_rows).to_csv(results_dir / "strict_random_baseline_distribution.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stricter matched random baselines v1-v4.")
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.trials, args.results_dir)
    print(f"Strict random baselines written to {args.results_dir}")
