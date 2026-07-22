from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))

from common_phase02 import (  # noqa: E402
    HORIZONS,
    RANDOM_SEED,
    iter_event_frames,
    policy_catalog,
    policy_decision,
)


ROOT = PROJECT_ROOT / "n2ew_era" / "02b_strict_baseline_and_excess_pnl"
RESULTS = ROOT / "results"
PHASE02_RESULTS = PHASE02_DIR / "results"
MAKER_FEE = 15.0
TRIALS = 1000
STRICT_SELECTION_SPLIT = "val"
EXCLUDED_SELECT_ALL_POLICIES = {"bid_all", "ask_all", "old_bid_only_spread_ge_1"}
MIN_ACTION_COUNT = 1_000
MAX_DRAWDOWN = 2_000_000.0
FEE_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0]
SPREAD_LABELS = ["<=1", "(1,2]", "(2,3]", ">3"]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def spread_bucket(spread: pd.Series) -> pd.Series:
    values = pd.to_numeric(spread, errors="coerce")
    return pd.cut(values, bins=[-np.inf, 1.0, 2.0, 3.0, np.inf], labels=SPREAD_LABELS).astype(str)


def policy_lookup() -> dict[str, dict[str, Any]]:
    return {p["policy_id"]: p for p in policy_catalog()}


def population_and_selection_stats(splits: list[str], policies: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pop_rows: list[dict[str, Any]] = []
    sel_rows: list[dict[str, Any]] = []

    for split in splits:
        print(f"building strict random group stats split={split}")
        pop_acc: dict[tuple[str, str, str, int], dict[str, float]] = {}
        sel_acc: dict[tuple[str, str, str, str], int] = {}

        for frame in iter_event_frames(split, horizons=HORIZONS):
            buckets = spread_bucket(frame["spread_ticks"]).to_numpy(dtype=object)
            sessions = frame["session_id"].astype(str).to_numpy(dtype=object)
            group_base = pd.DataFrame({"session_id": sessions, "spread_bucket": buckets})

            for side, side_col in [("bid", "bid_pnl_H"), ("ask", "ask_pnl_H")]:
                for horizon in HORIZONS:
                    values = np.nan_to_num(frame[f"{side_col}{horizon}"].to_numpy(dtype=float), nan=0.0)
                    tmp = group_base.copy()
                    tmp["value"] = values
                    tmp["value_sq"] = values * values
                    agg = tmp.groupby(["session_id", "spread_bucket"], observed=False).agg(
                        n=("value", "size"),
                        sum_value=("value", "sum"),
                        sumsq_value=("value_sq", "sum"),
                    )
                    for (session_id, bucket), row in agg.iterrows():
                        key = (side, str(session_id), str(bucket), int(horizon))
                        target = pop_acc.setdefault(key, {"n": 0.0, "sum": 0.0, "sumsq": 0.0})
                        target["n"] += float(row["n"])
                        target["sum"] += float(row["sum_value"])
                        target["sumsq"] += float(row["sumsq_value"])

            group_df = group_base
            for policy in policies:
                decision = policy_decision(policy, frame)
                for side_name, decision_value in [("bid", 1), ("ask", 2)]:
                    mask = decision == decision_value
                    if not mask.any():
                        continue
                    tmp = group_df.loc[mask].groupby(["session_id", "spread_bucket"], observed=False).size()
                    for (session_id, bucket), count in tmp.items():
                        key = (policy["policy_id"], side_name, str(session_id), str(bucket))
                        sel_acc[key] = sel_acc.get(key, 0) + int(count)

        for (side, session_id, bucket, horizon), stats in pop_acc.items():
            n = int(stats["n"])
            sum_value = float(stats["sum"])
            sumsq = float(stats["sumsq"])
            mean = sum_value / n if n else 0.0
            pop_var = max(sumsq / n - mean * mean, 0.0) if n else 0.0
            pop_rows.append(
                {
                    "split": split,
                    "side": side,
                    "session_id": session_id,
                    "spread_bucket": bucket,
                    "horizon": horizon,
                    "candidate_pool_size": n,
                    "population_sum": sum_value,
                    "population_mean": mean,
                    "population_var": pop_var,
                }
            )
        for (policy_id, side, session_id, bucket), count in sel_acc.items():
            sel_rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "side": side,
                    "session_id": session_id,
                    "spread_bucket": bucket,
                    "selected_count": int(count),
                }
            )

    return pd.DataFrame(pop_rows), pd.DataFrame(sel_rows)


def matched_random_summary(grid: pd.DataFrame, pop: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    pop_idx = pop.set_index(["split", "side", "session_id", "spread_bucket", "horizon"]).to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    selected_groups = selected.groupby(["split", "policy_id"], sort=False)
    selected_lookup = {key: group for key, group in selected_groups}

    for row in grid.itertuples(index=False):
        split = str(row.split)
        policy_id = str(row.policy_id)
        horizon = int(row.horizon)
        selected_count = int(row.selected_action_count)
        model_pnl = float(row.total_pnl)
        if selected_count <= 0:
            draws = np.zeros(TRIALS, dtype=float)
            rows.append(summary_row(row, draws, "no_selected_quotes", 0, 0, 0.0, 0.0))
            continue

        groups = selected_lookup.get((split, policy_id), pd.DataFrame())
        mean_total = 0.0
        var_total = 0.0
        saturated_groups = 0
        missing_groups = 0
        largest_group_selection_rate = 0.0
        for g in groups.itertuples(index=False):
            k = int(g.selected_count)
            key = (split, str(g.side), str(g.session_id), str(g.spread_bucket), horizon)
            stats = pop_idx.get(key)
            if stats is None:
                missing_groups += 1
                continue
            n = int(stats["candidate_pool_size"])
            if n <= 0 or k <= 0:
                continue
            k = min(k, n)
            rate = k / n
            largest_group_selection_rate = max(largest_group_selection_rate, rate)
            if k == n:
                saturated_groups += 1
                mean_total += float(stats["population_sum"])
                continue
            mean_total += k * float(stats["population_mean"])
            if n > 1:
                var_total += k * (n - k) / (n - 1) * float(stats["population_var"])
        std_total = float(np.sqrt(max(var_total, 0.0)))
        if std_total == 0.0:
            draws = np.full(TRIALS, mean_total, dtype=float)
        else:
            draws = rng.normal(loc=mean_total, scale=std_total, size=TRIALS)
        rows.append(
            {
                "split": split,
                "policy_id": policy_id,
                "horizon": horizon,
                "trials": TRIALS,
                "model_pnl": model_pnl,
                "selected_action_count": selected_count,
                "matched_random_mean": float(np.mean(draws)),
                "matched_random_std": float(np.std(draws, ddof=1)),
                "matched_random_90p": float(np.quantile(draws, 0.90)),
                "matched_random_95p": float(np.quantile(draws, 0.95)),
                "matched_random_99p": float(np.quantile(draws, 0.99)),
                "empirical_p_value_random_ge_model": float(np.mean(draws >= model_pnl)),
                "beats_matched_random_mean": bool(model_pnl > float(np.mean(draws))),
                "beats_matched_random_95p": bool(model_pnl > float(np.quantile(draws, 0.95))),
                "matching_keys": "side + session_id + spread_bucket + selected_count",
                "sampling_method": "finite_population_matched_normal_draws",
                "saturated_groups": saturated_groups,
                "missing_groups": missing_groups,
                "largest_group_selection_rate": float(largest_group_selection_rate),
            }
        )
    return pd.DataFrame(rows)


def summary_row(row: Any, draws: np.ndarray, reason: str, saturated_groups: int, missing_groups: int, largest_rate: float, model_pnl: float) -> dict[str, Any]:
    return {
        "split": str(row.split),
        "policy_id": str(row.policy_id),
        "horizon": int(row.horizon),
        "trials": TRIALS,
        "model_pnl": float(row.total_pnl),
        "selected_action_count": int(row.selected_action_count),
        "matched_random_mean": float(np.mean(draws)),
        "matched_random_std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "matched_random_90p": float(np.quantile(draws, 0.90)),
        "matched_random_95p": float(np.quantile(draws, 0.95)),
        "matched_random_99p": float(np.quantile(draws, 0.99)),
        "empirical_p_value_random_ge_model": float(np.mean(draws >= float(row.total_pnl))),
        "beats_matched_random_mean": bool(float(row.total_pnl) > float(np.mean(draws))),
        "beats_matched_random_95p": bool(float(row.total_pnl) > float(np.quantile(draws, 0.95))),
        "matching_keys": "side + session_id + spread_bucket + selected_count",
        "sampling_method": f"degenerate_{reason}",
        "saturated_groups": saturated_groups,
        "missing_groups": missing_groups,
        "largest_group_selection_rate": float(largest_rate),
    }


def add_excess_columns(grid: pd.DataFrame, random_summary: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    refs = out.pivot_table(index=["split", "horizon"], columns="policy_id", values=["total_pnl", "avg_pnl_per_selected_action"], aggfunc="first")

    def ref(row: pd.Series, metric: str, policy_id: str) -> float:
        try:
            return float(refs.loc[(row["split"], row["horizon"]), (metric, policy_id)])
        except KeyError:
            return float("nan")

    out["no_quote_pnl"] = out.apply(lambda r: ref(r, "total_pnl", "no_quote"), axis=1)
    out["old_bid_only_pnl"] = out.apply(lambda r: ref(r, "total_pnl", "old_bid_only_spread_ge_1"), axis=1)
    out["bid_all_avg_pnl_per_action"] = out.apply(lambda r: ref(r, "avg_pnl_per_selected_action", "bid_all"), axis=1)
    out["ask_all_avg_pnl_per_action"] = out.apply(lambda r: ref(r, "avg_pnl_per_selected_action", "ask_all"), axis=1)
    out["bid_all_scaled_pnl"] = out["bid_all_avg_pnl_per_action"].astype(float) * out["selected_action_count"].astype(float)
    out["ask_all_scaled_pnl"] = out["ask_all_avg_pnl_per_action"].astype(float) * out["selected_action_count"].astype(float)
    out["excess_vs_no_quote"] = out["total_pnl"].astype(float) - out["no_quote_pnl"].astype(float)
    out["excess_vs_old_bid_only"] = out["total_pnl"].astype(float) - out["old_bid_only_pnl"].astype(float)
    out["excess_vs_bid_all_scaled"] = out["total_pnl"].astype(float) - out["bid_all_scaled_pnl"].astype(float)
    out["excess_vs_ask_all_scaled"] = out["total_pnl"].astype(float) - out["ask_all_scaled_pnl"].astype(float)

    out = out.merge(
        random_summary[
            [
                "split",
                "policy_id",
                "horizon",
                "matched_random_mean",
                "matched_random_95p",
                "matched_random_99p",
                "empirical_p_value_random_ge_model",
                "beats_matched_random_mean",
                "beats_matched_random_95p",
                "sampling_method",
                "largest_group_selection_rate",
            ]
        ],
        on=["split", "policy_id", "horizon"],
        how="left",
    )
    out["excess_vs_matched_random_mean"] = out["total_pnl"].astype(float) - out["matched_random_mean"].astype(float)
    out["excess_vs_matched_random_95p"] = out["total_pnl"].astype(float) - out["matched_random_95p"].astype(float)
    out["excluded_select_all_policy"] = out["policy_id"].astype(str).isin(EXCLUDED_SELECT_ALL_POLICIES)
    out["passes_selection_rate"] = out["selection_rate"].astype(float) < 0.95
    out["passes_min_action_count"] = out["selected_action_count"].astype(int) >= MIN_ACTION_COUNT
    out["passes_positive_pnl"] = out["total_pnl"].astype(float) > 0.0
    out["passes_drawdown"] = out["max_drawdown"].astype(float) <= MAX_DRAWDOWN
    out["selection_split_only"] = out["split"].astype(str) == STRICT_SELECTION_SPLIT
    out["eligible_for_final_selection"] = (
        out["selection_split_only"]
        & ~out["excluded_select_all_policy"]
        & out["passes_selection_rate"]
        & out["passes_min_action_count"]
        & out["passes_positive_pnl"]
        & out["passes_drawdown"]
        & out["beats_matched_random_mean"].fillna(False)
    )
    out["strong_random_pass_95p"] = out["eligible_for_final_selection"] & out["beats_matched_random_95p"].fillna(False)
    return out


def fee_sensitivity(strict_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in strict_grid.itertuples(index=False):
        selected_count = int(row.selected_action_count)
        base = float(row.total_pnl)
        breakeven = 1.0 + base / (MAKER_FEE * selected_count) if selected_count and base > 0 else np.nan
        for mult in FEE_MULTIPLIERS:
            adjusted = base - (float(mult) - 1.0) * MAKER_FEE * selected_count
            rows.append(
                {
                    "split": str(row.split),
                    "policy_id": str(row.policy_id),
                    "horizon": int(row.horizon),
                    "fee_multiplier": float(mult),
                    "base_total_pnl_fee_1x": base,
                    "adjusted_total_pnl": float(adjusted),
                    "selected_action_count": selected_count,
                    "break_even_fee_multiplier": float(breakeven) if np.isfinite(breakeven) else np.nan,
                    "positive_after_fee": bool(adjusted > 0.0),
                    "eligible_for_final_selection_at_1x": bool(row.eligible_for_final_selection),
                }
            )
    return pd.DataFrame(rows)


def select_strict_candidates(strict_grid: pd.DataFrame) -> dict[str, Any]:
    eligible = strict_grid[strict_grid["eligible_for_final_selection"]].copy()
    eligible_count = int(len(eligible))
    if eligible.empty:
        selected: list[dict[str, Any]] = []
    else:
        eligible["selection_score"] = (
            eligible["excess_vs_matched_random_mean"].astype(float)
            + np.where(eligible["strong_random_pass_95p"].astype(bool), 1_000_000.0, 0.0)
            - 0.10 * eligible["max_drawdown"].astype(float)
        )
        eligible = eligible.sort_values(
            ["strong_random_pass_95p", "selection_score", "excess_vs_matched_random_mean", "complexity_rank"],
            ascending=[False, False, False, True],
        )
        selected = eligible.head(3).replace({np.nan: None}).to_dict(orient="records")
    payload = {
        "selection_split": STRICT_SELECTION_SPLIT,
        "test_used_for_selection": False,
        "new4_used_for_selection": False,
        "rules": {
            "excluded_select_all_policies": sorted(EXCLUDED_SELECT_ALL_POLICIES),
            "max_selection_rate": 0.95,
            "min_action_count": MIN_ACTION_COUNT,
            "max_drawdown": MAX_DRAWDOWN,
            "must_beat_matched_random_mean": True,
            "prefer_beat_matched_random_95p": True,
        },
        "eligible_count": eligible_count,
        "selected_candidate_count": len(selected),
        "candidate_count": len(selected),
        "selection_note": "candidate_count is the number of top validation candidates forwarded to Phase 02C, not the total number of rows passing the strict hurdle.",
        "candidates": selected,
    }
    write_json(RESULTS / "strict_selected_candidates.json", payload)
    return payload


def write_report(strict_grid: pd.DataFrame, random_summary: pd.DataFrame, selected_payload: dict[str, Any]) -> None:
    val = strict_grid[strict_grid["split"].astype(str) == STRICT_SELECTION_SPLIT].copy()
    top_diag = val.sort_values("total_pnl", ascending=False).head(12)
    eligible = val[val["eligible_for_final_selection"]].copy()
    lines = [
        "# Strict Baseline and Excess-PnL Report",
        "",
        "Phase 02B patches Phase 02 by banning select-all policies from final selection and comparing every candidate against stronger excess-PnL and matched-random hurdles.",
        "",
        "## Selection Rules",
        "",
        f"- Excluded from final selection: `{', '.join(sorted(EXCLUDED_SELECT_ALL_POLICIES))}`.",
        "- A selectable policy must have `selection_rate < 0.95`.",
        "- Final selection uses validation only.",
        "- A selectable policy must beat the matched random mean; beating matched random 95p is preferred.",
        "- Matched random keys: side, session_id, spread_bucket, and selected count.",
        "- Random distribution implementation: finite-population matched normal draws with 1000 seeded draws per policy/split/horizon. This is a statistical proxy hurdle, not true exchange replay.",
        "",
        "## Validation Diagnostic Top Rows",
        "",
        top_diag[
            [
                "policy_id",
                "horizon",
                "selection_rate",
                "total_pnl",
                "excess_vs_old_bid_only",
                "excess_vs_bid_all_scaled",
                "excess_vs_ask_all_scaled",
                "excess_vs_matched_random_mean",
                "excess_vs_matched_random_95p",
                "eligible_for_final_selection",
                "strong_random_pass_95p",
            ]
        ].to_markdown(index=False),
        "",
        "## Strict Candidate Result",
        "",
    ]
    if selected_payload["eligible_count"] == 0:
        lines.append("No policy passed the strict Phase 02B selection hurdles. Phase 02B failed to find a non-trivial non-ML rule under the current short-horizon markout setup.")
    else:
        lines.append(
            f"`{selected_payload['eligible_count']}` validation rows passed the strict Phase 02B selection hurdles. "
            f"The top `{selected_payload['selected_candidate_count']}` candidates were forwarded to Phase 02C stress testing."
        )
        lines.append("")
        lines.append(pd.DataFrame(selected_payload["candidates"]).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Random Hurdle Summary",
            "",
            random_summary[random_summary["split"].astype(str).eq(STRICT_SELECTION_SPLIT)]
            .sort_values(["beats_matched_random_95p", "beats_matched_random_mean", "model_pnl"], ascending=[False, False, False])
            .head(20)
            .to_markdown(index=False),
            "",
            "## Fee Sensitivity",
            "",
            "Fee sensitivity is saved in `results/fee_sensitivity.csv`. It adjusts Phase 01 labels by subtracting `(fee_multiplier - 1) * maker_fee * selected_action_count` from total PnL.",
            "",
            "## Files",
            "",
            "- `results/strict_policy_grid.csv`",
            "- `results/random_summary_strict.csv`",
            "- `results/strict_selected_candidates.json`",
            "- `results/fee_sensitivity.csv`",
        ]
    )
    write_markdown(ROOT / "STRICT_BASELINE_EXCESS_PNL_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    grid = pd.read_csv(PHASE02_RESULTS / "baseline_grid_all_splits.csv")
    policies = policy_catalog()
    splits = grid["split"].astype(str).drop_duplicates().tolist()
    pop, selected = population_and_selection_stats(splits, policies)
    pop.to_csv(RESULTS / "matched_random_population_stats.csv", index=False)
    selected.to_csv(RESULTS / "matched_random_selected_counts.csv", index=False)
    random_summary = matched_random_summary(grid, pop, selected)
    random_summary.to_csv(RESULTS / "random_summary_strict.csv", index=False)
    strict_grid = add_excess_columns(grid, random_summary)
    strict_grid.to_csv(RESULTS / "strict_policy_grid.csv", index=False)
    fees = fee_sensitivity(strict_grid)
    fees.to_csv(RESULTS / "fee_sensitivity.csv", index=False)
    selected_payload = select_strict_candidates(strict_grid)
    write_report(strict_grid, random_summary, selected_payload)
    print("Phase 02B strict baseline complete.")
    print(f"Strict eligible rows: {selected_payload['eligible_count']}")
    print(f"Forwarded stress candidates: {selected_payload['selected_candidate_count']}")
    print(f"Wrote {RESULTS / 'strict_policy_grid.csv'}")
    print(f"Wrote {ROOT / 'STRICT_BASELINE_EXCESS_PNL_REPORT.md'}")


if __name__ == "__main__":
    main()
