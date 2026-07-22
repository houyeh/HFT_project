from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "33_corrected_final_falsification_audits"
RESULTS = ROOT / "results"

V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V31_DIR = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"
V29_DIR = PROJECT_ROOT / "new_era" / "29_bid_only_all_unused_oos"
V32_DIR = PROJECT_ROOT / "new_era" / "32_final_falsification_audits"

for path in [V26_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import apply_non_ml_policy  # noqa: E402


FULL_ACTION_UNIVERSE = V31_DIR / "data" / "full_action_universe.parquet"
UNUSED_OOS_ACTIONS = V29_DIR / "results" / "bid_only_unused_oos_action_details.parquet"
PHASE032_RANDOM_SUMMARY = V32_DIR / "results" / "matched_random_quote_summary.csv"

RANDOM_TRIALS = 5000
RANDOM_SEED = 20260733
DELAY_TICKS = [0, 1, 2, 3, 5, 10]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def _add_eval_columns(df: pd.DataFrame, *, source: str, eval_group: pd.Series | str) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["eval_group"] = eval_group
    out["side"] = out["side"].astype(str)
    out["session_id"] = out["session_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["local_row_id"] = out["local_row_id"].astype(np.int64)
    out["spread_ticks"] = out["spread_ticks"].astype(float)
    out["quote_ev_moderate"] = out["quote_ev_moderate"].astype(float)
    out["selected_baseline"] = ((out["side"] == "bid") & (out["spread_ticks"] >= 1.0)).astype(bool)
    if "spread_bucket" not in out.columns:
        out["spread_bucket"] = pd.cut(
            out["spread_ticks"],
            bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
            labels=["<=1", "(1,2]", "(2,3]", ">3"],
        ).astype(str)
    else:
        out["spread_bucket"] = out["spread_bucket"].astype(str)
    return out


def load_phase031_eval_actions() -> pd.DataFrame:
    df = pd.read_parquet(FULL_ACTION_UNIVERSE)
    df = df[df["phase031_split"].isin(["test", "new4"])].reset_index(drop=True)
    df = _add_eval_columns(
        df,
        source="phase031_full_action_universe",
        eval_group="phase031_" + df["phase031_split"].astype(str),
    )
    keep = [
        "source",
        "eval_group",
        "phase031_split",
        "dataset",
        "symbol",
        "session_id",
        "local_row_id",
        "action_id",
        "side",
        "price",
        "spread_ticks",
        "spread_bucket",
        "future_mid_return_H_ticks",
        "quote_ev_moderate",
        "selected_baseline",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def load_unused_oos_actions() -> pd.DataFrame:
    df = pd.read_parquet(UNUSED_OOS_ACTIONS).reset_index(drop=True)
    df = df.rename(columns={"quote_value_moderate": "quote_ev_moderate"})
    df["dataset"] = df["data_dir"].astype(str)
    df["phase031_split"] = "unused_oos"
    group = "unused_" + df["data_dir"].astype(str) + "_" + df["symbol"].astype(str)
    df = _add_eval_columns(df, source="unused_oos_rebuild_new_to_new4", eval_group=group)
    keep = [
        "source",
        "eval_group",
        "phase031_split",
        "data_dir",
        "dataset",
        "symbol",
        "session_id",
        "local_row_id",
        "action_id",
        "side",
        "price",
        "spread_ticks",
        "spread_bucket",
        "future_mid_return_H_ticks",
        "quote_ev_moderate",
        "selected_baseline",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def load_all_eval_actions() -> pd.DataFrame:
    return pd.concat([load_phase031_eval_actions(), load_unused_oos_actions()], ignore_index=True)


def baseline_policy_dict() -> dict[str, Any]:
    return {"policy_id": "bid_only_spread_ge_1", "policy_type": "bid_only_spread", "min_spread_ticks": 1.0}


def baseline_mask(actions: pd.DataFrame) -> np.ndarray:
    return ((actions["side"].astype(str) == "bid") & (actions["spread_ticks"].astype(float) >= 1.0)).to_numpy(dtype=bool)


def eval_selection(actions: pd.DataFrame, selected: np.ndarray, policy_name: str) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    values = actions["quote_ev_moderate"].astype(float).to_numpy()
    positive = values > 0.0
    negative = values < 0.0
    return {
        "policy": policy_name,
        "action_count": int(len(actions)),
        "selected_quote_count": int(selected.sum()),
        "selection_rate": float(selected.mean()) if len(selected) else 0.0,
        "selected_quote_pnl": float(values[selected].sum()),
        "avg_selected_quote_pnl": float(values[selected].mean()) if selected.any() else np.nan,
        "selected_positive_count": int((selected & positive).sum()),
        "selected_negative_count": int((selected & negative).sum()),
        "selected_negative_loss": float(-values[selected & negative].sum()),
        "missed_positive_pnl": float(values[(~selected) & positive].sum()),
        "avoided_negative_loss": float(-values[(~selected) & negative].sum()),
        "positive_precision": float((selected & positive).sum() / selected.sum()) if selected.sum() else np.nan,
        "positive_recall": float((selected & positive).sum() / positive.sum()) if positive.sum() else np.nan,
    }


def add_lagged_spreads(actions: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    keys = ["eval_group", "symbol", "session_id", "local_row_id"]
    row_level = (
        actions[keys + ["spread_ticks"]]
        .drop_duplicates(keys)
        .sort_values(["eval_group", "symbol", "session_id", "local_row_id"])
        .reset_index(drop=True)
    )
    for lag in lags:
        row_level[f"spread_lag_{lag}"] = row_level.groupby(["eval_group", "symbol", "session_id"], sort=False)["spread_ticks"].shift(lag)
    return actions.merge(row_level[keys + [f"spread_lag_{lag}" for lag in lags]], on=keys, how="left", validate="many_to_one")


def delayed_entry_values(actions: pd.DataFrame, selected: np.ndarray, delay: int) -> tuple[np.ndarray, int]:
    selected = np.asarray(selected, dtype=bool)
    decisions = actions.loc[selected, ["eval_group", "symbol", "session_id", "local_row_id", "side"]].copy()
    decisions["entry_local_row_id"] = decisions["local_row_id"].astype(np.int64) + int(delay)
    decisions["_decision_index"] = decisions.index.to_numpy(dtype=np.int64)
    entry = actions[["eval_group", "symbol", "session_id", "local_row_id", "side", "quote_ev_moderate"]].rename(
        columns={"local_row_id": "entry_local_row_id", "quote_ev_moderate": "entry_quote_ev_moderate"}
    )
    merged = decisions.merge(
        entry,
        on=["eval_group", "symbol", "session_id", "entry_local_row_id", "side"],
        how="left",
        validate="many_to_one",
    )
    delayed_values = np.zeros(len(actions), dtype=float)
    valid = merged["entry_quote_ev_moderate"].notna().to_numpy(dtype=bool)
    delayed_values[merged.loc[valid, "_decision_index"].to_numpy(dtype=np.int64)] = merged.loc[valid, "entry_quote_ev_moderate"].astype(float).to_numpy()
    missing = int((~valid).sum())
    return delayed_values, missing


def matched_random_pnls(
    actions: pd.DataFrame,
    selected: np.ndarray,
    *,
    group_cols: list[str],
    candidate_mask: np.ndarray,
    trials: int = RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    work = actions.reset_index(drop=True).copy()
    selected = np.asarray(selected, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    if not group_cols:
        work["_all_group"] = "all"
        group_cols = ["_all_group"]
    groups = work.groupby(group_cols, observed=False, sort=True).groups
    specs: list[tuple[np.ndarray, int]] = []
    audit_rows: list[dict[str, Any]] = []
    for key, idx in groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        candidates = idx_arr[candidate_mask[idx_arr]]
        k = int(selected[idx_arr].sum())
        if not isinstance(key, tuple):
            key = (key,)
        specs.append((candidates, k))
        audit_rows.append(
            {
                **dict(zip(group_cols, key)),
                "required_baseline_quotes": k,
                "candidate_pool_size": int(len(candidates)),
                "saturated_pool": bool(k > 0 and k == len(candidates)),
                "insufficient_candidate_pool": bool(k > len(candidates)),
            }
        )
    pnls = np.zeros(int(trials), dtype=float)
    for trial in range(int(trials)):
        mask = np.zeros(len(work), dtype=bool)
        for candidates, k in specs:
            if k <= 0 or len(candidates) <= 0:
                continue
            size = min(k, len(candidates))
            mask[rng.choice(candidates, size=size, replace=False)] = True
        pnls[trial] = float(values[mask].sum())
    return pnls, pd.DataFrame(audit_rows)


def summarize_random_distribution(
    *,
    eval_group: str,
    scenario: str,
    model_pnl: float,
    model_count: int,
    pnls: np.ndarray,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    std = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    return {
        "eval_group": eval_group,
        "scenario": scenario,
        "trials": int(len(pnls)),
        "baseline_pnl": float(model_pnl),
        "baseline_selected_count": int(model_count),
        "random_mean": float(np.mean(pnls)),
        "random_std": std,
        "random_90p": float(np.quantile(pnls, 0.90)),
        "random_95p": float(np.quantile(pnls, 0.95)),
        "random_99p": float(np.quantile(pnls, 0.99)),
        "empirical_p_value_random_ge_baseline": float(np.mean(pnls >= model_pnl)),
        "baseline_minus_random_mean": float(model_pnl - np.mean(pnls)),
        "baseline_minus_random_95p": float(model_pnl - np.quantile(pnls, 0.95)),
        "baseline_minus_random_99p": float(model_pnl - np.quantile(pnls, 0.99)),
        "zero_variance": bool(std == 0.0),
        "saturated_bucket_count": int(audit["saturated_pool"].sum()) if "saturated_pool" in audit.columns else 0,
        "insufficient_bucket_count": int(audit["insufficient_candidate_pool"].sum()) if "insufficient_candidate_pool" in audit.columns else 0,
    }


def rule_mask(actions: pd.DataFrame, rule: str) -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    spread = actions["spread_ticks"].astype(float).to_numpy()
    if rule == "no_quote":
        return np.zeros(len(actions), dtype=bool)
    if rule == "all_actions":
        return np.ones(len(actions), dtype=bool)
    if rule == "bid_all":
        return side == "bid"
    if rule == "ask_all":
        return side == "ask"
    for prefix in ["spread_ge_", "bid_spread_ge_", "ask_spread_ge_"]:
        if rule.startswith(prefix):
            thr = float(rule[len(prefix) :])
            if prefix == "spread_ge_":
                return spread >= thr
            if prefix == "bid_spread_ge_":
                return (side == "bid") & (spread >= thr)
            return (side == "ask") & (spread >= thr)
    raise ValueError(rule)


def decision_source_audit() -> dict[str, Any]:
    source = inspect.getsource(apply_non_ml_policy)
    forbidden = ["quote_ev", "quote_value", "future_", "toxic", "oracle", "pnl", "return_H"]
    found = [token for token in forbidden if token in source]
    return {
        "source_function": "baseline_utils.apply_non_ml_policy",
        "forbidden_future_or_outcome_tokens_found": ",".join(found),
        "passes_no_future_token_check": len(found) == 0,
        "source_excerpt": " ".join(source.split()),
    }


def row_price_sequence_hash(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        frame.pivot_table(
            index=["data_dir", "symbol", "session_id", "local_row_id", "spread_ticks"],
            columns="side",
            values="price",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"bid": "bid_price", "ask": "ask_price"})
    )
    for col in ["bid_price", "ask_price"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    def row_hash(row: pd.Series) -> str:
        text = f"{row.symbol}|{row.session_id}|{int(row.local_row_id)}|{row.bid_price:.8f}|{row.ask_price:.8f}|{row.spread_ticks:.8f}"
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    pivot["row_price_hash"] = pivot.apply(row_hash, axis=1)
    session_rows = []
    for keys, group in pivot.sort_values("local_row_id").groupby(["data_dir", "symbol", "session_id"], sort=True):
        joined = "|".join(group["row_price_hash"].astype(str).tolist())
        session_rows.append(
            {
                "data_dir": keys[0],
                "symbol": keys[1],
                "session_id": keys[2],
                "row_count": int(len(group)),
                "price_sequence_hash": hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest(),
            }
        )
    return pd.DataFrame(session_rows)
