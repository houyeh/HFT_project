from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"
RESULTS = ROOT / "results"
DATA_NEW4 = PROJECT_ROOT / "data" / "NEW4"
DATA_HIST = PROJECT_ROOT / "data" / "NEW"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"

for path in [PROJECT_ROOT, V26_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import (  # noqa: E402
    MAKER_FEE,
    POINT_VALUE,
    apply_non_ml_policy,
    evaluate_non_ml,
    load_selected_non_ml_policy,
    stable_u,
    val_test_actions,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from new_era018_helpers import detect_symbol_from_raw_dir, infer_tick_size_simple  # noqa: E402
from queue_stress_core import build_action_table  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402


STRESS_SPECS: dict[str, dict[str, float]] = {
    "base": {"neutral_fill_haircut": 1.0, "toxic_multiplier": 1.0},
    "mild": {"neutral_fill_haircut": 0.75, "toxic_multiplier": 1.25},
    "moderate": {"neutral_fill_haircut": 0.5, "toxic_multiplier": 1.5},
}


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
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_historical_label_columns(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet", columns=columns)


def first_csv_header(data_dir: Path) -> tuple[Path | None, list[str]]:
    paths = sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir)))
    if not paths:
        return None, []
    with paths[0].open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return paths[0], header


def raw_csv_inventory(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir))):
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                sample_rows = 0
                valid_l1 = 0
                symbols: dict[str, int] = {}
                for row in reader:
                    sample_rows += 1
                    if len(row) > 3:
                        symbols[row[3].strip()] = symbols.get(row[3].strip(), 0) + 1
                    if len(row) > 22:
                        try:
                            bid = float(row[21])
                            ask = float(row[22])
                            valid_l1 += int(bid > 0 and ask > 0 and ask >= bid)
                        except ValueError:
                            pass
                    if sample_rows >= 20000:
                        break
        except OSError:
            continue
        rows.append(
            {
                "dataset_dir": str(data_dir.relative_to(PROJECT_ROOT)),
                "file": str(csv_path.relative_to(data_dir)),
                "column_count": len(header),
                "sample_rows_checked": sample_rows,
                "valid_l1_rows_in_sample": valid_l1,
                "dominant_symbol_in_sample": max(symbols.items(), key=lambda kv: kv[1])[0] if symbols else "",
            }
        )
    return pd.DataFrame(rows)


def build_new4_frame(horizon_ticks: int = 10) -> tuple[pd.DataFrame, str, float]:
    symbol = detect_symbol_from_raw_dir(DATA_NEW4, preferred="TXFA6")
    lob_frames, _ = load_new_tw_data(DATA_NEW4, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"No LOB frame loaded for {symbol} under {DATA_NEW4}")
    frame = lob_frames[symbol].sort_values(["session_id", "timestamp"]).reset_index(drop=True).copy()
    tick_size = infer_tick_size_simple(frame)
    frame["dataset"] = "NEW4"
    frame["symbol"] = symbol
    frame["split"] = "external_oos_new4"
    frame["local_row_id"] = frame.groupby("session_id", sort=False).cumcount().astype(np.int64)
    frame["tick_size"] = float(tick_size)
    frame["mid_price"] = (frame["bid_px_1"].astype(float) + frame["ask_px_1"].astype(float)) / 2.0
    frame["spread_ticks"] = (frame["ask_px_1"].astype(float) - frame["bid_px_1"].astype(float)) / float(tick_size)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)

    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("session_id", sort=False):
        g = group.copy()
        future_mid = g["mid_price"].shift(-int(horizon_ticks))
        g["future_mid_return_H_ticks"] = (future_mid - g["mid_price"]) / float(tick_size)
        parts.append(g)
    frame = pd.concat(parts, ignore_index=True)
    return add_market_buckets(frame), symbol, float(tick_size)


def build_new4_actions(frame: pd.DataFrame) -> pd.DataFrame:
    actions = build_action_table(frame)
    if actions.empty:
        return actions
    actions = actions[np.isfinite(actions["future_mid_return_H_ticks"].astype(float))].reset_index(drop=True)
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    actions["event_id"] = frame.loc[idx, "event_id"].to_numpy(dtype=np.int64)
    actions["spread_ticks"] = frame.loc[idx, "spread_ticks"].to_numpy(dtype=float)
    actions["spread_bucket"] = frame.loc[idx, "spread_bucket"].astype(str).to_numpy()
    actions["coarse_spread_bucket"] = frame.loc[idx, "coarse_spread_bucket"].astype(str).to_numpy()
    actions["side_toxic"] = np.where(
        side == "bid",
        actions["future_mid_return_H_ticks"].astype(float).to_numpy() < 0.0,
        actions["future_mid_return_H_ticks"].astype(float).to_numpy() > 0.0,
    )
    final_mid = frame.groupby("session_id", sort=True)["mid_price"].last().astype(float).to_dict()
    actions["final_mid"] = actions["session_id"].map(final_mid).astype(float)
    actions["neutral_u"] = [stable_u(eid, s, "NEW4_external_oos_neutral") for eid, s in zip(actions["event_id"], side)]
    actions["good_u"] = [stable_u(eid, s, "NEW4_external_oos_good") for eid, s in zip(actions["event_id"], side)]
    for stress_name, spec in STRESS_SPECS.items():
        actions[f"quote_value_{stress_name}"] = quote_value_for_stress(
            actions,
            neutral_fill_haircut=spec["neutral_fill_haircut"],
            toxic_multiplier=spec["toxic_multiplier"],
        )
    actions["quote_value_moderate"] = actions["quote_value_moderate"].astype(float)
    actions["oracle_positive_quote"] = actions["quote_value_moderate"] > 0.0
    return actions


def quote_value_for_stress(actions: pd.DataFrame, *, neutral_fill_haircut: float, toxic_multiplier: float) -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(
        side == "bid",
        actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(),
        actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy(),
    )
    adverse = np.maximum(0.0, -side_markout)
    eligible = actions["neutral_u"].astype(float).to_numpy() <= float(neutral_fill_haircut)
    value = terminal_ticks * POINT_VALUE - MAKER_FEE - max(float(toxic_multiplier) - 1.0, 0.0) * adverse * POINT_VALUE
    return np.where(eligible, value, 0.0).astype(float)


def load_actions_for_split(split: str) -> pd.DataFrame:
    if split == "new4":
        frame, _, _ = build_new4_frame()
        return build_new4_actions(frame)
    actions = val_test_actions(split).copy()
    if "quote_value_base" not in actions.columns:
        for stress_name, spec in STRESS_SPECS.items():
            actions[f"quote_value_{stress_name}"] = quote_value_for_stress(
                actions,
                neutral_fill_haircut=spec["neutral_fill_haircut"],
                toxic_multiplier=spec["toxic_multiplier"],
            )
    actions["quote_value_moderate"] = actions["quote_value_moderate"].astype(float)
    return actions


def evaluate_policy_on_stress(actions: pd.DataFrame, policy: dict[str, Any], split: str, stress_name: str) -> dict[str, Any]:
    selected = apply_non_ml_policy(actions, policy)
    work = actions.copy()
    work["quote_value_moderate"] = work[f"quote_value_{stress_name}"].astype(float)
    row = evaluate_non_ml(work, policy, split=split)
    row["stress_name"] = stress_name
    row["quote_value_column"] = f"quote_value_{stress_name}"
    return row


def candidate_spread_policies() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for thr in [1.0, 2.0, 3.0, 5.0]:
        policies.append({"policy_id": f"spread_only_ge_{thr:g}", "policy_type": "spread_only", "min_spread_ticks": thr})
        policies.append({"policy_id": f"bid_only_spread_ge_{thr:g}", "policy_type": "bid_only_spread", "min_spread_ticks": thr})
    for bid_thr in [1.0, 2.0, 3.0]:
        for ask_thr in [1.0, 2.0, 3.0, 5.0]:
            policies.append(
                {
                    "policy_id": f"side_specific_spread_bid{bid_thr:g}_ask{ask_thr:g}",
                    "policy_type": "side_specific_spread_threshold",
                    "bid_min_spread_ticks": bid_thr,
                    "ask_min_spread_ticks": ask_thr,
                }
            )
    return policies


def matched_random_quote_pnls(
    actions: pd.DataFrame,
    selected: np.ndarray,
    *,
    trials: int = 5000,
    seed: int = 20260525,
    group_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    group_cols = group_cols or ["side", "session_id", "spread_bucket"]
    work = actions.reset_index(drop=True).copy()
    selected = np.asarray(selected, dtype=bool)
    values = work["quote_value_moderate"].astype(float).to_numpy()
    groups = work.groupby(group_cols, observed=False, sort=True).groups
    rng = np.random.default_rng(seed)
    pnls = np.zeros(int(trials), dtype=float)
    audit_rows: list[dict[str, Any]] = []

    group_specs: list[tuple[np.ndarray, int, tuple[Any, ...]]] = []
    for key, idx in groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        k = int(selected[idx_arr].sum())
        if not isinstance(key, tuple):
            key = (key,)
        group_specs.append((idx_arr, k, key))
        audit_rows.append(
            {
                **dict(zip(group_cols, key)),
                "candidate_pool_size": int(len(idx_arr)),
                "required_quote_count": k,
                "selection_rate": float(k / len(idx_arr)) if len(idx_arr) else 0.0,
            }
        )

    for t in range(int(trials)):
        mask = np.zeros(len(work), dtype=bool)
        for idx_arr, k, _ in group_specs:
            if k:
                mask[rng.choice(idx_arr, size=min(k, len(idx_arr)), replace=False)] = True
        pnls[t] = float(values[mask].sum())
    return pnls, pd.DataFrame(audit_rows)


def random_summary_row(
    split: str,
    actions: pd.DataFrame,
    selected: np.ndarray,
    random_pnls: np.ndarray,
    *,
    match_keys: str,
    random_match_level: str,
) -> dict[str, Any]:
    values = actions["quote_value_moderate"].astype(float).to_numpy()
    model_pnl = float(values[np.asarray(selected, dtype=bool)].sum())
    return {
        "split": split,
        "trials": int(len(random_pnls)),
        "model_quote_pnl": model_pnl,
        "model_quote_count": int(np.asarray(selected, dtype=bool).sum()),
        "random_mean": float(np.mean(random_pnls)),
        "random_std": float(np.std(random_pnls, ddof=1)) if len(random_pnls) > 1 else 0.0,
        "random_90p": float(np.quantile(random_pnls, 0.90)),
        "random_95p": float(np.quantile(random_pnls, 0.95)),
        "random_99p": float(np.quantile(random_pnls, 0.99)),
        "empirical_p_value_random_ge_model": float(np.mean(random_pnls >= model_pnl)),
        "match_keys": match_keys,
        "random_match_level": random_match_level,
    }


def stable_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
