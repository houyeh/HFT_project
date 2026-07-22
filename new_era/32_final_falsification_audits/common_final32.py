from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "32_final_falsification_audits"
RESULTS = ROOT / "results"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V27_DIR = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"
V31_DIR = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"

for path in [V26_DIR, V27_DIR, V31_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import MAKER_FEE, POINT_VALUE, apply_non_ml_policy, load_selected_non_ml_policy  # noqa: E402
from common_027 import build_new4_frame  # noqa: E402


FULL_ACTION_UNIVERSE = V31_DIR / "data" / "full_action_universe.parquet"
RANDOM_SEED = 20260702
RANDOM_TRIALS = 5000


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_eval_actions() -> pd.DataFrame:
    df = pd.read_parquet(FULL_ACTION_UNIVERSE)
    df = df[df["phase031_split"].isin(["test", "new4"])].reset_index(drop=True).copy()
    df["quote_ev_moderate"] = df["quote_ev_moderate"].astype(float)
    df["selected_by_baseline"] = df["selected_by_baseline"].astype(bool)
    return df


def load_book_lookup(split: str) -> pd.DataFrame:
    if split == "test":
        cols = ["session_id", "local_row_id", "spread_ticks", "bid_px_1", "ask_px_1"]
        frame = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet", columns=cols + ["split"])
        frame = frame[frame["split"].astype(str) == "test"][cols].sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
    elif split == "new4":
        frame, _, _ = build_new4_frame()
        frame = frame[["session_id", "local_row_id", "spread_ticks", "bid_px_1", "ask_px_1"]].sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
    else:
        raise ValueError(split)
    for lag in [0, 1, 2, 3, 5, 10]:
        frame[f"spread_lag_{lag}"] = frame.groupby("session_id", sort=False)["spread_ticks"].shift(lag)
    return frame


def attach_lagged_spreads(actions: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split, group in actions.groupby("phase031_split", sort=False):
        lookup = load_book_lookup(str(split))
        cols = ["session_id", "local_row_id"] + [f"spread_lag_{lag}" for lag in lags]
        merged = group.merge(lookup[cols], on=["session_id", "local_row_id"], how="left", validate="many_to_one")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def baseline_decision(actions: pd.DataFrame, spread_col: str = "spread_ticks") -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    spread = actions[spread_col].astype(float).to_numpy()
    return (side == "bid") & np.isfinite(spread) & (spread >= 1.0)


def eval_selected(actions: pd.DataFrame, selected: np.ndarray, policy_name: str) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    values = actions["quote_ev_moderate"].astype(float).to_numpy()
    pos = values > 0.0
    neg = values < 0.0
    return {
        "policy": policy_name,
        "sample_count": int(len(actions)),
        "selected_count": int(selected.sum()),
        "selection_rate": float(selected.mean()) if len(selected) else 0.0,
        "selected_quote_pnl": float(values[selected].sum()),
        "avg_selected_quote_pnl": float(values[selected].mean()) if selected.any() else np.nan,
        "selected_positive_count": int((selected & pos).sum()),
        "selected_negative_count": int((selected & neg).sum()),
        "oracle_positive_count": int(pos.sum()),
        "oracle_positive_pnl": float(values[pos].sum()),
        "oracle_recall": float((selected & pos).sum() / pos.sum()) if pos.sum() else np.nan,
        "oracle_precision": float((selected & pos).sum() / selected.sum()) if selected.sum() else np.nan,
        "selected_negative_loss": float(-values[selected & neg].sum()),
        "missed_oracle_positive_pnl": float(values[(~selected) & pos].sum()),
    }


def matched_random_pnls(
    actions: pd.DataFrame,
    selected: np.ndarray,
    *,
    group_cols: list[str],
    candidate_mask: np.ndarray | None = None,
    trials: int = RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    work = actions.reset_index(drop=True).copy()
    selected = np.asarray(selected, dtype=bool)
    if candidate_mask is None:
        candidate_mask = np.ones(len(work), dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    groups = work.groupby(group_cols, observed=False, sort=True).groups
    specs: list[tuple[np.ndarray, int, tuple[Any, ...]]] = []
    audit_rows: list[dict[str, Any]] = []
    for key, idx in groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        candidates = idx_arr[candidate_mask[idx_arr]]
        k = int(selected[idx_arr].sum())
        if not isinstance(key, tuple):
            key = (key,)
        specs.append((candidates, k, key))
        audit_rows.append(
            {
                **dict(zip(group_cols, key)),
                "required_selected_count": k,
                "candidate_pool_size": int(len(candidates)),
                "saturated_pool": bool(k == len(candidates) and k > 0),
            }
        )
    pnls = np.zeros(int(trials), dtype=float)
    for t in range(int(trials)):
        mask = np.zeros(len(work), dtype=bool)
        for candidates, k, _ in specs:
            if k and len(candidates):
                mask[rng.choice(candidates, size=min(k, len(candidates)), replace=False)] = True
        pnls[t] = float(values[mask].sum())
    return pnls, pd.DataFrame(audit_rows)


def decision_source_audit() -> dict[str, Any]:
    source = inspect.getsource(apply_non_ml_policy)
    forbidden_tokens = ["quote_ev", "future_", "toxic", "oracle", "pnl", "return_H"]
    return {
        "source_function": "baseline_utils.apply_non_ml_policy",
        "source_contains_forbidden_future_tokens": any(tok in source for tok in forbidden_tokens),
        "forbidden_tokens_found": ",".join(tok for tok in forbidden_tokens if tok in source),
        "source_excerpt": " ".join(source.split()),
    }


def stressed_quote_values(
    actions: pd.DataFrame,
    *,
    fee_multiplier: float,
    winning_fill_keep_rate: float,
    adverse_multiplier: float,
    neutral_fill_haircut: float = 0.5,
) -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(
        side == "bid",
        actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(),
        actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy(),
    )
    adverse = np.maximum(0.0, -side_markout)
    base_value = terminal_ticks * POINT_VALUE - fee_multiplier * MAKER_FEE - max(float(adverse_multiplier) - 1.0, 0.0) * adverse * POINT_VALUE
    neutral_u = actions["neutral_u"].astype(float).to_numpy() if "neutral_u" in actions.columns else np.zeros(len(actions))
    good_u = actions["good_u"].astype(float).to_numpy() if "good_u" in actions.columns else np.zeros(len(actions))
    filled = neutral_u <= float(neutral_fill_haircut)
    winners = base_value > 0.0
    filled &= ~(winners & (good_u > float(winning_fill_keep_rate)))
    return np.where(filled, base_value, 0.0)

