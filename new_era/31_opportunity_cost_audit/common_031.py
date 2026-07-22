from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V28_DIR = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"

for path in [V26_DIR, V28_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import apply_non_ml_policy, load_selected_non_ml_policy  # noqa: E402
from common_028 import FEATURE_COLUMNS, build_historical_actions_with_features, build_new4_actions_with_features  # noqa: E402


RANDOM_SEED = 20260701
RANDOM_TRIALS = 5000
FULL_ACTION_FILE = DATA / "full_action_universe.parquet"
FULL_ACTION_FEATURE_COLUMNS = FEATURE_COLUMNS + ["side_is_bid"]


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def build_full_action_universe() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        actions = build_historical_actions_with_features(split)
        actions["phase031_split"] = split
        parts.append(actions)
    new4 = build_new4_actions_with_features()
    new4["phase031_split"] = "new4"
    parts.append(new4)
    out = pd.concat(parts, ignore_index=True)
    policy = load_selected_non_ml_policy()
    out["selected_by_baseline"] = apply_non_ml_policy(out, policy)
    out["baseline_policy_id"] = policy["policy_id"]
    out["quote_ev_moderate"] = out["quote_value_moderate"].astype(float)
    out["side_is_bid"] = (out["side"].astype(str) == "bid").astype(float)
    out["positive_ev_moderate"] = (out["quote_ev_moderate"] > 0.0).astype(int)
    out["negative_ev_moderate"] = (out["quote_ev_moderate"] < 0.0).astype(int)
    out["missed_positive_by_baseline"] = (~out["selected_by_baseline"].astype(bool)) & (out["quote_ev_moderate"] > 0.0)
    out["avoided_negative_by_baseline"] = (~out["selected_by_baseline"].astype(bool)) & (out["quote_ev_moderate"] < 0.0)
    out["sample_id"] = [f"{split}_{i:08d}" for split, i in zip(out["phase031_split"].astype(str), out.groupby("phase031_split", sort=False).cumcount())]
    out["event_rate_bucket"] = bucket_by_train_val(out, "event_rate_past_50", ["low", "mid", "high"])
    out["volatility_bucket"] = bucket_by_train_val(out, "volatility_past_50", ["low", "mid", "high"])
    return out.reset_index(drop=True)


def bucket_by_train_val(df: pd.DataFrame, col: str, labels: list[str]) -> pd.Series:
    source = df[df["phase031_split"].isin(["train", "val"])][col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if source.nunique() < len(labels):
        return pd.Series(np.repeat(labels[0], len(df)), index=df.index, dtype=object)
    qs = np.unique(np.quantile(source.to_numpy(), np.linspace(0.0, 1.0, len(labels) + 1)))
    if len(qs) <= 2:
        return pd.Series(np.repeat(labels[0], len(df)), index=df.index, dtype=object)
    qs[0] = -np.inf
    qs[-1] = np.inf
    used_labels = labels[: len(qs) - 1]
    return pd.cut(df[col].astype(float), bins=qs, labels=used_labels, include_lowest=True).astype(str)


def load_full_actions() -> pd.DataFrame:
    return pd.read_parquet(FULL_ACTION_FILE)


def split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["phase031_split"].astype(str) == split].reset_index(drop=True)


def eval_selection(df: pd.DataFrame, selected: np.ndarray, policy_name: str) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    pos = values > 0.0
    neg = values < 0.0
    return {
        "policy": policy_name,
        "sample_count": int(len(df)),
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
        "captured_oracle_positive_pnl": float(values[selected & pos].sum()),
        "missed_oracle_positive_pnl": float(values[(~selected) & pos].sum()),
        "selected_negative_loss": float(-values[selected & neg].sum()),
        "avoided_negative_ev_pnl": float(-values[(~selected) & neg].sum()),
    }


def matched_random_selected_pnls(
    df: pd.DataFrame,
    selected: np.ndarray,
    *,
    group_cols: list[str],
    trials: int = RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
    candidate_mask: np.ndarray | None = None,
    base_pnl: float = 0.0,
) -> np.ndarray:
    work = df.reset_index(drop=True).copy()
    selected = np.asarray(selected, dtype=bool)
    if candidate_mask is None:
        candidate_mask = np.ones(len(work), dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    groups = work.groupby(group_cols, observed=False, sort=True).groups
    rng = np.random.default_rng(seed)
    pnls = np.zeros(int(trials), dtype=float)
    specs: list[tuple[np.ndarray, int]] = []
    for _, idx in groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        candidates = idx_arr[candidate_mask[idx_arr]]
        k = int(selected[idx_arr].sum())
        specs.append((candidates, k))
    for t in range(int(trials)):
        mask = np.zeros(len(work), dtype=bool)
        for candidates, k in specs:
            if k and len(candidates):
                mask[rng.choice(candidates, size=min(k, len(candidates)), replace=False)] = True
        pnls[t] = float(base_pnl + values[mask].sum())
    return pnls


def model_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[FULL_ACTION_FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y_cls = df["positive_ev_moderate"].astype(int).to_numpy()
    y_reg = df["quote_ev_moderate"].astype(float).to_numpy()
    return x, y_cls, y_reg


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_quote_ev": float(np.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan,
        "mae_quote_ev": float(mean_absolute_error(y, pred)) if len(y) else np.nan,
        "spearman_pred_ev_vs_ev": float(pd.Series(pred).corr(pd.Series(y), method="spearman")) if len(y) else np.nan,
    }


def threshold_curve_for_selector(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    scores = df[score_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(-np.inf).to_numpy()
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    finite = scores[np.isfinite(scores)]
    thresholds = np.unique(np.quantile(finite, np.linspace(0.50, 0.995, 80))) if len(finite) else np.asarray([0.0])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = scores >= float(threshold)
        row = eval_selection(df, selected, "candidate")
        row.update({"threshold": float(threshold), "selected_quote_pnl": float(values[selected].sum())})
        rows.append(row)
    return pd.DataFrame(rows)
