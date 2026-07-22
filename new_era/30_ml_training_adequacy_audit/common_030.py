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
ROOT = PROJECT_ROOT / "new_era" / "30_ml_training_adequacy_audit"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
EXPORT = ROOT / "export" / "ml_training_pack"

V28_DIR = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
V28_DATA = V28_DIR / "data"
V28_RESULTS = V28_DIR / "results"

if str(V28_DIR) not in sys.path:
    sys.path.insert(0, str(V28_DIR))

from common_028 import FEATURE_COLUMNS, matched_random_veto_after_pnls  # noqa: E402


MAX_VETO_RATE = 0.05
RANDOM_TRIALS = 5000
RANDOM_SEED = 20260630


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, EXPORT]:
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


def load_028_split(split: str) -> pd.DataFrame:
    if split == "new4":
        return pd.read_parquet(V28_DATA / "new4_baseline_universe_quote_ev.parquet").reset_index(drop=True)
    df = pd.read_parquet(V28_DATA / "baseline_universe_quote_ev.parquet")
    return df[df["split"].astype(str) == split].reset_index(drop=True)


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: load_028_split(split) for split in ["train", "val", "test", "new4"]}


def prepare_export_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["phase030_split"] = split
    out["sample_id"] = [f"{split}_{i:08d}" for i in range(len(out))]
    out["target_row_id"] = out["local_row_id"].astype(int)
    out["quote_ev_moderate"] = out["quote_ev_moderate"].astype(float)
    out["negative_ev_moderate"] = out["negative_ev_moderate"].astype(int)
    out["severe_negative_ev_moderate"] = (out["quote_ev_moderate"] <= -5000.0).astype(int)
    return out


def model_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y_cls = df["negative_ev_moderate"].astype(int).to_numpy()
    y_reg = df["quote_ev_moderate"].astype(float).to_numpy()
    return x, y_cls, y_reg


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan
    mae = float(mean_absolute_error(y, pred)) if len(y) else np.nan
    corr = float(pd.Series(pred).corr(pd.Series(y), method="spearman")) if len(y) else np.nan
    return {"rmse_quote_ev": rmse, "mae_quote_ev": mae, "spearman_pred_ev_vs_ev": corr}


def threshold_curve(df: pd.DataFrame, score_col: str, *, max_veto_rate: float = MAX_VETO_RATE) -> pd.DataFrame:
    work = df.reset_index(drop=True).copy()
    scores = work[score_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(-np.inf).to_numpy()
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    base_pnl = float(values.sum())
    finite = scores[np.isfinite(scores)]
    if len(finite):
        thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))
        no_veto_threshold = float(np.max(finite) + 1e-6)
    else:
        thresholds = np.asarray([], dtype=float)
        no_veto_threshold = 1.000001
    thresholds = np.unique(np.concatenate([[no_veto_threshold], thresholds]))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        veto = scores >= float(threshold)
        veto_rate = float(veto.mean()) if len(veto) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "veto_count": int(veto.sum()),
                "veto_rate": veto_rate,
                "max_veto_rate": max_veto_rate,
                "constraints_pass": bool(veto_rate <= max_veto_rate),
                "baseline_pnl": base_pnl,
                "vetoed_quote_ev": float(values[veto].sum()),
                "after_veto_pnl": base_pnl - float(values[veto].sum()),
                "incremental_vs_baseline": -float(values[veto].sum()),
                "negative_ev_rate_vetoed": float((values[veto] < 0.0).mean()) if veto.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_threshold_on_val(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    curve = threshold_curve(df, score_col, max_veto_rate=MAX_VETO_RATE)
    eligible = curve[curve["constraints_pass"].astype(bool)].copy()
    if eligible.empty:
        eligible = curve.copy()
    selected = eligible.sort_values(["after_veto_pnl", "veto_count"], ascending=[False, True]).iloc[0].to_dict()
    selected["selection_split"] = "validation"
    selected["test_used_for_selection"] = False
    return selected


def evaluate_veto(df: pd.DataFrame, score_col: str, threshold: float) -> dict[str, Any]:
    scores = df[score_col].astype(float).to_numpy()
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    veto = scores >= float(threshold)
    return {
        "sample_count": int(len(df)),
        "baseline_pnl": float(values.sum()),
        "veto_count": int(veto.sum()),
        "veto_rate": float(veto.mean()) if len(veto) else 0.0,
        "vetoed_quote_ev": float(values[veto].sum()),
        "after_veto_pnl": float(values.sum() - values[veto].sum()),
        "incremental_vs_baseline": float(-values[veto].sum()),
        "negative_ev_rate_vetoed": float((values[veto] < 0.0).mean()) if veto.any() else np.nan,
    }


def evaluate_with_random_hurdle(df: pd.DataFrame, score_col: str, threshold: float, *, trials: int = RANDOM_TRIALS) -> dict[str, Any]:
    base = evaluate_veto(df, score_col, threshold)
    veto = df[score_col].astype(float).to_numpy() >= float(threshold)
    random_pnls = matched_random_veto_after_pnls(df, veto, trials=trials, seed=RANDOM_SEED)
    base.update(
        {
            "random_trials": int(trials),
            "random_veto_mean_pnl": float(np.mean(random_pnls)),
            "random_veto_95p_pnl": float(np.quantile(random_pnls, 0.95)),
            "random_veto_99p_pnl": float(np.quantile(random_pnls, 0.99)),
            "beats_random_veto_95p": bool(base["after_veto_pnl"] > float(np.quantile(random_pnls, 0.95))),
            "beats_random_veto_99p": bool(base["after_veto_pnl"] > float(np.quantile(random_pnls, 0.99))),
            "improves_baseline": bool(base["after_veto_pnl"] > base["baseline_pnl"]),
        }
    )
    return base

