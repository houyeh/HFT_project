from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "24_calibration_robust_policy"
RESULTS_DIR = ROOT / "results"

V23_DIR = PROJECT_ROOT / "new_era" / "23_root_cause_stress_profitability"
if str(V23_DIR) not in sys.path:
    sys.path.insert(0, str(V23_DIR))

from root_cause_utils import (  # noqa: E402
    alignment_metrics,
    eval_quote_selection,
    load_context,
    score_specs as v23_score_specs,
    write_markdown,
)


WINDOWS = [25, 50, 100, 200]
QUOTE_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70]
SIDE_QUOTE_FRACTIONS = [0.10, 0.20, 0.30, 0.50]


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def score_specs(include_spread_only: bool = True) -> dict[str, dict[str, Any]]:
    specs = v23_score_specs()
    out = {
        "015": specs["015"],
        "016_v2": specs["016_v2"],
        "ensemble_015_016": specs["ensemble_015_016"],
        "021_ev_score": specs["021_ev_score"],
    }
    if include_spread_only:
        out["spread_only"] = specs["spread_only"] | {"eligible_for_main_selection": False}
    for name, spec in out.items():
        spec.setdefault("eligible_for_main_selection", name != "spread_only")
    return out


def _threshold_quantile(quote_fraction: float, higher_is_better: bool) -> float:
    quote_fraction = float(quote_fraction)
    return 1.0 - quote_fraction if higher_is_better else quote_fraction


def rolling_quantile_select(
    actions: pd.DataFrame,
    *,
    score_col: str,
    window: int,
    quote_fraction: float,
    higher_is_better: bool,
    group_cols: list[str],
    side_quote_fractions: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    work = actions.reset_index(drop=True).copy()
    work["_orig_pos"] = np.arange(len(work), dtype=np.int64)
    work["_score"] = work[score_col].astype(float).replace([np.inf, -np.inf], np.nan)
    selected = np.zeros(len(work), dtype=bool)
    thresholds = np.full(len(work), np.nan, dtype=float)
    min_periods = max(5, int(window // 5))

    for _, group in work.sort_values(["session_id", "row_index", "action_id"]).groupby(group_cols, observed=False, sort=False):
        hist = group["_score"].shift(1).rolling(window=int(window), min_periods=min_periods)
        side = str(group["side"].iloc[0]) if "side" in group_cols and len(group) else ""
        q_frac = float(side_quote_fractions.get(side, quote_fraction)) if side_quote_fractions else float(quote_fraction)
        q = _threshold_quantile(q_frac, higher_is_better)
        thr = hist.quantile(q)
        score = group["_score"]
        sel = (score >= thr) if higher_is_better else (score <= thr)
        pos = group["_orig_pos"].to_numpy(dtype=np.int64)
        selected[pos] = sel.fillna(False).to_numpy(dtype=bool)
        thresholds[pos] = thr.to_numpy(dtype=float)
    return selected, thresholds


def evaluate_candidate(
    actions: pd.DataFrame,
    selected: np.ndarray,
    *,
    policy_id: str,
    family: str,
    score_name: str,
    score_col: str,
    window: int,
    quote_fraction: float | None,
    group_cols: list[str],
    higher_is_better: bool,
    split: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = eval_quote_selection(actions, selected, policy_id)
    selected_bool = np.asarray(selected, dtype=bool)
    side = actions["side"].astype(str)
    session = actions["session_id"].astype(str)
    side_counts = side[selected_bool].value_counts()
    session_pnl = actions.assign(_selected=selected_bool).groupby("session_id", observed=False).apply(
        lambda g: float(g.loc[g["_selected"], "quote_value_moderate"].sum()),
        include_groups=False,
    )
    row = {
        "policy_id": policy_id,
        "family": family,
        "score_name": score_name,
        "score_column": score_col,
        "window": int(window),
        "quote_fraction": np.nan if quote_fraction is None else float(quote_fraction),
        "higher_is_better": bool(higher_is_better),
        "group_cols": "|".join(group_cols),
        "split": split,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "bid_selected_count": int(side_counts.get("bid", 0)),
        "ask_selected_count": int(side_counts.get("ask", 0)),
        "positive_session_count": int((session_pnl > 0).sum()) if len(session_pnl) else 0,
        "session_count": int(session.nunique()),
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def build_grid(
    actions: pd.DataFrame,
    *,
    family: str,
    group_cols: list[str],
    windows: list[int] | None = None,
    quote_fractions: list[float] | None = None,
    include_spread_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    windows = windows or WINDOWS
    quote_fractions = quote_fractions or QUOTE_FRACTIONS
    for score_name, spec in score_specs(include_spread_only=include_spread_only).items():
        col = str(spec["column"])
        higher = bool(spec["higher_is_better"])
        for window in windows:
            for quote_fraction in quote_fractions:
                policy_id = f"{family}__{score_name}__w{window}__quote{quote_fraction:.2f}"
                selected, _ = rolling_quantile_select(
                    actions,
                    score_col=col,
                    window=window,
                    quote_fraction=quote_fraction,
                    higher_is_better=higher,
                    group_cols=group_cols,
                )
                masks[policy_id] = selected
                rows.append(
                    evaluate_candidate(
                        actions,
                        selected,
                        policy_id=policy_id,
                        family=family,
                        score_name=score_name,
                        score_col=col,
                        window=window,
                        quote_fraction=quote_fraction,
                        group_cols=group_cols,
                        higher_is_better=higher,
                        split="validation",
                        extra={"eligible_for_main_selection": bool(spec.get("eligible_for_main_selection", True))},
                    )
                )
    return pd.DataFrame(rows), masks


def build_side_specific_grid(actions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for score_name, spec in score_specs(include_spread_only=True).items():
        col = str(spec["column"])
        higher = bool(spec["higher_is_better"])
        for window in WINDOWS:
            for bid_q in SIDE_QUOTE_FRACTIONS:
                for ask_q in SIDE_QUOTE_FRACTIONS:
                    policy_id = f"side_specific__{score_name}__w{window}__bid{bid_q:.2f}__ask{ask_q:.2f}"
                    selected, _ = rolling_quantile_select(
                        actions,
                        score_col=col,
                        window=window,
                        quote_fraction=0.0,
                        higher_is_better=higher,
                        group_cols=["session_id", "side"],
                        side_quote_fractions={"bid": bid_q, "ask": ask_q},
                    )
                    masks[policy_id] = selected
                    rows.append(
                        evaluate_candidate(
                            actions,
                            selected,
                            policy_id=policy_id,
                            family="side_specific",
                            score_name=score_name,
                            score_col=col,
                            window=window,
                            quote_fraction=None,
                            group_cols=["session_id", "side"],
                            higher_is_better=higher,
                            split="validation",
                            extra={
                                "bid_quote_fraction": bid_q,
                                "ask_quote_fraction": ask_q,
                                "bid_cancel_quantile": 1.0 - bid_q,
                                "ask_cancel_quantile": 1.0 - ask_q,
                                "eligible_for_main_selection": bool(spec.get("eligible_for_main_selection", True)),
                            },
                        )
                    )
    return pd.DataFrame(rows), masks


def candidate_passes(row: pd.Series) -> bool:
    if not bool(row.get("eligible_for_main_selection", True)):
        return False
    if int(row.get("selected_quote_count", 0)) < 20:
        return False
    if float(row.get("selection_rate", 0.0)) < 0.05 or float(row.get("selection_rate", 0.0)) > 0.75:
        return False
    if int(row.get("bid_selected_count", 0)) <= 0 or int(row.get("ask_selected_count", 0)) <= 0:
        return False
    return True


def select_validation_policy(grid: pd.DataFrame) -> pd.Series:
    work = grid.copy()
    work["constraints_pass"] = work.apply(candidate_passes, axis=1)
    eligible = work[work["constraints_pass"]].copy()
    if eligible.empty:
        eligible = work[work["eligible_for_main_selection"].astype(bool)].copy()
    eligible["selection_objective"] = eligible["selected_quote_pnl"] - 0.02 * eligible["false_positive_loss"]
    return eligible.sort_values(
        ["selection_objective", "selected_quote_pnl", "oracle_precision"],
        ascending=[False, False, False],
    ).iloc[0]


def policy_from_row(row: pd.Series) -> dict[str, Any]:
    policy = {
        "policy_id": str(row["policy_id"]),
        "family": str(row["family"]),
        "score_name": str(row["score_name"]),
        "score_column": str(row["score_column"]),
        "window": int(row["window"]),
        "quote_fraction": None if pd.isna(row.get("quote_fraction")) else float(row["quote_fraction"]),
        "higher_is_better": bool(row["higher_is_better"]),
        "group_cols": str(row["group_cols"]).split("|"),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "moderate_stress": "neutral_fill_haircut=0.5,toxic_multiplier=1.5,delay=0",
    }
    if str(row["family"]) == "side_specific":
        policy.update(
            {
                "bid_quote_fraction": float(row["bid_quote_fraction"]),
                "ask_quote_fraction": float(row["ask_quote_fraction"]),
                "bid_cancel_quantile": float(row["bid_cancel_quantile"]),
                "ask_cancel_quantile": float(row["ask_cancel_quantile"]),
            }
        )
    return policy


def apply_policy(actions: pd.DataFrame, policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    side_fracs = None
    quote_fraction = float(policy.get("quote_fraction") or 0.0)
    if policy["family"] == "side_specific":
        side_fracs = {"bid": float(policy["bid_quote_fraction"]), "ask": float(policy["ask_quote_fraction"])}
    return rolling_quantile_select(
        actions,
        score_col=str(policy["score_column"]),
        window=int(policy["window"]),
        quote_fraction=quote_fraction,
        higher_is_better=bool(policy["higher_is_better"]),
        group_cols=list(policy["group_cols"]),
        side_quote_fractions=side_fracs,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


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


def selected_policy_detail(actions: pd.DataFrame, selected: np.ndarray, policy_id: str, split: str) -> pd.DataFrame:
    out = alignment_metrics(actions, selected, policy_id, ["session_id", "side", "spread_bucket"])
    out["split"] = split
    out["selection_split"] = "validation"
    out["test_used_for_selection"] = False
    return out
