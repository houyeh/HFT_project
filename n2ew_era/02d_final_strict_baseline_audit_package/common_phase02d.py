from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
PHASE02B_DIR = PROJECT_ROOT / "n2ew_era" / "02b_strict_baseline_and_excess_pnl"
PHASE02C_DIR = PROJECT_ROOT / "n2ew_era" / "02c_queue_aware_stress"

if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))

ROOT = PROJECT_ROOT / "n2ew_era" / "02d_final_strict_baseline_audit_package"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SELECTED_POLICY_ID = "side_by_queue_imbalance_abs_0p1"
SELECTED_HORIZON = 20
SELECTED_RULE_DEFINITION = "If queue_imbalance > +0.1 quote bid; if queue_imbalance < -0.1 quote ask; otherwise no_quote."
EVAL_SPLITS = ["val", "test", "new4"]
FROZEN_OOS_SPLITS = ["test", "new4"]
MODERATE_FILL_KEEP = 0.5
MODERATE_ADVERSE_MULTIPLIER = 1.5
MODERATE_DELAY = 0


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def selected_strict_rows() -> pd.DataFrame:
    grid = pd.read_csv(PHASE02B_DIR / "results" / "strict_policy_grid.csv")
    return grid[
        (grid["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (grid["horizon"].astype(int) == SELECTED_HORIZON)
        & (grid["split"].astype(str).isin(EVAL_SPLITS))
    ].copy()


def phase02b_random_rows() -> pd.DataFrame:
    random_df = read_csv_if_exists(PHASE02B_DIR / "results" / "random_summary_strict.csv")
    if random_df.empty:
        return random_df
    return random_df[
        (random_df["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (random_df["horizon"].astype(int) == SELECTED_HORIZON)
        & (random_df["split"].astype(str).isin(EVAL_SPLITS))
    ].copy()


def exact_random_rows() -> pd.DataFrame:
    rows = []
    val_path = PHASE02B_DIR / "results" / "random_summary_strict_exact_top_candidates.csv"
    oos_path = RESULTS / "exact_random_oos_summary.csv"
    for path in [val_path, oos_path]:
        df = read_csv_if_exists(path)
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[
        (out["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (out["horizon"].astype(int) == SELECTED_HORIZON)
        & (out["split"].astype(str).isin(EVAL_SPLITS))
    ].copy()


def selected_control_table() -> pd.DataFrame:
    grid = selected_strict_rows()
    random_df = phase02b_random_rows()
    if random_df.empty:
        return grid
    return grid.merge(
        random_df[
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
            ]
        ],
        on=["split", "policy_id", "horizon"],
        how="left",
        suffixes=("", "_phase02b"),
    )


def html_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<p class='muted'>No rows available.</p>"
    work = df.head(max_rows).copy() if max_rows else df.copy()
    headers = "".join(f"<th>{html_escape(col)}</th>" for col in work.columns)
    body_rows = []
    for _, row in work.iterrows():
        tds = "".join(f"<td>{html_escape(format_value(val))}</td>" for val in row)
        body_rows.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        return f"{float(value):.4g}"
    return str(value)
