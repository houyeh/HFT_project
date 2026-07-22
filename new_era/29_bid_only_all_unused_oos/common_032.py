from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "32_bid_only_all_unused_oos"
RESULTS = ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V27_DIR = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"

for path in [PROJECT_ROOT, V26_DIR, V27_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import MAKER_FEE, POINT_VALUE, apply_non_ml_policy, load_selected_non_ml_policy, stable_u  # noqa: E402
from new_era.models_playaround_15.data_loading import SESSION_NS_SPACING, parse_new_tw_timestamp  # noqa: E402
from new_era018_helpers import infer_tick_size_simple  # noqa: E402
from queue_stress_core import build_action_table  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402


DATA_DIR_NAMES = ["NEW", "NEW1", "NEW2", "NEW3", "NEW4"]
HORIZON_TICKS = 10


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def training_selection_sessions() -> pd.DataFrame:
    cols = ["dataset", "symbol", "session_id", "split"]
    labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet", columns=cols)
    rows = (
        labels.groupby(["dataset", "symbol", "split", "session_id"], observed=False)
        .size()
        .reset_index(name="label_rows")
        .sort_values(["split", "session_id"])
    )
    rows["used_for_bid_only_policy"] = rows["split"].map({"train": "training_context", "val": "validation_selection", "test": "not_training_or_selection"}).fillna("unknown")
    return rows


def excluded_session_map() -> dict[str, str]:
    provenance = training_selection_sessions()
    out: dict[str, str] = {}
    for row in provenance.itertuples(index=False):
        split = str(row.split)
        sid = str(row.session_id)
        if split == "train":
            out[sid] = "excluded_train_split_used_by_016_026_context"
        elif split == "val":
            out[sid] = "excluded_validation_selection_for_bid_only_spread_ge_1"
    return out


def csv_session_inventory(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir))):
        session_id = str(csv_path.parent.relative_to(data_dir)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem
        rows.append(
            {
                "data_dir": data_dir.name,
                "session_id": session_id,
                "file": str(csv_path.relative_to(data_dir)),
                "bytes": int(csv_path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)


def detect_symbols(data_dir: Path, excluded: dict[str, str], *, max_rows_per_file: int = 20_000) -> pd.DataFrame:
    counts: dict[str, int] = {}
    sessions: dict[str, set[str]] = {}
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir))):
        session_id = str(csv_path.parent.relative_to(data_dir)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem
        if session_id in excluded:
            continue
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            scanned = 0
            for row in reader:
                scanned += 1
                if len(row) < 34:
                    if scanned >= max_rows_per_file:
                        break
                    continue
                try:
                    bid = float(row[21])
                    ask = float(row[22])
                except ValueError:
                    if scanned >= max_rows_per_file:
                        break
                    continue
                if bid <= 0 or ask <= 0 or ask < bid:
                    if scanned >= max_rows_per_file:
                        break
                    continue
                symbol = row[3].strip()
                counts[symbol] = counts.get(symbol, 0) + 1
                sessions.setdefault(symbol, set()).add(session_id)
                if scanned >= max_rows_per_file:
                    break
    return pd.DataFrame(
        [
            {"data_dir": data_dir.name, "symbol": symbol, "valid_l1_rows": count, "session_count": len(sessions.get(symbol, set()))}
            for symbol, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ]
    )


def load_filtered_lob(data_dir: Path, symbol: str, excluded: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    csv_paths = []
    for csv_path in sorted(data_dir.rglob("*.csv"), key=lambda p: str(p.relative_to(data_dir))):
        session_id = str(csv_path.parent.relative_to(data_dir)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem
        if session_id in excluded:
            continue
        csv_paths.append((csv_path, session_id))

    session_order_map: dict[str, int] = {}
    for csv_path, session_id in csv_paths:
        session_order = session_order_map.setdefault(session_id, len(session_order_map))
        session_offset = session_order * SESSION_NS_SPACING
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 34 or row[3].strip() != symbol:
                    continue
                try:
                    timestamp = session_offset + parse_new_tw_timestamp(row[0].strip())
                except ValueError:
                    continue
                try:
                    bid_px_1 = float(row[21])
                    ask_px_1 = float(row[22])
                    if bid_px_1 <= 0 or ask_px_1 <= 0:
                        continue
                    records.append(
                        {
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "bid_px_1": bid_px_1,
                            "bid_vol_1": int(float(row[20])),
                            "bid_px_2": float(row[19]),
                            "bid_vol_2": int(float(row[18])),
                            "bid_px_3": float(row[17]),
                            "bid_vol_3": int(float(row[16])),
                            "bid_px_4": float(row[15]),
                            "bid_vol_4": int(float(row[14])),
                            "bid_px_5": float(row[13]),
                            "bid_vol_5": int(float(row[12])),
                            "ask_px_1": ask_px_1,
                            "ask_vol_1": int(float(row[23])),
                            "ask_px_2": float(row[24]),
                            "ask_vol_2": int(float(row[25])),
                            "ask_px_3": float(row[26]),
                            "ask_vol_3": int(float(row[27])),
                            "ask_px_4": float(row[28]),
                            "ask_vol_4": int(float(row[29])),
                            "ask_px_5": float(row[30]),
                            "ask_vol_5": int(float(row[31])),
                        }
                    )
                except ValueError:
                    continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def build_frame_for_symbol(data_dir: Path, symbol: str, excluded: dict[str, str]) -> tuple[pd.DataFrame, float]:
    frame = load_filtered_lob(data_dir, symbol, excluded)
    if frame.empty:
        return pd.DataFrame(), 1.0
    tick_size = infer_tick_size_simple(frame)
    frame["data_dir"] = data_dir.name
    frame["dataset"] = data_dir.name
    frame["symbol"] = symbol
    frame["split"] = "unused_oos"
    frame["local_row_id"] = frame.groupby("session_id", sort=False).cumcount().astype(np.int64)
    frame["tick_size"] = float(tick_size)
    frame["mid_price"] = (frame["bid_px_1"].astype(float) + frame["ask_px_1"].astype(float)) / 2.0
    frame["spread_ticks"] = (frame["ask_px_1"].astype(float) - frame["bid_px_1"].astype(float)) / float(tick_size)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("session_id", sort=False):
        g = group.copy()
        future_mid = g["mid_price"].shift(-HORIZON_TICKS)
        g["future_mid_return_H_ticks"] = (future_mid - g["mid_price"]) / float(tick_size)
        parts.append(g)
    frame = pd.concat(parts, ignore_index=True)
    return add_market_buckets(frame), float(tick_size)


def actions_from_frame(frame: pd.DataFrame, *, salt_prefix: str) -> pd.DataFrame:
    actions = build_action_table(frame)
    if actions.empty:
        return actions
    actions = actions[np.isfinite(actions["future_mid_return_H_ticks"].astype(float))].reset_index(drop=True)
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    actions["event_id"] = frame.loc[idx, "event_id"].to_numpy(dtype=np.int64)
    actions["data_dir"] = frame.loc[idx, "data_dir"].astype(str).to_numpy()
    actions["symbol"] = frame.loc[idx, "symbol"].astype(str).to_numpy()
    actions["spread_ticks"] = frame.loc[idx, "spread_ticks"].to_numpy(dtype=float)
    actions["spread_bucket"] = frame.loc[idx, "spread_bucket"].astype(str).to_numpy()
    actions["local_row_id"] = frame.loc[idx, "local_row_id"].to_numpy(dtype=np.int64)
    actions["side_toxic"] = np.where(
        side == "bid",
        actions["future_mid_return_H_ticks"].astype(float).to_numpy() < 0.0,
        actions["future_mid_return_H_ticks"].astype(float).to_numpy() > 0.0,
    )
    final_mid = frame.groupby("session_id", sort=True)["mid_price"].last().astype(float).to_dict()
    actions["final_mid"] = actions["session_id"].map(final_mid).astype(float)
    actions["neutral_u"] = [stable_u(eid, s, f"{salt_prefix}_neutral") for eid, s in zip(actions["event_id"], side)]
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(
        side == "bid",
        actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(),
        actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy(),
    )
    adverse = np.maximum(0.0, -side_markout)
    eligible = actions["neutral_u"].astype(float).to_numpy() <= 0.5
    value = terminal_ticks * POINT_VALUE - MAKER_FEE - 0.5 * adverse * POINT_VALUE
    actions["quote_value_moderate"] = np.where(eligible, value, 0.0)
    actions["oracle_positive_quote"] = actions["quote_value_moderate"].astype(float) > 0.0
    return actions


def evaluate_actions(actions: pd.DataFrame, policy: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = apply_non_ml_policy(actions, policy)
    values = actions["quote_value_moderate"].astype(float).to_numpy()
    pos = values > 0
    neg = values < 0
    summary = {
        "policy_id": policy["policy_id"],
        "action_count": int(len(actions)),
        "selected_quote_count": int(selected.sum()),
        "selection_rate": float(selected.mean()) if len(actions) else 0.0,
        "selected_quote_pnl": float(values[selected].sum()),
        "avg_selected_quote_pnl": float(values[selected].mean()) if selected.any() else np.nan,
        "oracle_positive_count": int(pos.sum()),
        "oracle_positive_pnl": float(values[pos].sum()),
        "oracle_recall": float((selected & pos).sum() / pos.sum()) if pos.sum() else np.nan,
        "oracle_precision": float((selected & pos).sum() / selected.sum()) if selected.sum() else np.nan,
        "selected_negative_loss": float(-values[selected & neg].sum()),
        "avoided_negative_pnl": float(-values[(~selected) & neg].sum()),
        "missed_oracle_positive_pnl": float(values[(~selected) & pos].sum()),
    }
    detail = actions.copy()
    detail["selected_by_bid_only_spread_ge_1"] = selected
    return summary, detail


def evaluate_data_dir(data_dir: Path, excluded: dict[str, str]) -> tuple[list[dict[str, Any]], list[pd.DataFrame], pd.DataFrame]:
    policy = load_selected_non_ml_policy()
    symbol_table = detect_symbols(data_dir, excluded)
    rows: list[dict[str, Any]] = []
    detail_parts: list[pd.DataFrame] = []
    if symbol_table.empty:
        return rows, detail_parts, symbol_table
    # Evaluate every detected symbol in unused sessions so "all unused DATA" is literal.
    symbols = symbol_table.sort_values("valid_l1_rows", ascending=False)["symbol"].astype(str).tolist()
    for symbol in symbols:
        frame, tick_size = build_frame_for_symbol(data_dir, symbol, excluded)
        if frame.empty:
            rows.append(
                {
                    "data_dir": data_dir.name,
                    "symbol": symbol,
                    "tick_size": tick_size,
                    "included_session_count": 0,
                    "status": "no_unused_sessions_after_exclusion",
                    "policy_id": policy["policy_id"],
                    "action_count": 0,
                    "selected_quote_count": 0,
                    "selected_quote_pnl": 0.0,
                }
            )
            continue
        salt_prefix = "NEW4_external_oos" if data_dir.name == "NEW4" else f"{data_dir.name}_{symbol}_external_oos"
        actions = actions_from_frame(frame, salt_prefix=salt_prefix)
        if actions.empty:
            rows.append(
                {
                    "data_dir": data_dir.name,
                    "symbol": symbol,
                    "tick_size": tick_size,
                    "included_session_count": int(frame["session_id"].nunique()),
                    "status": "no_candidate_actions",
                    "policy_id": policy["policy_id"],
                    "action_count": 0,
                    "selected_quote_count": 0,
                    "selected_quote_pnl": 0.0,
                }
            )
            continue
        summary, detail = evaluate_actions(actions, policy)
        summary.update(
            {
                "data_dir": data_dir.name,
                "symbol": symbol,
                "tick_size": tick_size,
                "included_session_count": int(frame["session_id"].nunique()),
                "included_sessions": ",".join(sorted(frame["session_id"].astype(str).unique())),
                "status": "evaluated_unused_oos",
            }
        )
        rows.append(summary)
        detail_parts.append(detail)
        del frame, actions, detail
        gc.collect()
    return rows, detail_parts, symbol_table
