from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
RESULTS = ROOT / "results"
DATA_NEW4 = PROJECT_ROOT / "data" / "NEW4"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_utils import (  # noqa: E402
    POINT_VALUE,
    MAKER_FEE,
    apply_non_ml_policy,
    evaluate_non_ml,
    load_selected_non_ml_policy,
    stable_u,
    write_markdown,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from new_era018_helpers import detect_symbol_from_raw_dir, infer_tick_size_simple  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402
from queue_stress_core import build_action_table  # noqa: E402


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


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
    parts = []
    for _, group in frame.groupby("session_id", sort=False):
        g = group.copy()
        future_mid = g["mid_price"].shift(-int(horizon_ticks))
        g["future_mid_return_H_ticks"] = (future_mid - g["mid_price"]) / float(tick_size)
        parts.append(g)
    frame = pd.concat(parts, ignore_index=True)
    return add_market_buckets(frame), symbol, float(tick_size)


def new4_actions(frame: pd.DataFrame) -> pd.DataFrame:
    actions = build_action_table(frame)
    actions = actions[np.isfinite(actions["future_mid_return_H_ticks"].astype(float))].reset_index(drop=True)
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    actions["event_id"] = frame.loc[idx, "event_id"].to_numpy(dtype=np.int64)
    actions["spread_ticks"] = frame.loc[idx, "spread_ticks"].to_numpy(dtype=float)
    actions["spread_bucket"] = frame.loc[idx, "spread_bucket"].astype(str).to_numpy()
    actions["side_toxic"] = np.where(side == "bid", actions["future_mid_return_H_ticks"].astype(float).to_numpy() < 0.0, actions["future_mid_return_H_ticks"].astype(float).to_numpy() > 0.0)
    final_mid = frame.groupby("session_id", sort=True)["mid_price"].last().astype(float).to_dict()
    actions["final_mid"] = actions["session_id"].map(final_mid).astype(float)
    actions["neutral_u"] = [stable_u(eid, s, "NEW4_external_oos_neutral") for eid, s in zip(actions["event_id"], side)]
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


def run() -> None:
    ensure_dirs()
    policy = load_selected_non_ml_policy()
    frame, symbol, tick_size = build_new4_frame(horizon_ticks=10)
    actions = new4_actions(frame)
    selected = apply_non_ml_policy(actions, policy)
    summary = evaluate_non_ml(actions, policy, split="external_oos_new4", extra={"dataset": "NEW4", "symbol": symbol, "tick_size": tick_size})
    pd.DataFrame([summary]).to_csv(RESULTS / "new4_oos_non_ml.csv", index=False)

    detail = actions.assign(selected_quote=selected)
    by_session = (
        detail[detail["selected_quote"].astype(bool)]
        .groupby(["session_id", "side"], observed=False)
        .agg(
            selected_quote_count=("action_id", "count"),
            selected_quote_pnl=("quote_value_moderate", "sum"),
            avg_quote_pnl=("quote_value_moderate", "mean"),
            toxic_rate=("side_toxic", "mean"),
            positive_pnl_rate=("quote_value_moderate", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    by_session.to_csv(RESULTS / "new4_oos_by_session_side.csv", index=False)

    all_actions_summary = (
        actions.groupby(["session_id", "side"], observed=False)
        .agg(
            quote_action_count=("action_id", "count"),
            all_action_pnl=("quote_value_moderate", "sum"),
            all_toxic_rate=("side_toxic", "mean"),
        )
        .reset_index()
    )
    all_actions_summary.to_csv(RESULTS / "new4_action_universe_by_session_side.csv", index=False)
    write_markdown(
        RESULTS / "new4_oos_notes.md",
        [
            "# NEW4 OOS Notes",
            "",
            f"- Source: `{DATA_NEW4}`.",
            f"- Detected symbol: `{symbol}`.",
            f"- Inferred tick size: `{tick_size}`.",
            "- Evaluation role: external OOS only.",
            "- Selection policy is frozen from 026 validation: no NEW4 tuning was performed.",
            "- Raw NEW4 has no true aggressor side or exchange queue position; PnL remains the same simulator-derived moderate-stress quote proxy used in 023-026.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote NEW4 OOS outputs to {RESULTS}")

