from __future__ import annotations

"""
08 - NEW2 baseline backtest（資料盤點 + baseline-only 回測）
=========================================================

你說的「先 baseline 就好」版本：
- 讀取 `data/NEW2/` 的 raw CSV
- 盤點資料狀態（sessions/rows/基本缺失與異常）
- 只跑 maker-only baseline（不使用模型、不需要 checkpoint）

輸出資料夾（依你的指定）：
`new_era/backtest_outputs/NEW2_result/`

建議用法：
`python new_era/15_models_playaround/08_new2_baseline_backtest.py --symbol TXFB6`
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _ensure_project_root_on_path, build_cfg

_ensure_project_root_on_path()

from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from src.data.tw_replay import infer_tick_size_from_lob  # noqa: E402


def _max_drawdown(pnl_path: np.ndarray) -> float:
    pnl_path = np.asarray(pnl_path, dtype=np.float64)
    if pnl_path.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(pnl_path)
    drawdowns = peaks - pnl_path
    return float(np.max(drawdowns))


def _future_min_max(mid: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    mid = np.asarray(mid, dtype=np.float64)
    if len(mid) <= horizon:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(mid[1:], window_shape=horizon)
    return view.min(axis=1), view.max(axis=1)


def run_baseline_maker_only(
    *,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    tick_size: float,
    point_value: float,
    maker_fee_twd: float,
    taker_fee_twd: float,
    max_inventory: int,
    horizon: int,
    return_paths: bool = False,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray] | None]:
    bid_px = np.asarray(bid_px, dtype=np.float64)
    ask_px = np.asarray(ask_px, dtype=np.float64)
    mid = (ask_px + bid_px) / 2.0

    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0
    quote_sides = 0
    spread_capture_ticks: list[float] = []
    adverse_selection_ticks: list[float] = []

    fut_min, fut_max = _future_min_max(mid, horizon=horizon)

    pnl_path: list[float] = []
    inv_path: list[int] = []

    for i in range(len(mid) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
            inv_path.append(inventory)
            continue
        if ask <= bid:
            pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
            inv_path.append(inventory)
            continue

        quote_bid: float | None = bid if inventory < max_inventory else None
        quote_ask: float | None = ask if inventory > -max_inventory else None
        if quote_bid is not None:
            quote_sides += 1
        if quote_ask is not None:
            quote_sides += 1

        spread_capture_ticks_now = (ask - bid) / (2.0 * tick_size) if tick_size > 0 else 0.0

        if quote_bid is not None and next_ask <= quote_bid:
            gross_cash -= quote_bid * point_value
            fees_paid += maker_fee_twd
            inventory += 1
            maker_trades += 1
            spread_capture_ticks.append(float(spread_capture_ticks_now))
            if i < len(fut_min) and tick_size > 0:
                adverse = max(0.0, (quote_bid - float(fut_min[i])) / tick_size)
                adverse_selection_ticks.append(float(adverse))

        if quote_ask is not None and next_bid >= quote_ask:
            gross_cash += quote_ask * point_value
            fees_paid += maker_fee_twd
            inventory -= 1
            maker_trades += 1
            spread_capture_ticks.append(float(spread_capture_ticks_now))
            if i < len(fut_max) and tick_size > 0:
                adverse = max(0.0, (float(fut_max[i]) - quote_ask) / tick_size)
                adverse_selection_ticks.append(float(adverse))

        pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
        inv_path.append(inventory)

    final_mid = float(mid[-1]) if len(mid) else 0.0
    final_bid = float(bid_px[-1]) if len(bid_px) else final_mid
    final_ask = float(ask_px[-1]) if len(ask_px) else final_mid
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid

    # Force-flatten inventory at the end (conservative taker liquidation).
    flatten_cash = float(gross_cash)
    flatten_fees = float(fees_paid)
    forced_liq_trades = 0
    if inventory > 0:
        forced_liq_trades = int(inventory)
        flatten_cash += float(inventory) * final_bid * point_value
        flatten_fees += float(abs(inventory)) * taker_fee_twd
    elif inventory < 0:
        forced_liq_trades = int(abs(inventory))
        flatten_cash -= float(abs(inventory)) * final_ask * point_value
        flatten_fees += float(abs(inventory)) * taker_fee_twd

    pnl_after_fees_flatten = float(flatten_cash - flatten_fees)
    pnl_arr = np.asarray(pnl_path, dtype=np.float64) if pnl_path else np.zeros((0,), dtype=np.float64)
    inv_arr = np.asarray(inv_path, dtype=np.int64) if inv_path else np.zeros((0,), dtype=np.int64)

    result: dict[str, float | int | str] = {
        "strategy": "baseline",
        "pnl": float(gross_pnl),
        "pnl_after_fees": float(net_pnl),
        "pnl_after_fees_flatten": float(pnl_after_fees_flatten),
        "fees_paid": float(fees_paid),
        "fees_paid_flatten": float(flatten_fees),
        "trades": int(maker_trades),
        "maker_trades": int(maker_trades),
        "taker_trades": int(0),
        "ending_inventory": int(inventory),
        "ending_inventory_flatten": int(0),
        "forced_liquidation_trades": int(forced_liq_trades),
        "max_abs_inventory": int(np.max(np.abs(inv_arr))) if inv_arr.size else 0,
        "fill_rate": float(maker_trades / max(quote_sides, 1)),
        "avg_spread_capture": float(np.mean(spread_capture_ticks)) if spread_capture_ticks else 0.0,
        "avg_adverse_selection": float(np.mean(adverse_selection_ticks)) if adverse_selection_ticks else 0.0,
        "max_drawdown": float(_max_drawdown(pnl_arr)),
    }
    paths_out = {"pnl_path": pnl_arr, "inventory_path": inv_arr} if return_paths else None
    return result, paths_out


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="TXFB6")
    parser.add_argument("--out-dir", type=str, default="new_era/backtest_outputs/NEW2_result")
    args = parser.parse_args()

    project_root = _ensure_project_root_on_path()
    data_dir = project_root / "data" / "NEW2"
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg()
    cfg = replace(cfg, data=replace(cfg.data, data_dir=data_dir, symbols=(str(args.symbol),)))

    symbol = str(args.symbol)
    print(f"[NEW2] data_dir={data_dir} symbol={symbol}")

    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    lob = lob_frames.get(symbol)
    trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    if lob is None or lob.empty:
        raise RuntimeError(f"No LOB rows found for symbol={symbol} in {data_dir}")

    tick_size = float(infer_tick_size_from_lob(lob))

    # ---------------------------------------------------------------------
    # 1) Data status / quality checks
    # ---------------------------------------------------------------------
    sessions = list(lob["session_id"].drop_duplicates())
    session_rows = lob.groupby("session_id").size().rename("lob_rows").reset_index()
    trade_rows = trades.groupby("session_id").size().rename("trade_rows").reset_index() if not trades.empty else None
    if trade_rows is not None:
        session_rows = session_rows.merge(trade_rows, on="session_id", how="left").fillna({"trade_rows": 0})
    else:
        session_rows["trade_rows"] = 0
    session_rows["trade_rows"] = session_rows["trade_rows"].astype(int)

    # Basic anomaly counters (per session)
    qc_rows: list[dict[str, object]] = []
    for sid in sessions:
        s = lob[lob["session_id"] == sid].copy()
        bid = s["bid_px_1"].to_numpy(dtype=np.float64)
        ask = s["ask_px_1"].to_numpy(dtype=np.float64)
        spread = ask - bid

        qc_rows.append(
            {
                "session_id": sid,
                "lob_rows": int(len(s)),
                "num_nonfinite_bid_or_ask": int((~np.isfinite(bid) | ~np.isfinite(ask)).sum()),
                "num_bid_le_0": int((bid <= 0).sum()),
                "num_ask_le_0": int((ask <= 0).sum()),
                "num_ask_le_bid": int((ask <= bid).sum()),
                "spread_min": float(np.nanmin(spread)) if len(spread) else float("nan"),
                "spread_median": float(np.nanmedian(spread)) if len(spread) else float("nan"),
                "spread_mean": float(np.nanmean(spread)) if len(spread) else float("nan"),
                "spread_max": float(np.nanmax(spread)) if len(spread) else float("nan"),
            }
        )

    qc_df = pd.DataFrame(qc_rows).sort_values("session_id").reset_index(drop=True)
    session_rows.to_csv(out_dir / "new2_session_row_counts.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(out_dir / "new2_data_quality_by_session.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_dir": str(data_dir),
        "symbol": symbol,
        "tick_size": tick_size,
        "sessions": sessions,
        "total_lob_rows": int(len(lob)),
        "total_trade_rows": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,
    }
    _write_json(out_dir / "new2_data_summary.json", summary)

    # ---------------------------------------------------------------------
    # 2) Baseline backtest (per session + concatenated)
    # ---------------------------------------------------------------------
    bt_cfg = cfg.backtest
    point_value = float(bt_cfg.point_value_twd_per_point)
    maker_fee = float(bt_cfg.maker_fee_twd)
    taker_fee = float(bt_cfg.taker_fee_twd)
    max_inv = int(bt_cfg.max_inventory)
    horizon = int(cfg.window.horizon)

    per_session_results: list[dict[str, object]] = []
    pnl_paths: dict[str, np.ndarray] = {}
    inv_paths: dict[str, np.ndarray] = {}

    for sid in sessions:
        s = lob[lob["session_id"] == sid]
        bid = s["bid_px_1"].to_numpy(dtype=np.float64)
        ask = s["ask_px_1"].to_numpy(dtype=np.float64)

        res, paths = run_baseline_maker_only(
            bid_px=bid,
            ask_px=ask,
            tick_size=tick_size,
            point_value=point_value,
            maker_fee_twd=maker_fee,
            taker_fee_twd=taker_fee,
            max_inventory=max_inv,
            horizon=horizon,
            return_paths=True,
        )
        row = {"session_id": sid, **res}
        per_session_results.append(row)
        if paths:
            pnl_paths[sid] = paths["pnl_path"]
            inv_paths[sid] = paths["inventory_path"]

    per_df = pd.DataFrame(per_session_results).sort_values("session_id").reset_index(drop=True)
    per_df.to_csv(out_dir / "baseline_results_by_session.csv", index=False, encoding="utf-8-sig")

    # Session-reset aggregate: treat each session as an independent run (inventory/cash reset, forced flatten each session).
    session_reset_total = {
        "scope": "sum_of_sessions(flatten_each_session)",
        "strategy": "baseline",
        "pnl": float(pd.to_numeric(per_df["pnl"], errors="coerce").sum()),
        "pnl_after_fees": float(pd.to_numeric(per_df["pnl_after_fees"], errors="coerce").sum()),
        "pnl_after_fees_flatten": float(pd.to_numeric(per_df["pnl_after_fees_flatten"], errors="coerce").sum()),
        "fees_paid": float(pd.to_numeric(per_df["fees_paid"], errors="coerce").sum()),
        "fees_paid_flatten": float(pd.to_numeric(per_df["fees_paid_flatten"], errors="coerce").sum()),
        "trades": int(pd.to_numeric(per_df["trades"], errors="coerce").sum()),
        "maker_trades": int(pd.to_numeric(per_df["maker_trades"], errors="coerce").sum()),
        "taker_trades": 0,
        "ending_inventory": 0,
        "ending_inventory_flatten": 0,
        "forced_liquidation_trades": int(pd.to_numeric(per_df["forced_liquidation_trades"], errors="coerce").sum()),
        "max_abs_inventory": int(pd.to_numeric(per_df["max_abs_inventory"], errors="coerce").max()),
        "fill_rate": float("nan"),
        "avg_spread_capture": float("nan"),
        "avg_adverse_selection": float("nan"),
        "max_drawdown": float("nan"),
    }

    # Concatenated (matches the existing 015 backtest behavior more closely).
    bid_all = lob["bid_px_1"].to_numpy(dtype=np.float64)
    ask_all = lob["ask_px_1"].to_numpy(dtype=np.float64)
    total_res, total_paths = run_baseline_maker_only(
        bid_px=bid_all,
        ask_px=ask_all,
        tick_size=tick_size,
        point_value=point_value,
        maker_fee_twd=maker_fee,
        taker_fee_twd=taker_fee,
        max_inventory=max_inv,
        horizon=horizon,
        return_paths=True,
    )
    pd.DataFrame([session_reset_total, {"scope": "all_sessions_concatenated", **total_res}]).to_csv(
        out_dir / "baseline_results_total.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if total_paths:
        np.save(out_dir / "baseline_pnl_path.npy", total_paths["pnl_path"])
        np.save(out_dir / "baseline_inventory_path.npy", total_paths["inventory_path"])

    print(f"[NEW2] wrote -> {out_dir}")
    print(f"[NEW2] sessions={len(sessions)} lob_rows={len(lob):,} trades={len(trades):,} tick_size={tick_size}")
    print(
        f"[NEW2] baseline(all sessions) pnl_after_fees={float(total_res['pnl_after_fees']):.2f} "
        f"flatten={float(total_res['pnl_after_fees_flatten']):.2f} trades={int(total_res['trades'])}"
    )


if __name__ == "__main__":
    main()
