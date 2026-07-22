"""Walk-forward optimization CLI for sweeping saved prediction files chronologically."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.hybrid_mm_helpers import alpha_profiles_from_names, parse_time_to_ns, quantile_thresholds
from src.backtest.hybrid_market_maker import run_hybrid_market_maker_backtest
from src.data.tw_replay import load_tw_replay_trades


@dataclass
class FoldSlice:
    fold: int
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


REGIME_DEFS = {
    "none": {},
    "vol_guard_q0.90": {"vol_q": 0.90},
    "cancel_guard_q0.80": {"cancel_q": 0.80},
    "vol_cancel_q0.90_0.80": {"vol_q": 0.90, "cancel_q": 0.80},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward optimization for hybrid market-making rules.")
    parser.add_argument("--run-dir", default="artifacts/02_model_runs/tw_execution_aware_valcal/20260319_185026")
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--trade-symbols", default="TXFJ5,MXFJ5")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--warmup-frac", type=float, default=0.5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score-thresholds", default="0.004,0.006")
    parser.add_argument("--one-sided-thresholds", default="0.008")
    parser.add_argument("--quote-off-thresholds", default="0.004,0.006")
    parser.add_argument("--stationary-cutoffs", default="0.338,0.336")
    parser.add_argument("--alpha-profiles", default="flow_queue,flow_cancel")
    parser.add_argument("--inventory-penalties", default="0.5")
    parser.add_argument("--quote-confirmation-steps", default="0,1")
    parser.add_argument("--alpha-smoothing-windows", default="1,2")
    parser.add_argument("--stop-loss-ticks", default="4")
    parser.add_argument("--take-profit-ticks", default="0,8")
    parser.add_argument("--alpha-decay-thresholds", default="0.0,0.002")
    parser.add_argument("--directional-gate", default="false,true")
    parser.add_argument("--force-directional-one-sided", default="false")
    parser.add_argument("--regime-profiles", default="none,cancel_guard_q0.80,vol_guard_q0.90,vol_cancel_q0.90_0.80")
    parser.add_argument("--end-time", default="12:00:00")
    parser.add_argument("--max-holding-steps", type=int, default=20)
    parser.add_argument("--adverse-score-threshold", type=float, default=0.010)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_text_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return [mapping[item.strip().lower()] for item in raw.split(",") if item.strip()]


def build_walk_forward_slices(predictions: pd.DataFrame, folds: int, warmup_frac: float) -> list[FoldSlice]:
    symbols = sorted(predictions["symbol"].unique().tolist())
    per_symbol_splits: dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}

    for symbol in symbols:
        sdf = predictions.loc[predictions["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        n = len(sdf)
        warmup = max(1, int(n * warmup_frac))
        remaining = max(0, n - warmup)
        if remaining < folds:
            raise ValueError(f"Not enough rows for walk-forward folds on symbol {symbol}: rows={n}, folds={folds}")
        step = max(1, remaining // folds)

        symbol_slices: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for fold in range(folds):
            train_end = warmup + fold * step
            test_end = n if fold == folds - 1 else min(n, train_end + step)
            train_df = sdf.iloc[:train_end].copy()
            test_df = sdf.iloc[train_end:test_end].copy()
            symbol_slices.append((train_df, test_df))
        per_symbol_splits[symbol] = symbol_slices

    slices: list[FoldSlice] = []
    for fold in range(folds):
        train_frames = [per_symbol_splits[symbol][fold][0] for symbol in symbols]
        test_frames = [per_symbol_splits[symbol][fold][1] for symbol in symbols]
        slices.append(
            FoldSlice(
                fold=fold + 1,
                train_predictions=pd.concat(train_frames, ignore_index=True),
                test_predictions=pd.concat(test_frames, ignore_index=True),
            )
        )
    return slices


def evaluate_symbol_config(
    symbol: str,
    train_predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    config: dict,
    alpha_profiles: dict[str, dict[str, float]],
) -> dict:
    symbol_predictions = train_predictions.loc[train_predictions["symbol"] == symbol].copy().reset_index(drop=True)
    if symbol_predictions.empty:
        return {
            "symbol": symbol,
            **config,
            "realized_trades": 0,
            "fills": 0,
            "quote_updates": 0,
            "net_pnl_ticks": 0.0,
            "avg_pnl_ticks": 0.0,
            "win_rate": 0.0,
            "avg_holding_steps": 0.0,
            "sharpe_like": 0.0,
        }
    thresholds = quantile_thresholds(symbol_predictions, REGIME_DEFS[config["regime_profile"]])
    _, summary, _, _ = run_hybrid_market_maker_backtest(
        symbol_predictions,
        trade_tapes={symbol: trade_tapes.get(symbol, pd.DataFrame())},
        score_threshold=config["score_threshold"],
        one_sided_threshold=config["one_sided_threshold"],
        quote_off_score_threshold=config["quote_off_threshold"],
        stationary_probability_cutoff=config["stationary_cutoff"],
        alpha_coefficients=alpha_profiles[config["alpha_profile"]],
        inventory_penalty_ticks=config["inventory_penalty_ticks"],
        max_inventory=1,
        max_holding_steps=config["max_holding_steps"],
        adverse_score_threshold=config["adverse_score_threshold"],
        quote_confirmation_steps=config["quote_confirmation_steps"],
        alpha_smoothing_window=config["alpha_smoothing_window"],
        inventory_stop_loss_ticks=config["stop_loss_ticks"] or None,
        inventory_take_profit_ticks=config["take_profit_ticks"] or None,
        inventory_flatten_score_threshold=config["alpha_decay_threshold"],
        require_directional_gate=config["require_directional_gate"],
        force_directional_one_sided=config["force_directional_one_sided"],
        symbol_end_times={symbol: config["end_time_ns"]},
        regime_volatility_thresholds=(
            {symbol: thresholds["regime_volatility_thresholds"]}
            if "regime_volatility_thresholds" in thresholds
            else None
        ),
        regime_max_cancel_pressure=(
            {symbol: thresholds["regime_max_cancel_pressure"]}
            if "regime_max_cancel_pressure" in thresholds
            else None
        ),
        allowed_symbols={symbol},
    )
    return {"symbol": symbol, **config, **summary}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.concat(
        [
            pd.read_csv(run_dir / "val_predictions.csv"),
            pd.read_csv(run_dir / "test_predictions.csv"),
        ],
        ignore_index=True,
    ).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    trade_symbols = parse_text_list(args.trade_symbols)
    predictions = predictions.loc[predictions["symbol"].isin(trade_symbols)].copy().reset_index(drop=True)
    trade_tapes = load_tw_replay_trades(args.replay_path, trade_symbols)
    alpha_profiles = alpha_profiles_from_names(parse_text_list(args.alpha_profiles))

    config_grid: list[dict] = []
    for score_threshold in parse_float_list(args.score_thresholds):
        for one_sided_threshold in parse_float_list(args.one_sided_thresholds):
            for quote_off_threshold in parse_float_list(args.quote_off_thresholds):
                if quote_off_threshold > score_threshold:
                    continue
                for stationary_cutoff in parse_float_list(args.stationary_cutoffs):
                    for alpha_profile in alpha_profiles.keys():
                        for inventory_penalty_ticks in parse_float_list(args.inventory_penalties):
                            for quote_confirmation_steps in parse_int_list(args.quote_confirmation_steps):
                                for alpha_smoothing_window in parse_int_list(args.alpha_smoothing_windows):
                                    for stop_loss_ticks in parse_float_list(args.stop_loss_ticks):
                                        for take_profit_ticks in parse_float_list(args.take_profit_ticks):
                                            for alpha_decay_threshold in parse_float_list(args.alpha_decay_thresholds):
                                                for require_directional_gate in parse_bool_list(args.directional_gate):
                                                    for force_directional_one_sided in parse_bool_list(args.force_directional_one_sided):
                                                        for regime_profile in parse_text_list(args.regime_profiles):
                                                            config_grid.append(
                                                                {
                                                                    "score_threshold": score_threshold,
                                                                    "one_sided_threshold": one_sided_threshold,
                                                                    "quote_off_threshold": quote_off_threshold,
                                                                    "stationary_cutoff": stationary_cutoff,
                                                                    "alpha_profile": alpha_profile,
                                                                    "inventory_penalty_ticks": inventory_penalty_ticks,
                                                                    "quote_confirmation_steps": quote_confirmation_steps,
                                                                    "alpha_smoothing_window": alpha_smoothing_window,
                                                                    "stop_loss_ticks": stop_loss_ticks,
                                                                    "take_profit_ticks": take_profit_ticks,
                                                                    "alpha_decay_threshold": alpha_decay_threshold,
                                                                    "require_directional_gate": require_directional_gate,
                                                                    "force_directional_one_sided": force_directional_one_sided,
                                                                    "regime_profile": regime_profile,
                                                                    "max_holding_steps": args.max_holding_steps,
                                                                    "adverse_score_threshold": args.adverse_score_threshold,
                                                                    "end_time_ns": parse_time_to_ns(args.end_time),
                                                                }
                                                            )

    slices = build_walk_forward_slices(predictions, folds=args.folds, warmup_frac=args.warmup_frac)
    fold_rows: list[dict] = []
    all_trade_logs: list[pd.DataFrame] = []

    for fold_slice in slices:
        best_configs: list[dict] = []
        for symbol in trade_symbols:
            rows = [
                evaluate_symbol_config(
                    symbol=symbol,
                    train_predictions=fold_slice.train_predictions,
                    trade_tapes=trade_tapes,
                    config=config,
                    alpha_profiles=alpha_profiles,
                )
                for config in config_grid
            ]
            result = pd.DataFrame(rows)
            if result.empty:
                result = pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "score_threshold": 0.0,
                            "one_sided_threshold": 0.0,
                            "quote_off_threshold": 0.0,
                            "stationary_cutoff": 1.0,
                            "alpha_profile": next(iter(alpha_profiles.keys())),
                            "inventory_penalty_ticks": 0.0,
                            "quote_confirmation_steps": 0,
                            "alpha_smoothing_window": 1,
                            "stop_loss_ticks": 0.0,
                            "take_profit_ticks": 0.0,
                            "alpha_decay_threshold": 0.0,
                            "require_directional_gate": False,
                            "force_directional_one_sided": False,
                            "regime_profile": "none",
                            "max_holding_steps": args.max_holding_steps,
                            "adverse_score_threshold": args.adverse_score_threshold,
                            "end_time_ns": parse_time_to_ns(args.end_time),
                            "realized_trades": 0,
                            "fills": 0,
                            "quote_updates": 0,
                            "net_pnl_ticks": 0.0,
                            "avg_pnl_ticks": 0.0,
                            "win_rate": 0.0,
                            "avg_holding_steps": 0.0,
                            "sharpe_like": 0.0,
                        }
                    ]
                )
            else:
                result = result.sort_values(
                    ["net_pnl_ticks", "avg_pnl_ticks", "win_rate"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)
            result.to_csv(output_dir / f"fold_{fold_slice.fold}_{symbol}_train_search.csv", index=False)
            best_configs.append(result.iloc[0].to_dict())

        best_df = pd.DataFrame(best_configs)
        best_df.to_csv(output_dir / f"fold_{fold_slice.fold}_best_by_symbol.csv", index=False)

        symbol_score_thresholds = dict(zip(best_df["symbol"], best_df["score_threshold"]))
        symbol_one_sided_thresholds = dict(zip(best_df["symbol"], best_df["one_sided_threshold"]))
        symbol_quote_off_thresholds = dict(zip(best_df["symbol"], best_df["quote_off_threshold"]))
        symbol_stationary_cutoffs = dict(zip(best_df["symbol"], best_df["stationary_cutoff"]))
        symbol_inventory_penalties = dict(zip(best_df["symbol"], best_df["inventory_penalty_ticks"]))
        symbol_quote_confirmation = dict(zip(best_df["symbol"], best_df["quote_confirmation_steps"]))
        symbol_alpha_smoothing = dict(zip(best_df["symbol"], best_df["alpha_smoothing_window"]))
        symbol_stop_loss = dict(zip(best_df["symbol"], best_df["stop_loss_ticks"]))
        symbol_take_profit = dict(zip(best_df["symbol"], best_df["take_profit_ticks"]))
        symbol_alpha_decay = dict(zip(best_df["symbol"], best_df["alpha_decay_threshold"]))
        symbol_alpha_coefficients = {
            row["symbol"]: alpha_profiles[row["alpha_profile"]]
            for _, row in best_df.iterrows()
        }
        symbol_end_times = {row["symbol"]: int(row["end_time_ns"]) for _, row in best_df.iterrows()}
        require_directional_gate = bool(best_df["require_directional_gate"].any())
        force_directional_one_sided = bool(best_df["force_directional_one_sided"].any())

        regime_volatility_thresholds: dict[str, float] = {}
        regime_max_cancel_pressure: dict[str, float] = {}
        for _, row in best_df.iterrows():
            symbol = row["symbol"]
            thresholds = quantile_thresholds(
                fold_slice.train_predictions.loc[fold_slice.train_predictions["symbol"] == symbol],
                REGIME_DEFS[row["regime_profile"]],
            )
            if "regime_volatility_thresholds" in thresholds:
                regime_volatility_thresholds[symbol] = thresholds["regime_volatility_thresholds"]
            if "regime_max_cancel_pressure" in thresholds:
                regime_max_cancel_pressure[symbol] = thresholds["regime_max_cancel_pressure"]

        trade_log, summary, by_symbol, quote_log = run_hybrid_market_maker_backtest(
            fold_slice.test_predictions,
            trade_tapes=trade_tapes,
            max_holding_steps=args.max_holding_steps,
            adverse_score_threshold=args.adverse_score_threshold,
            symbol_score_thresholds=symbol_score_thresholds,
            symbol_one_sided_thresholds=symbol_one_sided_thresholds,
            symbol_quote_off_thresholds=symbol_quote_off_thresholds,
            symbol_stationary_cutoffs=symbol_stationary_cutoffs,
            symbol_quote_confirmation_steps=symbol_quote_confirmation,
            symbol_alpha_smoothing_windows=symbol_alpha_smoothing,
            symbol_inventory_penalties=symbol_inventory_penalties,
            symbol_end_times=symbol_end_times,
            symbol_alpha_coefficients=symbol_alpha_coefficients,
            symbol_stop_loss_ticks=symbol_stop_loss,
            symbol_take_profit_ticks=symbol_take_profit,
            symbol_alpha_decay_thresholds=symbol_alpha_decay,
            require_directional_gate=require_directional_gate,
            force_directional_one_sided=force_directional_one_sided,
            regime_volatility_thresholds=regime_volatility_thresholds or None,
            regime_max_cancel_pressure=regime_max_cancel_pressure or None,
            allowed_symbols=set(trade_symbols),
        )
        trade_log.to_csv(output_dir / f"fold_{fold_slice.fold}_trade_log.csv", index=False)
        by_symbol.to_csv(output_dir / f"fold_{fold_slice.fold}_by_symbol.csv", index=False)
        quote_log.to_csv(output_dir / f"fold_{fold_slice.fold}_quote_log.csv", index=False)

        fold_row = {
            "fold": fold_slice.fold,
            "train_rows": int(len(fold_slice.train_predictions)),
            "test_rows": int(len(fold_slice.test_predictions)),
            **summary,
        }
        fold_rows.append(fold_row)

        if not trade_log.empty:
            trade_log = trade_log.copy()
            trade_log["fold"] = fold_slice.fold
            all_trade_logs.append(trade_log)

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "walk_forward_fold_summary.csv", index=False)

    if all_trade_logs:
        combined_trade_log = pd.concat(all_trade_logs, ignore_index=True)
        combined_trade_log.to_csv(output_dir / "walk_forward_trade_log.csv", index=False)
        combined_by_symbol = (
            combined_trade_log.groupby("symbol", as_index=False)
            .agg(
                realized_trades=("symbol", "size"),
                net_pnl_ticks=("pnl_ticks", "sum"),
                avg_pnl_ticks=("pnl_ticks", "mean"),
            )
            .sort_values("net_pnl_ticks", ascending=False)
            .reset_index(drop=True)
        )
    else:
        combined_trade_log = pd.DataFrame()
        combined_by_symbol = pd.DataFrame(columns=["symbol", "realized_trades", "net_pnl_ticks", "avg_pnl_ticks"])

    combined_by_symbol.to_csv(output_dir / "walk_forward_by_symbol.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "trade_symbols": trade_symbols,
        "folds": args.folds,
        "warmup_frac": args.warmup_frac,
        "total_realized_trades": int(fold_summary["realized_trades"].sum()),
        "total_net_pnl_ticks": float(fold_summary["net_pnl_ticks"].sum()),
        "avg_fold_pnl_ticks": float(fold_summary["net_pnl_ticks"].mean()),
        "positive_folds": int((fold_summary["net_pnl_ticks"] > 0).sum()),
    }
    with (output_dir / "walk_forward_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
