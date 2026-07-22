"""Search hybrid market-making settings and quantify latency degradation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.hybrid_mm_helpers import (
    alpha_profiles_from_names,
    build_regime_profiles,
    parse_time_to_ns,
    quantile_thresholds,
)
from src.backtest.hybrid_market_maker import run_hybrid_market_maker_backtest
from src.data.tw_replay import load_tw_replay_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency proof sweep for hybrid MM.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-path", default="replay1_TW.csv")
    parser.add_argument("--trade-symbols", default="TXFJ5,MXFJ5")
    parser.add_argument("--score-thresholds", default="0.006")
    parser.add_argument("--one-sided-thresholds", default="0.008")
    parser.add_argument("--quote-off-thresholds", default="0.006,0.007")
    parser.add_argument("--toxicity-thresholds", default="0.04,0.06")
    parser.add_argument("--one-sided-toxicity-thresholds", default="0.02,0.04")
    parser.add_argument("--toxicity-freeze-thresholds", default="0.08,0.12")
    parser.add_argument("--toxicity-freeze-steps", default="5,8")
    parser.add_argument("--inventory-widen-ticks", default="0.5,1.0")
    parser.add_argument("--inventory-tighten-ticks", default="0.5,1.0")
    parser.add_argument("--stationary-cutoffs", default="0.336")
    parser.add_argument("--inventory-penalties", default="0.5")
    parser.add_argument("--max-inventories", default="1")
    parser.add_argument("--end-times", default="12:00:00")
    parser.add_argument("--latency-steps", default="0,1,2,4")
    parser.add_argument("--quote-confirmation-steps", default="0,1")
    parser.add_argument("--alpha-smoothing-windows", default="1,2")
    parser.add_argument("--stop-loss-ticks", default="4")
    parser.add_argument("--take-profit-ticks", default="0")
    parser.add_argument("--alpha-decay-thresholds", default="0.002")
    parser.add_argument("--quote-step-ticks", type=float, default=1.0)
    parser.add_argument("--max-holding-steps", type=int, default=20)
    parser.add_argument("--adverse-score-threshold", type=float, default=0.010)
    parser.add_argument("--vol-quantiles", default="0.90")
    parser.add_argument("--trade-quantiles", default="")
    parser.add_argument("--cancel-quantiles", default="0.80")
    parser.add_argument("--liq-quantiles", default="")
    parser.add_argument("--alpha-profiles", default="flow_cancel,toxic_flow")
    parser.add_argument("--output-dir", default="artifacts/03_latency_proof/latency_proof")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_text_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _search_symbol(
    symbol: str,
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    score_thresholds: list[float],
    one_sided_thresholds: list[float],
    quote_off_thresholds: list[float],
    toxicity_thresholds: list[float],
    one_sided_toxicity_thresholds: list[float],
    toxicity_freeze_thresholds: list[float],
    toxicity_freeze_steps_list: list[int],
    inventory_widen_ticks_list: list[float],
    inventory_tighten_ticks_list: list[float],
    stationary_cutoffs: list[float],
    inventory_penalties: list[float],
    max_inventories: list[int],
    end_times: list[str],
    quote_confirmation_steps_list: list[int],
    alpha_smoothing_windows: list[int],
    stop_loss_ticks_list: list[float],
    take_profit_ticks_list: list[float],
    alpha_decay_thresholds: list[float],
    quote_step_ticks: float,
    max_holding_steps: int,
    adverse_score_threshold: float,
    regime_profiles: list[dict],
    alpha_profiles: dict[str, dict[str, float]],
) -> pd.DataFrame:
    symbol_predictions = predictions.loc[predictions["symbol"] == symbol].copy().reset_index(drop=True)
    rows: list[dict] = []
    for score_threshold in score_thresholds:
        for one_sided_threshold in one_sided_thresholds:
            if one_sided_threshold < score_threshold:
                continue
            for quote_off_threshold in quote_off_thresholds:
                if quote_off_threshold > one_sided_threshold:
                    continue
                for toxicity_threshold in toxicity_thresholds:
                    for one_sided_toxicity_threshold in one_sided_toxicity_thresholds:
                        if one_sided_toxicity_threshold > toxicity_threshold:
                            continue
                        for toxicity_freeze_threshold in toxicity_freeze_thresholds:
                            for toxicity_freeze_steps in toxicity_freeze_steps_list:
                                for inventory_widen_ticks in inventory_widen_ticks_list:
                                    for inventory_tighten_ticks in inventory_tighten_ticks_list:
                                        for stationary_cutoff in stationary_cutoffs:
                                            for alpha_name, alpha_coefficients in alpha_profiles.items():
                                                for inventory_penalty in inventory_penalties:
                                                    for max_inventory in max_inventories:
                                                        for end_time in end_times:
                                                            for quote_confirmation_steps in quote_confirmation_steps_list:
                                                                for alpha_smoothing_window in alpha_smoothing_windows:
                                                                    for stop_loss_ticks in stop_loss_ticks_list:
                                                                        for take_profit_ticks in take_profit_ticks_list:
                                                                            for alpha_decay_threshold in alpha_decay_thresholds:
                                                                                for profile in regime_profiles:
                                                                                    thresholds = quantile_thresholds(symbol_predictions, profile)
                                                                                    trade_log, summary, _, _ = run_hybrid_market_maker_backtest(
                                                                                        symbol_predictions,
                                                                                        trade_tapes={symbol: trade_tapes.get(symbol, pd.DataFrame())},
                                                                                        score_threshold=score_threshold,
                                                                                        one_sided_threshold=one_sided_threshold,
                                                                                        quote_off_score_threshold=quote_off_threshold,
                                                                                        toxicity_threshold=toxicity_threshold,
                                                                                        one_sided_toxicity_threshold=one_sided_toxicity_threshold,
                                                                                        toxicity_freeze_threshold=toxicity_freeze_threshold,
                                                                                        toxicity_freeze_steps=toxicity_freeze_steps,
                                                                                        stationary_probability_cutoff=stationary_cutoff,
                                                                                        alpha_coefficients=alpha_coefficients,
                                                                                        inventory_penalty_ticks=inventory_penalty,
                                                                                        inventory_widen_ticks=inventory_widen_ticks,
                                                                                        inventory_tighten_ticks=inventory_tighten_ticks,
                                                                                        quote_step_ticks=quote_step_ticks,
                                                                                        max_inventory=max_inventory,
                                                                                        max_holding_steps=max_holding_steps,
                                                                                        adverse_score_threshold=adverse_score_threshold,
                                                                                        quote_confirmation_steps=quote_confirmation_steps,
                                                                                        alpha_smoothing_window=alpha_smoothing_window,
                                                                                        inventory_stop_loss_ticks=(stop_loss_ticks if stop_loss_ticks > 0 else None),
                                                                                        inventory_take_profit_ticks=(take_profit_ticks if take_profit_ticks > 0 else None),
                                                                                        inventory_flatten_score_threshold=alpha_decay_threshold,
                                                                                        symbol_end_times={symbol: parse_time_to_ns(end_time)},
                                                                                        regime_volatility_thresholds=(
                                                                                            {symbol: thresholds["regime_volatility_thresholds"]}
                                                                                            if "regime_volatility_thresholds" in thresholds
                                                                                            else None
                                                                                        ),
                                                                                        regime_trade_intensity_thresholds=(
                                                                                            {symbol: thresholds["regime_trade_intensity_thresholds"]}
                                                                                            if "regime_trade_intensity_thresholds" in thresholds
                                                                                            else None
                                                                                        ),
                                                                                        regime_min_visible_queue=(
                                                                                            {symbol: thresholds["regime_min_visible_queue"]}
                                                                                            if "regime_min_visible_queue" in thresholds
                                                                                            else None
                                                                                        ),
                                                                                        regime_max_cancel_pressure=(
                                                                                            {symbol: thresholds["regime_max_cancel_pressure"]}
                                                                                            if "regime_max_cancel_pressure" in thresholds
                                                                                            else None
                                                                                        ),
                                                                                        allowed_symbols={symbol},
                                                                                    )
                                                                                    negative_pnl = (
                                                                                        float(-trade_log.loc[trade_log["pnl_ticks"] < 0, "pnl_ticks"].sum())
                                                                                        if not trade_log.empty
                                                                                        else 0.0
                                                                                    )
                                                                                    rows.append(
                                                                                        {
                                                                                            "symbol": symbol,
                                                                                            "score_threshold": score_threshold,
                                                                                            "one_sided_threshold": one_sided_threshold,
                                                                                            "quote_off_threshold": quote_off_threshold,
                                                                                            "toxicity_threshold": toxicity_threshold,
                                                                                            "one_sided_toxicity_threshold": one_sided_toxicity_threshold,
                                                                                            "toxicity_freeze_threshold": toxicity_freeze_threshold,
                                                                                            "toxicity_freeze_steps": toxicity_freeze_steps,
                                                                                            "inventory_widen_ticks": inventory_widen_ticks,
                                                                                            "inventory_tighten_ticks": inventory_tighten_ticks,
                                                                                            "stationary_cutoff": stationary_cutoff,
                                                                                            "alpha_profile": alpha_name,
                                                                                            "inventory_penalty_ticks": inventory_penalty,
                                                                                            "max_inventory": max_inventory,
                                                                                            "end_time": end_time,
                                                                                            "quote_confirmation_steps": quote_confirmation_steps,
                                                                                            "alpha_smoothing_window": alpha_smoothing_window,
                                                                                            "stop_loss_ticks": stop_loss_ticks,
                                                                                            "take_profit_ticks": take_profit_ticks,
                                                                                            "alpha_decay_threshold": alpha_decay_threshold,
                                                                                            "regime_profile": profile["name"],
                                                                                            "fill_rate": float(summary["fills"] / max(summary["quote_updates"], 1)),
                                                                                            "adverse_selection_loss_ticks": negative_pnl,
                                                                                            **summary,
                                                                                        }
                                                                                    )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["net_pnl_ticks", "avg_pnl_ticks", "fill_rate", "win_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _regime_threshold_maps(best_by_symbol: pd.DataFrame, predictions: pd.DataFrame, regime_profiles: list[dict]) -> dict[str, dict[str, float]]:
    thresholds_by_name = {
        "regime_volatility_thresholds": {},
        "regime_trade_intensity_thresholds": {},
        "regime_min_visible_queue": {},
        "regime_max_cancel_pressure": {},
    }
    for _, row in best_by_symbol.iterrows():
        profile = next(profile for profile in regime_profiles if profile["name"] == row["regime_profile"])
        symbol_df = predictions.loc[predictions["symbol"] == row["symbol"]]
        thresholds = quantile_thresholds(symbol_df, profile)
        for key, value in thresholds.items():
            thresholds_by_name[key][row["symbol"]] = value
    return thresholds_by_name


def _run_combined(
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame],
    best_by_symbol: pd.DataFrame,
    alpha_profiles: dict[str, dict[str, float]],
    latency_steps: int,
    regime_thresholds: dict[str, dict[str, float]],
    quote_step_ticks: float,
    max_holding_steps: int,
    adverse_score_threshold: float,
    trade_symbols: list[str],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    return run_hybrid_market_maker_backtest(
        predictions.loc[predictions["symbol"].isin(trade_symbols)].copy(),
        trade_tapes=trade_tapes,
        score_threshold=float(best_by_symbol["score_threshold"].min()),
        one_sided_threshold=float(best_by_symbol["one_sided_threshold"].max()),
        quote_off_score_threshold=float(best_by_symbol["quote_off_threshold"].max()),
        toxicity_threshold=float(best_by_symbol["toxicity_threshold"].max()),
        one_sided_toxicity_threshold=float(best_by_symbol["one_sided_toxicity_threshold"].max()),
        toxicity_freeze_threshold=float(best_by_symbol["toxicity_freeze_threshold"].max()),
        toxicity_freeze_steps=int(best_by_symbol["toxicity_freeze_steps"].max()),
        stationary_probability_cutoff=float(best_by_symbol["stationary_cutoff"].max()),
        inventory_penalty_ticks=float(best_by_symbol["inventory_penalty_ticks"].max()),
        inventory_widen_ticks=float(best_by_symbol["inventory_widen_ticks"].max()),
        inventory_tighten_ticks=float(best_by_symbol["inventory_tighten_ticks"].max()),
        quote_step_ticks=quote_step_ticks,
        max_inventory=int(best_by_symbol["max_inventory"].max()),
        max_holding_steps=max_holding_steps,
        adverse_score_threshold=adverse_score_threshold,
        latency_steps=latency_steps,
        quote_confirmation_steps=0,
        alpha_smoothing_window=1,
        symbol_score_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["score_threshold"])),
        symbol_one_sided_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["one_sided_threshold"])),
        symbol_quote_off_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["quote_off_threshold"])),
        symbol_toxicity_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["toxicity_threshold"])),
        symbol_one_sided_toxicity_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["one_sided_toxicity_threshold"])),
        symbol_toxicity_freeze_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["toxicity_freeze_threshold"])),
        symbol_toxicity_freeze_steps=dict(zip(best_by_symbol["symbol"], best_by_symbol["toxicity_freeze_steps"])),
        symbol_stationary_cutoffs=dict(zip(best_by_symbol["symbol"], best_by_symbol["stationary_cutoff"])),
        symbol_quote_confirmation_steps=dict(zip(best_by_symbol["symbol"], best_by_symbol["quote_confirmation_steps"])),
        symbol_alpha_smoothing_windows=dict(zip(best_by_symbol["symbol"], best_by_symbol["alpha_smoothing_window"])),
        symbol_inventory_penalties=dict(zip(best_by_symbol["symbol"], best_by_symbol["inventory_penalty_ticks"])),
        symbol_max_inventory=dict(zip(best_by_symbol["symbol"], best_by_symbol["max_inventory"])),
        symbol_end_times={row["symbol"]: parse_time_to_ns(row["end_time"]) for _, row in best_by_symbol.iterrows()},
        symbol_alpha_coefficients={row["symbol"]: alpha_profiles[row["alpha_profile"]] for _, row in best_by_symbol.iterrows()},
        symbol_inventory_widen_ticks=dict(zip(best_by_symbol["symbol"], best_by_symbol["inventory_widen_ticks"])),
        symbol_inventory_tighten_ticks=dict(zip(best_by_symbol["symbol"], best_by_symbol["inventory_tighten_ticks"])),
        symbol_stop_loss_ticks=dict(zip(best_by_symbol["symbol"], best_by_symbol["stop_loss_ticks"])),
        symbol_take_profit_ticks=dict(zip(best_by_symbol["symbol"], best_by_symbol["take_profit_ticks"])),
        symbol_alpha_decay_thresholds=dict(zip(best_by_symbol["symbol"], best_by_symbol["alpha_decay_threshold"])),
        regime_volatility_thresholds=regime_thresholds["regime_volatility_thresholds"] or None,
        regime_trade_intensity_thresholds=regime_thresholds["regime_trade_intensity_thresholds"] or None,
        regime_min_visible_queue=regime_thresholds["regime_min_visible_queue"] or None,
        regime_max_cancel_pressure=regime_thresholds["regime_max_cancel_pressure"] or None,
        allowed_symbols=set(trade_symbols),
    )


def _freeze_model(run_dir: Path, output_dir: Path, latency_df: pd.DataFrame, hardware_profile_path: Path | None) -> dict:
    frozen_model_path = output_dir / "frozen_tcn_best_model.pt"
    shutil.copy2(run_dir / "best_model.pt", frozen_model_path)
    delay_zero_pnl = float(latency_df.loc[latency_df["latency_steps"] == 0, "net_pnl_ticks"].iloc[0])
    delay_four_pnl = float(latency_df.loc[latency_df["latency_steps"] == 4, "net_pnl_ticks"].iloc[0])
    manifest = {
        "source_run_dir": str(run_dir),
        "frozen_model_path": str(frozen_model_path),
        "freeze_reason": "Selected for latency proof benchmark and FPGA handoff.",
        "proof_reference_delay": 0,
        "latency_gap_0_vs_4_ticks": float(delay_zero_pnl - delay_four_pnl),
    }
    if hardware_profile_path and hardware_profile_path.exists():
        with hardware_profile_path.open("r", encoding="utf-8") as handle:
            manifest["hardware_profile"] = json.load(handle)
    with (output_dir / "model_freeze_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _write_latency_report(
    output_dir: Path,
    best_by_symbol: pd.DataFrame,
    latency_df: pd.DataFrame,
    freeze_manifest: dict,
) -> None:
    best_table = best_by_symbol.to_string(index=False)
    latency_table = latency_df.to_string(index=False)
    lines = [
        "# Latency Proof Report",
        "",
        "## Validation Winner",
        "",
        "```text",
        best_table,
        "```",
        "",
        "## Test Latency Sweep",
        "",
        "```text",
        latency_table,
        "```",
        "",
        "## Notes",
        "",
        "- Fill rate is computed as `fills / quote_updates`.",
        "- Adverse selection loss is defined here as the magnitude of realized negative pnl ticks on closed trades.",
        f"- Monotonic decreasing pnl by latency: `{latency_df['net_pnl_ticks'].is_monotonic_decreasing}`",
        "",
        "## Frozen Model",
        "",
        f"- Frozen model path: `{freeze_manifest['frozen_model_path']}`",
    ]
    if "hardware_profile" in freeze_manifest:
        profile = freeze_manifest["hardware_profile"]
        input_bytes = int(profile["num_input_features"] * profile["window_size"] * 4)
        lines.extend(
            [
                f"- MACs per sample: `{profile['macs_per_sample']}`",
                f"- FP32 weight bytes: `{profile['weight_bytes_fp32']}`",
                f"- INT8 weight bytes: `{profile['weight_bytes_int8']}`",
                f"- Approximate FP32 input bytes per inference: `{input_bytes}`",
                f"- Approximate INT8 streamed weight bytes per inference: `{profile['weight_bytes_int8']}`",
            ]
        )
    (output_dir / "latency_proof_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validation predictions are used to choose per-symbol controls before the
    # selected settings are replayed on test data under several latency levels.
    trade_symbols = parse_text_list(args.trade_symbols)
    trade_tapes = load_tw_replay_trades(args.replay_path, trade_symbols)
    val_predictions = pd.read_csv(run_dir / "val_predictions.csv")
    test_predictions = pd.read_csv(run_dir / "test_predictions.csv")

    alpha_profiles = alpha_profiles_from_names(parse_text_list(args.alpha_profiles))
    regime_profiles = build_regime_profiles(
        vol_quantiles=parse_float_list(args.vol_quantiles),
        trade_quantiles=parse_float_list(args.trade_quantiles),
        cancel_quantiles=parse_float_list(args.cancel_quantiles),
        liq_quantiles=parse_float_list(args.liq_quantiles),
    )

    best_rows: list[dict] = []
    for symbol in trade_symbols:
        # Search each symbol independently so the final combined latency sweep
        # can preserve symbol-specific quote, toxicity, and inventory settings.
        result = _search_symbol(
            symbol=symbol,
            predictions=val_predictions,
            trade_tapes=trade_tapes,
            score_thresholds=parse_float_list(args.score_thresholds),
            one_sided_thresholds=parse_float_list(args.one_sided_thresholds),
            quote_off_thresholds=parse_float_list(args.quote_off_thresholds),
            toxicity_thresholds=parse_float_list(args.toxicity_thresholds),
            one_sided_toxicity_thresholds=parse_float_list(args.one_sided_toxicity_thresholds),
            toxicity_freeze_thresholds=parse_float_list(args.toxicity_freeze_thresholds),
            toxicity_freeze_steps_list=parse_int_list(args.toxicity_freeze_steps),
            inventory_widen_ticks_list=parse_float_list(args.inventory_widen_ticks),
            inventory_tighten_ticks_list=parse_float_list(args.inventory_tighten_ticks),
            stationary_cutoffs=parse_float_list(args.stationary_cutoffs),
            inventory_penalties=parse_float_list(args.inventory_penalties),
            max_inventories=parse_int_list(args.max_inventories),
            end_times=parse_text_list(args.end_times),
            quote_confirmation_steps_list=parse_int_list(args.quote_confirmation_steps),
            alpha_smoothing_windows=parse_int_list(args.alpha_smoothing_windows),
            stop_loss_ticks_list=parse_float_list(args.stop_loss_ticks),
            take_profit_ticks_list=parse_float_list(args.take_profit_ticks),
            alpha_decay_thresholds=parse_float_list(args.alpha_decay_thresholds),
            quote_step_ticks=args.quote_step_ticks,
            max_holding_steps=args.max_holding_steps,
            adverse_score_threshold=args.adverse_score_threshold,
            regime_profiles=regime_profiles,
            alpha_profiles=alpha_profiles,
        )
        result.to_csv(output_dir / f"val_search_{symbol}.csv", index=False)
        best_rows.append(result.iloc[0].to_dict())

    best_by_symbol = pd.DataFrame(best_rows)
    best_by_symbol.to_csv(output_dir / "val_best_by_symbol.csv", index=False)

    regime_thresholds = _regime_threshold_maps(best_by_symbol, val_predictions, regime_profiles)
    latency_rows: list[dict] = []
    for latency in parse_int_list(args.latency_steps):
        # Re-run the combined market maker while injecting additional decision
        # delay to show how quickly queue alpha decays.
        trade_log, summary, by_symbol, quote_log = _run_combined(
            predictions=test_predictions,
            trade_tapes=trade_tapes,
            best_by_symbol=best_by_symbol,
            alpha_profiles=alpha_profiles,
            latency_steps=latency,
            regime_thresholds=regime_thresholds,
            quote_step_ticks=args.quote_step_ticks,
            max_holding_steps=args.max_holding_steps,
            adverse_score_threshold=args.adverse_score_threshold,
            trade_symbols=trade_symbols,
        )
        latency_row = {
            "latency_steps": latency,
            "fill_rate": float(summary["fills"] / max(summary["quote_updates"], 1)),
            "adverse_selection_loss_ticks": (
                float(-trade_log.loc[trade_log["pnl_ticks"] < 0, "pnl_ticks"].sum()) if not trade_log.empty else 0.0
            ),
            **summary,
        }
        latency_rows.append(latency_row)
        prefix = f"test_latency_{latency}"
        trade_log.to_csv(output_dir / f"{prefix}_trade_log.csv", index=False)
        by_symbol.to_csv(output_dir / f"{prefix}_by_symbol.csv", index=False)
        quote_log.to_csv(output_dir / f"{prefix}_quote_log.csv", index=False)
        with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(latency_row, handle, indent=2)

    latency_df = pd.DataFrame(latency_rows).sort_values("latency_steps").reset_index(drop=True)
    latency_df.to_csv(output_dir / "latency_sweep.csv", index=False)

    hardware_profile_path = run_dir / "hardware_quant" / "tcn_hardware_profile.json"
    # Freeze the winning FP32 checkpoint so the hardware handoff and HLS work
    # reference the same model used in the latency proof.
    freeze_manifest = _freeze_model(run_dir, output_dir, latency_df, hardware_profile_path)
    _write_latency_report(output_dir, best_by_symbol, latency_df, freeze_manifest)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "delay_zero_positive": bool(
            not latency_df.loc[latency_df["latency_steps"] == 0].empty
            and float(latency_df.loc[latency_df["latency_steps"] == 0, "net_pnl_ticks"].iloc[0]) > 0
        ),
        "monotonic_decreasing_pnl": bool(latency_df["net_pnl_ticks"].is_monotonic_decreasing),
        "best_by_symbol": best_by_symbol.to_dict(orient="records"),
    }
    with (output_dir / "latency_proof_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
