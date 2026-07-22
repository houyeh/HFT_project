from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fix_common import (
    ASYM_PAIRS,
    BACKTESTS_DIR,
    COMPARISON_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    THRESHOLDS,
    build_processed_frame,
    ensure_fix_dirs,
    import_cfg,
    import_simulator,
    load_model,
    load_new_split_frames,
    matched_random_baseline,
    predict_all_event_probs,
    run_backtest_from_probs,
)


@dataclass
class ModelVersion:
    name: str
    bid_model_dir: Path
    ask_model_dir: Path


def _candidate_pairs() -> list[tuple[float, float]]:
    pairs = [(float(t), float(t)) for t in THRESHOLDS] + [(float(b), float(a)) for b, a in ASYM_PAIRS]
    seen = set()
    out = []
    for b, a in pairs:
        key = (round(b, 6), round(a, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((b, a))
    return out


def _load_versions() -> list[ModelVersion]:
    return [
        ModelVersion(
            name="conditional_only_primary",
            bid_model_dir=OUTPUTS_DIR / "models" / "bid_primary_fw1_h10_posw",
            ask_model_dir=OUTPUTS_DIR / "models" / "ask_primary_fw1_h10_posw",
        ),
        ModelVersion(
            name="all_event_primary",
            bid_model_dir=MODELS_DIR / "bid_all_event_fw1_h10_posw",
            ask_model_dir=MODELS_DIR / "ask_all_event_fw1_h10_posw",
        ),
    ]


def _score(val_net_pnl: float, val_max_drawdown: float, val_net_cancel_value: float) -> float:
    return float(val_net_pnl - 0.2 * abs(val_max_drawdown) + 0.1 * val_net_cancel_value)


def _bool_as_float(v: bool) -> float:
    return 1.0 if bool(v) else 0.0


def main() -> None:
    ensure_fix_dirs()
    simulate_with_metrics = import_simulator()
    cfg = import_cfg()
    split_frames = load_new_split_frames()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_val, tick_val = split_frames["NEW_val"]
    new_test, tick_test = split_frames["NEW_test"]
    new3_full, tick_new3 = split_frames["NEW3_full"]

    # Baseline drawdown constraints reference.
    base_val = simulate_with_metrics(
        cfg,
        prob=None,
        ask_px=new_val["ask_px_1"].to_numpy(dtype=np.float64),
        bid_px=new_val["bid_px_1"].to_numpy(dtype=np.float64),
        tick_size=float(tick_val),
        threshold=1.1,
    )
    base_val_mdd_limit = float(base_val["max_drawdown"]) * 1.2

    rows = []
    selected_rows = []
    pair_list = _candidate_pairs()

    for version in _load_versions():
        if (not version.bid_model_dir.exists()) or (not version.ask_model_dir.exists()):
            print(f"Skip {version.name}: missing model dirs.")
            continue

        bid_model, _ = load_model(version.bid_model_dir, device)
        ask_model, _ = load_model(version.ask_model_dir, device)

        print(f"[{version.name}] predicting val/test/new3...")
        val_bid_probs = predict_all_event_probs(bid_model, new_val, device)
        val_ask_probs = predict_all_event_probs(ask_model, new_val, device)
        test_bid_probs = predict_all_event_probs(bid_model, new_test, device)
        test_ask_probs = predict_all_event_probs(ask_model, new_test, device)
        new3_bid_probs = predict_all_event_probs(bid_model, new3_full, device)
        new3_ask_probs = predict_all_event_probs(ask_model, new3_full, device)

        version_rows = []
        for thr_bid, thr_ask in pair_list:
            vm = run_backtest_from_probs(
                frame=new_val,
                tick_size=tick_val,
                bid_probs=val_bid_probs,
                ask_probs=val_ask_probs,
                thr_bid=thr_bid,
                thr_ask=thr_ask,
                cfg=cfg,
                simulate_with_metrics=simulate_with_metrics,
            )
            c_trade_300 = vm["trade_count"] >= 300
            c_active = vm["active_event_rate"] >= 0.50
            c_bid = vm["bid_signal_rate"] <= 0.50
            c_ask = vm["ask_signal_rate"] <= 0.50
            c_mdd = vm["max_drawdown"] <= base_val_mdd_limit
            caveat = ""
            passed = bool(c_trade_300 and c_active and c_bid and c_ask and c_mdd)
            row = {
                "model_version": version.name,
                "threshold_bid": float(thr_bid),
                "threshold_ask": float(thr_ask),
                "val_net_pnl": float(vm["net_pnl"]),
                "val_trade_count": int(vm["trade_count"]),
                "val_active_event_rate": float(vm["active_event_rate"]),
                "val_bid_signal_rate": float(vm["bid_signal_rate"]),
                "val_ask_signal_rate": float(vm["ask_signal_rate"]),
                "val_max_drawdown": float(vm["max_drawdown"]),
                "val_net_cancel_value": float(vm["net_cancel_value"]),
                "selection_score": float(_score(vm["net_pnl"], vm["max_drawdown"], vm["net_cancel_value"])),
                "selected": 0,
                "selection_caveat": caveat,
                "_passed_strict": passed,
                "_c_trade_300": c_trade_300,
                "_c_active": c_active,
                "_c_bid": c_bid,
                "_c_ask": c_ask,
                "_c_mdd": c_mdd,
            }
            version_rows.append(row)

        vdf = pd.DataFrame(version_rows)
        strict_ok = vdf[vdf["_passed_strict"]].copy()
        selected_caveat = ""
        if strict_ok.empty:
            # Relax trade constraint only.
            relaxed = vdf[
                (vdf["val_trade_count"] >= 150)
                & (vdf["val_active_event_rate"] >= 0.50)
                & (vdf["val_bid_signal_rate"] <= 0.50)
                & (vdf["val_ask_signal_rate"] <= 0.50)
                & (vdf["val_max_drawdown"] <= base_val_mdd_limit)
            ].copy()
            selected_caveat = "LOW_TRADE_COUNT_CAVEAT"
            if relaxed.empty:
                # Fall back to highest selection score, with explicit caveat.
                relaxed = vdf.copy()
                selected_caveat = "NO_CONSTRAINT_SATISFIED"
            chosen_idx = relaxed.sort_values("selection_score", ascending=False).index[0]
        else:
            chosen_idx = strict_ok.sort_values("selection_score", ascending=False).index[0]

        vdf.loc[chosen_idx, "selected"] = 1
        if selected_caveat:
            vdf.loc[chosen_idx, "selection_caveat"] = selected_caveat

        chosen = vdf.loc[chosen_idx]
        thr_bid = float(chosen["threshold_bid"])
        thr_ask = float(chosen["threshold_ask"])

        # Test once on NEW_test and NEW3_full for this selected threshold.
        test_m = run_backtest_from_probs(
            frame=new_test,
            tick_size=tick_test,
            bid_probs=test_bid_probs,
            ask_probs=test_ask_probs,
            thr_bid=thr_bid,
            thr_ask=thr_ask,
            cfg=cfg,
            simulate_with_metrics=simulate_with_metrics,
        )
        new3_m = run_backtest_from_probs(
            frame=new3_full,
            tick_size=tick_new3,
            bid_probs=new3_bid_probs,
            ask_probs=new3_ask_probs,
            thr_bid=thr_bid,
            thr_ask=thr_ask,
            cfg=cfg,
            simulate_with_metrics=simulate_with_metrics,
        )

        rand_new = matched_random_baseline(
            frame=new_test,
            tick_size=tick_test,
            signal_rate_bid=float(test_m["bid_signal_rate"]),
            signal_rate_ask=float(test_m["ask_signal_rate"]),
            cfg=cfg,
            simulate_with_metrics=simulate_with_metrics,
            seeds=20,
        )
        # NEW3 random baseline intentionally skipped (too expensive and less meaningful for selection), set NaN.
        rand_new3 = {"random_mean": np.nan, "random_std": np.nan, "random_mean_plus_1std": np.nan}

        smoothed_new = 84980.0
        smoothed_new3 = -551705.0

        beats_new_mean = bool(test_m["net_pnl"] > rand_new["random_mean"])
        beats_new_plus = bool(test_m["net_pnl"] > rand_new["random_mean_plus_1std"])
        beats_new_smoothed = bool(test_m["net_pnl"] > smoothed_new)
        verdict_new = "not_ready"
        if beats_new_plus and beats_new_smoothed and (test_m["trade_count"] >= 300):
            verdict_new = "replace_smoothed_baseline"
        elif beats_new_mean and (test_m["trade_count"] >= 150):
            verdict_new = "auxiliary_candidate"

        selected_rows.append(
            {
                "dataset": "NEW_test",
                "model_version": version.name,
                "threshold_bid": thr_bid,
                "threshold_ask": thr_ask,
                "net_pnl": float(test_m["net_pnl"]),
                "trade_count": int(test_m["trade_count"]),
                "active_event_rate": float(test_m["active_event_rate"]),
                "bid_signal_rate": float(test_m["bid_signal_rate"]),
                "ask_signal_rate": float(test_m["ask_signal_rate"]),
                "max_drawdown": float(test_m["max_drawdown"]),
                "net_cancel_value": float(test_m["net_cancel_value"]),
                "random_mean": float(rand_new["random_mean"]),
                "random_std": float(rand_new["random_std"]),
                "random_mean_plus_1std": float(rand_new["random_mean_plus_1std"]),
                "beats_random_mean": beats_new_mean,
                "beats_random_plus_1std": beats_new_plus,
                "smoothed_baseline_net_pnl": smoothed_new,
                "beats_smoothed_baseline": beats_new_smoothed,
                "verdict": verdict_new,
            }
        )
        selected_rows.append(
            {
                "dataset": "NEW3_full",
                "model_version": version.name,
                "threshold_bid": thr_bid,
                "threshold_ask": thr_ask,
                "net_pnl": float(new3_m["net_pnl"]),
                "trade_count": int(new3_m["trade_count"]),
                "active_event_rate": float(new3_m["active_event_rate"]),
                "bid_signal_rate": float(new3_m["bid_signal_rate"]),
                "ask_signal_rate": float(new3_m["ask_signal_rate"]),
                "max_drawdown": float(new3_m["max_drawdown"]),
                "net_cancel_value": float(new3_m["net_cancel_value"]),
                "random_mean": rand_new3["random_mean"],
                "random_std": rand_new3["random_std"],
                "random_mean_plus_1std": rand_new3["random_mean_plus_1std"],
                "beats_random_mean": np.nan,
                "beats_random_plus_1std": np.nan,
                "smoothed_baseline_net_pnl": smoothed_new3,
                "beats_smoothed_baseline": bool(new3_m["net_pnl"] > smoothed_new3),
                "verdict": "stress_check_only",
            }
        )

        rows.append(vdf)

        # For all-event model, also emit full threshold summary required by task 6.
        if version.name == "all_event_primary":
            ae_rows = []
            for thr in THRESHOLDS:
                m_new = run_backtest_from_probs(
                    frame=new_test,
                    tick_size=tick_test,
                    bid_probs=test_bid_probs,
                    ask_probs=test_ask_probs,
                    thr_bid=float(thr),
                    thr_ask=float(thr),
                    cfg=cfg,
                    simulate_with_metrics=simulate_with_metrics,
                )
                rand = matched_random_baseline(
                    frame=new_test,
                    tick_size=tick_test,
                    signal_rate_bid=float(m_new["bid_signal_rate"]),
                    signal_rate_ask=float(m_new["ask_signal_rate"]),
                    cfg=cfg,
                    simulate_with_metrics=simulate_with_metrics,
                    seeds=20,
                )
                ae_rows.append(
                    {
                        "dataset": "NEW_test",
                        "model_version": version.name,
                        "threshold_bid": float(thr),
                        "threshold_ask": float(thr),
                        "net_pnl": float(m_new["net_pnl"]),
                        "trade_count": int(m_new["trade_count"]),
                        "active_event_rate": float(m_new["active_event_rate"]),
                        "bid_signal_rate": float(m_new["bid_signal_rate"]),
                        "ask_signal_rate": float(m_new["ask_signal_rate"]),
                        "max_drawdown": float(m_new["max_drawdown"]),
                        "avg_pnl_per_fill": float(m_new["avg_pnl_per_fill"]),
                        "net_cancel_value": float(m_new["net_cancel_value"]),
                        "random_mean": float(rand["random_mean"]),
                        "random_std": float(rand["random_std"]),
                        "random_mean_plus_1std": float(rand["random_mean_plus_1std"]),
                        "beats_random_mean": bool(m_new["net_pnl"] > rand["random_mean"]),
                        "beats_random_plus_1std": bool(m_new["net_pnl"] > rand["random_mean_plus_1std"]),
                        "smoothed_baseline_net_pnl": smoothed_new,
                        "beats_smoothed_baseline": bool(m_new["net_pnl"] > smoothed_new),
                        "notes": "",
                    }
                )
                m_new3 = run_backtest_from_probs(
                    frame=new3_full,
                    tick_size=tick_new3,
                    bid_probs=new3_bid_probs,
                    ask_probs=new3_ask_probs,
                    thr_bid=float(thr),
                    thr_ask=float(thr),
                    cfg=cfg,
                    simulate_with_metrics=simulate_with_metrics,
                )
                ae_rows.append(
                    {
                        "dataset": "NEW3_full",
                        "model_version": version.name,
                        "threshold_bid": float(thr),
                        "threshold_ask": float(thr),
                        "net_pnl": float(m_new3["net_pnl"]),
                        "trade_count": int(m_new3["trade_count"]),
                        "active_event_rate": float(m_new3["active_event_rate"]),
                        "bid_signal_rate": float(m_new3["bid_signal_rate"]),
                        "ask_signal_rate": float(m_new3["ask_signal_rate"]),
                        "max_drawdown": float(m_new3["max_drawdown"]),
                        "avg_pnl_per_fill": float(m_new3["avg_pnl_per_fill"]),
                        "net_cancel_value": float(m_new3["net_cancel_value"]),
                        "random_mean": np.nan,
                        "random_std": np.nan,
                        "random_mean_plus_1std": np.nan,
                        "beats_random_mean": np.nan,
                        "beats_random_plus_1std": np.nan,
                        "smoothed_baseline_net_pnl": smoothed_new3,
                        "beats_smoothed_baseline": bool(m_new3["net_pnl"] > smoothed_new3),
                        "notes": "random baseline not computed for NEW3_full",
                    }
                )
            pd.DataFrame(ae_rows).to_csv(BACKTESTS_DIR / "all_event_binary_backtest_summary.csv", index=False)

    if rows:
        val_df = pd.concat(rows, ignore_index=True)
    else:
        val_df = pd.DataFrame(
            columns=[
                "model_version",
                "threshold_bid",
                "threshold_ask",
                "val_net_pnl",
                "val_trade_count",
                "val_active_event_rate",
                "val_bid_signal_rate",
                "val_ask_signal_rate",
                "val_max_drawdown",
                "val_net_cancel_value",
                "selection_score",
                "selected",
                "selection_caveat",
            ]
        )
    keep_cols = [
        "model_version",
        "threshold_bid",
        "threshold_ask",
        "val_net_pnl",
        "val_trade_count",
        "val_active_event_rate",
        "val_bid_signal_rate",
        "val_ask_signal_rate",
        "val_max_drawdown",
        "val_net_cancel_value",
        "selection_score",
        "selected",
        "selection_caveat",
    ]
    val_df[keep_cols].to_csv(COMPARISON_DIR / "val_threshold_selection_binary.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(BACKTESTS_DIR / "val_selected_binary_test_result.csv", index=False)

    print("Wrote:")
    print(COMPARISON_DIR / "val_threshold_selection_binary.csv")
    print(BACKTESTS_DIR / "val_selected_binary_test_result.csv")
    print(BACKTESTS_DIR / "all_event_binary_backtest_summary.csv")


if __name__ == "__main__":
    main()

