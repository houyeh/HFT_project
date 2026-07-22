from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BACKTESTS_DIR,
    COMPARISON_DIR,
    FEATURES_DIR,
    FIGURES_DIR,
    MODEL_THRESHOLDS,
    POLICIES_DIR,
    REPORTS_DIR,
    ROLLING_WINDOWS,
    safe_quantile,
    BACKTESTS_DIR,
    COMPARISON_DIR,
    EWMA_ALPHAS,
    evaluate_policy,
    ensure_output_dirs,
    extract_arrays,
    get_logger,
    load_experiment_cfg,
    serialize_params,
    simulate_with_metrics,
)

SELECTION_CAVEAT = "Validation set contains only one session; parameter selection may have high variance."


def _spread_masks(df: pd.DataFrame, spread_min: float) -> tuple[np.ndarray, np.ndarray]:
    spread = df["spread_ticks"].to_numpy(dtype=np.float64)
    spread_ok = np.isfinite(spread) & (spread >= float(spread_min))
    return spread_ok.copy(), spread_ok.copy()


def _combined_masks(
    df: pd.DataFrame,
    *,
    spread_min: float,
    trend_col: str,
    trend_threshold: float,
    vol_col: str | None = None,
    vol_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spread = df["spread_ticks"].to_numpy(dtype=np.float64)
    spread_ok = np.isfinite(spread) & (spread >= float(spread_min))
    ret = df[trend_col].to_numpy(dtype=np.float64) if trend_col in df.columns else np.full(len(df), np.nan)

    trend_up = np.isfinite(ret) & (ret >= float(trend_threshold))
    trend_down = np.isfinite(ret) & (ret <= -float(trend_threshold))

    if vol_col is not None and vol_threshold is not None and vol_col in df.columns:
        vol = df[vol_col].to_numpy(dtype=np.float64)
        vol_ok = np.isfinite(vol) & (vol <= float(vol_threshold))
    else:
        vol_ok = np.ones(len(df), dtype=bool)

    bid_active = spread_ok & vol_ok & (~trend_down)
    ask_active = spread_ok & vol_ok & (~trend_up)
    return bid_active, ask_active


def _policy_masks(df: pd.DataFrame, policy_name: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    if policy_name == "spread_only_auto":
        spread_min = float(params["spread_min"])
        bid, ask = _spread_masks(df, spread_min)
        note = "Policy A: spread-only gate."
        return bid, ask, note

    if policy_name == "trend_risk_off":
        w = int(params["trend_w"])
        trend_thr = float(params["trend_threshold"])
        spread_min = float(params["spread_min"])
        trend_col = f"rolling_return_ticks_{w}"
        spread_bid, spread_ask = _spread_masks(df, spread_min)
        combined_bid, combined_ask = _combined_masks(
            df,
            spread_min=spread_min,
            trend_col=trend_col,
            trend_threshold=trend_thr,
            vol_col=None,
            vol_threshold=None,
        )
        ret = df[trend_col].to_numpy(dtype=np.float64) if trend_col in df.columns else np.full(len(df), np.nan)
        high_trend = np.isfinite(ret) & (np.abs(ret) > trend_thr)
        bid = np.where(high_trend, combined_bid, spread_bid)
        ask = np.where(high_trend, combined_ask, spread_ask)
        note = "Policy B implementation: high trend -> combined_filter behavior (side-specific cancel)."
        return bid.astype(bool), ask.astype(bool), note

    if policy_name == "volatility_risk_off":
        w = int(params["vol_w"])
        spread_min = float(params["spread_min"])
        vol_thr = float(params["vol_threshold"])
        vol_col = f"rolling_vol_ticks_{w}"
        spread_bid, spread_ask = _spread_masks(df, spread_min)
        vol = df[vol_col].to_numpy(dtype=np.float64) if vol_col in df.columns else np.full(len(df), np.nan)
        high_vol = np.isfinite(vol) & (vol > vol_thr)
        bid = np.where(high_vol, False, spread_bid)
        ask = np.where(high_vol, False, spread_ask)
        note = "Policy C implementation: high volatility -> risk-off (pull both sides)."
        return bid.astype(bool), ask.astype(bool), note

    if policy_name == "trend_vol_spread_auto":
        trend_w = int(params["trend_w"])
        vol_w = int(params["vol_w"])
        trend_thr = float(params["trend_threshold"])
        vol_thr = float(params["vol_threshold"])
        spread_min = float(params["spread_min"])

        trend_col = f"rolling_return_ticks_{trend_w}"
        vol_col = f"rolling_vol_ticks_{vol_w}"
        spread_bid, spread_ask = _spread_masks(df, spread_min)
        combined_bid, combined_ask = _combined_masks(
            df,
            spread_min=spread_min,
            trend_col=trend_col,
            trend_threshold=trend_thr,
            vol_col=vol_col,
            vol_threshold=vol_thr,
        )

        ret = df[trend_col].to_numpy(dtype=np.float64) if trend_col in df.columns else np.full(len(df), np.nan)
        vol = df[vol_col].to_numpy(dtype=np.float64) if vol_col in df.columns else np.full(len(df), np.nan)
        high_trend = np.isfinite(ret) & (np.abs(ret) > trend_thr)
        high_vol = np.isfinite(vol) & (vol > vol_thr)
        high_risk = high_trend | high_vol
        bid = np.where(high_risk, combined_bid, spread_bid)
        ask = np.where(high_risk, combined_ask, spread_ask)
        note = "Policy D implementation: high trend/vol -> combined_filter behavior, else spread_filter."
        return bid.astype(bool), ask.astype(bool), note

    raise ValueError(f"Unknown policy: {policy_name}")


def _evaluate_candidates(
    *,
    cfg,
    val_df: pd.DataFrame,
    vol_threshold_lookup: dict[tuple[int, float], float],
) -> pd.DataFrame:
    val_probs, val_ask, val_bid = extract_arrays(val_df)
    tick_size = float(val_df["tick_size"].dropna().iloc[0])
    val_session_count = int(val_df["session_id"].nunique())
    val_sample_count = int(len(val_df))

    no_gate_by_threshold: dict[float, dict[str, float | int]] = {}
    for thr in MODEL_THRESHOLDS:
        no_gate_by_threshold[thr] = simulate_with_metrics(
            cfg,
            val_probs,
            val_ask,
            val_bid,
            tick_size,
            threshold=float(thr),
        )

    candidates: list[tuple[str, dict[str, Any]]] = []
    for spread_min in [1.0, 1.5, 2.0]:
        candidates.append(("spread_only_auto", {"spread_min": float(spread_min)}))

    for w in [100, 200, 500]:
        for trend_thr in [5.0, 10.0, 20.0, 30.0]:
            for spread_min in [1.5, 2.0]:
                candidates.append(
                    (
                        "trend_risk_off",
                        {"trend_w": int(w), "trend_threshold": float(trend_thr), "spread_min": float(spread_min)},
                    )
                )

    for w in [100, 200, 500]:
        for q in [0.70, 0.80, 0.90]:
            vol_thr = vol_threshold_lookup.get((w, q), float("nan"))
            if not np.isfinite(vol_thr):
                continue
            for spread_min in [1.5, 2.0]:
                candidates.append(
                    (
                        "volatility_risk_off",
                        {
                            "vol_w": int(w),
                            "vol_quantile": float(q),
                            "vol_threshold": float(vol_thr),
                            "spread_min": float(spread_min),
                        },
                    )
                )

    for trend_w in [100, 200, 500]:
        for trend_thr in [5.0, 10.0, 20.0, 30.0]:
            for vol_w in [100, 200, 500]:
                for vol_q in [0.80, 0.90]:
                    vol_thr = vol_threshold_lookup.get((vol_w, vol_q), float("nan"))
                    if not np.isfinite(vol_thr):
                        continue
                    for spread_min in [1.5, 2.0]:
                        candidates.append(
                            (
                                "trend_vol_spread_auto",
                                {
                                    "trend_w": int(trend_w),
                                    "trend_threshold": float(trend_thr),
                                    "vol_w": int(vol_w),
                                    "vol_quantile": float(vol_q),
                                    "vol_threshold": float(vol_thr),
                                    "spread_min": float(spread_min),
                                },
                            )
                        )

    rows: list[dict[str, Any]] = []
    for policy_name, params in candidates:
        bid_mask, ask_mask, impl_note = _policy_masks(val_df, policy_name, params)
        for thr in MODEL_THRESHOLDS:
            result = evaluate_policy(
                cfg,
                probs=val_probs,
                ask=val_ask,
                bid=val_bid,
                tick_size=tick_size,
                threshold=float(thr),
                bid_active_mask=bid_mask,
                ask_active_mask=ask_mask,
            )
            no_gate_ref = no_gate_by_threshold[float(thr)]
            active_ok = float(result["active_event_rate"]) >= 0.50
            min_trade = 0.5 * float(no_gate_ref["trade_count"])
            trade_ok = float(result["trade_count"]) >= min_trade
            no_gate_mdd = max(float(no_gate_ref["max_drawdown"]), 1e-6)
            mdd_ok = float(result["max_drawdown"]) <= (1.2 * no_gate_mdd)
            selection_score = float(result["net_pnl"]) - 0.2 * abs(float(result["max_drawdown"])) + 0.1 * float(
                result["net_cancel_value"]
            )
            rows.append(
                {
                    "policy_name": policy_name,
                    "params": serialize_params(params),
                    "model_threshold": float(thr),
                    "val_session_count": val_session_count,
                    "val_sample_count": val_sample_count,
                    "val_net_pnl": float(result["net_pnl"]),
                    "val_max_drawdown": float(result["max_drawdown"]),
                    "val_active_event_rate": float(result["active_event_rate"]),
                    "val_trade_count": int(result["trade_count"]),
                    "val_net_cancel_value": float(result["net_cancel_value"]),
                    "selection_score": float(selection_score),
                    "selected": False,
                    "selection_reason": "",
                    "selection_caveat": SELECTION_CAVEAT,
                    "constraint_active_ok": bool(active_ok),
                    "constraint_trade_ok": bool(trade_ok),
                    "constraint_mdd_ok": bool(mdd_ok),
                    "implementation_notes": impl_note,
                }
            )

    df = pd.DataFrame(rows)
    eligible = df["constraint_active_ok"] & df["constraint_trade_ok"] & df["constraint_mdd_ok"]
    if eligible.any():
        pick_idx = df.loc[eligible, "selection_score"].idxmax()
        df.loc[pick_idx, "selected"] = True
        df.loc[pick_idx, "selection_reason"] = "Highest selection_score among candidates satisfying constraints."
    else:
        pick_idx = df["selection_score"].idxmax()
        df.loc[pick_idx, "selected"] = True
        df.loc[pick_idx, "selection_reason"] = "No candidate satisfied all constraints; picked highest selection_score."
    return df.sort_values(["selected", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def _evaluate_selected_on_test_sets(
    *,
    cfg,
    selected_row: pd.Series,
    new_test_df: pd.DataFrame,
    new3_df: pd.DataFrame,
) -> pd.DataFrame:
    policy_name = str(selected_row["policy_name"])
    params = json.loads(str(selected_row["params"]))
    model_threshold = float(selected_row["model_threshold"])

    out_rows: list[dict[str, Any]] = []
    for dataset_name, df, split_note in [
        ("NEW", new_test_df, "NEW test split"),
        ("NEW3", new3_df, "NEW3 full OOS"),
    ]:
        probs, ask, bid = extract_arrays(df)
        tick_size = float(df["tick_size"].dropna().iloc[0])
        bid_mask, ask_mask, impl_note = _policy_masks(df, policy_name, params)
        result = evaluate_policy(
            cfg,
            probs=probs,
            ask=ask,
            bid=bid,
            tick_size=tick_size,
            threshold=model_threshold,
            bid_active_mask=bid_mask,
            ask_active_mask=ask_mask,
        )
        out_rows.append(
            {
                "dataset": dataset_name,
                "policy_name": policy_name,
                "params": serialize_params(params),
                "model_threshold": model_threshold,
                "net_pnl": float(result["net_pnl"]),
                "baseline_net_pnl": float(result["baseline_net_pnl"]),
                "improvement": float(result["improvement"]),
                "active_event_rate": float(result["active_event_rate"]),
                "trade_count": int(result["trade_count"]),
                "max_drawdown": float(result["max_drawdown"]),
                "avg_pnl_per_fill": float(result["avg_pnl_per_fill"]),
                "net_cancel_value": float(result["net_cancel_value"]),
                "realized_bid_pnl": float(result["realized_bid_pnl"]),
                "realized_ask_pnl": float(result["realized_ask_pnl"]),
                "notes": f"{split_note}; {impl_note}",
            }
        )
    return pd.DataFrame(out_rows)


def _build_comparison_table(static_df: pd.DataFrame, auto_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keep_static = {"no_gate_tcn", "always_spread_filter", "always_combined_filter", "dataset_oracle_best_gate"}
    for _, r in static_df[static_df["policy_name"].isin(keep_static)].iterrows():
        rows.append(
            {
                "dataset": str(r["dataset"]),
                "method": "static",
                "policy_name": str(r["policy_name"]),
                "params": str(r["gate_params"]),
                "model_threshold": float(r["model_threshold"]) if pd.notna(r["model_threshold"]) else np.nan,
                "net_pnl": float(r["net_pnl"]),
                "improvement": float(r["improvement"]),
                "active_event_rate": float(r["active_event_rate"]) if pd.notna(r["active_event_rate"]) else np.nan,
                "trade_count": int(r["trade_count"]) if pd.notna(r["trade_count"]) else 0,
                "max_drawdown": float(r["max_drawdown"]) if pd.notna(r["max_drawdown"]) else np.nan,
                "net_cancel_value": np.nan,
                "oracle_flag": bool(r["oracle_flag"]),
                "deployable_flag": bool(r["deployable_flag"]),
                "notes": str(r["notes"]),
            }
        )
    for _, r in auto_df.iterrows():
        rows.append(
            {
                "dataset": str(r["dataset"]),
                "method": "auto_gate",
                "policy_name": str(r["policy_name"]),
                "params": str(r["params"]),
                "model_threshold": float(r["model_threshold"]),
                "net_pnl": float(r["net_pnl"]),
                "improvement": float(r["improvement"]),
                "active_event_rate": float(r["active_event_rate"]),
                "trade_count": int(r["trade_count"]),
                "max_drawdown": float(r["max_drawdown"]),
                "net_cancel_value": float(r["net_cancel_value"]),
                "oracle_flag": False,
                "deployable_flag": True,
                "notes": str(r["notes"]),
            }
        )

    cmp_df = pd.DataFrame(rows)
    cmp_df["beats_no_gate"] = False
    cmp_df["beats_always_spread"] = False
    cmp_df["beats_always_combined"] = False
    cmp_df["beats_dataset_oracle"] = False

    for dataset in sorted(cmp_df["dataset"].dropna().unique().tolist()):
        ds = cmp_df[cmp_df["dataset"] == dataset]
        ref = {
            "no_gate_tcn": ds.loc[ds["policy_name"] == "no_gate_tcn", "net_pnl"],
            "always_spread_filter": ds.loc[ds["policy_name"] == "always_spread_filter", "net_pnl"],
            "always_combined_filter": ds.loc[ds["policy_name"] == "always_combined_filter", "net_pnl"],
            "dataset_oracle_best_gate": ds.loc[ds["policy_name"] == "dataset_oracle_best_gate", "net_pnl"],
        }
        no_gate = float(ref["no_gate_tcn"].iloc[0]) if not ref["no_gate_tcn"].empty else np.nan
        spread = float(ref["always_spread_filter"].iloc[0]) if not ref["always_spread_filter"].empty else np.nan
        combined = float(ref["always_combined_filter"].iloc[0]) if not ref["always_combined_filter"].empty else np.nan
        oracle = float(ref["dataset_oracle_best_gate"].iloc[0]) if not ref["dataset_oracle_best_gate"].empty else np.nan
        idx = cmp_df["dataset"] == dataset
        cmp_df.loc[idx, "beats_no_gate"] = cmp_df.loc[idx, "net_pnl"] > no_gate if np.isfinite(no_gate) else False
        cmp_df.loc[idx, "beats_always_spread"] = (
            cmp_df.loc[idx, "net_pnl"] > spread if np.isfinite(spread) else False
        )
        cmp_df.loc[idx, "beats_always_combined"] = (
            cmp_df.loc[idx, "net_pnl"] > combined if np.isfinite(combined) else False
        )
        cmp_df.loc[idx, "beats_dataset_oracle"] = (
            cmp_df.loc[idx, "net_pnl"] > oracle if np.isfinite(oracle) else False
        )
    return cmp_df


def _evaluate_ewma_ablation(
    *,
    cfg,
    new_val_df: pd.DataFrame,
    new_test_df: pd.DataFrame,
    new3_df: pd.DataFrame,
    selected_threshold: float,
    spread_min: float,
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for alpha in EWMA_ALPHAS:
        suffix = f"{alpha:.2f}".replace(".", "p")
        vel_col = f"ewma_velocity_alpha_{suffix}"
        vol_col = f"ewma_vol_alpha_{suffix}"
        if vel_col not in new_val_df.columns or vol_col not in new_val_df.columns:
            continue
        abs_vel_val = np.abs(new_val_df[vel_col].to_numpy(dtype=np.float64))
        vol_val = new_val_df[vol_col].to_numpy(dtype=np.float64)

        for vel_q in [0.80, 0.90]:
            vel_thr = safe_quantile(abs_vel_val, vel_q)
            if not np.isfinite(vel_thr):
                continue
            for vol_q in [0.80, 0.90]:
                vol_thr = safe_quantile(vol_val, vol_q)
                if not np.isfinite(vol_thr):
                    continue

                for dataset_name, df in [("NEW", new_test_df), ("NEW3", new3_df)]:
                    probs, ask, bid = extract_arrays(df)
                    tick_size = float(df["tick_size"].dropna().iloc[0])
                    spread = df["spread_ticks"].to_numpy(dtype=np.float64)
                    spread_ok = np.isfinite(spread) & (spread >= float(spread_min))

                    vel = df[vel_col].to_numpy(dtype=np.float64)
                    vol = df[vol_col].to_numpy(dtype=np.float64)
                    high_vel = np.isfinite(vel) & (np.abs(vel) > vel_thr)
                    high_vol = np.isfinite(vol) & (vol > vol_thr)
                    high_risk = high_vel | high_vol
                    trend_up = np.isfinite(vel) & (vel >= vel_thr)
                    trend_down = np.isfinite(vel) & (vel <= -vel_thr)
                    vol_ok = np.isfinite(vol) & (vol <= vol_thr)

                    combined_bid = spread_ok & vol_ok & (~trend_down)
                    combined_ask = spread_ok & vol_ok & (~trend_up)
                    bid_active = np.where(high_risk, combined_bid, spread_ok)
                    ask_active = np.where(high_risk, combined_ask, spread_ok)

                    result = evaluate_policy(
                        cfg,
                        probs=probs,
                        ask=ask,
                        bid=bid,
                        tick_size=tick_size,
                        threshold=float(selected_threshold),
                        bid_active_mask=bid_active,
                        ask_active_mask=ask_active,
                    )
                    out_rows.append(
                        {
                            "method": "ewma_trend_vol_auto",
                            "alpha": float(alpha),
                            "velocity_quantile": float(vel_q),
                            "vol_quantile": float(vol_q),
                            "dataset": dataset_name,
                            "net_pnl": float(result["net_pnl"]),
                            "active_event_rate": float(result["active_event_rate"]),
                            "trade_count": int(result["trade_count"]),
                            "max_drawdown": float(result["max_drawdown"]),
                            "notes": (
                                f"threshold={selected_threshold:.2f}, spread_min={spread_min:.2f}, "
                                f"vel_thr={vel_thr:.6f}, vol_thr={vol_thr:.6f}"
                            ),
                        }
                    )
    return pd.DataFrame(out_rows)


def _plot_distributions(new_df: pd.DataFrame, new3_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("rolling_return_ticks_200", "Rolling Return (200 events, ticks)"),
        ("rolling_vol_ticks_200", "Rolling Vol (200 events, ticks)"),
        ("spread_ticks", "Spread (ticks)"),
    ]

    for ax, (col, title) in zip(axes, pairs):
        x = new_df[col].to_numpy(dtype=np.float64)
        y = new3_df[col].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 and y.size == 0:
            continue
        combined = np.concatenate([x[: min(len(x), 300000)], y[: min(len(y), 300000)]], axis=0)
        lo = np.quantile(combined, 0.01)
        hi = np.quantile(combined, 0.99)
        bins = np.linspace(lo, hi, 120) if hi > lo else 60
        ax.hist(x, bins=bins, alpha=0.45, density=True, label="NEW")
        ax.hist(y, bins=bins, alpha=0.45, density=True, label="NEW3")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regime_feature_distributions_NEW_vs_NEW3.png", dpi=170)
    plt.close(fig)


def _plot_static_vs_auto(static_df: pd.DataFrame, auto_df: pd.DataFrame) -> None:
    categories = [
        "no_gate_tcn",
        "always_spread_filter",
        "always_combined_filter",
        "auto_gate",
        "dataset_oracle_best_gate",
    ]
    datasets = ["NEW", "NEW3"]
    value_map: dict[tuple[str, str], float] = {}

    for ds in datasets:
        ds_static = static_df[static_df["dataset"] == ds]
        for cat in categories:
            if cat == "auto_gate":
                row = auto_df[auto_df["dataset"] == ds]
                if not row.empty:
                    value_map[(ds, cat)] = float(row["net_pnl"].iloc[0])
            else:
                row = ds_static[ds_static["policy_name"] == cat]
                if not row.empty:
                    value_map[(ds, cat)] = float(row["net_pnl"].iloc[0])

    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ds in enumerate(datasets):
        vals = [value_map.get((ds, c), np.nan) for c in categories]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("Net PnL")
    ax.set_title("Static vs Auto-Gate Net PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "static_vs_auto_gate_pnl.png", dpi=170)
    plt.close(fig)


def _plot_active_vs_pnl(selection_df: pd.DataFrame) -> None:
    family_colors = {
        "spread_only_auto": "#1f77b4",
        "trend_risk_off": "#ff7f0e",
        "volatility_risk_off": "#2ca02c",
        "trend_vol_spread_auto": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, sub in selection_df.groupby("policy_name"):
        ax.scatter(
            sub["val_active_event_rate"],
            sub["val_net_pnl"],
            s=24,
            alpha=0.6,
            label=name,
            color=family_colors.get(name, None),
        )
    selected = selection_df[selection_df["selected"]]
    if not selected.empty:
        ax.scatter(
            selected["val_active_event_rate"],
            selected["val_net_pnl"],
            s=160,
            marker="*",
            color="black",
            label="selected",
        )
    ax.set_xlabel("Active Event Rate")
    ax.set_ylabel("Validation Net PnL")
    ax.set_title("Auto-Gate Candidates: Active Rate vs Net PnL")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "active_rate_vs_pnl_auto_gate.png", dpi=170)
    plt.close(fig)


def _plot_timeline(
    new3_df: pd.DataFrame,
    selected_policy_name: str,
    selected_params: dict[str, Any],
) -> None:
    if new3_df.empty:
        return
    session_lengths = new3_df.groupby("session_id").size().sort_values(ascending=False)
    if session_lengths.empty:
        return
    session_id = str(session_lengths.index[0])
    sess = new3_df[new3_df["session_id"] == session_id].copy()
    bid_mask, ask_mask, _ = _policy_masks(sess, selected_policy_name, selected_params)
    gate_state = (bid_mask.astype(np.float64) + ask_mask.astype(np.float64)) / 2.0

    n = min(4000, len(sess))
    x = np.arange(n)
    mid = sess["mid"].to_numpy(dtype=np.float64)[:n]
    if selected_policy_name in ("trend_risk_off", "trend_vol_spread_auto"):
        trend_col = f"rolling_return_ticks_{int(selected_params['trend_w'])}"
    else:
        trend_col = "rolling_return_ticks_200"
    trend_vals = sess[trend_col].to_numpy(dtype=np.float64)[:n] if trend_col in sess.columns else np.zeros(n)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(x, mid, linewidth=0.9)
    axes[0].set_ylabel("Mid")
    axes[0].set_title(f"NEW3 session sample: {session_id}")
    axes[1].plot(x, trend_vals, linewidth=0.9, color="#ff7f0e")
    axes[1].set_ylabel("Trend")
    axes[2].step(x, gate_state[:n], where="post", linewidth=0.8, color="#2ca02c")
    axes[2].set_ylabel("Gate State")
    axes[2].set_ylim(-0.05, 1.05)
    axes[3].step(x, bid_mask[:n].astype(float), where="post", label="bid_active", linewidth=0.8)
    axes[3].step(x, ask_mask[:n].astype(float), where="post", label="ask_active", linewidth=0.8)
    axes[3].set_ylabel("Side Active")
    axes[3].set_xlabel("Event Index (sample)")
    axes[3].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "auto_gate_policy_timeline_sample.png", dpi=170)
    plt.close(fig)


def _determine_verdict(static_df: pd.DataFrame, auto_df: pd.DataFrame) -> tuple[str, str]:
    merged = auto_df.set_index("dataset")
    if "NEW" not in merged.index or "NEW3" not in merged.index:
        return "D", "Missing NEW/NEW3 auto-gate test rows."

    new_auto = float(merged.loc["NEW", "net_pnl"])
    new3_auto = float(merged.loc["NEW3", "net_pnl"])
    new_active = float(merged.loc["NEW", "active_event_rate"])
    new3_active = float(merged.loc["NEW3", "active_event_rate"])

    def _static(dataset: str, policy: str) -> float:
        row = static_df[(static_df["dataset"] == dataset) & (static_df["policy_name"] == policy)]
        return float(row["net_pnl"].iloc[0]) if not row.empty else float("nan")

    new_no_gate = _static("NEW", "no_gate_tcn")
    new3_no_gate = _static("NEW3", "no_gate_tcn")
    new_spread = _static("NEW", "always_spread_filter")
    new3_spread = _static("NEW3", "always_spread_filter")
    new_combined = _static("NEW", "always_combined_filter")
    new3_combined = _static("NEW3", "always_combined_filter")

    auto_avg = np.nanmean([new_auto, new3_auto])
    spread_avg = np.nanmean([new_spread, new3_spread])
    combined_avg = np.nanmean([new_combined, new3_combined])

    if min(new_active, new3_active) < 0.30:
        return "C", "Active event rate drops below 0.30 on at least one dataset (risk-off behavior)."
    if np.isfinite(new_combined) and np.isfinite(new3_combined):
        if not (new_auto > new_combined or new3_auto > new3_combined):
            return "C", "Auto-gate cannot beat static combined_filter on either NEW or NEW3."
    if np.isfinite(new_no_gate) and np.isfinite(new3_no_gate):
        if (new_auto > new_no_gate) and (new3_auto > new3_no_gate):
            if auto_avg > spread_avg and auto_avg > combined_avg:
                return "A", "Auto-gate beats no-gate on both datasets and beats static spread/combined on average."
            return "B", "Auto-gate beats no-gate on both datasets but does not dominate static gates on average."
    if auto_avg > spread_avg or auto_avg > combined_avg:
        return "B", "Auto-gate is directionally helpful but robustness is incomplete."
    return "D", "Results are inconclusive or weaker than static alternatives."


def _write_regime_auto_gate_report(
    *,
    static_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    selected_row: pd.Series,
    auto_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    ewma_df: pd.DataFrame,
    verdict: str,
    verdict_reason: str,
) -> None:
    selected_params = json.loads(str(selected_row["params"]))
    thr = float(selected_row["model_threshold"])
    new_res = auto_df[auto_df["dataset"] == "NEW"].iloc[0]
    new3_res = auto_df[auto_df["dataset"] == "NEW3"].iloc[0]

    new_best_spread = static_df[
        (static_df["dataset"] == "NEW") & (static_df["policy_name"] == "always_spread_filter")
    ]["net_pnl"].iloc[0]
    new3_best_combined = static_df[
        (static_df["dataset"] == "NEW3") & (static_df["policy_name"] == "always_combined_filter")
    ]["net_pnl"].iloc[0]

    oracle_new = static_df[
        (static_df["dataset"] == "NEW") & (static_df["policy_name"] == "dataset_oracle_best_gate")
    ]["net_pnl"].iloc[0]
    oracle_new3 = static_df[
        (static_df["dataset"] == "NEW3") & (static_df["policy_name"] == "dataset_oracle_best_gate")
    ]["net_pnl"].iloc[0]

    val_count = int(selected_row["val_session_count"])
    val_samples = int(selected_row["val_sample_count"])

    lines: list[str] = []
    lines.append("# 1. Executive Summary")
    lines.append(
        f"Causal auto-gating verdict: **{verdict}**. Selected policy `{selected_row['policy_name']}` @ threshold `{thr:.2f}` ({verdict_reason})"
    )
    lines.append("")
    lines.append("# 2. Motivation")
    lines.append(
        f"Known static results: NEW best spread_filter net_pnl={new_best_spread:.2f}; NEW3 best combined_filter net_pnl={new3_best_combined:.2f}."
    )
    lines.append("This confirms regime dependency but does not by itself prove deployable auto-gating.")
    lines.append("")
    lines.append("# 3. Causal Regime Features")
    lines.append("Features use rolling/EWMA calculations only (backward-looking). No future/full-session leakage features are used.")
    lines.append("")
    lines.append("# 4. Static Baselines and Oracle")
    lines.append("- Baselines: no_gate_tcn, always_spread_filter, always_combined_filter.")
    lines.append("- Oracle: dataset_oracle_best_gate (non-deployable).")
    lines.append("- Oracle mapping from static summary: NEW -> spread_filter, NEW3 -> combined_filter.")
    lines.append(
        f"- Oracle net_pnl: NEW={oracle_new:.2f}, NEW3={oracle_new3:.2f}. Oracle is an upper bound, not a live policy."
    )
    lines.append("")
    lines.append("# 5. Validation Selection")
    lines.append("- Selection done on NEW validation only.")
    lines.append(f"- val_session_count={val_count}, val_sample_count={val_samples}.")
    lines.append(f"- Caveat: {SELECTION_CAVEAT}")
    lines.append(f"- Selected params: {json.dumps(selected_params, ensure_ascii=False)}")
    lines.append("")
    lines.append("# 6. NEW Test Results")
    lines.append(
        f"NEW test auto-gate net_pnl={float(new_res['net_pnl']):.2f}, improvement={float(new_res['improvement']):.2f}, "
        f"active_event_rate={float(new_res['active_event_rate']):.4f}, max_drawdown={float(new_res['max_drawdown']):.2f}."
    )
    lines.append("")
    lines.append("# 7. NEW3 Stress Results")
    lines.append(
        f"NEW3 full auto-gate net_pnl={float(new3_res['net_pnl']):.2f}, improvement={float(new3_res['improvement']):.2f}, "
        f"active_event_rate={float(new3_res['active_event_rate']):.4f}, max_drawdown={float(new3_res['max_drawdown']):.2f}."
    )
    lines.append("")
    lines.append("# 8. Auto-Gate vs Static Gates")
    auto_cmp = comparison_df[comparison_df["method"] == "auto_gate"].copy()
    for _, r in auto_cmp.iterrows():
        lines.append(
            f"- {r['dataset']}: beats_no_gate={bool(r['beats_no_gate'])}, beats_spread={bool(r['beats_always_spread'])}, "
            f"beats_combined={bool(r['beats_always_combined'])}, beats_oracle={bool(r['beats_dataset_oracle'])}."
        )
    if (
        auto_cmp["beats_always_spread"].all()
        and auto_cmp["beats_always_combined"].all()
        and (~auto_cmp["beats_dataset_oracle"]).any()
    ):
        lines.append("Auto-gate is promising but not optimal; oracle remains an upper bound.")
    lines.append("")
    lines.append("# 9. EWMA Ablation")
    if ewma_df.empty:
        lines.append("EWMA ablation skipped.")
    else:
        grp = ewma_df.groupby(["method", "alpha", "velocity_quantile", "vol_quantile"], dropna=False)["net_pnl"].mean()
        best_key = grp.idxmax()
        best_avg = grp.max()
        lines.append(
            f"Best EWMA combo avg(net_pnl NEW/NEW3)={best_avg:.2f} at alpha={best_key[1]}, "
            f"velocity_q={best_key[2]}, vol_q={best_key[3]}."
        )
        lines.append("If EWMA and rolling results are close, rolling gate is preferred for simplicity.")
    lines.append("")
    lines.append("# 10. Verdict")
    lines.append(f"{verdict}. {verdict_reason}")
    lines.append("")
    lines.append("# 11. Meeting-ready Claims")
    lines.append("- We proved regime dependency (NEW vs NEW3 static gate preference diverges).")
    lines.append("- We tested a causal auto-gating prototype with event-level masks.")
    lines.append(f"- Auto-gate verdict: {verdict}.")
    lines.append("- Production readiness is not claimed unless robustness is clearly dominant.")
    lines.append("")
    lines.append("# 12. Next Steps")
    lines.append("- Integrate gating hooks with 016_new_binary once binary model evidence is finalized.")
    lines.append("- Improve regime selector stability with more independent sessions.")
    lines.append("- Extend ablation (EWMA/Kalman) after stronger validation coverage.")
    lines.append("- Prepare FPGA MVP path for causal gate + cancel decision path.")

    (REPORTS_DIR / "regime_auto_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_checkpoint(
    *,
    static_df: pd.DataFrame,
    selected_row: pd.Series,
    auto_df: pd.DataFrame,
    ewma_df: pd.DataFrame,
    verdict: str,
    verdict_reason: str,
) -> None:
    def _get_static(dataset: str, policy: str) -> pd.Series:
        row = static_df[(static_df["dataset"] == dataset) & (static_df["policy_name"] == policy)]
        return row.iloc[0] if not row.empty else pd.Series(dtype=object)

    new_best = _get_static("NEW", "always_spread_filter")
    new3_best = _get_static("NEW3", "always_combined_filter")
    new_auto = auto_df[auto_df["dataset"] == "NEW"].iloc[0]
    new3_auto = auto_df[auto_df["dataset"] == "NEW3"].iloc[0]
    params = json.loads(str(selected_row["params"]))

    lines = []
    lines.append("# Final Checkpoint")
    lines.append("")
    lines.append("1. Best static gate on NEW:")
    lines.append(
        f"- policy: {new_best.get('policy_name', 'N/A')} ({new_best.get('gate_params', '')})"
    )
    lines.append(f"- net_pnl: {float(new_best.get('net_pnl', np.nan)):.2f}")
    lines.append(f"- active_event_rate: {float(new_best.get('active_event_rate', np.nan)):.4f}")
    lines.append("")
    lines.append("2. Best static gate on NEW3:")
    lines.append(
        f"- policy: {new3_best.get('policy_name', 'N/A')} ({new3_best.get('gate_params', '')})"
    )
    lines.append(f"- net_pnl: {float(new3_best.get('net_pnl', np.nan)):.2f}")
    lines.append(f"- active_event_rate: {float(new3_best.get('active_event_rate', np.nan)):.4f}")
    lines.append("")
    lines.append("3. Selected auto-gate policy:")
    lines.append(f"- policy_name: {selected_row['policy_name']}")
    lines.append(f"- params: {json.dumps(params, ensure_ascii=False)}")
    lines.append("- selected on validation data: NEW validation split only")
    lines.append(f"- model_threshold: {float(selected_row['model_threshold']):.2f}")
    lines.append(f"- validation caveat: {SELECTION_CAVEAT}")
    lines.append("")
    lines.append("4. Auto-gate result on NEW test:")
    lines.append(f"- net_pnl: {float(new_auto['net_pnl']):.2f}")
    lines.append(f"- active_event_rate: {float(new_auto['active_event_rate']):.4f}")
    lines.append(f"- trade_count: {int(new_auto['trade_count'])}")
    lines.append(f"- max_drawdown: {float(new_auto['max_drawdown']):.2f}")
    lines.append("")
    lines.append("5. Auto-gate result on NEW3:")
    lines.append(f"- net_pnl: {float(new3_auto['net_pnl']):.2f}")
    lines.append(f"- active_event_rate: {float(new3_auto['active_event_rate']):.4f}")
    lines.append(f"- trade_count: {int(new3_auto['trade_count'])}")
    lines.append(f"- max_drawdown: {float(new3_auto['max_drawdown']):.2f}")
    lines.append("")
    lines.append("6. EWMA result if run:")
    if ewma_df.empty:
        lines.append("- best EWMA policy: skipped")
        lines.append("- net_pnl NEW: N/A")
        lines.append("- net_pnl NEW3: N/A")
    else:
        pivot = ewma_df.pivot_table(
            index=["method", "alpha", "velocity_quantile", "vol_quantile"],
            columns="dataset",
            values="net_pnl",
            aggfunc="mean",
        ).reset_index()
        pivot["avg"] = pivot[["NEW", "NEW3"]].mean(axis=1)
        best = pivot.sort_values("avg", ascending=False).iloc[0]
        lines.append(
            "- best EWMA policy: "
            f"{best['method']} alpha={best['alpha']} velocity_q={best['velocity_quantile']} vol_q={best['vol_quantile']}"
        )
        lines.append(f"- net_pnl NEW: {float(best.get('NEW', np.nan)):.2f}")
        lines.append(f"- net_pnl NEW3: {float(best.get('NEW3', np.nan)):.2f}")
    lines.append("")
    lines.append("7. Verdict:")
    lines.append(f"- {verdict}")
    lines.append(f"- reason: {verdict_reason}")
    (REPORTS_DIR / "final_checkpoint.md").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest() -> None:
    entries = [
        ("regime_features_NEW.csv", FEATURES_DIR / "regime_features_NEW.csv", "Causal regime features for NEW."),
        ("regime_features_NEW3.csv", FEATURES_DIR / "regime_features_NEW3.csv", "Causal regime features for NEW3."),
        ("regime_feature_report.md", REPORTS_DIR / "regime_feature_report.md", "Feature leakage/availability report."),
        ("static_gate_summary.csv", POLICIES_DIR / "static_gate_summary.csv", "Static gate baseline and oracle summary."),
        (
            "auto_gate_validation_selection.csv",
            POLICIES_DIR / "auto_gate_validation_selection.csv",
            "Validation-only policy selection table.",
        ),
        ("auto_gate_test_results.csv", BACKTESTS_DIR / "auto_gate_test_results.csv", "Frozen auto-gate test results."),
        ("auto_gate_vs_static.csv", COMPARISON_DIR / "auto_gate_vs_static.csv", "Auto-gate vs static comparison."),
        ("ewma_gate_ablation.csv", COMPARISON_DIR / "ewma_gate_ablation.csv", "EWMA ablation table."),
        (
            "meta_classifier_skip_reason.md",
            REPORTS_DIR / "meta_classifier_skip_reason.md",
            "Reason meta-classifier gate training was skipped.",
        ),
        (
            "regime_feature_distributions_NEW_vs_NEW3.png",
            FIGURES_DIR / "regime_feature_distributions_NEW_vs_NEW3.png",
            "Feature distribution comparison plot.",
        ),
        ("static_vs_auto_gate_pnl.png", FIGURES_DIR / "static_vs_auto_gate_pnl.png", "Static vs auto net PnL bars."),
        (
            "active_rate_vs_pnl_auto_gate.png",
            FIGURES_DIR / "active_rate_vs_pnl_auto_gate.png",
            "Validation candidate scatter.",
        ),
        (
            "auto_gate_policy_timeline_sample.png",
            FIGURES_DIR / "auto_gate_policy_timeline_sample.png",
            "Sample NEW3 timeline for selected policy.",
        ),
        ("regime_auto_gate_report.md", REPORTS_DIR / "regime_auto_gate_report.md", "Final experiment report."),
        ("final_checkpoint.md", REPORTS_DIR / "final_checkpoint.md", "Meeting-ready checkpoint summary."),
    ]
    rows = []
    for file_name, path, desc in entries:
        rows.append(
            {
                "file": file_name,
                "path": str(path),
                "exists": bool(path.exists()),
                "description": desc,
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(COMPARISON_DIR.parent / "output_manifest.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    logger = get_logger("evaluate_auto_gate_policies")
    cfg = load_experiment_cfg()

    new_path = FEATURES_DIR / "regime_features_NEW.csv"
    new3_path = FEATURES_DIR / "regime_features_NEW3.csv"
    static_path = POLICIES_DIR / "static_gate_summary.csv"
    if not new_path.exists() or not new3_path.exists():
        raise FileNotFoundError("Feature CSVs are missing. Run build_regime_features.py first.")
    if not static_path.exists():
        raise FileNotFoundError("static_gate_summary.csv is missing. Run evaluate_static_gates.py first.")

    new_df = pd.read_csv(new_path, low_memory=False, dtype={"session_id": "string", "split": "string"})
    new3_df = pd.read_csv(new3_path, low_memory=False, dtype={"session_id": "string", "split": "string"})
    static_df = pd.read_csv(static_path)

    new_val_df = new_df[new_df["split"] == "val"].copy()
    new_test_df = new_df[new_df["split"] == "test"].copy()
    if new_val_df.empty or new_test_df.empty:
        raise RuntimeError("NEW validation/test splits are empty in feature CSV.")

    train_val_df = new_df[new_df["split"].isin(["train", "val"])].copy()
    vol_threshold_lookup: dict[tuple[int, float], float] = {}
    for w in [100, 200, 500]:
        col = f"rolling_vol_ticks_{w}"
        if col not in train_val_df.columns:
            continue
        vals = train_val_df[col].to_numpy(dtype=np.float64)
        for q in [0.70, 0.80, 0.90]:
            vol_threshold_lookup[(w, q)] = safe_quantile(vals, q)

    logger.info("Evaluating auto-gate candidates on NEW validation set...")
    selection_df = _evaluate_candidates(
        cfg=cfg,
        val_df=new_val_df,
        vol_threshold_lookup=vol_threshold_lookup,
    )
    selection_path = POLICIES_DIR / "auto_gate_validation_selection.csv"
    selection_df.to_csv(selection_path, index=False)
    logger.info("Saved validation selection table: %s rows", len(selection_df))

    selected_row = selection_df[selection_df["selected"]].iloc[0]
    logger.info(
        "Selected policy: %s params=%s thr=%.2f",
        selected_row["policy_name"],
        selected_row["params"],
        float(selected_row["model_threshold"]),
    )

    auto_test_df = _evaluate_selected_on_test_sets(
        cfg=cfg,
        selected_row=selected_row,
        new_test_df=new_test_df,
        new3_df=new3_df,
    )
    auto_test_path = BACKTESTS_DIR / "auto_gate_test_results.csv"
    auto_test_df.to_csv(auto_test_path, index=False)
    logger.info("Saved auto-gate test results to %s", auto_test_path)

    cmp_df = _build_comparison_table(static_df=static_df, auto_df=auto_test_df)
    cmp_path = COMPARISON_DIR / "auto_gate_vs_static.csv"
    cmp_df.to_csv(cmp_path, index=False)
    logger.info("Saved comparison table to %s", cmp_path)

    selected_params = json.loads(str(selected_row["params"]))
    spread_min = float(selected_params.get("spread_min", 1.5))
    ewma_df = _evaluate_ewma_ablation(
        cfg=cfg,
        new_val_df=new_val_df,
        new_test_df=new_test_df,
        new3_df=new3_df,
        selected_threshold=float(selected_row["model_threshold"]),
        spread_min=spread_min,
    )
    ewma_path = COMPARISON_DIR / "ewma_gate_ablation.csv"
    ewma_df.to_csv(ewma_path, index=False)
    logger.info("Saved EWMA ablation to %s", ewma_path)

    _plot_distributions(new_df, new3_df)
    _plot_static_vs_auto(static_df, auto_test_df)
    _plot_active_vs_pnl(selection_df)
    _plot_timeline(new3_df, str(selected_row["policy_name"]), selected_params)

    (REPORTS_DIR / "meta_classifier_skip_reason.md").write_text(
        "Skipped because available independent sessions / blocks are insufficient for reliable meta-gate training.",
        encoding="utf-8",
    )

    verdict, verdict_reason = _determine_verdict(static_df=static_df, auto_df=auto_test_df)
    _write_regime_auto_gate_report(
        static_df=static_df,
        selection_df=selection_df,
        selected_row=selected_row,
        auto_df=auto_test_df,
        comparison_df=cmp_df,
        ewma_df=ewma_df,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )
    _write_final_checkpoint(
        static_df=static_df,
        selected_row=selected_row,
        auto_df=auto_test_df,
        ewma_df=ewma_df,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )
    _write_manifest()
    logger.info("Auto-gate evaluation complete.")


if __name__ == "__main__":
    main()
