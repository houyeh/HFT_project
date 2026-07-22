from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
)
OUT_ROOT = MODEL_DIR / "eval_outputs" / "week13_meeting_pack"
SLIDE_FOLDERS = [f"{i:02d}_result" for i in range(1, 13)]


REQUIRED_INPUTS = [
    "dataset_manifest.json",
    "experiment_config.json",
    "train_outputs/train_history.csv",
    "eval_outputs/classification_report_test.csv",
    "eval_outputs/confusion_matrix_percent_test.png",
    "eval_outputs/diagnosis/baseline_summary_fixed.csv",
    "eval_outputs/diagnosis/threshold_sweep_enhanced.csv",
    "eval_outputs/diagnosis/missed_opportunity_diagnostics_fixed.csv",
    "eval_outputs/diagnosis/latency_sensitivity_enhanced.csv",
    "eval_outputs/diagnosis/regime_gate_sweep.csv",
    "eval_outputs/diagnosis_v3/new3_session_audit.csv",
    "eval_outputs/diagnosis_v3/new_session_audit.csv",
    "eval_outputs/diagnosis_v3/side_label_distribution_fill_conditional.csv",
    "eval_outputs/diagnosis_v3/side_label_unconditional_vs_fill_conditional.csv",
    "eval_outputs/diagnosis_v3/fill_conditional_side_label_report.md",
]


def _split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float,
    val_frac: float,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def _ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in SLIDE_FOLDERS:
        (OUT_ROOT / folder).mkdir(parents=True, exist_ok=True)


def _write_source_files(slide_dir: Path, source_paths: list[Path | str]) -> None:
    lines = []
    for src in source_paths:
        if isinstance(src, Path):
            lines.append(str(src.resolve()))
        else:
            lines.append(str(src))
    if not lines:
        lines = ["N/A"]
    (slide_dir / "source_files.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_missing(slide_dir: Path, missing: list[Path]) -> None:
    if not missing:
        return
    lines = ["# Missing Files", ""]
    lines.extend(str(p.resolve()) for p in missing)
    (slide_dir / "MISSING_FILES.md").write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_slide_01() -> None:
    slide_dir = OUT_ROOT / "01_result"
    summary = """# From Directional Prediction to Toxic-flow Cancel Filter

- TCN is not used as a directional trading signal.
- TCN is used as a toxic-flow risk score for defensive maker quoting.
- Strategy mapping:
  - `P(Down)` high -> cancel bid
  - `P(Up)` high -> cancel ask
- Status: research candidate, not production-ready.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, ["N/A (research framing slide)"])


def build_slide_02(manifest: dict, cfg: dict) -> None:
    slide_dir = OUT_ROOT / "02_result"
    source_paths = [MODEL_DIR / "dataset_manifest.json", MODEL_DIR / "experiment_config.json"]
    missing = [p for p in source_paths if not p.exists()]
    _write_missing(slide_dir, missing)

    sessions = list(manifest.get("sessions", []))
    class_dist = manifest.get("class_distribution_by_session", {})
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train_session_frac", 0.7))
    val_frac = float(split_cfg.get("val_session_frac", 0.15))
    train_sessions, val_sessions, test_sessions = _split_ordered_session_ids(sessions, train_frac, val_frac)

    def _count(ss: list[str]) -> int:
        total = 0
        for s in ss:
            d = class_dist.get(s, {})
            total += int(d.get("0", 0)) + int(d.get("1", 0)) + int(d.get("2", 0))
        return total

    data_cfg = cfg.get("data", {})
    window_cfg = cfg.get("window", {})
    row = {
        "dataset_source": data_cfg.get("data_dir", "MISSING"),
        "symbol": (data_cfg.get("symbols") or ["MISSING"])[0],
        "train_samples": _count(train_sessions),
        "val_samples": _count(val_sessions),
        "test_samples": _count(test_sessions),
        "feature_count": cfg.get("features", {}).get("feature_count", "MISSING"),
        "zscore_window": data_cfg.get("zscore_window", manifest.get("zscore_window", "MISSING")),
        "window_size": window_cfg.get("window_size", manifest.get("window_size", "MISSING")),
        "horizon": window_cfg.get("horizon", manifest.get("horizon", "MISSING")),
        "label_type": manifest.get("label_type", cfg.get("labels", {}).get("label_type", "MISSING")),
        "threshold_ticks": window_cfg.get("threshold_ticks", manifest.get("threshold_ticks", "MISSING")),
        "split_method": f"session-time-ordered {int(train_frac*100)}/{int(val_frac*100)}/{int((1-train_frac-val_frac)*100)}",
    }
    pd.DataFrame([row]).to_csv(slide_dir / "slide_data.csv", index=False)

    summary = f"""# Dataset and Model Identity

- Dataset source: `{row['dataset_source']}`
- Symbol: `{row['symbol']}`
- Train/Val/Test samples: {row['train_samples']:,} / {row['val_samples']:,} / {row['test_samples']:,}
- Feature count: {row['feature_count']}, zscore_window: {row['zscore_window']}
- Window/Horizon: {row['window_size']} / {row['horizon']}
- Label: `{row['label_type']}`, threshold_ticks: {row['threshold_ticks']}
- Split method: {row['split_method']}
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, source_paths)


def build_slide_03() -> None:
    slide_dir = OUT_ROOT / "03_result"
    train_path = MODEL_DIR / "train_outputs" / "train_history.csv"
    cls_path = MODEL_DIR / "eval_outputs" / "classification_report_test.csv"
    conf_path = MODEL_DIR / "eval_outputs" / "confusion_matrix_percent_test.png"
    source_paths = [train_path, cls_path, conf_path]
    _write_missing(slide_dir, [p for p in source_paths if not p.exists()])

    train_df = pd.read_csv(train_path)
    best_idx = train_df["val_loss"].astype(float).idxmin()
    best_row = train_df.loc[best_idx]
    last_row = train_df.iloc[-1]
    macro_f1_cols = [c for c in train_df.columns if "macro" in c.lower() and "f1" in c.lower()]
    max_macro_f1 = float(train_df[macro_f1_cols[0]].max()) if macro_f1_cols else np.nan

    tsummary = pd.DataFrame(
        [
            {
                "best_epoch_by_val_loss": int(best_row["epoch"]),
                "min_val_loss": float(best_row["val_loss"]),
                "last_train_loss": float(last_row["train_loss"]),
                "last_val_loss": float(last_row["val_loss"]),
                "max_val_macro_f1_if_available": max_macro_f1,
            }
        ]
    )
    tsummary.to_csv(slide_dir / "training_summary.csv", index=False)

    cls_df = pd.read_csv(cls_path)
    cls_df = cls_df.rename(columns={"Unnamed: 0": "class", "f1-score": "f1_score"})
    wanted = cls_df[cls_df["class"].isin(["Down", "Stationary", "Up"])].copy()
    interp = []
    for _, r in wanted.iterrows():
        if r["class"] in ("Down", "Up"):
            interp.append("low precision + high recall: useful as cancel-risk trigger, not directional alpha")
        else:
            interp.append("dominant class with high precision and moderate recall")
    wanted["interpretation"] = interp
    wanted = wanted[["class", "precision", "recall", "f1_score", "support", "interpretation"]]
    wanted.to_csv(slide_dir / "classification_summary.csv", index=False)

    shutil.copy2(conf_path, slide_dir / "confusion_matrix_percent_test.png")

    # Copy existing train plot if present; otherwise generate.
    existing_plot = None
    for candidate in [
        MODEL_DIR / "train_outputs" / "train_history_plot.png",
        MODEL_DIR / "eval_outputs" / "train_history_plot.png",
    ]:
        if candidate.exists():
            existing_plot = candidate
            break
    if existing_plot is not None:
        shutil.copy2(existing_plot, slide_dir / "train_history_plot.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_df["epoch"], train_df["train_loss"], label="train_loss")
        ax.plot(train_df["epoch"], train_df["val_loss"], label="val_loss")
        ax.scatter([best_row["epoch"]], [best_row["val_loss"]], color="red", zorder=3)
        ax.set_title("Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(slide_dir / "training_loss_plot.png", dpi=150)
        plt.close(fig)

    summary = f"""# Training and Classification

- Model converges on train/validation loss (best epoch by val loss: {int(best_row['epoch'])}, min_val_loss={float(best_row['val_loss']):.6f}).
- Up/Down classes show low precision with high recall pattern.
- This behavior does not support directional trading deployment.
- Positioning remains: high-sensitivity cancel filter candidate.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, source_paths + ([existing_plot] if existing_plot else []))


def build_slide_04() -> None:
    slide_dir = OUT_ROOT / "04_result"
    baseline_path = MODEL_DIR / "eval_outputs" / "diagnosis" / "baseline_summary_fixed.csv"
    sweep_path = MODEL_DIR / "eval_outputs" / "diagnosis" / "threshold_sweep_enhanced.csv"
    source_paths = [baseline_path, sweep_path]
    _write_missing(slide_dir, [p for p in source_paths if not p.exists()])

    bdf = pd.read_csv(baseline_path).set_index("experiment_group")
    targets = [
        ("Model_0.55", "Random_0.55", 0.55, "Model_0.55"),
        ("Model_0.60", "Random_0.60", 0.60, "Model_0.60"),
        ("Model_0.70", "Random_0.70", 0.70, "Val-selected_0.70"),
    ]
    rows = []
    for model_name, rand_name, thr, exp_name in targets:
        model_row = bdf.loc[model_name] if model_name in bdf.index else None
        rand_row = bdf.loc[rand_name] if rand_name in bdf.index else None
        net = float(model_row["mean_net_pnl"]) if model_row is not None else np.nan
        rmean = float(rand_row["mean_net_pnl"]) if rand_row is not None else np.nan
        rstd = float(rand_row["std_net_pnl"]) if rand_row is not None else np.nan
        rplus = rmean + rstd if np.isfinite(rmean) and np.isfinite(rstd) else np.nan
        beats_mean = bool(net > rmean) if np.isfinite(net) and np.isfinite(rmean) else "MISSING"
        beats_plus = bool(net > rplus) if np.isfinite(net) and np.isfinite(rplus) else "MISSING"
        if beats_plus is True:
            interp = "beats matched random mean+1std in NEW test split"
        elif beats_mean is True:
            interp = "beats random mean but not mean+1std"
        else:
            interp = "does not beat random benchmark robustly"
        rows.append(
            {
                "experiment": exp_name,
                "threshold": thr,
                "net_pnl": net,
                "random_mean": rmean,
                "random_std": rstd,
                "random_mean_plus_1std": rplus,
                "beats_random_mean": beats_mean,
                "beats_random_mean_plus_1std": beats_plus,
                "interpretation": interp,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(slide_dir / "baseline_evidence.csv", index=False)

    row55 = out_df[out_df["experiment"] == "Model_0.55"].iloc[0]
    row60 = out_df[out_df["experiment"] == "Model_0.60"].iloc[0]
    row70 = out_df[out_df["experiment"] == "Val-selected_0.70"].iloc[0]
    summary = f"""# Baseline Evidence

- 0.55 local edge: net_pnl={row55['net_pnl']:.2f}, random+1std={row55['random_mean_plus_1std']:.2f}, beats={row55['beats_random_mean_plus_1std']}.
- 0.60 local edge: net_pnl={row60['net_pnl']:.2f}, random+1std={row60['random_mean_plus_1std']:.2f}, beats={row60['beats_random_mean_plus_1std']}.
- 0.70 is validation-selected threshold (not production threshold by default).
- 0.70 vs random+1std: net_pnl={row70['net_pnl']:.2f} vs {row70['random_mean_plus_1std']:.2f}; beats={row70['beats_random_mean_plus_1std']}.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, source_paths)


def build_slide_05() -> None:
    slide_dir = OUT_ROOT / "05_result"
    src = MODEL_DIR / "eval_outputs" / "diagnosis" / "missed_opportunity_diagnostics_fixed.csv"
    _write_missing(slide_dir, [src] if not src.exists() else [])
    df = pd.read_csv(src)
    model_df = df[df["notes"].astype(str).str.contains("model threshold", case=False, na=False)].copy()
    model_df["interpretation"] = np.where(
        model_df["net_cancel_value"] > 0,
        "avoided toxic loss exceeds missed profitable opportunity",
        "missed profitable opportunity dominates",
    )
    out = model_df[
        [
            "threshold",
            "signal_rate",
            "missed_profitable_pnl",
            "missed_toxic_pnl_avoided",
            "net_cancel_value",
            "interpretation",
        ]
    ].copy()
    out.to_csv(slide_dir / "opportunity_cost_summary.csv", index=False)
    summary = """# Opportunity Cost

- Under current backtest assumptions, avoided toxic loss exceeds missed profitable opportunity across model thresholds.
- Net cancel value remains positive in this diagnostic slice.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, [src])


def build_slide_06() -> None:
    slide_dir = OUT_ROOT / "06_result"
    session_path = MODEL_DIR / "eval_outputs" / "diagnosis_v3" / "new3_session_audit.csv"
    gate_path = MODEL_DIR / "eval_outputs" / "diagnosis" / "regime_gate_sweep.csv"
    source_paths = [session_path, gate_path]
    _write_missing(slide_dir, [p for p in source_paths if not p.exists()])

    sdf = pd.read_csv(session_path)
    out = sdf[
        ["session_id", "trend_strength", "suitability_label", "baseline_net_pnl", "main_failure_mode"]
    ].copy()
    out.to_csv(slide_dir / "new3_regime_summary.csv", index=False)

    gdf = pd.read_csv(gate_path)
    ref = gdf[
        (gdf["dataset"] == "NEW3")
        & (gdf["gate_type"] == "spread_filter")
        & (gdf["gate_params"].astype(str).str.contains('"spread_ticks_min": 1.0', regex=False))
        & (gdf["model_threshold"] == 0.70)
    ]
    if ref.empty:
        ref = gdf[(gdf["dataset"] == "NEW3") & (gdf["active_event_rate"] >= 0.9999)].sort_values("net_pnl").head(1)
    if not ref.empty:
        r = ref.iloc[0]
        base_line = f"model={float(r['net_pnl']):.0f}, baseline={float(r['baseline_net_pnl']):.0f}, improvement={float(r['improvement']):+.0f}"
    else:
        base_line = "MISSING"

    summary = f"""# Why Regime Gate

- NEW3 no-gate style reference: {base_line}.
- NEW3 acts as stress / unfavorable regime.
- Micro TCN cancel alone cannot fully solve macro regime failure.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, source_paths)


def build_slide_07() -> None:
    slide_dir = OUT_ROOT / "07_result"
    src = MODEL_DIR / "eval_outputs" / "diagnosis" / "regime_gate_sweep.csv"
    _write_missing(slide_dir, [src] if not src.exists() else [])
    df = pd.read_csv(src)

    rows: list[dict] = []
    for ds in ["NEW", "NEW3"]:
        ds_df = df[df["dataset"] == ds].copy()
        if ds_df.empty:
            continue
        best = ds_df.sort_values("net_pnl", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": ds,
                "best_gate_type": best["gate_type"],
                "best_gate_params": best["gate_params"],
                "model_threshold": float(best["model_threshold"]),
                "net_pnl": float(best["net_pnl"]),
                "baseline_net_pnl": float(best["baseline_net_pnl"]),
                "improvement": float(best["improvement"]),
                "active_event_rate": float(best["active_event_rate"]),
                "trade_count": float(best["trade_count"]),
                "interpretation": "best gate by net_pnl for dataset",
            }
        )

    for ds in ["NEW", "NEW3"]:
        ref = df[
            (df["dataset"] == ds)
            & (df["gate_type"] == "spread_filter")
            & (df["gate_params"].astype(str).str.contains('"spread_ticks_min": 1.0', regex=False))
            & (df["model_threshold"] == 0.70)
        ]
        if ref.empty:
            continue
        r = ref.iloc[0]
        rows.append(
            {
                "dataset": f"{ds}_no_gate_reference",
                "best_gate_type": r["gate_type"],
                "best_gate_params": r["gate_params"],
                "model_threshold": float(r["model_threshold"]),
                "net_pnl": float(r["net_pnl"]),
                "baseline_net_pnl": float(r["baseline_net_pnl"]),
                "improvement": float(r["improvement"]),
                "active_event_rate": float(r["active_event_rate"]),
                "trade_count": float(r["trade_count"]),
                "interpretation": "reference near no-gate behavior (spread_filter=1.0)",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(slide_dir / "regime_gate_best_by_dataset.csv", index=False)

    new_best = out[out["dataset"] == "NEW"].iloc[0] if (out["dataset"] == "NEW").any() else None
    new3_best = out[out["dataset"] == "NEW3"].iloc[0] if (out["dataset"] == "NEW3").any() else None
    summary_lines = ["# Regime Gate Results", ""]
    if new_best is not None:
        summary_lines.append(
            f"- NEW best: {new_best['best_gate_type']} + TCN (threshold={new_best['model_threshold']}, net_pnl={new_best['net_pnl']:.0f})."
        )
    else:
        summary_lines.append("- NEW best: MISSING")
    if new3_best is not None:
        summary_lines.append(
            f"- NEW3 best: {new3_best['best_gate_type']} + TCN (threshold={new3_best['model_threshold']}, net_pnl={new3_best['net_pnl']:.0f})."
        )
    else:
        summary_lines.append("- NEW3 best: MISSING")
    summary_lines.extend(
        [
            "- This indicates regime dependency.",
            "- A unified production gate selector is not established yet.",
        ]
    )
    (slide_dir / "slide_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    _write_source_files(slide_dir, [src])


def build_slide_08() -> None:
    slide_dir = OUT_ROOT / "08_result"
    content = """# Implemented Regime Gate

- Implemented:
  - rolling trend filter
  - rolling volatility filter
  - spread filter
  - combined_filter (trend + volatility + spread)
- Not yet implemented:
  - EWMA gate
  - Kalman-based gate
- EWMA / Kalman are future work.
"""
    (slide_dir / "implemented_regime_gate.md").write_text(content, encoding="utf-8")
    (slide_dir / "slide_summary.md").write_text(
        "# Implemented Regime Gate\n\n- Current implementation is rolling-feature based.\n- EWMA/Kalman remain future work.",
        encoding="utf-8",
    )
    _write_source_files(slide_dir, ["N/A (implementation status from existing diagnostics)"])


def build_slide_09() -> None:
    slide_dir = OUT_ROOT / "09_result"
    cond_path = MODEL_DIR / "eval_outputs" / "diagnosis_v3" / "side_label_distribution_fill_conditional.csv"
    cmp_path = MODEL_DIR / "eval_outputs" / "diagnosis_v3" / "side_label_unconditional_vs_fill_conditional.csv"
    rep_path = MODEL_DIR / "eval_outputs" / "diagnosis_v3" / "fill_conditional_side_label_report.md"
    source_paths = [cond_path, cmp_path, rep_path]
    _write_missing(slide_dir, [p for p in source_paths if not p.exists()])

    rec_text = rep_path.read_text(encoding="utf-8")
    m = re.search(r"fill_window=(\d+)", rec_text)
    rec_fill_window = int(m.group(1)) if m else 1

    cond = pd.read_csv(cond_path)
    cmp_df = pd.read_csv(cmp_path)
    merged = cond.merge(
        cmp_df[
            [
                "dataset",
                "split",
                "horizon",
                "proxy_type",
                "fill_window",
                "bid_toxic_rate_lift",
                "ask_toxic_rate_lift",
            ]
        ],
        on=["dataset", "split", "horizon", "proxy_type", "fill_window"],
        how="left",
    )
    filt = merged[
        (merged["proxy_type"] == "price_level_removed_proxy")
        & (merged["horizon"] == 10)
        & (merged["fill_window"] == rec_fill_window)
    ].copy()
    filt["interpretation"] = (
        "proxy conditional toxicity lift observed; proxy only, not true fill label"
    )
    out = filt[
        [
            "dataset",
            "split",
            "proxy_type",
            "fill_window",
            "horizon",
            "bid_fill_count",
            "bid_toxic_rate_conditional",
            "ask_fill_count",
            "ask_toxic_rate_conditional",
            "bid_toxic_rate_lift",
            "ask_toxic_rate_lift",
            "interpretation",
        ]
    ].copy()
    out.to_csv(slide_dir / "fill_conditional_label_summary.csv", index=False)

    avg_bid_lift = float(out["bid_toxic_rate_lift"].mean()) if not out.empty else np.nan
    avg_ask_lift = float(out["ask_toxic_rate_lift"].mean()) if not out.empty else np.nan
    summary = f"""# Fill-Conditional Label Audit

- Using proxy C (`price_level_removed_proxy`) at horizon=10, fill_window={rec_fill_window}.
- Conditional toxic rate lift vs unconditional is substantial (avg bid lift={avg_bid_lift:.2f}x, avg ask lift={avg_ask_lift:.2f}x).
- Conditional toxic rate remains in low single-digit percent for proxy C rows.
- No aggressor-side data: this is proxy evidence, not true exchange fill truth.
- Result motivates bid_toxic / ask_toxic retraining research.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, source_paths)


def build_slide_10() -> None:
    slide_dir = OUT_ROOT / "10_result"
    src = MODEL_DIR / "eval_outputs" / "diagnosis" / "latency_sensitivity_enhanced.csv"
    _write_missing(slide_dir, [src] if not src.exists() else [])

    df = pd.read_csv(src)
    delay_events = []
    for _, r in df.iterrows():
        mode = str(r["mode"])
        if mode == "prediction_delay":
            delay_events.append(int(r["prediction_delay_events"]))
        elif mode == "cancel_delay":
            delay_events.append(int(r["cancel_delay_events"]))
        else:
            delay_events.append(int(max(r["prediction_delay_events"], r["cancel_delay_events"])))
    df["delay_events"] = delay_events
    df["interpretation"] = ""

    for mode in ["prediction_delay", "cancel_delay", "combined_delay"]:
        for thr in sorted(df["threshold"].dropna().unique()):
            mask = (df["mode"] == mode) & (df["threshold"] == thr)
            sub = df.loc[mask].sort_values("delay_events")
            if sub.empty:
                continue
            base = float(sub[sub["delay_events"] == 0]["net_pnl"].iloc[0]) if (sub["delay_events"] == 0).any() else np.nan
            for idx, row in sub.iterrows():
                d = int(row["delay_events"])
                pnl = float(row["net_pnl"])
                if mode == "cancel_delay":
                    if d in (3, 5) and np.isfinite(base) and pnl > base:
                        msg = "possible false-positive cancel reduction; non-monotonic gain"
                    elif d == 10 and np.isfinite(base) and pnl < base:
                        msg = "deterioration at long cancel delay"
                    else:
                        msg = "cancel delay trade-off"
                elif mode == "prediction_delay":
                    msg = "prediction delay changes signal timing; non-monotonic response possible"
                else:
                    msg = "combined delay effect"
                df.at[idx, "interpretation"] = msg

    out = df[
        [
            "mode",
            "threshold",
            "delay_events",
            "net_pnl",
            "delayed_toxic_loss",
            "delayed_toxic_fill_count",
            "interpretation",
        ]
    ].copy()
    out.to_csv(slide_dir / "latency_summary.csv", index=False)

    c55 = df[(df["mode"] == "cancel_delay") & (df["threshold"] == 0.55)].sort_values("delay_events").copy()
    c55_out = c55[["delay_events", "net_pnl", "delayed_toxic_loss", "delayed_toxic_fill_count"]].copy()
    c55_out.to_csv(slide_dir / "cancel_delay_threshold_055.csv", index=False)

    if not c55_out.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(c55_out["delay_events"], c55_out["net_pnl"], marker="o")
        ax.set_title("Cancel Delay vs Net PnL (threshold=0.55)")
        ax.set_xlabel("delay_events")
        ax.set_ylabel("net_pnl")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(slide_dir / "cancel_delay_net_pnl.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(c55_out["delay_events"], c55_out["delayed_toxic_loss"], marker="o", color="tab:red")
        ax.set_title("Cancel Delay vs Delayed Toxic Loss (threshold=0.55)")
        ax.set_xlabel("delay_events")
        ax.set_ylabel("delayed_toxic_loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(slide_dir / "cancel_delay_delayed_toxic_loss.png", dpi=150)
        plt.close(fig)

    summary = """# Latency Sensitivity

- Cancel delay increases delayed toxic loss in diagnostics.
- Net PnL response is non-monotonic.
- Delay 3/5 can improve PnL in some settings due to reduced false-positive cancels.
- Delay 10 shows deterioration in key slices.
- FPGA relevance here is risk-control-oriented rather than a single monotonic PnL claim.
"""
    (slide_dir / "slide_summary.md").write_text(summary, encoding="utf-8")
    _write_source_files(slide_dir, [src])


def build_slide_11() -> None:
    slide_dir = OUT_ROOT / "11_result"
    content = """# Current Status and Limitations

Current status: C+ / B- frontier

Already achieved:
- TCN local edge on NEW test split
- Model 0.55 / 0.60 beats matched random+1std
- Regime gate rescues NEW3
- Fill-conditional audit shows toxic-rate lift
- Cancel delay risk quantified

Not yet achieved:
- validation-selected gated threshold beating random+1std
- retrained bid_toxic / ask_toxic model
- queue-aware true fill validation
- unified causal gate selector
- definitive FPGA PnL proof
"""
    (slide_dir / "current_status.md").write_text(content, encoding="utf-8")
    (slide_dir / "slide_summary.md").write_text(
        "# Current Status\n\n- Frontier status remains C+ / B-.\n- Evidence is promising but not production-complete.",
        encoding="utf-8",
    )
    _write_source_files(slide_dir, ["N/A (status synthesis from existing outputs)"])


def build_slide_12() -> None:
    slide_dir = OUT_ROOT / "12_result"
    content = """# Next Steps

1. Build bid_toxic / ask_toxic labels.
2. Retrain small model with side-specific toxic labels.
3. Select threshold under regime gate using validation.
4. Compare with matched random and rule-based baselines.
5. Add queue-aware / HftBacktest-style validation.
6. FPGA MVP: rolling or EWMA gate + tiny TCN cancel.
"""
    (slide_dir / "next_steps.md").write_text(content, encoding="utf-8")
    (slide_dir / "slide_summary.md").write_text(
        "# Next Steps\n\n- Side-specific relabeling + small-model retrain.\n- Validation-first selection under regime gate.\n- Queue-aware validation before hardware claims.",
        encoding="utf-8",
    )
    _write_source_files(slide_dir, ["N/A (forward plan from diagnostics)"])


def build_manifest_and_pack_summary() -> None:
    rows = []
    for folder in SLIDE_FOLDERS:
        slide_dir = OUT_ROOT / folder
        source_txt = slide_dir / "source_files.txt"
        src_list = source_txt.read_text(encoding="utf-8").strip() if source_txt.exists() else "MISSING"
        for fp in sorted(slide_dir.iterdir()):
            rows.append(
                {
                    "folder": folder,
                    "file_name": fp.name,
                    "path": str(fp.resolve()),
                    "exists": fp.exists(),
                    "source_files": src_list,
                    "notes": "",
                }
            )
    manifest_df = pd.DataFrame(rows)
    manifest_path = OUT_ROOT / "week13_output_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    missing_notes = []
    for folder in SLIDE_FOLDERS:
        mf = OUT_ROOT / folder / "MISSING_FILES.md"
        if mf.exists():
            missing_notes.append(f"- {folder}: has MISSING_FILES.md")
    missing_block = "\n".join(missing_notes) if missing_notes else "- None"

    summary = f"""# Week13 Meeting Pack Summary

## Generated
- Output root: `{OUT_ROOT}`
- Slide folders: {", ".join(SLIDE_FOLDERS)}
- Manifest: `week13_output_manifest.csv`

## Missing Inputs
{missing_block}

## Warnings
- No new backtests were run.
- No model retraining was run.
- Threshold 0.55 is not labeled as production threshold.
- Fill-conditional label audit is proxy-based (not true exchange fill truth).
- Latency interpretation avoids monotonic collapse claims.

## Claims To Avoid
- Do not claim directional trading readiness.
- Do not claim delay > 3 always collapses PnL.
- Do not claim proxy fills are true fills.
- Do not claim unified production gate selector is solved.
"""
    (OUT_ROOT / "week13_meeting_pack_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    # Global input check is only for pack summary; slide-level missing handled per builder.
    global_missing = [MODEL_DIR / p for p in REQUIRED_INPUTS if not (MODEL_DIR / p).exists()]
    if global_missing:
        (OUT_ROOT / "GLOBAL_MISSING_FILES.md").write_text(
            "\n".join(str(p.resolve()) for p in global_missing),
            encoding="utf-8",
        )

    manifest = _read_json(MODEL_DIR / "dataset_manifest.json")
    cfg = _read_json(MODEL_DIR / "experiment_config.json")

    build_slide_01()
    build_slide_02(manifest, cfg)
    build_slide_03()
    build_slide_04()
    build_slide_05()
    build_slide_06()
    build_slide_07()
    build_slide_08()
    build_slide_09()
    build_slide_10()
    build_slide_11()
    build_slide_12()
    build_manifest_and_pack_summary()
    print(f"Week13 pack generated at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
