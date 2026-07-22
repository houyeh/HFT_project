from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure torch is importable from project-local .venv packages.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import torch

from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.tcn_pipeline_utils import apply_rolling_zscore
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from src.utils.tw_train_helpers import build_tw_feature_frame

from common import (
    BACKTESTS_DIR,
    COMPARISON_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    NEW3_DIR,
    NEW3_SYMBOL,
    NEW_ALL_DIR,
    NEW_SYMBOL,
    REPORTS_DIR,
    SELECTED_FEATURES,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZE,
    Z_FEATURES,
    ensure_output_dirs,
    save_json,
    split_ordered_session_ids,
)


PRIMARY_HORIZON = 10
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
ASYM_PAIRS = [(0.30, 0.50), (0.50, 0.30), (0.20, 0.40), (0.40, 0.20)]


def _import_simulator():
    path = PROJECT_ROOT / "new_era" / "15_models_playaround" / "run_diagnosis_1.py"
    parent = path.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("run_diagnosis_1_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load simulator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate_with_metrics


def _build_cfg():
    common_path = PROJECT_ROOT / "new_era" / "15_models_playaround"
    if str(common_path) not in sys.path:
        sys.path.insert(0, str(common_path))
    import _common  # type: ignore

    return _common.build_cfg()


def _load_symbol(data_dir: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"Missing symbol {symbol} in {data_dir}")
    lob = lob_frames[symbol].copy()
    trade = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])).copy()
    return lob, trade


def _build_processed_frames(
    lob: pd.DataFrame,
    trade: pd.DataFrame,
    symbol: str,
    session_ids: list[str] | None = None,
    zscore_window: int = 2000,
) -> tuple[pd.DataFrame, float]:
    frames = []
    tick_values = []
    session_set = set(session_ids) if session_ids is not None else None
    for sid, lob_s in lob.groupby("session_id", sort=True):
        if session_set is not None and sid not in session_set:
            continue
        trd_s = trade[trade["session_id"] == sid]
        tick = float(infer_tick_size_from_lob(lob_s))
        fdf = build_tw_feature_frame(
            lob_s.reset_index(drop=True),
            trd_s.reset_index(drop=True),
            symbol=symbol,
            tick_size=tick,
        ).dropna()
        zdf = apply_rolling_zscore(fdf, SELECTED_FEATURES, window=zscore_window).reset_index(drop=True)
        zdf["session_id"] = sid
        frames.append(zdf)
        tick_values.append(tick)
    if not frames:
        return pd.DataFrame(), float("nan")
    full = pd.concat(frames, ignore_index=True)
    tick_size = float(np.median(np.asarray(tick_values, dtype=np.float64)))
    return full, tick_size


def _load_model(side: str, device: torch.device):
    model_dir = MODELS_DIR / f"{side}_primary_fw1_h10_posw"
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    model = HLSFriendlyTCNClassifier(
        num_inputs=int(cfg["feature_count"]),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=WINDOW_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def _predict_all_event_probs(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    feat = frame[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
    n = len(frame)
    probs_full = np.zeros(n, dtype=np.float32)
    start_idx = WINDOW_SIZE - 1
    end_idx = n - PRIMARY_HORIZON
    if end_idx <= start_idx:
        return probs_full

    label_indices = np.arange(start_idx, end_idx, dtype=np.int64)
    for p0 in range(0, len(label_indices), batch_size):
        batch_idx = label_indices[p0 : p0 + batch_size]
        x = np.zeros((len(batch_idx), len(Z_FEATURES), WINDOW_SIZE), dtype=np.float32)
        for k, i in enumerate(batch_idx):
            st = int(i) - WINDOW_SIZE + 1
            ed = int(i) + 1
            x[k] = feat[st:ed].T
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs_full[batch_idx] = p
    return probs_full


def _run_backtest_for_thresholds(
    *,
    dataset_name: str,
    frame: pd.DataFrame,
    tick_size: float,
    bid_probs: np.ndarray,
    ask_probs: np.ndarray,
    simulate_with_metrics,
    cfg,
    bid_model_tag: str,
    ask_model_tag: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)

    combos = [(t, t) for t in THRESHOLDS] + ASYM_PAIRS
    for thr_bid, thr_ask in combos:
        pull_bid = bid_probs >= float(thr_bid)
        pull_ask = ask_probs >= float(thr_ask)
        signal_rate_bid = float(pull_bid.mean()) if pull_bid.size else float("nan")
        signal_rate_ask = float(pull_ask.mean()) if pull_ask.size else float("nan")

        res = simulate_with_metrics(
            cfg,
            prob=None,
            ask_px=ask,
            bid_px=bid,
            tick_size=tick_size,
            threshold=1.1,
            force_pull_bid_mask=pull_bid,
            force_pull_ask_mask=pull_ask,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "backtest_framework": "simulate_with_metrics_force_masks",
                "model_side_setup": "bid_model+ask_model",
                "bid_model_config": bid_model_tag,
                "ask_model_config": ask_model_tag,
                "threshold_bid": float(thr_bid),
                "threshold_ask": float(thr_ask),
                "signal_rate_bid": signal_rate_bid,
                "signal_rate_ask": signal_rate_ask,
                "trade_count": int(res["trade_count"]),
                "gross_pnl": float(res["gross_pnl"]),
                "net_pnl": float(res["net_pnl"]),
                "fee": float(res["fee"]),
                "max_drawdown": float(res["max_drawdown"]),
                "avg_pnl_per_fill": float(res["avg_pnl_per_fill"]),
                "missed_profitable_pnl": float(res["missed_profitable_pnl"]),
                "missed_toxic_pnl_avoided": float(res["missed_toxic_pnl_avoided"]),
                "net_cancel_value": float(res["net_cancel_value"]),
                "realized_bid_pnl": float(res["realized_bid_pnl"]),
                "realized_ask_pnl": float(res["realized_ask_pnl"]),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _matched_random_baseline(
    *,
    dataset_name: str,
    frame: pd.DataFrame,
    tick_size: float,
    signal_rate_bid: float,
    signal_rate_ask: float,
    simulate_with_metrics,
    cfg,
    seeds: int = 20,
) -> dict[str, float]:
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64)
    pnl = []
    mdd = []
    tc = []
    ncv = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        pull_bid = rng.random(len(bid)) < signal_rate_bid
        pull_ask = rng.random(len(ask)) < signal_rate_ask
        res = simulate_with_metrics(
            cfg,
            prob=None,
            ask_px=ask,
            bid_px=bid,
            tick_size=tick_size,
            threshold=1.1,
            force_pull_bid_mask=pull_bid,
            force_pull_ask_mask=pull_ask,
            seed=seed,
        )
        pnl.append(float(res["net_pnl"]))
        mdd.append(float(res["max_drawdown"]))
        tc.append(float(res["trade_count"]))
        ncv.append(float(res["net_cancel_value"]))
    return {
        "dataset": dataset_name,
        "num_runs": int(seeds),
        "mean_net_pnl": float(np.mean(pnl)),
        "std_net_pnl": float(np.std(pnl)),
        "mean_max_drawdown": float(np.mean(mdd)),
        "mean_trade_count": float(np.mean(tc)),
        "mean_net_cancel_value": float(np.mean(ncv)),
    }


def _write_comparison_reports(
    *,
    backtest_df: pd.DataFrame,
    new_test_frame: pd.DataFrame,
    new3_frame: pd.DataFrame,
    tick_new: float,
    tick_new3: float,
    simulate_with_metrics,
    cfg,
) -> None:
    # Choose best candidate on NEW test by net_pnl.
    new_df = backtest_df[backtest_df["dataset"] == "NEW_test"].copy()
    if new_df.empty:
        raise RuntimeError("No NEW_test rows in backtest summary.")
    best = new_df.sort_values("net_pnl", ascending=False).iloc[0]
    thr_bid = float(best["threshold_bid"])
    thr_ask = float(best["threshold_ask"])
    sig_bid = float(best["signal_rate_bid"])
    sig_ask = float(best["signal_rate_ask"])

    rand_new = _matched_random_baseline(
        dataset_name="NEW_test",
        frame=new_test_frame,
        tick_size=tick_new,
        signal_rate_bid=sig_bid,
        signal_rate_ask=sig_ask,
        simulate_with_metrics=simulate_with_metrics,
        cfg=cfg,
        seeds=20,
    )

    # No-model baseline under same framework.
    ask_new = new_test_frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid_new = new_test_frame["bid_px_1"].to_numpy(dtype=np.float64)
    base_new = simulate_with_metrics(cfg, None, ask_new, bid_new, tick_new, threshold=1.1)

    # NEW3 performance at the same best NEW threshold pair.
    new3_row = backtest_df[
        (backtest_df["dataset"] == "NEW3_full")
        & (backtest_df["threshold_bid"] == thr_bid)
        & (backtest_df["threshold_ask"] == thr_ask)
    ]
    new3_net = float(new3_row.iloc[0]["net_pnl"]) if not new3_row.empty else float("nan")

    smoothed_baseline_net = 84980.0
    smoothed_ref_text = "NEW test smoothed baseline reference (Model_0.55=84980)"
    random_plus_1std = rand_new["mean_net_pnl"] + rand_new["std_net_pnl"]

    comp_rows = []
    comp_rows.append(
        {
            "dataset": "NEW_test",
            "method": "binary_best",
            "threshold_setup": f"bid={thr_bid:.2f},ask={thr_ask:.2f}",
            "net_pnl": float(best["net_pnl"]),
            "random_mean": float(rand_new["mean_net_pnl"]),
            "random_std": float(rand_new["std_net_pnl"]),
            "random_mean_plus_1std": float(random_plus_1std),
            "beats_random_mean": bool(float(best["net_pnl"]) > float(rand_new["mean_net_pnl"])),
            "beats_random_mean_plus_1std": bool(float(best["net_pnl"]) > float(random_plus_1std)),
            "beats_smoothed_baseline": bool(float(best["net_pnl"]) > smoothed_baseline_net),
            "smoothed_baseline_reference": smoothed_ref_text,
            "max_drawdown": float(best["max_drawdown"]),
            "trade_count": float(best["trade_count"]),
            "net_cancel_value": float(best["net_cancel_value"]),
            "notes": "",
        }
    )
    comp_rows.append(
        {
            "dataset": "NEW_test",
            "method": "random_matched_20seed",
            "threshold_setup": f"matched_rates_bid={sig_bid:.6f},ask={sig_ask:.6f}",
            "net_pnl": float(rand_new["mean_net_pnl"]),
            "random_mean": float(rand_new["mean_net_pnl"]),
            "random_std": float(rand_new["std_net_pnl"]),
            "random_mean_plus_1std": float(random_plus_1std),
            "beats_random_mean": False,
            "beats_random_mean_plus_1std": False,
            "beats_smoothed_baseline": bool(float(rand_new["mean_net_pnl"]) > smoothed_baseline_net),
            "smoothed_baseline_reference": smoothed_ref_text,
            "max_drawdown": float(rand_new["mean_max_drawdown"]),
            "trade_count": float(rand_new["mean_trade_count"]),
            "net_cancel_value": float(rand_new["mean_net_cancel_value"]),
            "notes": "",
        }
    )
    comp_rows.append(
        {
            "dataset": "NEW_test",
            "method": "no_model_baseline",
            "threshold_setup": "none",
            "net_pnl": float(base_new["net_pnl"]),
            "random_mean": float(rand_new["mean_net_pnl"]),
            "random_std": float(rand_new["std_net_pnl"]),
            "random_mean_plus_1std": float(random_plus_1std),
            "beats_random_mean": bool(float(base_new["net_pnl"]) > float(rand_new["mean_net_pnl"])),
            "beats_random_mean_plus_1std": bool(float(base_new["net_pnl"]) > float(random_plus_1std)),
            "beats_smoothed_baseline": bool(float(base_new["net_pnl"]) > smoothed_baseline_net),
            "smoothed_baseline_reference": smoothed_ref_text,
            "max_drawdown": float(base_new["max_drawdown"]),
            "trade_count": float(base_new["trade_count"]),
            "net_cancel_value": float(base_new["net_cancel_value"]),
            "notes": "",
        }
    )
    if np.isfinite(new3_net):
        comp_rows.append(
            {
                "dataset": "NEW3_full",
                "method": "binary_best_from_new",
                "threshold_setup": f"bid={thr_bid:.2f},ask={thr_ask:.2f}",
                "net_pnl": float(new3_net),
                "random_mean": np.nan,
                "random_std": np.nan,
                "random_mean_plus_1std": np.nan,
                "beats_random_mean": np.nan,
                "beats_random_mean_plus_1std": np.nan,
                "beats_smoothed_baseline": np.nan,
                "smoothed_baseline_reference": "NEW3 baseline reference in prior diagnostics: model=-551705, baseline=-557165",
                "max_drawdown": float(new3_row.iloc[0]["max_drawdown"]),
                "trade_count": float(new3_row.iloc[0]["trade_count"]),
                "net_cancel_value": float(new3_row.iloc[0]["net_cancel_value"]),
                "notes": "compare with prior NEW3 diagnostics carefully (framework alignment required)",
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(COMPARISON_DIR / "binary_vs_baselines.csv", index=False)

    # Model comparison summary
    train_summary = pd.read_csv(MODELS_DIR / "binary_training_summary.csv")
    bid_row = train_summary[train_summary["side"] == "bid"].iloc[0]
    ask_row = train_summary[train_summary["side"] == "ask"].iloc[0]
    best_thr_text = f"bid={thr_bid:.2f},ask={thr_ask:.2f}"
    recommendation = "not_ready"
    if bool(comp_rows[0]["beats_random_mean_plus_1std"]) and bool(comp_rows[0]["beats_smoothed_baseline"]):
        recommendation = "replace_smoothed_baseline"
    elif bool(comp_rows[0]["beats_random_mean_plus_1std"]):
        recommendation = "keep_as_auxiliary_signal"

    mdf = pd.DataFrame(
        [
            {
                "model_name": "bid_quote_toxic_binary+ask_quote_toxic_binary",
                "label_type": "fill_conditional_quote_risk_binary_proxyC",
                "side_specific": True,
                "fill_conditional": True,
                "train_sample_count": int(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["train_samples"]),
                "val_sample_count": int(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["val_samples"]),
                "test_sample_count": int(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["test_samples"]),
                "positive_rate_train": float(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["positive_rate_train"]),
                "positive_rate_val": float(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["positive_rate_val"]),
                "positive_rate_test": float(json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))["positive_rate_test"]),
                "test_pr_auc": float((float(bid_row["test_auc_pr"]) + float(ask_row["test_auc_pr"])) / 2.0),
                "test_roc_auc": float((float(bid_row["test_auc_roc"]) + float(ask_row["test_auc_roc"])) / 2.0),
                "test_f1_best_threshold": float(best["avg_pnl_per_fill"]) if "avg_pnl_per_fill" in best else np.nan,
                "best_threshold": best_thr_text,
                "test_net_pnl": float(best["net_pnl"]),
                "test_vs_random_plus_1std": float(best["net_pnl"] - random_plus_1std),
                "test_vs_smoothed_baseline": float(best["net_pnl"] - smoothed_baseline_net),
                "NEW3_net_pnl": float(new3_net),
                "recommendation": recommendation,
            }
        ]
    )
    mdf.to_csv(COMPARISON_DIR / "model_comparison_summary.csv", index=False)

    # Long-form report
    lines = []
    lines.append("# New Binary Model Report")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("- Old toxic/adverse labels were either non-conditional or too sparse.")
    lines.append("- New labels are side-specific and fill-conditional using Proxy C.")
    lines.append("")
    lines.append("## 2. Label Definition")
    lines.append("- bid_quote_toxic_binary / ask_quote_toxic_binary")
    lines.append("- fill proxy: price_level_removed (LOB proxy, not true fill)")
    lines.append("- toxic threshold: markout_ticks < -0.5")
    lines.append("")
    lines.append("## 3. Label Distribution")
    meta = json.loads((DATASETS_DIR / "primary_dataset_metadata.json").read_text(encoding="utf-8"))
    for side in ["bid", "ask"]:
        lines.append(f"### {side}")
        for split_key in ["train_stats", "val_stats", "test_stats"]:
            st = meta[side][split_key]
            lines.append(
                f"- {split_key.replace('_stats','')}: samples={int(st['num_samples_after_conditioning'])}, "
                f"toxic_rate={st['toxic_rate']:.6f}, fill_proxy_rate={st['fill_proxy_rate']:.6f}"
            )
    lines.append("")
    lines.append("## 4. Training Convergence")
    for _, row in train_summary.iterrows():
        lines.append(
            f"- {row['side']}: best_epoch={int(row['best_epoch'])}, "
            f"best_val_loss={row['best_val_loss']:.6f}, val_pr_auc={row['val_auc_pr']:.6f}, test_pr_auc={row['test_auc_pr']:.6f}"
        )
    lines.append("")
    lines.append("## 5. Classification Results")
    lines.append("- See outputs/models/*/classification_report_test.csv and curve PNGs.")
    lines.append("")
    lines.append("## 6. Threshold Diagnostics")
    lines.append("- See outputs/models/*/threshold_diagnostics_test.csv.")
    lines.append("")
    lines.append("## 7. Trading Backtest")
    lines.append(
        f"- Best NEW test binary setup: {best_thr_text}, net_pnl={float(best['net_pnl']):.2f}, "
        f"random+1std={float(random_plus_1std):.2f}, smoothed_ref={smoothed_baseline_net:.2f}"
    )
    lines.append("")
    lines.append("## 8. OOS / NEW3 Stress Test")
    lines.append(f"- NEW3 net_pnl at best NEW setup: {new3_net:.2f}")
    lines.append("")
    lines.append("## 9. Verdict")
    if recommendation == "replace_smoothed_baseline":
        verdict = "A. New binary label is better than smoothed baseline."
    elif recommendation == "keep_as_auxiliary_signal":
        verdict = "B. New binary label is useful as auxiliary signal but not replacement."
    else:
        verdict = "C. New binary label converges but no trading edge."
    lines.append(f"- {verdict}")
    lines.append("")
    lines.append("## 10. Next Steps")
    if recommendation in {"replace_smoothed_baseline", "keep_as_auxiliary_signal"}:
        lines.append("- integrate with regime gate")
        lines.append("- validation threshold selection")
        lines.append("- test FPGA-friendly version")
    else:
        lines.append("- keep smoothed_mid_move_3class as main baseline")
        lines.append("- focus on regime gate / rule-based FPGA MVP")
    (REPORTS_DIR / "new_binary_model_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Final checkpoint
    checkpoint_lines = []
    checkpoint_lines.append("# Final Checkpoint")
    checkpoint_lines.append("")
    checkpoint_lines.append("## 1. Label distribution")
    for side in ["bid", "ask"]:
        st = meta[side]
        checkpoint_lines.append(
            f"- {side} toxic rate train/val/test: "
            f"{st['train_stats']['toxic_rate']:.6f} / {st['val_stats']['toxic_rate']:.6f} / {st['test_stats']['toxic_rate']:.6f}"
        )
    checkpoint_lines.append("")
    checkpoint_lines.append("## 2. Best bid model")
    checkpoint_lines.append(
        f"- val loss={float(bid_row['best_val_loss']):.6f}, val PR-AUC={float(bid_row['val_auc_pr']):.6f}, "
        f"test PR-AUC={float(bid_row['test_auc_pr']):.6f}, best threshold={float(bid_row['best_threshold_by_f1']):.2f}"
    )
    checkpoint_lines.append("")
    checkpoint_lines.append("## 3. Best ask model")
    checkpoint_lines.append(
        f"- val loss={float(ask_row['best_val_loss']):.6f}, val PR-AUC={float(ask_row['val_auc_pr']):.6f}, "
        f"test PR-AUC={float(ask_row['test_auc_pr']):.6f}, best threshold={float(ask_row['best_threshold_by_f1']):.2f}"
    )
    checkpoint_lines.append("")
    checkpoint_lines.append("## 4. Best NEW test trading result")
    checkpoint_lines.append(
        f"- net_pnl={float(best['net_pnl']):.2f}, random mean+1std={float(random_plus_1std):.2f}, "
        f"smoothed baseline net_pnl={smoothed_baseline_net:.2f}, recommendation={recommendation}"
    )
    checkpoint_lines.append("")
    checkpoint_lines.append("## 5. NEW3 result")
    checkpoint_lines.append(
        f"- net_pnl={new3_net:.2f}; compare with prior regime-gated benchmarks carefully."
    )
    checkpoint_lines.append("")
    checkpoint_lines.append("## 6. Final recommendation")
    checkpoint_lines.append(f"- {recommendation}")
    (REPORTS_DIR / "final_checkpoint.md").write_text("\n".join(checkpoint_lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    simulate_with_metrics = _import_simulator()
    cfg = _build_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bid_model, bid_cfg = _load_model("bid", device)
    ask_model, ask_cfg = _load_model("ask", device)
    bid_tag = f"{bid_cfg.get('side')}:{bid_cfg.get('config_name')}"
    ask_tag = f"{ask_cfg.get('side')}:{ask_cfg.get('config_name')}"

    # Prepare NEW split based on session order.
    new_lob, new_trade = _load_symbol(NEW_ALL_DIR, NEW_SYMBOL)
    all_sessions = sorted(new_lob["session_id"].unique().tolist())
    _, _, test_sessions = split_ordered_session_ids(all_sessions, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)
    new_test_frame, tick_new = _build_processed_frames(new_lob, new_trade, NEW_SYMBOL, session_ids=test_sessions)

    # Prepare NEW3 full.
    new3_lob, new3_trade = _load_symbol(NEW3_DIR, NEW3_SYMBOL)
    new3_frame, tick_new3 = _build_processed_frames(new3_lob, new3_trade, NEW3_SYMBOL, session_ids=None)

    # Predict event-level probabilities.
    print("Predicting NEW test probabilities...")
    new_bid_probs = _predict_all_event_probs(bid_model, new_test_frame, device=device, batch_size=2048)
    new_ask_probs = _predict_all_event_probs(ask_model, new_test_frame, device=device, batch_size=2048)

    print("Predicting NEW3 full probabilities...")
    new3_bid_probs = _predict_all_event_probs(bid_model, new3_frame, device=device, batch_size=2048)
    new3_ask_probs = _predict_all_event_probs(ask_model, new3_frame, device=device, batch_size=2048)

    save_json(
        BACKTESTS_DIR / "prediction_shapes.json",
        {
            "NEW_test_events": int(len(new_test_frame)),
            "NEW3_full_events": int(len(new3_frame)),
            "NEW_tick_size": float(tick_new),
            "NEW3_tick_size": float(tick_new3),
        },
    )

    df_new = _run_backtest_for_thresholds(
        dataset_name="NEW_test",
        frame=new_test_frame,
        tick_size=tick_new,
        bid_probs=new_bid_probs,
        ask_probs=new_ask_probs,
        simulate_with_metrics=simulate_with_metrics,
        cfg=cfg,
        bid_model_tag=bid_tag,
        ask_model_tag=ask_tag,
    )
    df_new3 = _run_backtest_for_thresholds(
        dataset_name="NEW3_full",
        frame=new3_frame,
        tick_size=tick_new3,
        bid_probs=new3_bid_probs,
        ask_probs=new3_ask_probs,
        simulate_with_metrics=simulate_with_metrics,
        cfg=cfg,
        bid_model_tag=bid_tag,
        ask_model_tag=ask_tag,
    )
    summary_df = pd.concat([df_new, df_new3], ignore_index=True)
    summary_df.to_csv(BACKTESTS_DIR / "binary_backtest_summary.csv", index=False)

    _write_comparison_reports(
        backtest_df=summary_df,
        new_test_frame=new_test_frame,
        new3_frame=new3_frame,
        tick_new=tick_new,
        tick_new3=tick_new3,
        simulate_with_metrics=simulate_with_metrics,
        cfg=cfg,
    )

    print("Binary backtest done.")
    print(f"Summary: {BACKTESTS_DIR / 'binary_backtest_summary.csv'}")


if __name__ == "__main__":
    main()
