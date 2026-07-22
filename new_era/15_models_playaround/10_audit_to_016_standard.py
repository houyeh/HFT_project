from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(V16_DIR) not in sys.path:
    sys.path.insert(0, str(V16_DIR))
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from round2_audit_utils import RESULTS as V16_RESULTS
from round2_audit_utils import load_test_frame, simulate_cancel_policy, threshold_for_target_rate
from src.models.tcn import HLSFriendlyTCNClassifier


F8_Z_FEATURES = [
    "obi_z",
    "queue_imbalance_z",
    "weighted_mid_deviation_z",
    "trade_intensity_short_z",
    "deep_obi_z",
    "queue_imbalance_change_z",
    "normalized_spread_z",
    "cancel_pressure_short_z",
]

DEFAULT_015_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "new_era" / "18_queue_aware" / "015_common_audit"


def _load_strict_random_module():
    sys.path.insert(0, str(V16_DIR))
    path = V16_DIR / "15_strict_random_baseline.py"
    spec = importlib.util.spec_from_file_location("strict_random_common_015", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strict_random_common_015"] = module
    spec.loader.exec_module(module)
    return module


def _predict_015_on_common(frame: pd.DataFrame, model_dir: Path, batch_size: int, device_arg: str) -> pd.DataFrame:
    selected = json.loads((model_dir / "backtest_outputs" / "selected_strategy_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "experiment_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))

    model = HLSFriendlyTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=list(cfg["model"]["channel_list"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        seq_length=int(cfg["window"]["window_size"]),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "checkpoints" / "tcn_weights.pth", map_location=device, weights_only=True))
    model.eval()

    out_frames = []
    window_size = int(cfg["window"]["window_size"])
    for session_id, group in frame.groupby("session_id", sort=True):
        g = group.sort_values("local_row_id").reset_index(drop=True).copy()
        features = g[F8_Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
        probs = np.zeros((len(g), 3), dtype=np.float32)
        available = np.zeros(len(g), dtype=bool)
        if len(g) >= window_size:
            views = np.lib.stride_tricks.sliding_window_view(features, window_shape=window_size, axis=0)
            positions = np.arange(window_size - 1, len(g), dtype=np.int64)
            with torch.no_grad():
                for start in range(0, len(positions), int(batch_size)):
                    pos = positions[start : start + int(batch_size)]
                    window_start = pos - window_size + 1
                    x = torch.from_numpy(views[window_start].copy()).to(device)
                    logits = model(x)
                    probs[pos] = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
                    available[pos] = True

        g["prob_down_015"] = probs[:, 0]
        g["prob_stationary_015"] = probs[:, 1]
        g["prob_up_015"] = probs[:, 2]
        g["prediction_available_015"] = available

        pthr = float(selected["prob_threshold"])
        ethr = float(selected["edge_threshold"])
        max_spread = float(selected["max_spread_ticks"])
        edge = g["prob_up_015"].to_numpy(dtype=float) - g["prob_down_015"].to_numpy(dtype=float)
        spread = g["spread_ticks"].to_numpy(dtype=float)
        wide = spread > max_spread
        pull_bid = wide | ((g["prob_down_015"].to_numpy(dtype=float) >= pthr) & (edge <= -ethr))
        pull_ask = wide | ((g["prob_up_015"].to_numpy(dtype=float) >= pthr) & (edge >= ethr))
        pull_bid &= available
        pull_ask &= available
        g["pull_bid_015_common"] = pull_bid
        g["pull_ask_015_common"] = pull_ask
        g["score_bid_015"] = g["prob_down_015"]
        g["score_ask_015"] = g["prob_up_015"]
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def _score_lift(frame: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for side, score_col, toxic_col in [
        ("bid", "score_bid_015", "bid_toxic"),
        ("ask", "score_ask_015", "ask_toxic"),
    ]:
        scores = frame[score_col].to_numpy(dtype=float)
        y = frame[toxic_col].to_numpy(dtype=int)
        available = frame["prediction_available_015"].to_numpy(dtype=bool)
        base = float(np.mean(y[available])) if available.any() else float("nan")
        for frac, bucket in [(0.10, "top_10pct"), (0.05, "top_5pct"), (0.01, "top_1pct")]:
            sc = scores[available]
            threshold = np.sort(sc)[::-1][max(0, min(len(sc) - 1, int(np.ceil(frac * len(sc))) - 1))]
            mask = available & (scores >= threshold)
            toxic_rate = float(np.mean(y[mask])) if mask.any() else float("nan")
            rows.append(
                {
                    "track": "015_common_universe",
                    "side": side,
                    "bucket": bucket,
                    "num_samples": int(mask.sum()),
                    "mean_score": float(np.mean(scores[mask])) if mask.any() else float("nan"),
                    "toxic_rate": toxic_rate,
                    "base_toxic_rate": base,
                    "lift": float(toxic_rate / base) if base > 0 and np.isfinite(toxic_rate) else float("nan"),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "score_bucket_toxic_lift_015_common.csv", index=False)


def _strict_random_v3(frame: pd.DataFrame, out_dir: Path, trials: int) -> None:
    mod = _load_strict_random_module()
    tmp = frame.copy()
    tmp["pull_bid"] = tmp["pull_bid_015_common"].astype(bool)
    tmp["pull_ask"] = tmp["pull_ask_015_common"].astype(bool)
    action_df, by_session, final_mid = mod._build_actions(tmp)
    specs, health = mod._build_standard_specs(tmp, action_df, method="random_v3", key_cols=["session_id", "spread_bucket"])
    rng = np.random.default_rng(20260524)
    health_df = pd.DataFrame(health)

    trials_path = out_dir / "strict_random_015_common_trials.csv"
    sessions_path = out_dir / "strict_random_015_common_session_trials.csv"
    dist_path = out_dir / "strict_random_015_common_distribution.csv"
    if trials_path.exists() and sessions_path.exists():
        existing_trials = pd.read_csv(trials_path)
        if len(existing_trials) == int(trials):
            model_summary, model_sessions = simulate_cancel_policy(
                tmp,
                tmp["pull_bid_015_common"].to_numpy(dtype=bool),
                tmp["pull_ask_015_common"].to_numpy(dtype=bool),
                method="015_common_universe",
            )
            if not dist_path.exists():
                pnl = existing_trials["net_pnl"].to_numpy(dtype=float)
                pd.DataFrame(
                    [
                        {
                            "track": "015_common_universe",
                            "method": "random_v3",
                            "trials": int(trials),
                            "model_net_pnl": float(model_summary["net_pnl"]),
                            "mean": float(np.mean(pnl)),
                            "std": float(np.std(pnl, ddof=0)),
                            "p95": float(np.percentile(pnl, 95)),
                            "p99": float(np.percentile(pnl, 99)),
                            "empirical_p_value_random_ge_model": float(np.mean(pnl >= float(model_summary["net_pnl"]))),
                        }
                    ]
                ).to_csv(dist_path, index=False)
            health_df.to_csv(out_dir / "strict_random_015_common_health.csv", index=False)
            pd.DataFrame([model_summary]).to_csv(out_dir / "pnl_summary_015_common.csv", index=False)
            model_sessions.to_csv(out_dir / "pnl_by_session_015_common.csv", index=False)
            return

    trial_rows = []
    session_rows = []
    for trial in range(int(trials)):
        canceled = mod._sample_cancels(specs, rng, len(action_df))
        agg, session_df = mod._simulate_actions(tmp, by_session, final_mid, canceled)
        trial_rows.append(
            {
                "method": "random_v3",
                "trial": trial,
                "net_pnl": agg["net_pnl"],
                "trade_count": agg["trade_count"],
                "max_drawdown": agg["max_drawdown"],
            }
        )
        for _, row in session_df.iterrows():
            session_rows.append(
                {
                    "method": "random_v3",
                    "trial": trial,
                    "session_id": row["session_id"],
                    "net_pnl": row["net_pnl"],
                    "trade_count": row["trade_count"],
                    "max_drawdown": row["max_drawdown"],
                }
            )

    trials_df = pd.DataFrame(trial_rows)
    model_summary, model_sessions = simulate_cancel_policy(
        tmp,
        tmp["pull_bid_015_common"].to_numpy(dtype=bool),
        tmp["pull_ask_015_common"].to_numpy(dtype=bool),
        method="015_common_universe",
    )
    pnl = trials_df["net_pnl"].to_numpy(dtype=float)
    dist = pd.DataFrame(
        [
            {
                "track": "015_common_universe",
                "method": "random_v3",
                "trials": int(trials),
                "model_net_pnl": float(model_summary["net_pnl"]),
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl, ddof=0)),
                "p95": float(np.percentile(pnl, 95)),
                "p99": float(np.percentile(pnl, 99)),
                "empirical_p_value_random_ge_model": float(np.mean(pnl >= float(model_summary["net_pnl"]))),
            }
        ]
    )
    dist.to_csv(out_dir / "strict_random_015_common_distribution.csv", index=False)
    trials_df.to_csv(out_dir / "strict_random_015_common_trials.csv", index=False)
    pd.DataFrame(session_rows).to_csv(out_dir / "strict_random_015_common_session_trials.csv", index=False)
    health_df.to_csv(out_dir / "strict_random_015_common_health.csv", index=False)
    pd.DataFrame([model_summary]).to_csv(out_dir / "pnl_summary_015_common.csv", index=False)
    model_sessions.to_csv(out_dir / "pnl_by_session_015_common.csv", index=False)


def _per_session(frame: pd.DataFrame, out_dir: Path) -> None:
    model_sessions = pd.read_csv(out_dir / "pnl_by_session_015_common.csv")
    random_sessions = pd.read_csv(out_dir / "strict_random_015_common_session_trials.csv")
    rows = []
    total = float(model_sessions["net_pnl"].sum())
    for session_id, group in random_sessions.groupby("session_id", sort=True):
        model_pnl = float(model_sessions[model_sessions["session_id"].astype(str) == str(session_id)].iloc[0]["net_pnl"])
        pnl = group["net_pnl"].to_numpy(dtype=float)
        rows.append(
            {
                "track": "015_common_universe",
                "session_id": str(session_id),
                "model_net_pnl": model_pnl,
                "model_share_of_total_pnl": float(model_pnl / total) if total else float("nan"),
                "random_v3_mean": float(np.mean(pnl)),
                "random_v3_p95": float(np.percentile(pnl, 95)),
                "random_v3_p99": float(np.percentile(pnl, 99)),
                "model_beats_random_v3_mean": bool(model_pnl > np.mean(pnl)),
                "model_beats_random_v3_p95": bool(model_pnl > np.percentile(pnl, 95)),
                "model_beats_random_v3_p99": bool(model_pnl > np.percentile(pnl, 99)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "pnl_by_session_model_vs_random_015_common.csv", index=False)


def _threshold_sensitivity(frame: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for target in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        available = frame["prediction_available_015"].to_numpy(dtype=bool)
        bid_thr = threshold_for_target_rate(frame.loc[available, "score_bid_015"].to_numpy(dtype=float), target)
        ask_thr = threshold_for_target_rate(frame.loc[available, "score_ask_015"].to_numpy(dtype=float), target)
        pull_bid = available & (frame["score_bid_015"].to_numpy(dtype=float) >= bid_thr)
        pull_ask = available & (frame["score_ask_015"].to_numpy(dtype=float) >= ask_thr)
        summary, _ = simulate_cancel_policy(frame, pull_bid, pull_ask, method=f"015_common_target_{target:.3f}")
        rows.append(
            {
                "track": "015_common_universe",
                "target_cancel_rate": target,
                "bid_threshold": bid_thr,
                "ask_threshold": ask_thr,
                "bid_cancel_rate": float(np.mean(pull_bid)),
                "ask_cancel_rate": float(np.mean(pull_ask)),
                "net_pnl": float(summary["net_pnl"]),
                "trade_count": int(summary["trade_count"]),
                "is_main_claim": False,
                "used_for_model_selection": False,
                "diagnostic_note": "COMMON-UNIVERSE DIAGNOSTIC ONLY",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "threshold_sensitivity_015_common.csv", index=False)


def _plot_score_distribution(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, side, score, toxic in [
        (axes[0], "bid", "score_bid_015", "bid_toxic"),
        (axes[1], "ask", "score_ask_015", "ask_toxic"),
    ]:
        y = frame[toxic].astype(bool).to_numpy()
        s = frame[score].to_numpy(dtype=float)
        avail = frame["prediction_available_015"].to_numpy(dtype=bool)
        ax.hist(s[avail & ~y], bins=80, density=True, alpha=0.55, label="non-toxic")
        ax.hist(s[avail & y], bins=80, density=True, alpha=0.65, label="toxic")
        ax.set_title(f"015 Common {side.capitalize()} Score")
        ax.set_xlabel("Score")
        ax.legend()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution_pos_neg_015_common.png", dpi=150)
    plt.close(fig)


def run(out_dir: Path, model_dir: Path, trials: int, batch_size: int, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_test_frame()
    missing_features = [c for c in F8_Z_FEATURES if c not in frame.columns]
    if missing_features:
        labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
        labels["event_id"] = np.arange(len(labels), dtype=np.int64)
        frame = frame.merge(labels[["event_id", *F8_Z_FEATURES]], on="event_id", how="left")
        still_missing = [c for c in F8_Z_FEATURES if c not in frame.columns]
        if still_missing:
            raise KeyError(f"Missing required 015 F8 feature columns after label merge: {still_missing}")
    common_path = out_dir / "015_common_predictions.parquet"
    if common_path.exists():
        common = pd.read_parquet(common_path)
    else:
        common = _predict_015_on_common(frame, model_dir, batch_size=batch_size, device_arg=device)
        common.to_parquet(common_path, index=False)

    manifest = json.loads((model_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    original_test_count = int(
        sum(int(np.load(model_dir / "dataset_sequences.npz")[f"TXFB6__{sid}_y"].shape[0]) for sid in ["20260127Day", "20260127Night"])
    )
    common_count = int(len(common))
    prediction_count = int(common["prediction_available_015"].sum())
    alignment = pd.DataFrame(
        [
            {
                "model": "015_smoothed",
                "track": "015_original_universe",
                "original_event_count": original_test_count,
                "common_universe_event_count": np.nan,
                "overlap_count": original_test_count,
                "coverage_rate": 1.0,
                "missing_prediction_rate": 0.0,
                "dropped_event_count": 0,
                "notes": "Historical 015 audit universe from dataset_sequences.npz test sessions.",
            },
            {
                "model": "015_smoothed",
                "track": "015_common_universe",
                "original_event_count": original_test_count,
                "common_universe_event_count": common_count,
                "overlap_count": prediction_count,
                "coverage_rate": float(prediction_count / max(common_count, 1)),
                "missing_prediction_rate": float(1.0 - prediction_count / max(common_count, 1)),
                "dropped_event_count": int(common_count - prediction_count),
                "notes": "015 weights re-scored on 016_v2 quote-eligible test universe; first window-1 rows per session lack prediction.",
            },
        ]
    )
    alignment.to_csv(out_dir / "universe_alignment_report.csv", index=False)

    _score_lift(common, out_dir)
    _strict_random_v3(common, out_dir, trials=trials)
    _per_session(common, out_dir)
    _threshold_sensitivity(common, out_dir)
    _plot_score_distribution(common, out_dir)

    metadata = {
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "trials_random_v3": int(trials),
        "common_predictions": str(out_dir / "015_common_predictions.parquet"),
        "selected_features": F8_Z_FEATURES,
        "original_manifest_sessions": manifest.get("sessions", []),
    }
    (out_dir / "015_common_audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 015 on historical and 016 common universes.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_015_DIR)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.out_dir, args.model_dir, args.trials, args.batch_size, args.device)
    print(f"015 two-track audit written to {args.out_dir}")
