from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))


import torch

from src.models.tcn import HLSFriendlyTCNClassifier


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs"
DEFAULT_LABELS = DEFAULT_OUT_DIR / "binary_labels_v2.parquet"
DEFAULT_THRESHOLDS = DEFAULT_OUT_DIR / "selected_thresholds.json"

Z_FEATURES = [
    "obi_z",
    "queue_imbalance_z",
    "weighted_mid_deviation_z",
    "trade_intensity_short_z",
    "deep_obi_z",
    "queue_imbalance_change_z",
    "normalized_spread_z",
    "cancel_pressure_short_z",
]
SIDES = ["bid", "ask"]


@dataclass
class SessionPredictions:
    frame: pd.DataFrame
    bid_probs: np.ndarray
    ask_probs: np.ndarray
    pull_bid: np.ndarray
    pull_ask: np.ndarray


def _import_simulator_and_cfg():
    diag_path = PROJECT_ROOT / "new_era" / "15_models_playaround" / "run_diagnosis_1.py"
    diag_parent = diag_path.parent
    if str(diag_parent) not in sys.path:
        sys.path.insert(0, str(diag_parent))
    spec = importlib.util.spec_from_file_location("run_diagnosis_1_binary_v2", diag_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import simulator from {diag_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import _common  # type: ignore

    return module.simulate_with_metrics, _common.build_cfg()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_model(model_dir: Path, side: str, device: torch.device):
    side_dir = model_dir / f"{side}_model"
    config = json.loads((side_dir / "model_config.json").read_text(encoding="utf-8"))
    model = HLSFriendlyTCNClassifier(
        num_inputs=len(Z_FEATURES),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=int(config["window_size"]),
    ).to(device)
    model.load_state_dict(torch.load(side_dir / "best_model.pt", map_location=device))
    model.eval()
    return model, config


def _predict_session(
    *,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features = frame[Z_FEATURES].to_numpy(dtype=np.float32, copy=True)
    n = len(frame)
    probs = np.zeros(n, dtype=np.float32)
    if n < window_size:
        return probs

    views = np.lib.stride_tricks.sliding_window_view(features, window_shape=int(window_size), axis=0)
    positions = np.arange(int(window_size) - 1, n, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(positions), int(batch_size)):
            pos = positions[start : start + int(batch_size)]
            window_start = pos - int(window_size) + 1
            x_np = views[window_start]
            x = torch.from_numpy(x_np.copy()).to(device)
            logits = model(x).squeeze(1)
            probs[pos] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return probs


def _predict_test_sessions(
    *,
    labels: pd.DataFrame,
    model_dir: Path,
    thresholds: dict[str, float],
    device: torch.device,
    batch_size: int,
) -> list[SessionPredictions]:
    bid_model, bid_cfg = _load_model(model_dir, "bid", device)
    ask_model, ask_cfg = _load_model(model_dir, "ask", device)
    if int(bid_cfg["window_size"]) != int(ask_cfg["window_size"]):
        raise RuntimeError("Bid and ask model window sizes differ.")
    window_size = int(bid_cfg["window_size"])

    test = labels[labels["split"] == "test"].copy()
    if test.empty:
        raise RuntimeError("No test rows in labels parquet.")

    out: list[SessionPredictions] = []
    for session_id, group in test.groupby("session_id", sort=True):
        frame = group.sort_values("local_row_id").reset_index(drop=True)
        bid_probs = _predict_session(
            model=bid_model,
            frame=frame,
            window_size=window_size,
            batch_size=batch_size,
            device=device,
        )
        ask_probs = _predict_session(
            model=ask_model,
            frame=frame,
            window_size=window_size,
            batch_size=batch_size,
            device=device,
        )
        pull_bid = bid_probs >= float(thresholds["bid"])
        pull_ask = ask_probs >= float(thresholds["ask"])
        out.append(
            SessionPredictions(
                frame=frame,
                bid_probs=bid_probs,
                ask_probs=ask_probs,
                pull_bid=pull_bid,
                pull_ask=pull_ask,
            )
        )
    return out


def _run_one_session(
    *,
    pred: SessionPredictions,
    cfg,
    simulate_with_metrics,
    pull_bid: np.ndarray,
    pull_ask: np.ndarray,
    method: str,
    seed: int | None = None,
) -> dict[str, Any]:
    frame = pred.frame
    ask = frame["ask_px_1"].to_numpy(dtype=np.float64, copy=False)
    bid = frame["bid_px_1"].to_numpy(dtype=np.float64, copy=False)
    tick_size = float(frame["tick_size"].median())
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
    return {
        "method": method,
        "seed": seed if seed is not None else "",
        "dataset": str(frame["dataset"].iloc[0]),
        "symbol": str(frame["symbol"].iloc[0]),
        "split": "test",
        "session_id": str(frame["session_id"].iloc[0]),
        "rows": int(len(frame)),
        "bid_signal_rate": float(np.mean(pull_bid)) if len(pull_bid) else float("nan"),
        "ask_signal_rate": float(np.mean(pull_ask)) if len(pull_ask) else float("nan"),
        "any_side_signal_rate": float(np.mean(pull_bid | pull_ask)) if len(pull_bid) else float("nan"),
        "trade_count": int(res["trade_count"]),
        "gross_pnl": float(res["gross_pnl"]),
        "net_pnl": float(res["net_pnl"]),
        "fee": float(res["fee"]),
        "max_drawdown": float(res["max_drawdown"]),
        "avg_pnl_per_fill": float(res["avg_pnl_per_fill"]),
        "net_cancel_value": float(res["net_cancel_value"]),
        "missed_toxic_pnl_avoided": float(res["missed_toxic_pnl_avoided"]),
        "missed_profitable_pnl": float(res["missed_profitable_pnl"]),
        "realized_bid_pnl": float(res["realized_bid_pnl"]),
        "realized_ask_pnl": float(res["realized_ask_pnl"]),
        "bid_fill_count": int(res["bid_fill_count"]),
        "ask_fill_count": int(res["ask_fill_count"]),
    }


def _aggregate_session_rows(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    return {
        "method": method,
        "sessions": int(len(df)),
        "rows": int(df["rows"].sum()),
        "bid_signal_rate": float(np.average(df["bid_signal_rate"], weights=df["rows"])),
        "ask_signal_rate": float(np.average(df["ask_signal_rate"], weights=df["rows"])),
        "any_side_signal_rate": float(np.average(df["any_side_signal_rate"], weights=df["rows"])),
        "trade_count": int(df["trade_count"].sum()),
        "gross_pnl": float(df["gross_pnl"].sum()),
        "net_pnl": float(df["net_pnl"].sum()),
        "fee": float(df["fee"].sum()),
        "max_session_drawdown": float(df["max_drawdown"].max()),
        "net_cancel_value": float(df["net_cancel_value"].sum()),
        "missed_toxic_pnl_avoided": float(df["missed_toxic_pnl_avoided"].sum()),
        "missed_profitable_pnl": float(df["missed_profitable_pnl"].sum()),
        "realized_bid_pnl": float(df["realized_bid_pnl"].sum()),
        "realized_ask_pnl": float(df["realized_ask_pnl"].sum()),
        "bid_fill_count": int(df["bid_fill_count"].sum()),
        "ask_fill_count": int(df["ask_fill_count"].sum()),
    }


def _matched_random_runs(
    *,
    predictions: list[SessionPredictions],
    cfg,
    simulate_with_metrics,
    bid_signal_rate: float,
    ask_signal_rate: float,
    seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for seed in range(int(seeds)):
        rng = np.random.default_rng(seed)
        rows_this_seed = []
        for pred in predictions:
            n = len(pred.frame)
            pull_bid = rng.random(n) < float(bid_signal_rate)
            pull_ask = rng.random(n) < float(ask_signal_rate)
            row = _run_one_session(
                pred=pred,
                cfg=cfg,
                simulate_with_metrics=simulate_with_metrics,
                pull_bid=pull_bid,
                pull_ask=pull_ask,
                method="matched_random",
                seed=seed,
            )
            rows_this_seed.append(row)
            session_rows.append(row)
        agg = _aggregate_session_rows(rows_this_seed, method="matched_random")
        agg["seed"] = seed
        seed_rows.append(agg)

    return pd.DataFrame(seed_rows), pd.DataFrame(session_rows)


def run_backtest(
    *,
    labels_path: Path,
    thresholds_path: Path,
    out_dir: Path,
    random_seeds: int,
    batch_size: int,
    device_arg: str,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds = threshold_payload["selected_thresholds"]
    model_dir = Path(threshold_payload["model_dir"])
    if threshold_payload.get("selection_split") != "val" or bool(threshold_payload.get("test_used")):
        raise RuntimeError("selected_thresholds.json does not certify validation-only selection.")

    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)

    device = _resolve_device(device_arg)
    simulate_with_metrics, cfg = _import_simulator_and_cfg()
    predictions = _predict_test_sessions(
        labels=labels,
        model_dir=model_dir,
        thresholds={"bid": float(thresholds["bid"]), "ask": float(thresholds["ask"])},
        device=device,
        batch_size=batch_size,
    )

    model_session_rows = [
        _run_one_session(
            pred=pred,
            cfg=cfg,
            simulate_with_metrics=simulate_with_metrics,
            pull_bid=pred.pull_bid,
            pull_ask=pred.pull_ask,
            method="model_frozen_val_threshold",
        )
        for pred in predictions
    ]
    model_agg = _aggregate_session_rows(model_session_rows, method="model_frozen_val_threshold")
    model_agg["threshold_bid"] = float(thresholds["bid"])
    model_agg["threshold_ask"] = float(thresholds["ask"])
    model_agg["selection_split"] = "val"
    model_agg["test_evaluation_count"] = 1

    random_seed_df, random_session_df = _matched_random_runs(
        predictions=predictions,
        cfg=cfg,
        simulate_with_metrics=simulate_with_metrics,
        bid_signal_rate=float(model_agg["bid_signal_rate"]),
        ask_signal_rate=float(model_agg["ask_signal_rate"]),
        seeds=random_seeds,
    )
    random_summary = {
        "method": "matched_random_cancel",
        "num_runs": int(random_seeds),
        "matched_bid_signal_rate": float(model_agg["bid_signal_rate"]),
        "matched_ask_signal_rate": float(model_agg["ask_signal_rate"]),
        "mean_net_pnl": float(random_seed_df["net_pnl"].mean()),
        "std_net_pnl": float(random_seed_df["net_pnl"].std(ddof=0)),
        "mean_plus_1std_net_pnl": float(random_seed_df["net_pnl"].mean() + random_seed_df["net_pnl"].std(ddof=0)),
        "mean_trade_count": float(random_seed_df["trade_count"].mean()),
        "mean_net_cancel_value": float(random_seed_df["net_cancel_value"].mean()),
    }
    model_agg["random_mean_net_pnl"] = random_summary["mean_net_pnl"]
    model_agg["random_std_net_pnl"] = random_summary["std_net_pnl"]
    model_agg["random_mean_plus_1std_net_pnl"] = random_summary["mean_plus_1std_net_pnl"]
    model_agg["beats_random_mean"] = bool(model_agg["net_pnl"] > random_summary["mean_net_pnl"])
    model_agg["beats_random_plus_1std"] = bool(model_agg["net_pnl"] > random_summary["mean_plus_1std_net_pnl"])

    test_summary_path = out_dir / "test_backtest_summary.csv"
    random_summary_path = out_dir / "random_baseline_summary.csv"
    pnl_by_session_path = out_dir / "pnl_by_session.csv"
    predictions_path = out_dir / "test_predictions_binary_cancel.parquet"
    random_runs_path = out_dir / "random_baseline_runs.csv"

    pd.DataFrame([model_agg]).to_csv(test_summary_path, index=False)
    pd.DataFrame([random_summary]).to_csv(random_summary_path, index=False)
    pd.DataFrame(model_session_rows).to_csv(pnl_by_session_path, index=False)
    random_seed_df.to_csv(random_runs_path, index=False)

    pred_frames = []
    for pred in predictions:
        pf = pred.frame[
            ["event_id", "dataset", "symbol", "session_id", "split", "local_row_id", "bid_px_1", "ask_px_1", "mid_price"]
        ].copy()
        pf["bid_prob_toxic"] = pred.bid_probs
        pf["ask_prob_toxic"] = pred.ask_probs
        pf["pull_bid"] = pred.pull_bid
        pf["pull_ask"] = pred.pull_ask
        pred_frames.append(pf)
    pd.concat(pred_frames, ignore_index=True).to_parquet(predictions_path, index=False)

    metadata = {
        "labels_path": str(labels_path),
        "thresholds_path": str(thresholds_path),
        "model_dir": str(model_dir),
        "thresholds": {"bid": float(thresholds["bid"]), "ask": float(thresholds["ask"])},
        "random_seeds": int(random_seeds),
        "test_evaluation_count": 1,
        "test_policy": "Thresholds loaded from selected_thresholds.json and applied blindly to test.",
        "outputs": {
            "test_backtest_summary": str(test_summary_path),
            "random_baseline_summary": str(random_summary_path),
            "pnl_by_session": str(pnl_by_session_path),
            "random_baseline_runs": str(random_runs_path),
            "test_predictions": str(predictions_path),
        },
    }
    (out_dir / "test_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return test_summary_path, random_summary_path, pnl_by_session_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen validation thresholds to test and compare with matched random.")
    parser.add_argument("--labels_path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--thresholds_path", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random_seeds", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_summary, random_summary, pnl_by_session = run_backtest(
        labels_path=args.labels_path,
        thresholds_path=args.thresholds_path,
        out_dir=args.out_dir,
        random_seeds=args.random_seeds,
        batch_size=args.batch_size,
        device_arg=args.device,
    )
    print("Frozen-threshold test backtest complete:")
    print(f"- {test_summary}")
    print(f"- {random_summary}")
    print(f"- {pnl_by_session}")


if __name__ == "__main__":
    main()
