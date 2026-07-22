from __future__ import annotations

import argparse
import json
import sys
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


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs"


def _load_latest_model_dir(out_dir: Path) -> Path:
    latest_path = out_dir / "latest_training_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing {latest_path}. Run 04_train_binary_model.py first or pass --model_dir."
        )
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    model_dir = Path(latest["model_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory from latest_training_run.json does not exist: {model_dir}")
    return model_dir


def _parse_rates(text: str) -> list[float]:
    rates = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Cancel-rate targets must be between 0 and 1, got {value}")
        rates.append(value)
    if not rates:
        raise ValueError("At least one target cancel rate is required")
    return rates


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _side_markout_ticks(frame: pd.DataFrame, side: str) -> np.ndarray:
    future_mid = frame["future_mid_return_H_ticks"].to_numpy(dtype=np.float64, copy=False)
    if side == "bid":
        return future_mid
    if side == "ask":
        return -future_mid
    raise ValueError(f"Unknown side: {side}")


def _evaluate_side_threshold(frame: pd.DataFrame, side: str, threshold: float) -> dict[str, Any]:
    probs = frame["prob_toxic"].to_numpy(dtype=np.float64, copy=False)
    y_true = frame["y_true"].to_numpy(dtype=np.int64, copy=False)
    fill = frame["fill_proxy"].to_numpy(dtype=bool, copy=False)
    markout = _side_markout_ticks(frame, side)
    valid = np.isfinite(probs) & np.isfinite(markout)
    cancel = valid & (probs >= float(threshold))

    cancel_fill = cancel & fill
    toxic = y_true == 1
    toxic_cancel = cancel & toxic
    non_toxic_fill_cancel = cancel_fill & ~toxic
    proxy_cancel_value_ticks = np.where(cancel_fill, -markout, 0.0)

    return {
        "side": side,
        "threshold": float(threshold),
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "cancel_count": int(cancel.sum()),
        "cancel_rate": float(cancel.mean()) if len(cancel) else float("nan"),
        "fill_cancel_count": int(cancel_fill.sum()),
        "toxic_count": int(toxic.sum()),
        "toxic_cancel_count": int(toxic_cancel.sum()),
        "non_toxic_fill_cancel_count": int(non_toxic_fill_cancel.sum()),
        "toxic_precision_per_cancel": float(toxic_cancel.sum() / max(cancel.sum(), 1)),
        "toxic_recall": float(toxic_cancel.sum() / max(toxic.sum(), 1)),
        "proxy_cancel_value_ticks": float(np.nansum(proxy_cancel_value_ticks)),
        "proxy_cancel_value_ticks_per_cancel": float(np.nansum(proxy_cancel_value_ticks) / max(cancel.sum(), 1)),
        "proxy_cancel_value_ticks_per_fill_cancel": float(np.nansum(proxy_cancel_value_ticks) / max(cancel_fill.sum(), 1)),
    }


def _threshold_for_target_rate(probs: np.ndarray, target_rate: float) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return float("nan")
    if target_rate <= 0:
        return float(np.inf)
    if target_rate >= 1:
        return float(-np.inf)
    sorted_probs = np.sort(probs)[::-1]
    kth = max(0, min(len(sorted_probs) - 1, int(np.ceil(target_rate * len(sorted_probs))) - 1))
    return float(sorted_probs[kth])


def _build_side_candidates(
    frame: pd.DataFrame,
    side: str,
    *,
    target_rates: list[float],
    grid_size: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    probs = frame["prob_toxic"].to_numpy(dtype=np.float64, copy=False)
    candidates: list[dict[str, Any]] = []

    for rate in target_rates:
        threshold = _threshold_for_target_rate(probs, rate)
        row = _evaluate_side_threshold(frame, side, threshold)
        row["candidate_type"] = "target_cancel_rate"
        row["target_cancel_rate"] = float(rate)
        candidates.append(row)

    finite_probs = probs[np.isfinite(probs)]
    if finite_probs.size:
        grid = np.unique(np.quantile(finite_probs, np.linspace(0.01, 0.99, int(grid_size))))
    else:
        grid = np.array([0.5], dtype=np.float64)

    for threshold in grid:
        row = _evaluate_side_threshold(frame, side, float(threshold))
        row["candidate_type"] = "proxy_value_grid"
        row["target_cancel_rate"] = np.nan
        candidates.append(row)

    candidate_df = pd.DataFrame(candidates)
    best_proxy = candidate_df[candidate_df["candidate_type"] == "proxy_value_grid"].sort_values(
        ["proxy_cancel_value_ticks", "toxic_recall"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    target_rows = candidate_df[candidate_df["candidate_type"] == "target_cancel_rate"].copy()
    target_by_rate = {
        f"{float(row['target_cancel_rate']):.4f}": row.to_dict()
        for _, row in target_rows.iterrows()
    }
    return candidate_df, best_proxy, target_by_rate


def _combined_summary(
    bid_frame: pd.DataFrame,
    ask_frame: pd.DataFrame,
    *,
    bid_threshold: float,
    ask_threshold: float,
) -> dict[str, Any]:
    bid = bid_frame[["event_id", "prob_toxic", "fill_proxy", "y_true", "future_mid_return_H_ticks"]].copy()
    ask = ask_frame[["event_id", "prob_toxic", "fill_proxy", "y_true", "future_mid_return_H_ticks"]].copy()
    bid.rename(
        columns={
            "prob_toxic": "bid_prob",
            "fill_proxy": "bid_fill_proxy",
            "y_true": "bid_y_true",
            "future_mid_return_H_ticks": "future_mid_return_H_ticks",
        },
        inplace=True,
    )
    ask.rename(
        columns={
            "prob_toxic": "ask_prob",
            "fill_proxy": "ask_fill_proxy",
            "y_true": "ask_y_true",
        },
        inplace=True,
    )
    merged = bid.merge(ask, on="event_id", how="inner", suffixes=("", "_ask"))
    future_mid = merged["future_mid_return_H_ticks"].to_numpy(dtype=np.float64, copy=False)

    bid_cancel = merged["bid_prob"].to_numpy(dtype=np.float64) >= float(bid_threshold)
    ask_cancel = merged["ask_prob"].to_numpy(dtype=np.float64) >= float(ask_threshold)
    bid_fill = merged["bid_fill_proxy"].to_numpy(dtype=bool)
    ask_fill = merged["ask_fill_proxy"].to_numpy(dtype=bool)
    bid_value = np.where(bid_cancel & bid_fill, -future_mid, 0.0)
    ask_value = np.where(ask_cancel & ask_fill, future_mid, 0.0)

    actions = 2 * len(merged)
    return {
        "rows": int(len(merged)),
        "bid_threshold": float(bid_threshold),
        "ask_threshold": float(ask_threshold),
        "bid_cancel_rate": float(bid_cancel.mean()) if len(merged) else float("nan"),
        "ask_cancel_rate": float(ask_cancel.mean()) if len(merged) else float("nan"),
        "combined_action_cancel_rate": float((bid_cancel.sum() + ask_cancel.sum()) / max(actions, 1)),
        "any_side_cancel_rate": float((bid_cancel | ask_cancel).mean()) if len(merged) else float("nan"),
        "proxy_cancel_value_ticks": float(np.nansum(bid_value + ask_value)),
        "bid_proxy_cancel_value_ticks": float(np.nansum(bid_value)),
        "ask_proxy_cancel_value_ticks": float(np.nansum(ask_value)),
    }


def select_thresholds(
    *,
    out_dir: Path,
    model_dir: Path | None,
    selection_mode: str,
    target_cancel_rate: float,
    target_rates: list[float],
    grid_size: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_dir = model_dir if model_dir is not None else _load_latest_model_dir(out_dir)

    bid_frame = pd.read_parquet(resolved_model_dir / "val_predictions_bid.parquet")
    ask_frame = pd.read_parquet(resolved_model_dir / "val_predictions_ask.parquet")

    bid_candidates, bid_best_proxy, bid_targets = _build_side_candidates(
        bid_frame,
        "bid",
        target_rates=target_rates,
        grid_size=grid_size,
    )
    ask_candidates, ask_best_proxy, ask_targets = _build_side_candidates(
        ask_frame,
        "ask",
        target_rates=target_rates,
        grid_size=grid_size,
    )

    candidates = pd.concat([bid_candidates, ask_candidates], ignore_index=True)
    candidates.to_csv(out_dir / "val_threshold_candidates.csv", index=False)

    target_key = f"{float(target_cancel_rate):.4f}"
    if selection_mode == "target_cancel_rate":
        if target_key not in bid_targets or target_key not in ask_targets:
            raise ValueError(f"target_cancel_rate={target_cancel_rate} was not included in --target_rates")
        selected_bid = bid_targets[target_key]
        selected_ask = ask_targets[target_key]
    elif selection_mode == "maximize_proxy_value":
        selected_bid = bid_best_proxy
        selected_ask = ask_best_proxy
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    combined = _combined_summary(
        bid_frame,
        ask_frame,
        bid_threshold=float(selected_bid["threshold"]),
        ask_threshold=float(selected_ask["threshold"]),
    )

    payload = {
        "selection_split": "val",
        "test_used": False,
        "model_dir": str(resolved_model_dir),
        "selection_mode": selection_mode,
        "target_cancel_rate": float(target_cancel_rate),
        "selected_thresholds": {
            "bid": float(selected_bid["threshold"]),
            "ask": float(selected_ask["threshold"]),
        },
        "selected_val_metrics": {
            "bid": selected_bid,
            "ask": selected_ask,
            "combined": combined,
        },
        "target_cancel_rate_candidates": {
            "bid": bid_targets,
            "ask": ask_targets,
        },
        "max_proxy_value_candidates": {
            "bid": bid_best_proxy,
            "ask": ask_best_proxy,
        },
        "notes": [
            "Thresholds are selected exclusively from validation predictions.",
            "Proxy cancel value is measured in ticks using fill_proxy and future_mid_return_H_ticks.",
            "The test split is not read or evaluated in this script.",
        ],
    }

    out_path = out_dir / "selected_thresholds.json"
    out_path.write_text(json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select bid/ask cancel thresholds on validation only.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument(
        "--selection_mode",
        choices=["target_cancel_rate", "maximize_proxy_value"],
        default="target_cancel_rate",
    )
    parser.add_argument("--target_cancel_rate", type=float, default=0.10)
    parser.add_argument("--target_rates", type=str, default="0.05,0.10")
    parser.add_argument("--grid_size", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = select_thresholds(
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        selection_mode=args.selection_mode,
        target_cancel_rate=args.target_cancel_rate,
        target_rates=_parse_rates(args.target_rates),
        grid_size=args.grid_size,
    )
    print("Validation-only threshold selection complete:")
    print(f"- {out_path}")


if __name__ == "__main__":
    main()
