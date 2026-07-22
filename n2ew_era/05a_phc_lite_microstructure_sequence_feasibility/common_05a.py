from __future__ import annotations

import argparse
import json
import math
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from models.phc_lite import PHCLite, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
PHASE04B_ACTIONS = PHASE04B / "data" / "txf_only_action_universe.parquet"
PHASE04B1 = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
HGB_MODEL_PATH = PHASE04B1 / "models" / "hgb_regressor_iter_120.joblib"
HGB_CLOSURE_PATH = PHASE04B1 / "results" / "txf_hgb_convergence_closure.json"

HORIZON = 20
SEQ_LEN = 128
STRIDE = 32
RANDOM_TRIALS = 1000
RANDOM_SEED = 20260602
ACTION_BID = 0
ACTION_ASK = 1
ACTION_NO_QUOTE = 2
DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2
ALL_SPLITS = ["train", "val", "new3", "new4"]
OOS_SPLITS = ["new3", "new4"]

BASE_EVENT_COLUMNS = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
]

PHC_FEATURES = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
    "log1p_bid_sz_1",
    "log1p_ask_sz_1",
    "log1p_top_depth",
    "clipped_inter_arrival_ms",
    "zscore_volatility_past_50_by_train_stats",
]

TARGET_RAW_COLUMNS = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "abs_return_past_50",
    "imbalance_change_past",
    "spread_change_past",
    "hawkes_excite",
]

HGB_ML_FEATURES = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "abs_return_past_50",
    "imbalance_change_past",
    "spread_change_past",
    "hawkes_excite",
    "action_is_bid",
    "action_is_ask",
    "side_sign",
    "side_queue_imbalance",
    "side_deep_imbalance",
    "side_imbalance_change_past",
]

FORBIDDEN_FEATURE_PATTERNS = [
    "best_bid",
    "best_ask",
    "mid_price",
    "quote_price",
    "product_id",
    "future_mid",
    "raw_pnl",
    "excess_pnl",
    "positive_excess",
    "terminal_quote_ev",
    "session_final_ev",
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def part_paths(split: str) -> list[Path]:
    base = PHASE04B_ACTIONS / f"split={split}"
    if not base.exists():
        return []
    return sorted(base.glob("*.parquet"))


def assert_no_new5_path(path: Path) -> None:
    text = str(path).lower()
    if "new5" in text:
        raise RuntimeError(f"Phase 05A hard rule violation: NEW5 path touched: {path}")


def numeric_series(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.zeros(len(frame), dtype=np.float32)
    return (
        pd.to_numeric(frame[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def read_event_frame(path: Path) -> pd.DataFrame:
    assert_no_new5_path(path)
    cols = [
        "split",
        "session_id",
        "calendar_session_id",
        "symbol",
        "event_index",
        "timestamp_us",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    raw = pd.read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(ACTION_ASK)].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int).eq(ACTION_NO_QUOTE)].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet mismatch in {path}")
    id_cols = ["session_id", "symbol", "event_index"]
    for col in id_cols:
        if not bid[col].reset_index(drop=True).equals(ask[col].reset_index(drop=True)):
            raise RuntimeError(f"Bid/ask {col} alignment mismatch in {path}")
        if not bid[col].reset_index(drop=True).equals(no_quote[col].reset_index(drop=True)):
            raise RuntimeError(f"Bid/no_quote {col} alignment mismatch in {path}")
    out = bid.drop(columns=["action_slot", "excess_pnl_H20"]).copy()
    out["bid_excess_pnl_H20"] = pd.to_numeric(bid["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["ask_excess_pnl_H20"] = pd.to_numeric(ask["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def base_feature_frame(frame: pd.DataFrame, vol_mean: float = 0.0, vol_std: float = 1.0) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    for col in BASE_EVENT_COLUMNS:
        out[col] = numeric_series(frame, col)
    out["log1p_bid_sz_1"] = np.log1p(np.maximum(out["bid_sz_1"], 0.0)).astype(np.float32)
    out["log1p_ask_sz_1"] = np.log1p(np.maximum(out["ask_sz_1"], 0.0)).astype(np.float32)
    out["log1p_top_depth"] = np.log1p(np.maximum(out["top_depth"], 0.0)).astype(np.float32)
    out["clipped_inter_arrival_ms"] = np.clip(out["inter_arrival_ms"], 0.0, 5000.0).astype(np.float32)
    out["zscore_volatility_past_50_by_train_stats"] = ((out["volatility_past_50"] - vol_mean) / max(vol_std, 1e-6)).astype(np.float32)
    return pd.DataFrame(out, index=frame.index)


def fit_scaler() -> dict[str, Any]:
    ensure_dirs()
    vol_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    for path in part_paths("train"):
        frame = read_event_frame(path)
        vol_values.append(numeric_series(frame, "volatility_past_50"))
        target_values.append(frame[["bid_excess_pnl_H20", "ask_excess_pnl_H20"]].to_numpy(dtype=np.float32).reshape(-1))
    all_vol = np.concatenate(vol_values) if vol_values else np.array([0.0], dtype=np.float32)
    vol_mean = float(np.nanmean(all_vol))
    vol_std = float(np.nanstd(all_vol) if np.nanstd(all_vol) > 1e-6 else 1.0)
    feature_sum = np.zeros(len(PHC_FEATURES), dtype=np.float64)
    feature_sumsq = np.zeros(len(PHC_FEATURES), dtype=np.float64)
    feature_count = 0
    missing_total = 0
    for path in part_paths("train"):
        frame = read_event_frame(path)
        features = base_feature_frame(frame, vol_mean, vol_std)[PHC_FEATURES]
        arr = features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        missing_total += int(np.isnan(arr).sum())
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feature_sum += arr.sum(axis=0, dtype=np.float64)
        feature_sumsq += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
        feature_count += arr.shape[0]
    mean = feature_sum / max(feature_count, 1)
    var = np.maximum(feature_sumsq / max(feature_count, 1) - mean * mean, 1e-8)
    std = np.sqrt(var)
    target = np.concatenate(target_values) if target_values else np.array([1.0], dtype=np.float32)
    target_scale = float(np.nanstd(target))
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0
    stats = {
        "feature_names": PHC_FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "volatility_mean_train": vol_mean,
        "volatility_std_train": vol_std,
        "target_scale_train_std_no_mean_shift": target_scale,
        "train_feature_rows": int(feature_count),
        "train_feature_missing_values_after_coerce": int(missing_total),
        "normalization_fit_split": "train",
        "new3_used_for_scaler": False,
        "new4_used_for_scaler": False,
        "new5_used_for_scaler": False,
    }
    write_json(DATA / "phc_lite_feature_scaler.json", stats)
    return stats


def load_scaler() -> dict[str, Any]:
    path = DATA / "phc_lite_feature_scaler.json"
    if not path.exists():
        return fit_scaler()
    return read_json(path)


def normalized_features(frame: pd.DataFrame, scaler: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = base_feature_frame(
        frame,
        float(scaler["volatility_mean_train"]),
        float(scaler["volatility_std_train"]),
    )[PHC_FEATURES]
    arr = features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    norm = (arr - mean) / np.maximum(std, 1e-6)
    raw = frame[TARGET_RAW_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return norm.astype(np.float32), raw.astype(np.float32)


def build_windows_for_frame(frame: pd.DataFrame, scaler: dict[str, Any], seq_len: int, stride: int) -> dict[str, Any] | None:
    x, raw = normalized_features(frame, scaler)
    n = len(frame)
    if n < seq_len:
        return None
    starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
    window_count = len(starts)
    xw = np.empty((window_count, seq_len, x.shape[1]), dtype=np.float16)
    for i, start in enumerate(starts):
        xw[i] = x[start : start + seq_len].astype(np.float16)
    target_idx = starts + seq_len - 1
    sessions = frame["session_id"].astype(str).to_numpy()
    unique_sessions, session_code = np.unique(sessions[target_idx], return_inverse=True)
    out = {
        "x": xw,
        "y_bid": frame["bid_excess_pnl_H20"].to_numpy(dtype=np.float32)[target_idx],
        "y_ask": frame["ask_excess_pnl_H20"].to_numpy(dtype=np.float32)[target_idx],
        "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[target_idx],
        "timestamp_us": pd.to_numeric(frame["timestamp_us"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[target_idx],
        "target_raw": raw[target_idx].astype(np.float32),
        "session_code": session_code.astype(np.int32),
        "session_labels": unique_sessions.tolist(),
        "start_event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[starts],
    }
    return out


def merge_session_codes(chunks: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    labels: list[str] = []
    code_arrays = []
    label_to_code: dict[str, int] = {}
    for chunk in chunks:
        local_labels = chunk["session_labels"]
        codes = chunk["session_code"]
        mapped = np.empty_like(codes, dtype=np.int32)
        for i, label in enumerate(local_labels):
            if label not in label_to_code:
                label_to_code[label] = len(labels)
                labels.append(label)
            mapped[codes == i] = label_to_code[label]
        code_arrays.append(mapped)
    return np.concatenate(code_arrays).astype(np.int32), labels


def build_sequence_dataset(seq_len: int = SEQ_LEN, stride: int = STRIDE) -> pd.DataFrame:
    ensure_dirs()
    scaler = fit_scaler()
    manifest_rows: list[dict[str, Any]] = []
    for split in ALL_SPLITS:
        chunks: list[dict[str, Any]] = []
        source_events = 0
        for path in part_paths(split):
            frame = read_event_frame(path)
            source_events += len(frame)
            built = build_windows_for_frame(frame, scaler, seq_len, stride)
            if built is None:
                manifest_rows.append(
                    {
                        "split": split,
                        "part": path.name,
                        "event_rows": len(frame),
                        "window_rows": 0,
                        "skipped_reason": "fewer_rows_than_seq_len",
                    }
                )
                continue
            chunks.append(built)
            y_bid = built["y_bid"]
            y_ask = built["y_ask"]
            manifest_rows.append(
                {
                    "split": split,
                    "part": path.name,
                    "event_rows": len(frame),
                    "window_rows": int(len(y_bid)),
                    "seq_len": seq_len,
                    "stride": stride,
                    "feature_dim": len(PHC_FEATURES),
                    "bid_positive_rate": float((y_bid > 0).mean()) if len(y_bid) else 0.0,
                    "ask_positive_rate": float((y_ask > 0).mean()) if len(y_ask) else 0.0,
                    "target_mean_bid": float(np.mean(y_bid)) if len(y_bid) else 0.0,
                    "target_mean_ask": float(np.mean(y_ask)) if len(y_ask) else 0.0,
                    "target_std_bid": float(np.std(y_bid)) if len(y_bid) else 0.0,
                    "target_std_ask": float(np.std(y_ask)) if len(y_ask) else 0.0,
                    "skipped_reason": "",
                }
            )
            print(f"built split={split} part={path.name} windows={len(y_bid)}")
        if not chunks:
            continue
        session_code, session_labels = merge_session_codes(chunks)
        payload = {
            "x": torch.from_numpy(np.concatenate([c["x"] for c in chunks], axis=0)),
            "y_bid": torch.from_numpy(np.concatenate([c["y_bid"] for c in chunks]).astype(np.float32)),
            "y_ask": torch.from_numpy(np.concatenate([c["y_ask"] for c in chunks]).astype(np.float32)),
            "event_index": torch.from_numpy(np.concatenate([c["event_index"] for c in chunks]).astype(np.int64)),
            "timestamp_us": torch.from_numpy(np.concatenate([c["timestamp_us"] for c in chunks]).astype(np.int64)),
            "target_raw": torch.from_numpy(np.concatenate([c["target_raw"] for c in chunks], axis=0).astype(np.float32)),
            "session_code": torch.from_numpy(session_code.astype(np.int32)),
            "session_labels": session_labels,
            "feature_names": PHC_FEATURES,
            "target_raw_columns": TARGET_RAW_COLUMNS,
            "seq_len": seq_len,
            "stride": stride,
            "target_horizon": HORIZON,
            "source_event_rows": int(source_events),
            "split": split,
            "new5_used": False,
        }
        torch.save(payload, DATA / f"phc_lite_{split}_sequences.pt")
        pd.DataFrame(
            {
                "split": split,
                "window_id": np.arange(len(payload["y_bid"]), dtype=np.int64),
                "session_id": [session_labels[int(c)] for c in session_code],
                "event_index": payload["event_index"].numpy(),
            }
        ).to_csv(RESULTS / f"phc_lite_sequence_index_{split}.csv", index=False)
        print(f"saved split={split} total_windows={len(payload['y_bid'])}")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(RESULTS / "phc_lite_sequence_manifest.csv", index=False)
    write_sequence_dataset_report(manifest, scaler)
    return manifest


def load_sequences(split: str) -> dict[str, Any]:
    path = DATA / f"phc_lite_{split}_sequences.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence dataset: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("new5_used"):
        raise RuntimeError("Phase 05A hard rule violation: loaded payload marked new5_used=True")
    return payload


class SequenceTensorDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any], target_scale: float) -> None:
        self.x = payload["x"]
        self.y = torch.stack([payload["y_bid"], payload["y_ask"]], dim=1).float() / float(target_scale)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx].float(), self.y[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def safe_spearman(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 3 or np.nanstd(score) < 1e-12 or np.nanstd(target) < 1e-12:
        return float("nan")
    return float(pd.Series(score).corr(pd.Series(target), method="spearman"))


def safe_pr_auc(target_positive: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target_positive).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def safe_roc_auc(target_positive: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target_positive).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def huber_np(error: np.ndarray, delta: float = 1.0) -> float:
    abs_error = np.abs(error)
    loss = np.where(abs_error <= delta, 0.5 * error * error, delta * (abs_error - 0.5 * delta))
    return float(np.mean(loss))


def predict_scores(model: PHCLite, payload: dict[str, Any], batch_size: int = 1024, device: str = "cpu", target_scale: float = 1.0) -> np.ndarray:
    model.eval()
    x = payload["x"]
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            out = model(batch).detach().cpu().numpy().astype(np.float32)
            outs.append(out)
    return np.concatenate(outs, axis=0) * float(target_scale)


def policy_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    bid = scores[:, 0]
    ask = scores[:, 1]
    decision = np.zeros(len(scores), dtype=np.int8)
    decision[(bid >= ask) & (bid > threshold)] = DECISION_BID
    decision[(ask > bid) & (ask > threshold)] = DECISION_ASK
    return decision


def decision_pnl_from_payload(payload: dict[str, Any], decision: np.ndarray) -> np.ndarray:
    y_bid = payload["y_bid"].numpy()
    y_ask = payload["y_ask"].numpy()
    out = np.zeros(len(decision), dtype=np.float64)
    bid = decision == DECISION_BID
    ask = decision == DECISION_ASK
    out[bid] = y_bid[bid]
    out[ask] = y_ask[ask]
    out[~np.isfinite(out)] = 0.0
    return out


def metric_from_decision(split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray) -> dict[str, Any]:
    pnl = decision_pnl_from_payload(payload, decision)
    selected = decision != DECISION_NO_QUOTE
    selected_pnl = pnl[selected]
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0)) if len(equity) else np.array([0.0])
    dd = peak - equity if len(equity) else np.array([0.0])
    session_rows = []
    for code, label in enumerate(payload["session_labels"]):
        mask = payload["session_code"].numpy() == code
        session_rows.append(float(pnl[mask].sum()))
    return {
        "split": split,
        "policy_id": policy_id,
        "horizon": HORIZON,
        "state_count": int(len(decision)),
        "selected_action_count": int(selected.sum()),
        "selection_rate": float(selected.mean()) if len(selected) else 0.0,
        "bid_count": int((decision == DECISION_BID).sum()),
        "ask_count": int((decision == DECISION_ASK).sum()),
        "no_quote_count": int((decision == DECISION_NO_QUOTE).sum()),
        "total_excess_pnl_H20": float(selected_pnl.sum()) if len(selected_pnl) else 0.0,
        "avg_selected_excess": float(selected_pnl.mean()) if len(selected_pnl) else 0.0,
        "win_rate": float((selected_pnl > 0).mean()) if len(selected_pnl) else 0.0,
        "negative_rate": float((selected_pnl < 0).mean()) if len(selected_pnl) else 0.0,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "positive_session_count": int(sum(v > 0 for v in session_rows)),
        "session_count": int(len(session_rows)),
    }


def queue_rule_decision(payload: dict[str, Any]) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    q = raw[:, TARGET_RAW_COLUMNS.index("queue_imbalance")]
    decision = np.zeros(len(q), dtype=np.int8)
    finite = np.isfinite(q)
    decision[finite & (q > 0.1)] = DECISION_BID
    decision[finite & (q < -0.1)] = DECISION_ASK
    return decision


def old_bid_only_decision(payload: dict[str, Any]) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    spread = raw[:, TARGET_RAW_COLUMNS.index("spread_ticks")]
    decision = np.zeros(len(spread), dtype=np.int8)
    decision[np.isfinite(spread) & (spread >= 1.0)] = DECISION_BID
    return decision


def no_quote_decision(payload: dict[str, Any]) -> np.ndarray:
    return np.zeros(len(payload["y_bid"]), dtype=np.int8)


def hgb_feature_matrix(payload: dict[str, Any], action: str) -> pd.DataFrame:
    raw = payload["target_raw"].numpy()
    base = pd.DataFrame(raw, columns=TARGET_RAW_COLUMNS)
    slot = ACTION_BID if action == "bid" else ACTION_ASK
    side_sign = 1.0 if slot == ACTION_BID else -1.0
    out = base.copy()
    out["action_is_bid"] = 1.0 if slot == ACTION_BID else 0.0
    out["action_is_ask"] = 1.0 if slot == ACTION_ASK else 0.0
    out["side_sign"] = side_sign
    out["side_queue_imbalance"] = side_sign * out["queue_imbalance"]
    out["side_deep_imbalance"] = side_sign * out["deep_imbalance"]
    out["side_imbalance_change_past"] = side_sign * out["imbalance_change_past"]
    for col in HGB_ML_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out[HGB_ML_FEATURES]


def hgb_decision(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not HGB_MODEL_PATH.exists() or not HGB_CLOSURE_PATH.exists():
        return no_quote_decision(payload), np.zeros((len(payload["y_bid"]), 2), dtype=np.float32)
    closure = read_json(HGB_CLOSURE_PATH)
    threshold = float(closure["selected_threshold_on_validation"])
    model = joblib.load(HGB_MODEL_PATH)
    bid = model.predict(hgb_feature_matrix(payload, "bid").to_numpy(dtype=np.float32)).astype(np.float32)
    ask = model.predict(hgb_feature_matrix(payload, "ask").to_numpy(dtype=np.float32)).astype(np.float32)
    scores = np.column_stack([bid, ask])
    return policy_from_scores(scores, threshold), scores


def train_one_seed(
    seed: int = 0,
    hidden_dim: int = 16,
    mtl_steps: int = 1,
    dropout: float = 0.0,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    min_delta: float = 1e-3,
    overwrite_main: bool = True,
) -> pd.DataFrame:
    ensure_dirs()
    set_seed(seed)
    scaler = load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = load_sequences("train")
    val_payload = load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = torch.utils.data.DataLoader(
        SequenceTensorDataset(train_payload, target_scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    model = PHCLite(len(PHC_FEATURES), hidden_dim=hidden_dim, mtl_steps=mtl_steps, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = torch.nn.HuberLoss(delta=1.0)
    best_metric = -float("inf")
    best_epoch = -1
    early_counter = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        grad_norms = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            opt.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
        train_scores = predict_scores(model, train_payload, batch_size=batch_size, device=device, target_scale=target_scale)
        val_scores = predict_scores(model, val_payload, batch_size=batch_size, device=device, target_scale=target_scale)
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        train_loss = huber_np((train_scores - train_y) / target_scale)
        val_loss = huber_np((val_scores - val_y) / target_scale)
        val_decision = policy_from_scores(val_scores, 0.0)
        val_pnl = float(decision_pnl_from_payload(val_payload, val_decision).sum())
        val_selected = int((val_decision != DECISION_NO_QUOTE).sum())
        row = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "train_huber_loss": train_loss,
            "val_huber_loss": val_loss,
            "train_spearman_bid": safe_spearman(train_scores[:, 0], train_y[:, 0]),
            "val_spearman_bid": safe_spearman(val_scores[:, 0], val_y[:, 0]),
            "train_spearman_ask": safe_spearman(train_scores[:, 1], train_y[:, 1]),
            "val_spearman_ask": safe_spearman(val_scores[:, 1], val_y[:, 1]),
            "val_policy_excess_pnl": val_pnl,
            "val_selected_count": val_selected,
            "val_selection_rate": val_selected / len(val_decision) if len(val_decision) else 0.0,
            "val_pr_auc_bid": safe_pr_auc(val_y[:, 0] > 0, val_scores[:, 0]),
            "val_pr_auc_ask": safe_pr_auc(val_y[:, 1] > 0, val_scores[:, 1]),
            "learning_rate": lr,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "parameter_count": count_parameters(model),
            "device": device,
            "hidden_dim": hidden_dim,
            "mtl_steps": mtl_steps,
        }
        rows.append(row)
        if val_pnl > best_metric + min_delta:
            best_metric = val_pnl
            best_epoch = epoch
            early_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(PHC_FEATURES),
                "hidden_dim": hidden_dim,
                "mtl_steps": mtl_steps,
                "dropout": dropout,
                "seed": seed,
                "best_epoch": best_epoch,
                "target_scale": target_scale,
                "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
            torch.save(checkpoint, MODELS / f"phc_lite_seed{seed}_best_val.pt")
            if overwrite_main:
                torch.save(checkpoint, MODELS / "phc_lite_best_val.pt")
        else:
            early_counter += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(PHC_FEATURES),
                "hidden_dim": hidden_dim,
                "mtl_steps": mtl_steps,
                "dropout": dropout,
                "seed": seed,
                "epoch": epoch,
                "target_scale": target_scale,
            },
            MODELS / f"phc_lite_seed{seed}_last.pt",
        )
        if overwrite_main:
            shutil.copy2(MODELS / f"phc_lite_seed{seed}_last.pt", MODELS / "phc_lite_last.pt")
        print(f"seed={seed} epoch={epoch} val_pnl={val_pnl:.2f} selected={val_selected} grad={row['grad_norm']:.4f}")
        if early_counter >= patience:
            break
    curve = pd.DataFrame(rows)
    out_path = RESULTS / ("phc_lite_training_curve.csv" if overwrite_main else f"phc_lite_training_curve_seed{seed}.csv")
    curve.to_csv(out_path, index=False)
    plot_training_curves(curve, suffix="" if overwrite_main else f"_seed{seed}")
    write_training_report(curve, seed=seed, overwrite_main=overwrite_main)
    return curve


def load_model(checkpoint_path: Path | None = None) -> tuple[PHCLite, dict[str, Any]]:
    checkpoint_path = checkpoint_path or (MODELS / "phc_lite_best_val.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PHCLite(
        int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt["mtl_steps"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def select_policy_threshold() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    model, ckpt = load_model()
    val = load_sequences("val")
    scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    quantiles = np.unique(np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])) if len(finite) else np.array([0.0])
    thresholds = sorted(set([0.0, *quantiles.tolist()]))
    rows = []
    for threshold in thresholds:
        decision = policy_from_scores(scores, float(threshold))
        row = metric_from_decision("val", "phc_lite", val, decision)
        row["threshold"] = float(threshold)
        row["selection_split"] = "val"
        row["new3_used_for_selection"] = False
        row["new4_used_for_selection"] = False
        row["new5_used_for_selection"] = False
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).reset_index(drop=True)
    grid.to_csv(RESULTS / "phc_lite_threshold_search.csv", index=False)
    best = grid.iloc[0].to_dict()
    policy = {
        "model": "phc_lite",
        "selection_split": "val",
        "target": "excess_pnl_H20",
        "seq_len": int(ckpt.get("seq_len", SEQ_LEN)),
        "hidden_dim": int(ckpt["hidden_dim"]),
        "mtl_steps": int(ckpt["mtl_steps"]),
        "threshold": float(best["threshold"]),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "target_scale": float(ckpt["target_scale"]),
        "validation_metrics": json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    write_json(RESULTS / "phc_lite_selected_policy.json", policy)
    return grid, policy


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    policy = read_json(RESULTS / "phc_lite_selected_policy.json")
    model, ckpt = load_model()
    threshold = float(policy["threshold"])
    metric_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for split in ["val", "new3", "new4"]:
        payload = load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decisions = {
            "no_quote": no_quote_decision(payload),
            "old_bid_only_spread_ge_1": old_bid_only_decision(payload),
            "side_by_queue_imbalance_abs_0p1": queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": hgb_decision(payload)[0],
            "phc_lite": policy_from_scores(scores, threshold),
        }
        for policy_id, decision in decisions.items():
            metric_rows.append(metric_from_decision(split, policy_id, payload, decision))
            pnl = decision_pnl_from_payload(payload, decision)
            code = payload["session_code"].numpy()
            for session_code, session_id in enumerate(payload["session_labels"]):
                mask = code == session_code
                session_rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "session_id": session_id,
                        "session_excess_pnl_H20": float(pnl[mask].sum()),
                        "selected_action_count": int((decision[mask] != DECISION_NO_QUOTE).sum()),
                        "state_count": int(mask.sum()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    sessions = pd.DataFrame(session_rows)
    metrics.to_csv(RESULTS / "phc_lite_oos_policy_eval.csv", index=False)
    metrics.to_csv(RESULTS / "phc_lite_vs_hgb_vs_queue.csv", index=False)
    sessions.to_csv(RESULTS / "phc_lite_session_pnl.csv", index=False)
    plot_policy_comparison(metrics)
    write_oos_report(metrics)
    return metrics, sessions


def spread_bucket_from_payload(payload: dict[str, Any]) -> np.ndarray:
    spread = payload["target_raw"].numpy()[:, TARGET_RAW_COLUMNS.index("spread_ticks")]
    return np.asarray(
        pd.cut(spread, bins=[-np.inf, 0, 1, 2, 3, 5, np.inf], labels=["<=0", "1", "2", "3", "4-5", ">5"]).astype(str)
    )


def matched_random_for_split(split: str, decision: np.ndarray, n_trials: int = RANDOM_TRIALS, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = load_sequences(split)
    model_pnl = float(decision_pnl_from_payload(payload, decision).sum())
    session = payload["session_code"].numpy()
    spread_bucket = spread_bucket_from_payload(payload)
    y_bid = payload["y_bid"].numpy()
    y_ask = payload["y_ask"].numpy()
    rng = np.random.default_rng(seed + abs(hash(split)) % 100000)
    selected_groups: list[tuple[int, int, str, int]] = []
    for side in [DECISION_BID, DECISION_ASK]:
        mask_side = decision == side
        for sess in np.unique(session[mask_side]):
            mask_sess = mask_side & (session == sess)
            for bucket in np.unique(spread_bucket[mask_sess]):
                count = int((mask_sess & (spread_bucket == bucket)).sum())
                if count:
                    selected_groups.append((side, int(sess), str(bucket), count))
    trial_rows = []
    trial_values = np.zeros(n_trials, dtype=np.float64)
    for trial in range(n_trials):
        total = 0.0
        replacement_used = False
        fallback_count = 0
        for side, sess, bucket, count in selected_groups:
            pool = (session == sess) & (spread_bucket == bucket)
            pnl_pool = y_bid[pool] if side == DECISION_BID else y_ask[pool]
            if len(pnl_pool) < count:
                fallback_count += 1
                pool = session == sess
                pnl_pool = y_bid[pool] if side == DECISION_BID else y_ask[pool]
            replace = len(pnl_pool) < count
            replacement_used = replacement_used or replace
            if len(pnl_pool) == 0:
                continue
            idx = rng.choice(len(pnl_pool), size=count, replace=replace)
            total += float(np.asarray(pnl_pool)[idx].sum())
        trial_values[trial] = total
        trial_rows.append(
            {
                "split": split,
                "trial": trial,
                "random_total_excess_pnl_H20": total,
                "model_total_excess_pnl_H20": model_pnl,
                "replacement_used": bool(replacement_used),
                "fallback_group_count": int(fallback_count),
                "matching": "count_session_side_spread_bucket_with_session_side_fallback",
            }
        )
    p = float((trial_values >= model_pnl).mean())
    summary = pd.DataFrame(
        [
            {
                "split": split,
                "policy_id": "phc_lite",
                "model_total_excess_pnl_H20": model_pnl,
                "random_mean": float(trial_values.mean()),
                "random_std": float(trial_values.std()),
                "random_95p": float(np.quantile(trial_values, 0.95)),
                "random_99p": float(np.quantile(trial_values, 0.99)),
                "empirical_p_value_fraction_random_ge_model": p,
                "empirical_p_value_wording": "< 0.001" if p == 0.0 else f"{p:.6f}",
                "beats_random_95p": bool(model_pnl > np.quantile(trial_values, 0.95)),
                "n_trials": int(n_trials),
                "matching": "count_session_side_spread_bucket_with_session_side_fallback",
            }
        ]
    )
    return pd.DataFrame(trial_rows), summary


def run_matched_random(n_trials: int = RANDOM_TRIALS) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = read_json(RESULTS / "phc_lite_selected_policy.json")
    model, ckpt = load_model()
    all_trials = []
    all_summary = []
    for split in ["val", "new3", "new4"]:
        payload = load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = policy_from_scores(scores, float(policy["threshold"]))
        trials, summary = matched_random_for_split(split, decision, n_trials=n_trials)
        all_trials.append(trials)
        all_summary.append(summary)
        print(f"matched random split={split} model_pnl={summary.iloc[0]['model_total_excess_pnl_H20']:.2f}")
    trials_df = pd.concat(all_trials, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    trials_df.to_csv(RESULTS / "phc_lite_matched_random_trials.csv", index=False)
    summary_df.to_csv(RESULTS / "phc_lite_matched_random_summary.csv", index=False)
    write_matched_random_report(summary_df)
    return trials_df, summary_df


@dataclass
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def compute_fill_norms_from_train_sequences() -> FillNorms:
    payload = load_sequences("train")
    raw = payload["target_raw"].numpy()
    vals = {}
    for col in ["top_depth", "bid_sz_1", "ask_sz_1"]:
        arr = raw[:, TARGET_RAW_COLUMNS.index(col)]
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vals[col] = float(np.nanmedian(positive)) if len(positive) else 1.0
    return FillNorms(vals["top_depth"], vals["bid_sz_1"], vals["ask_sz_1"])


def stable_uniform(payload: dict[str, Any], decision: np.ndarray, salt: str) -> np.ndarray:
    session = payload["session_code"].numpy().astype(np.int64)
    event = payload["event_index"].numpy().astype(np.int64)
    keys = pd.DataFrame({"session_code": session, "event_index": event, "decision": decision.astype(np.int8)})
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(zlib.crc32(salt.encode("utf-8")))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def fill_probabilities(payload: dict[str, Any], decision: np.ndarray, norms: FillNorms) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    top_depth = raw[:, TARGET_RAW_COLUMNS.index("top_depth")]
    bid_depth = raw[:, TARGET_RAW_COLUMNS.index("bid_sz_1")]
    ask_depth = raw[:, TARGET_RAW_COLUMNS.index("ask_sz_1")]
    out = np.zeros(len(decision), dtype=np.float64)
    selected = decision != DECISION_NO_QUOTE
    if not selected.any():
        return out
    depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
    side_depth_norm = np.zeros(len(decision), dtype=np.float64)
    bid = decision == DECISION_BID
    ask = decision == DECISION_ASK
    side_depth_norm[bid] = bid_depth[bid] / max(norms.bid_sz_1_median, 1.0)
    side_depth_norm[ask] = ask_depth[ask] / max(norms.ask_sz_1_median, 1.0)
    side = 1.0 / (1.0 + np.maximum(side_depth_norm, 0.0))
    out[selected] = np.clip(0.5 * depth[selected] * side[selected], 0.01, 1.0)
    return out


def simulate_scheduled_by_event_index(payload: dict[str, Any], decision: np.ndarray, fills: np.ndarray, pnl: np.ndarray, inventory_cap: int = 1) -> tuple[float, int, int, int, float, int, int, np.ndarray]:
    n = len(decision)
    total_pnl = 0.0
    equity = np.zeros(n, dtype=np.float64)
    close_pnl = np.zeros(n, dtype=np.float64)
    close_side = np.zeros(n, dtype=np.int64)
    inventory = 0
    max_abs_inventory = 0
    forced = 0
    fill_count = 0
    win = 0
    loss = 0
    session_code = payload["session_code"].numpy()
    event_index = payload["event_index"].numpy()
    session_positions = {code: np.where(session_code == code)[0] for code in np.unique(session_code)}
    close_lookup: dict[tuple[int, int], int] = {}
    for code, positions in session_positions.items():
        events = event_index[positions]
        for pos in positions:
            target_event = event_index[pos] + HORIZON
            j = np.searchsorted(events, target_event, side="left")
            close_lookup[(int(code), int(pos))] = int(positions[min(j, len(positions) - 1)])
    peak = 0.0
    max_dd = 0.0
    for t in range(n):
        if close_pnl[t] != 0.0 or close_side[t] != 0:
            inventory -= int(close_side[t])
            total_pnl += float(close_pnl[t])
        if fills[t]:
            side_sign = 1 if decision[t] == DECISION_BID else -1
            if abs(inventory + side_sign) > inventory_cap:
                forced += 1
            else:
                trade_pnl = float(pnl[t])
                inventory += side_sign
                max_abs_inventory = max(max_abs_inventory, abs(inventory))
                close_idx = close_lookup[(int(session_code[t]), int(t))]
                close_side[close_idx] += side_sign
                close_pnl[close_idx] += trade_pnl
                fill_count += 1
                win += int(trade_pnl > 0.0)
                loss += int(trade_pnl < 0.0)
        equity[t] = total_pnl
        peak = max(peak, total_pnl)
        max_dd = max(max_dd, peak - total_pnl)
    return total_pnl, fill_count, forced, max_abs_inventory, max_dd, win, loss, equity


def causal_fill_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = read_json(RESULTS / "phc_lite_selected_policy.json")
    model, ckpt = load_model()
    norms = compute_fill_norms_from_train_sequences()
    rows = []
    equity_rows = []
    for split in ["val", "new3", "new4"]:
        payload = load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decisions = {
            "no_quote": no_quote_decision(payload),
            "old_bid_only_spread_ge_1": old_bid_only_decision(payload),
            "side_by_queue_imbalance_abs_0p1": queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": hgb_decision(payload)[0],
            "phc_lite": policy_from_scores(scores, float(policy["threshold"])),
        }
        for policy_id, decision in decisions.items():
            probs = fill_probabilities(payload, decision, norms)
            fills = (decision != DECISION_NO_QUOTE) & (stable_uniform(payload, decision, "05a_conservative_l2_stress") < probs)
            pnl = decision_pnl_from_payload(payload, decision)
            total, fill_count, forced, max_inv, max_dd, win, loss, equity = simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "horizon": HORIZON,
                    "fill_model": "conservative_l2_stress",
                    "inventory_cap": 1,
                    "sequential_excess_pnl": float(total),
                    "fill_count": int(fill_count),
                    "forced_no_quote_count": int(forced),
                    "max_abs_inventory": int(max_inv),
                    "max_drawdown": float(max_dd),
                    "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
                    "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
                    "avg_trade_excess_pnl": total / max(fill_count, 1) if fill_count else 0.0,
                    "fill_uses_future_pnl": False,
                    "new5_used": False,
                }
            )
            for i in range(0, len(equity), max(1, len(equity) // 500)):
                equity_rows.append({"split": split, "policy_id": policy_id, "row_id": i, "equity": float(equity[i])})
            print(f"causal-fill split={split} policy={policy_id} pnl={total:.2f}")
    out = pd.DataFrame(rows)
    eq = pd.DataFrame(equity_rows)
    out.to_csv(RESULTS / "phc_lite_causal_fill_sequential_eval.csv", index=False)
    eq.to_csv(RESULTS / "phc_lite_causal_fill_equity_curve.csv", index=False)
    plot_causal_fill(out)
    write_causal_fill_report(out)
    return out, eq


def decision_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = read_json(RESULTS / "phc_lite_selected_policy.json")
    model, ckpt = load_model()
    rows = []
    oracle_rows = []
    for split in ["val", "new3", "new4"]:
        payload = load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = policy_from_scores(scores, float(policy["threshold"]))
        pnl = decision_pnl_from_payload(payload, decision)
        selected = decision != DECISION_NO_QUOTE
        actual_positive = pnl > 0
        pred_positive = selected
        cm = confusion_matrix(actual_positive.astype(int), pred_positive.astype(int), labels=[0, 1])
        rows.append(
            {
                "split": split,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "selected_positive_definition": "model_score_above_validation_threshold",
                "actual_positive_definition": "selected_action_excess_pnl_H20_gt_0",
            }
        )
        y_bid = payload["y_bid"].numpy()
        y_ask = payload["y_ask"].numpy()
        oracle = np.zeros(len(decision), dtype=np.int8)
        oracle[(y_bid >= y_ask) & (y_bid > 0.0)] = DECISION_BID
        oracle[(y_ask > y_bid) & (y_ask > 0.0)] = DECISION_ASK
        labels = [DECISION_NO_QUOTE, DECISION_BID, DECISION_ASK]
        cm_oracle = confusion_matrix(oracle, decision, labels=labels)
        for i, oracle_action in enumerate(["no_quote", "bid", "ask"]):
            for j, model_action in enumerate(["no_quote", "bid", "ask"]):
                oracle_rows.append({"split": split, "oracle_action": oracle_action, "model_action": model_action, "count": int(cm_oracle[i, j])})
    action_df = pd.DataFrame(rows)
    oracle_df = pd.DataFrame(oracle_rows)
    action_df.to_csv(RESULTS / "phc_lite_action_confusion.csv", index=False)
    oracle_df.to_csv(RESULTS / "phc_lite_oracle_action_confusion.csv", index=False)
    plot_oracle_confusion(oracle_df)
    write_decision_report(action_df, oracle_df)
    return action_df, oracle_df


def score_bucket_lift() -> pd.DataFrame:
    policy = read_json(RESULTS / "phc_lite_selected_policy.json")
    model, ckpt = load_model()
    val = load_sequences("val")
    val_scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
    val_max = np.max(val_scores, axis=1)
    edges = np.quantile(val_max[np.isfinite(val_max)], [0.0, 0.50, 0.90, 0.95, 0.99, 1.0])
    rows = []
    for split in ["val", "new3", "new4"]:
        payload = load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        max_score = np.max(scores, axis=1)
        side = np.where(scores[:, 0] >= scores[:, 1], DECISION_BID, DECISION_ASK)
        chosen_pnl = np.where(side == DECISION_BID, payload["y_bid"].numpy(), payload["y_ask"].numpy())
        bucket_names = ["bottom/middle", "top_10", "top_5", "top_1", "max_edge"]
        bucket_id = np.digitize(max_score, edges[1:-1], right=False)
        for b, name in enumerate(bucket_names):
            mask = bucket_id == b
            if not mask.any():
                continue
            rows.append(
                {
                    "split": split,
                    "bucket": name,
                    "count": int(mask.sum()),
                    "avg_excess_pnl": float(chosen_pnl[mask].mean()),
                    "win_rate": float((chosen_pnl[mask] > 0).mean()),
                    "bid_ratio": float((side[mask] == DECISION_BID).mean()),
                    "ask_ratio": float((side[mask] == DECISION_ASK).mean()),
                    "mean_score": float(max_score[mask].mean()),
                    "bucket_edges_from": "validation",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_lite_score_bucket_lift.csv", index=False)
    plot_score_bucket(out)
    write_score_bucket_report(out)
    return out


def seed_stability(seeds: list[int] | None = None) -> pd.DataFrame:
    seeds = seeds or [0, 1, 2]
    rows = []
    queue_metrics = evaluate_or_load_metrics()
    queue_lookup = queue_metrics[queue_metrics["policy_id"].eq("side_by_queue_imbalance_abs_0p1")].set_index("split")["total_excess_pnl_H20"].to_dict()
    hgb_lookup = queue_metrics[queue_metrics["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].set_index("split")["total_excess_pnl_H20"].to_dict()
    causal_path = RESULTS / "phc_lite_causal_fill_sequential_eval.csv"
    causal_lookup: dict[tuple[str, str], float] = {}
    if causal_path.exists():
        causal_existing = pd.read_csv(causal_path)
        causal_lookup = {
            (str(r["split"]), str(r["policy_id"])): float(r["sequential_excess_pnl"])
            for _, r in causal_existing.iterrows()
        }
    norms = compute_fill_norms_from_train_sequences()
    for seed in seeds:
        curve_path = RESULTS / (f"phc_lite_training_curve_seed{seed}.csv" if seed != 0 else "phc_lite_training_curve.csv")
        checkpoint = MODELS / f"phc_lite_seed{seed}_best_val.pt"
        if not checkpoint.exists():
            train_one_seed(seed=seed, batch_size=4096, overwrite_main=(seed == 0))
        model, ckpt = load_model(checkpoint)
        val = load_sequences("val")
        val_scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        finite = np.max(val_scores, axis=1)
        thresholds = sorted(set([0.0, *np.quantile(finite[np.isfinite(finite)], [0.5, 0.75, 0.9, 0.95]).tolist()]))
        best_thr = 0.0
        best_pnl = -float("inf")
        for thr in thresholds:
            pnl = decision_pnl_from_payload(val, policy_from_scores(val_scores, float(thr))).sum()
            if pnl > best_pnl:
                best_pnl = float(pnl)
                best_thr = float(thr)
        for split in ["val", "new3", "new4"]:
            payload = load_sequences(split)
            scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
            decision = policy_from_scores(scores, best_thr)
            metric = metric_from_decision(split, f"phc_lite_seed{seed}", payload, decision)
            _, random_summary = matched_random_for_split(split, decision, n_trials=RANDOM_TRIALS, seed=RANDOM_SEED + seed * 10000)
            random_row = random_summary.iloc[0].to_dict()
            probs = fill_probabilities(payload, decision, norms)
            fills = (decision != DECISION_NO_QUOTE) & (stable_uniform(payload, decision, f"05a_seed_{seed}_conservative_l2_stress") < probs)
            pnl = decision_pnl_from_payload(payload, decision)
            total, fill_count, forced, max_inv, max_dd, win, loss, _ = simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
            queue_causal = causal_lookup.get((split, "side_by_queue_imbalance_abs_0p1"), float("nan"))
            hgb_causal = causal_lookup.get((split, "ml_hgb_regressor_shallow_convergence_selected"), float("nan"))
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "best_epoch": int(ckpt.get("best_epoch", -1)),
                    "threshold": best_thr,
                    "pnl": metric["total_excess_pnl_H20"],
                    "selected_action_count": metric["selected_action_count"],
                    "matched_random_95p": float(random_row["random_95p"]),
                    "matched_random_pass": bool(random_row["beats_random_95p"]),
                    "empirical_p_value_fraction_random_ge_model": float(random_row["empirical_p_value_fraction_random_ge_model"]),
                    "causal_fill_pnl": float(total),
                    "causal_fill_count": int(fill_count),
                    "causal_fill_max_drawdown": float(max_dd),
                    "causal_fill_beats_queue_rule": bool(total > queue_causal) if np.isfinite(queue_causal) else False,
                    "causal_fill_beats_hgb": bool(total > hgb_causal) if np.isfinite(hgb_causal) else False,
                    "beats_queue_rule": bool(metric["total_excess_pnl_H20"] > queue_lookup.get(split, -float("inf"))),
                    "beats_hgb": bool(metric["total_excess_pnl_H20"] > hgb_lookup.get(split, -float("inf"))),
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_lite_seed_stability.csv", index=False)
    write_seed_stability_report(out)
    return out


def evaluate_or_load_metrics() -> pd.DataFrame:
    path = RESULTS / "phc_lite_oos_policy_eval.csv"
    if path.exists():
        return pd.read_csv(path)
    return evaluate_oos()[0]


def hls_feasibility() -> tuple[pd.DataFrame, pd.DataFrame]:
    model, ckpt = load_model()
    params = count_parameters(model)
    rows = [
        {
            "model": "phc_lite",
            "feature_dim": len(PHC_FEATURES),
            "hidden_dim": int(ckpt["hidden_dim"]),
            "mtl_steps": int(ckpt["mtl_steps"]),
            "parameter_count": params,
            "state_values": int(ckpt["hidden_dim"]),
            "estimated_mac_per_event": int(len(PHC_FEATURES) * int(ckpt["hidden_dim"]) + 4 * int(ckpt["hidden_dim"]) * int(ckpt["hidden_dim"])),
            "target_under_20k_params": bool(params < 20000),
            "streaming_inference_plausible": bool(params < 20000),
        }
    ]
    param_df = pd.DataFrame(rows)
    param_df.to_csv(RESULTS / "phc_lite_parameter_count.csv", index=False)
    val = load_sequences("val")
    fp32 = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
    quantized = np.round(fp32 * 16.0) / 16.0
    threshold = float(read_json(RESULTS / "phc_lite_selected_policy.json")["threshold"])
    agree = policy_from_scores(fp32, threshold) == policy_from_scores(quantized, threshold)
    fixed_df = pd.DataFrame(
        [
            {
                "split": "val",
                "quantization": "score_round_to_1_16_proxy",
                "mean_abs_score_error": float(np.mean(np.abs(fp32 - quantized))),
                "max_abs_score_error": float(np.max(np.abs(fp32 - quantized))),
                "action_agreement_rate": float(agree.mean()),
                "diagnostic_only": True,
            }
        ]
    )
    fixed_df.to_csv(RESULTS / "phc_lite_fixed_point_emulation.csv", index=False)
    write_hls_report(param_df, fixed_df)
    return param_df, fixed_df


def make_plots_common() -> None:
    if (RESULTS / "phc_lite_training_curve.csv").exists():
        plot_training_curves(pd.read_csv(RESULTS / "phc_lite_training_curve.csv"))
    if (RESULTS / "phc_lite_oos_policy_eval.csv").exists():
        plot_policy_comparison(pd.read_csv(RESULTS / "phc_lite_oos_policy_eval.csv"))


def plot_training_curves(curve: pd.DataFrame, suffix: str = "") -> None:
    ensure_dirs()
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["train_loss"], label="train")
    ax.plot(curve["epoch"], curve["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("PHC-lite Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"phc_lite_loss_curve{suffix}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation policy excess PnL")
    ax.set_title("PHC-lite Validation Policy PnL by Epoch")
    fig.tight_layout()
    fig.savefig(FIGURES / f"phc_lite_val_policy_pnl_by_epoch{suffix}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["val_spearman_bid"], label="bid")
    ax.plot(curve["epoch"], curve["val_spearman_ask"], label="ask")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Spearman")
    ax.set_title("PHC-lite Spearman by Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"phc_lite_spearman_by_epoch{suffix}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["grad_norm"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm before clipping")
    ax.set_title("PHC-lite Gradient Norm by Epoch")
    fig.tight_layout()
    fig.savefig(FIGURES / f"phc_lite_grad_norm_by_epoch{suffix}.png", dpi=160)
    plt.close(fig)


def plot_policy_comparison(metrics: pd.DataFrame) -> None:
    subset = metrics[metrics["split"].isin(["val", "new3", "new4"])].copy()
    piv = subset.pivot_table(index="policy_id", columns="split", values="total_excess_pnl_H20", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(11, 6))
    piv.plot(kind="bar", ax=ax)
    ax.set_title("PHC-lite vs Frozen Baselines (Sequence Target Universe)")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Total excess PnL H20")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIGURES / "phc_lite_policy_comparison.png", dpi=160)
    plt.close(fig)
    if (RESULTS / "phc_lite_session_pnl.csv").exists():
        sessions = pd.read_csv(RESULTS / "phc_lite_session_pnl.csv")
        phc = sessions[sessions["policy_id"].eq("phc_lite")]
        fig, ax = plt.subplots(figsize=(12, 5))
        phc.pivot_table(index="session_id", columns="split", values="session_excess_pnl_H20", aggfunc="sum").plot(kind="bar", ax=ax)
        ax.set_title("PHC-lite Session PnL")
        ax.set_xlabel("Session")
        ax.set_ylabel("Session excess PnL H20")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        fig.savefig(FIGURES / "phc_lite_session_pnl.png", dpi=160)
        plt.close(fig)


def plot_causal_fill(out: pd.DataFrame) -> None:
    piv = out.pivot_table(index="policy_id", columns="split", values="sequential_excess_pnl", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(11, 6))
    piv.plot(kind="bar", ax=ax)
    ax.set_title("PHC-lite Causal-Fill Sequential Proxy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Sequential excess PnL")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIGURES / "phc_lite_causal_fill_comparison.png", dpi=160)
    plt.close(fig)


def plot_oracle_confusion(oracle_df: pd.DataFrame) -> None:
    splits = ["val", "new3", "new4"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, split in zip(axes, splits):
        sub = oracle_df[oracle_df["split"].eq(split)].pivot_table(index="oracle_action", columns="model_action", values="count", aggfunc="sum").fillna(0)
        im = ax.imshow(sub.to_numpy(), cmap="Blues")
        ax.set_title(split)
        ax.set_xticks(range(len(sub.columns)), sub.columns, rotation=30)
        ax.set_yticks(range(len(sub.index)), sub.index)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                ax.text(j, i, int(sub.iloc[i, j]), ha="center", va="center", fontsize=8)
    fig.suptitle("PHC-lite Oracle Action Confusion")
    fig.supxlabel("Model action")
    fig.supylabel("Oracle action")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    fig.tight_layout()
    fig.savefig(FIGURES / "phc_lite_oracle_action_confusion.png", dpi=160)
    plt.close(fig)


def plot_score_bucket(out: pd.DataFrame) -> None:
    if out.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    out.pivot_table(index="bucket", columns="split", values="avg_excess_pnl", aggfunc="mean").plot(kind="bar", ax=ax)
    ax.set_title("PHC-lite Score Bucket Economic Lift")
    ax.set_xlabel("Validation score bucket")
    ax.set_ylabel("Avg selected-side excess PnL")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIGURES / "phc_lite_score_bucket_lift.png", dpi=160)
    plt.close(fig)


def write_sequence_dataset_report(manifest: pd.DataFrame, scaler: dict[str, Any]) -> None:
    summary = manifest.groupby("split", as_index=False).agg(
        event_rows=("event_rows", "sum"),
        window_rows=("window_rows", "sum"),
        sessions=("part", "count"),
        bid_positive_rate=("bid_positive_rate", "mean"),
        ask_positive_rate=("ask_positive_rate", "mean"),
        target_mean_bid=("target_mean_bid", "mean"),
        target_mean_ask=("target_mean_ask", "mean"),
    )
    write_markdown(
        ROOT / "PHC_LITE_SEQUENCE_DATASET_REPORT.md",
        [
            "# PHC-lite Sequence Dataset Report",
            "",
            "Phase 05A uses the frozen Phase 04B TXF-only action universe. `NEW5` is not loaded. Development is `NEW2 -> train` and `NEW1 -> val`; diagnostic OOS is `NEW3` and `NEW4`.",
            "",
            f"Sequence setting: `seq_len={SEQ_LEN}`, `stride={STRIDE}`, target is the final timestep, target horizon is `H20`.",
            "",
            "Allowed PHC-lite input features:",
            "",
            ", ".join(PHC_FEATURES),
            "",
            "Forbidden columns are not used as inputs: best bid/ask, mid/quote price, product id, future_mid, raw_pnl, excess_pnl, positive labels, terminal/session-final EV.",
            "",
            "## Split Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Normalization",
            "",
            "Feature scaler is fit on train only and applied unchanged to val/NEW3/NEW4.",
            f"Train rows used for scaler: `{scaler['train_feature_rows']}`.",
            f"Target scale: `{scaler['target_scale_train_std_no_mean_shift']}`. The target is divided by this positive scale during optimization only; no mean shift is applied, so raw zero remains the no-quote reference.",
            "",
            "No NEW5 data is used for scaler fitting, architecture, early stopping, thresholding, or diagnostics in this phase.",
        ],
    )


def write_training_report(curve: pd.DataFrame, seed: int = 0, overwrite_main: bool = True) -> None:
    if curve.empty:
        return
    best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    train_decreased = bool(curve["train_loss"].iloc[-1] < curve["train_loss"].iloc[0])
    val_decreased = bool(curve["val_loss"].iloc[-1] <= curve["val_loss"].iloc[0])
    max_epoch_hit = bool(int(curve["epoch"].iloc[-1]) >= 50)
    grad_explosion = bool(curve["grad_norm"].max() > 100.0)
    path = ROOT / ("PHC_LITE_TRAINING_CONVERGENCE_REPORT.md" if overwrite_main else f"PHC_LITE_TRAINING_CONVERGENCE_REPORT_seed{seed}.md")
    write_markdown(
        path,
        [
            "# PHC-lite Training Convergence Report",
            "",
            f"Seed: `{seed}`.",
            f"Selected epoch: `{int(best['epoch'])}`, selected strictly by validation policy excess PnL at threshold 0.",
            f"Train loss decreased: `{train_decreased}`.",
            f"Validation loss decreased or plateaued: `{val_decreased}`.",
            f"Max epoch hit: `{max_epoch_hit}`.",
            f"Gradient explosion detected: `{grad_explosion}`.",
            f"Best validation policy excess PnL: `{float(best['val_policy_excess_pnl']):.6f}`.",
            f"Best validation selected count: `{int(best['val_selected_count'])}`.",
            "",
            "Validation only is used for epoch selection. NEW3/NEW4/NEW5 are not used for model selection or early stopping.",
        ],
    )


def write_oos_report(metrics: pd.DataFrame) -> None:
    phc = metrics[metrics["policy_id"].eq("phc_lite")]
    queue = metrics[metrics["policy_id"].eq("side_by_queue_imbalance_abs_0p1")]
    hgb = metrics[metrics["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")]
    merged = phc[["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "phc"}).merge(
        queue[["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "queue"}), on="split", how="left"
    ).merge(hgb[["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "hgb"}), on="split", how="left")
    merged["beats_queue"] = merged["phc"] > merged["queue"]
    merged["beats_hgb"] = merged["phc"] > merged["hgb"]
    write_markdown(
        ROOT / "PHC_LITE_OOS_EVALUATION_REPORT.md",
        [
            "# PHC-lite OOS Evaluation Report",
            "",
            "Evaluation is on the sequence target-state universe only. NEW5 is not loaded.",
            "",
            merged.to_markdown(index=False),
            "",
            "Success criterion is positive val/NEW3/NEW4 PnL, beating queue rule, and being competitive with the frozen Phase 04B.1 HGB. This report is diagnostic and does not claim real-world profitability.",
        ],
    )


def write_matched_random_report(summary: pd.DataFrame) -> None:
    lines = [
        "# PHC-lite Matched Random Report",
        "",
        "Matched random trials match selected count, session, side, and spread bucket when feasible, with session-side fallback for sparse buckets.",
        "",
        summary.to_markdown(index=False),
        "",
        "Empirical p-value is defined as the fraction of matched random trials whose PnL is greater than or equal to PHC-lite PnL. If no random trial beats the policy in 1000 trials, wording is `empirical p-value < 0.001`, not exactly zero.",
    ]
    write_markdown(ROOT / "PHC_LITE_MATCHED_RANDOM_REPORT.md", lines)


def write_causal_fill_report(out: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "PHC_LITE_CAUSAL_FILL_REPORT.md",
        [
            "# PHC-lite Causal-Fill Sequential Report",
            "",
            "This uses the same TXF-only conservative L2 causal-fill proxy family: H20 exit, inventory cap 1, fee already embedded in excess labels, and no future PnL is used for fill decisions. It is not true queue replay.",
            "",
            out.to_markdown(index=False),
            "",
            "The result is proxy-only and cannot prove real-world profitability or FPGA necessity.",
        ],
    )


def write_decision_report(action_df: pd.DataFrame, oracle_df: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "PHC_LITE_DECISION_DIAGNOSTICS_REPORT.md",
        [
            "# PHC-lite Decision Diagnostics Report",
            "",
            "The selected-action confusion matrix compares model selection to whether the chosen action has positive `excess_pnl_H20`. The oracle matrix compares the model action against an untradable future-information oracle `argmax(excess_bid, excess_ask, 0)`.",
            "",
            "## Selected Action Confusion",
            "",
            action_df.to_markdown(index=False),
            "",
            "## Oracle Action Confusion",
            "",
            oracle_df.head(50).to_markdown(index=False),
            "",
            "This is diagnostic only. It does not imply the model predicts the oracle perfectly.",
        ],
    )


def write_score_bucket_report(out: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "PHC_LITE_SCORE_BUCKET_REPORT.md",
        [
            "# PHC-lite Score Bucket Economic Lift Report",
            "",
            "Buckets are defined by validation score quantiles and applied unchanged to NEW3/NEW4.",
            "",
            out.to_markdown(index=False),
        ],
    )


def write_seed_stability_report(out: pd.DataFrame) -> None:
    summary = out.groupby("seed", as_index=False).agg(
        val_pnl=("pnl", lambda s: float(s[out.loc[s.index, "split"].eq("val")].iloc[0]) if any(out.loc[s.index, "split"].eq("val")) else float("nan")),
        new3_pnl=("pnl", lambda s: float(s[out.loc[s.index, "split"].eq("new3")].iloc[0]) if any(out.loc[s.index, "split"].eq("new3")) else float("nan")),
        new4_pnl=("pnl", lambda s: float(s[out.loc[s.index, "split"].eq("new4")].iloc[0]) if any(out.loc[s.index, "split"].eq("new4")) else float("nan")),
    )
    write_markdown(
        ROOT / "PHC_LITE_SEED_STABILITY_REPORT.md",
        [
            "# PHC-lite Seed Stability Report",
            "",
            "Seeds are trained using train only and selected using validation only. NEW3/NEW4 are diagnostic OOS; NEW5 is not used.",
            "",
            out.to_markdown(index=False),
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "If only one seed works, PHC-lite should not be accepted as a stable candidate.",
        ],
    )


def write_hls_report(param_df: pd.DataFrame, fixed_df: pd.DataFrame) -> None:
    write_markdown(
        ROOT / "PHC_LITE_HLS_FEASIBILITY_NOTE.md",
        [
            "# PHC-lite HLS / Streaming Feasibility Note",
            "",
            "This is a feasibility note only, not an HLS implementation and not a claim that FPGA improves PnL.",
            "",
            param_df.to_markdown(index=False),
            "",
            "## Fixed-Point Proxy Diagnostic",
            "",
            fixed_df.to_markdown(index=False),
            "",
            "The model keeps a small recurrent state and one within-timestep interaction loop, making streaming inference plausible if later validated with fixed-point and HLS tooling.",
        ],
    )


def final_report() -> str:
    metrics = pd.read_csv(RESULTS / "phc_lite_oos_policy_eval.csv") if (RESULTS / "phc_lite_oos_policy_eval.csv").exists() else pd.DataFrame()
    random_summary = pd.read_csv(RESULTS / "phc_lite_matched_random_summary.csv") if (RESULTS / "phc_lite_matched_random_summary.csv").exists() else pd.DataFrame()
    causal = pd.read_csv(RESULTS / "phc_lite_causal_fill_sequential_eval.csv") if (RESULTS / "phc_lite_causal_fill_sequential_eval.csv").exists() else pd.DataFrame()
    seed = pd.read_csv(RESULTS / "phc_lite_seed_stability.csv") if (RESULTS / "phc_lite_seed_stability.csv").exists() else pd.DataFrame()
    curve = pd.read_csv(RESULTS / "phc_lite_training_curve.csv") if (RESULTS / "phc_lite_training_curve.csv").exists() else pd.DataFrame()
    status = "phc_lite_not_ready"
    if not curve.empty and not metrics.empty:
        converged = bool(curve["train_loss"].iloc[-1] < curve["train_loss"].iloc[0]) and bool(curve["grad_norm"].max() < 100.0)
        phc = metrics[metrics["policy_id"].eq("phc_lite")].set_index("split")
        queue = metrics[metrics["policy_id"].eq("side_by_queue_imbalance_abs_0p1")].set_index("split")
        hgb = metrics[metrics["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].set_index("split")
        positive_all = all(phc.loc[s, "total_excess_pnl_H20"] > 0 for s in ["val", "new3", "new4"] if s in phc.index)
        beats_queue_all = all(phc.loc[s, "total_excess_pnl_H20"] > queue.loc[s, "total_excess_pnl_H20"] for s in ["val", "new3", "new4"] if s in phc.index and s in queue.index)
        beats_hgb_all = all(phc.loc[s, "total_excess_pnl_H20"] > hgb.loc[s, "total_excess_pnl_H20"] for s in ["val", "new3", "new4"] if s in phc.index and s in hgb.index)
        stable = False
        if not seed.empty:
            wide = seed.pivot_table(index="seed", columns="split", values="pnl", aggfunc="first")
            stable = len(wide) >= 3 and (wide[["val", "new3", "new4"]] > 0).all(axis=None)
        if not converged:
            status = "phc_lite_training_not_converged"
        elif not stable:
            status = "phc_lite_unstable_across_seeds"
        elif positive_all and beats_queue_all and beats_hgb_all:
            status = "phc_lite_beats_hgb_and_queue"
        elif positive_all:
            status = "phc_lite_signal_but_no_strategy_gain"
        else:
            status = "phc_lite_not_ready"
    lines = [
        "# PHC-lite Microstructure Feasibility Report",
        "",
        f"Final status: `{status}`.",
        "",
        "## 1. What PHC-lite Is",
        "",
        "PHC-lite is a PHC-inspired lightweight sequence model for TXF-only L2 microstructure states. It combines a temporal recurrent state with a small within-timestep multi-transmission interaction loop and emits bid/ask action-value scores.",
        "",
        "## 2. What It Is Not",
        "",
        "It is not full PHCSSM: ALIF dynamics, synaptic delay, STP, Dale's Law, and STDP are not implemented. It is not true exchange replay and does not prove real-world profitability.",
        "",
        "## 3. Data Protocol",
        "",
        "Only Phase 04B TXF-family action universe data is used. NEW2 is train, NEW1 is validation, NEW3/NEW4 are diagnostic OOS. NEW5 is not loaded, tuned on, or evaluated in Phase 05A.",
        "",
        "## 4. Training Diagnostics",
        "",
        curve.tail(8).to_markdown(index=False) if not curve.empty else "Training curve missing.",
        "",
        "## 5. Frozen OOS Policy Evaluation",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "OOS metrics missing.",
        "",
        "## 6. Matched Random",
        "",
        random_summary.to_markdown(index=False) if not random_summary.empty else "Matched random summary missing.",
        "",
        "## 7. Causal-Fill Sequential Proxy",
        "",
        causal.to_markdown(index=False) if not causal.empty else "Causal-fill sequential summary missing.",
        "",
        "## 8. Seed Stability",
        "",
        seed.to_markdown(index=False) if not seed.empty else "Seed stability summary missing.",
        "",
        "## 9. HLS-Style Streaming Feasibility",
        "",
        "Parameter count and fixed-point proxy diagnostics are provided in `results/phc_lite_parameter_count.csv` and `results/phc_lite_fixed_point_emulation.csv`. This only supports a future HLS feasibility discussion, not an FPGA PnL claim.",
        "",
        "## 10. Conservative Conclusion",
        "",
        "PHC-lite is evaluated as an advanced sequence-model feasibility test after the Phase 04C holdout. Because NEW5 is already burned, the rigorous next step for a positive PHC-lite result would require a fresh NEW6-style holdout. No Phase 05A result should be described as final holdout proof.",
    ]
    write_markdown(ROOT / "PHC_LITE_MICROSTRUCTURE_FEASIBILITY_REPORT.md", lines)
    return status


def run_all() -> None:
    build_sequence_dataset()
    train_one_seed(seed=0, overwrite_main=True)
    select_policy_threshold()
    evaluate_oos()
    run_matched_random()
    causal_fill_eval()
    decision_diagnostics()
    score_bucket_lift()
    seed_stability()
    hls_feasibility()
    final_report()


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--mtl-steps", type=int, default=1)
    parser.add_argument("--trials", type=int, default=RANDOM_TRIALS)
    return parser.parse_args()
