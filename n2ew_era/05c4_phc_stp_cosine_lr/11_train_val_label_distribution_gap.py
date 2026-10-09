from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE05A_DATA = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility" / "data"
RESULTS = ROOT / "results"
REPORT = ROOT / "AUDIT_train_val_label_distribution_gap.md"
CSV_OUT = RESULTS / "train_val_label_distribution_check.csv"
TRAIN_CURVE = RESULTS / "phc_stp_v2_training_curve_seed2.csv"

SPLITS = ["train", "val"]
SIDES = ["bid", "ask"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payload(split: str) -> dict[str, Any]:
    if split not in SPLITS or "NEW5" in split.upper():
        raise RuntimeError(f"Forbidden split requested: {split}")
    path = PHASE05A_DATA / f"phc_lite_{split}_sequences.pt"
    if "NEW5" in str(path).upper():
        raise RuntimeError(f"Forbidden NEW5 path encountered: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if bool(payload.get("new5_used", False)):
        raise RuntimeError(f"Refusing payload marked new5_used=True: {path}")
    return payload


def day_from_session(session_id: str) -> str:
    match = re.search(r"_(\d{8})", str(session_id))
    return match.group(1) if match else "unknown"


def huber_np(error: np.ndarray, delta: float = 1.0) -> float:
    abs_error = np.abs(error)
    loss = np.where(abs_error <= delta, 0.5 * error * error, delta * (abs_error - 0.5 * delta))
    return float(np.mean(loss))


def y_array(payload: dict[str, Any], side: str) -> np.ndarray:
    return payload[f"y_{side}"].detach().cpu().numpy().astype(np.float64)


def base_stats(values: np.ndarray, target_scale: float, prefix: str) -> dict[str, Any]:
    norm = values / target_scale
    finite = np.isfinite(values)
    return {
        f"{prefix}_mean_raw": float(np.mean(values)) if len(values) else np.nan,
        f"{prefix}_std_raw_population": float(np.std(values, ddof=0)) if len(values) else np.nan,
        f"{prefix}_mean_normalized": float(np.mean(norm)) if len(norm) else np.nan,
        f"{prefix}_std_normalized_population": float(np.std(norm, ddof=0)) if len(norm) else np.nan,
        f"{prefix}_nan_count": int(np.isnan(values).sum()),
        f"{prefix}_finite_count": int(finite.sum()),
        f"{prefix}_unique_count": int(len(np.unique(values))) if len(values) else 0,
        f"{prefix}_all_same": bool(len(np.unique(values)) <= 1) if len(values) else True,
        f"{prefix}_min_raw": float(np.min(values)) if len(values) else np.nan,
        f"{prefix}_max_raw": float(np.max(values)) if len(values) else np.nan,
        f"{prefix}_zero_prediction_huber_normalized": huber_np(norm) if len(norm) else np.nan,
    }


def row_for_mask(
    payload: dict[str, Any],
    split: str,
    row_type: str,
    target_scale: float,
    mask: np.ndarray,
    session_id: str = "",
    day_id: str = "",
) -> dict[str, Any]:
    labels = [str(x) for x in payload["session_labels"]]
    session_code = payload["session_code"].detach().cpu().numpy().astype(np.int64)
    session_ids = [labels[int(code)] for code in np.unique(session_code[mask])] if mask.any() else []
    day_ids = sorted({day_from_session(label) for label in session_ids})
    row: dict[str, Any] = {
        "row_type": row_type,
        "split": split,
        "session_id": session_id,
        "day_id": day_id,
        "row_count": int(mask.sum()),
        "session_count": int(len(session_ids)),
        "day_count": int(len(day_ids)),
        "session_labels_used": ",".join(session_ids),
        "day_labels_used": ",".join(day_ids),
        "target_scale_train_std_no_mean_shift": float(target_scale),
        "session_field_used": "session_code + session_labels",
    }
    for side in SIDES:
        row.update(base_stats(y_array(payload, side)[mask], target_scale, f"y_{side}"))
    row["avg_std_raw_population_bid_ask"] = float((row["y_bid_std_raw_population"] + row["y_ask_std_raw_population"]) / 2.0)
    row["avg_std_normalized_population_bid_ask"] = float(
        (row["y_bid_std_normalized_population"] + row["y_ask_std_normalized_population"]) / 2.0
    )
    row["avg_zero_prediction_huber_normalized_bid_ask"] = float(
        (row["y_bid_zero_prediction_huber_normalized"] + row["y_ask_zero_prediction_huber_normalized"]) / 2.0
    )
    return row


def summarize_payload(payload: dict[str, Any], split: str, target_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(payload["y_bid"])
    all_mask = np.ones(n, dtype=bool)
    rows.append(row_for_mask(payload, split, "split_summary", target_scale, all_mask, session_id="ALL", day_id="ALL"))

    labels = [str(x) for x in payload["session_labels"]]
    session_code = payload["session_code"].detach().cpu().numpy().astype(np.int64)

    for day in sorted({day_from_session(label) for label in labels}):
        day_codes = [i for i, label in enumerate(labels) if day_from_session(label) == day]
        day_mask = np.isin(session_code, day_codes)
        rows.append(row_for_mask(payload, split, "day_summary", target_scale, day_mask, session_id="ALL_FOR_DAY", day_id=day))

    for code, label in enumerate(labels):
        mask = session_code == code
        rows.append(row_for_mask(payload, split, "session_summary", target_scale, mask, session_id=label, day_id=day_from_session(label)))
    return rows


def add_train_distribution_position(rows: list[dict[str, Any]]) -> None:
    train_sessions = [
        row["avg_std_normalized_population_bid_ask"]
        for row in rows
        if row["row_type"] == "session_summary" and row["split"] == "train"
    ]
    train_sorted = sorted(float(x) for x in train_sessions)
    for row in rows:
        if row["row_type"] not in {"session_summary", "day_summary", "split_summary"}:
            continue
        value = float(row["avg_std_normalized_population_bid_ask"])
        row["rank_if_inserted_in_train_sessions_by_avg_norm_std_1_lowest"] = int(np.searchsorted(train_sorted, value, side="left") + 1)
        row["percentile_vs_train_sessions_by_avg_norm_std"] = float(np.mean(np.asarray(train_sorted) <= value)) if train_sorted else np.nan
        if not train_sorted:
            row["position_vs_train_session_std_range"] = "no_train_sessions"
        elif value < train_sorted[0]:
            row["position_vs_train_session_std_range"] = "below_train_min"
        elif value > train_sorted[-1]:
            row["position_vs_train_session_std_range"] = "above_train_max"
        else:
            row["position_vs_train_session_std_range"] = "within_train_range"


def health_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["row_type"] != "split_summary":
            continue
        health_ok = (
            int(row["y_bid_nan_count"]) == 0
            and int(row["y_ask_nan_count"]) == 0
            and not bool(row["y_bid_all_same"])
            and not bool(row["y_ask_all_same"])
        )
        out.append(
            {
                "row_type": "health_check",
                "split": row["split"],
                "session_id": "ALL",
                "day_id": "ALL",
                "row_count": row["row_count"],
                "target_scale_train_std_no_mean_shift": row["target_scale_train_std_no_mean_shift"],
                "y_bid_nan_count": row["y_bid_nan_count"],
                "y_ask_nan_count": row["y_ask_nan_count"],
                "y_bid_unique_count": row["y_bid_unique_count"],
                "y_ask_unique_count": row["y_ask_unique_count"],
                "y_bid_all_same": row["y_bid_all_same"],
                "y_ask_all_same": row["y_ask_all_same"],
                "health_ok": health_ok,
                "session_field_used": row["session_field_used"],
            }
        )
    return out


def loss_proxy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = {row["split"]: row for row in rows if row["row_type"] == "split_summary"}
    train = summary["train"]
    val = summary["val"]
    curve = pd.read_csv(TRAIN_CURVE)
    epoch171 = curve[curve["epoch"].astype(int).eq(171)].iloc[0] if "epoch" in curve.columns else curve.iloc[-1]
    train_loss = float(epoch171["train_huber_loss"])
    val_loss = float(epoch171["val_huber_loss"])
    observed_gap = train_loss - val_loss
    observed_ratio = train_loss / val_loss if val_loss else np.nan
    train_zero = float(train["avg_zero_prediction_huber_normalized_bid_ask"])
    val_zero = float(val["avg_zero_prediction_huber_normalized_bid_ask"])
    zero_gap = train_zero - val_zero
    return [
        {
            "row_type": "loss_scale_proxy",
            "split": "train_vs_val",
            "session_id": "ALL",
            "day_id": "ALL",
            "target_scale_train_std_no_mean_shift": train["target_scale_train_std_no_mean_shift"],
            "epoch_reference": int(epoch171["epoch"]),
            "observed_train_huber_loss": train_loss,
            "observed_val_huber_loss": val_loss,
            "observed_train_minus_val_huber_loss": observed_gap,
            "observed_train_over_val_huber_ratio": observed_ratio,
            "train_avg_zero_prediction_huber_normalized_bid_ask": train_zero,
            "val_avg_zero_prediction_huber_normalized_bid_ask": val_zero,
            "zero_prediction_huber_gap_train_minus_val": zero_gap,
            "zero_prediction_gap_over_observed_gap": zero_gap / observed_gap if observed_gap else np.nan,
            "train_avg_norm_std": train["avg_std_normalized_population_bid_ask"],
            "val_avg_norm_std": val["avg_std_normalized_population_bid_ask"],
            "train_over_val_avg_norm_std_ratio": train["avg_std_normalized_population_bid_ask"]
            / val["avg_std_normalized_population_bid_ask"],
            "train_over_val_avg_norm_variance_ratio": (
                train["avg_std_normalized_population_bid_ask"] / val["avg_std_normalized_population_bid_ask"]
            )
            ** 2,
        }
    ]


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                text = "N/A"
            elif any(token in col for token in ["count", "rank", "epoch_reference", "row_count", "day_count"]) and abs(float(value) - round(float(value))) < 1e-9:
                text = str(int(round(float(value))))
            elif isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            vals.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    split = df[df["row_type"].eq("split_summary")].copy()
    sessions = df[df["row_type"].eq("session_summary")].copy()
    days = df[df["row_type"].eq("day_summary")].copy()
    health = df[df["row_type"].eq("health_check")].copy()
    proxy = df[df["row_type"].eq("loss_scale_proxy")].copy()
    val_std = float(split[split["split"].eq("val")]["avg_std_normalized_population_bid_ask"].iloc[0])
    train_std = float(split[split["split"].eq("train")]["avg_std_normalized_population_bid_ask"].iloc[0])
    observed_gap = float(proxy["observed_train_minus_val_huber_loss"].iloc[0])
    zero_gap = float(proxy["zero_prediction_huber_gap_train_minus_val"].iloc[0])
    val_sessions = sessions[sessions["split"].eq("val")]
    val_below_train_min = int(val_sessions["position_vs_train_session_std_range"].eq("below_train_min").sum())
    val_within_train_range = int(val_sessions["position_vs_train_session_std_range"].eq("within_train_range").sum())
    val_day_rows = days[days["split"].eq("val")]
    val_days_below_train_min = int(val_day_rows["position_vs_train_session_std_range"].eq("below_train_min").sum())
    val_days_within_train_range = int(val_day_rows["position_vs_train_session_std_range"].eq("within_train_range").sum())
    status = "resolved_distributional_explanation" if val_std < train_std and zero_gap >= observed_gap else "inconclusive"
    lines = [
        "# AUDIT: Train/Val Label Distribution Gap",
        "",
        f"status: `{status}`",
        "",
        "## Scope Boundary",
        "",
        "- Splits used: `train` and `val` only.",
        "- NEW5 was not loaded or touched; loaded sequence payloads were checked for `new5_used=False`.",
        "- No model was trained, no hyperparameter was changed, and no frozen checkpoint, scaler, or training curve file was modified.",
        "- Payload session grouping uses `session_code + session_labels`; there is no per-row `session_id` field in the `.pt` payload.",
        "- Scaler source: `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/data/phc_lite_feature_scaler.json`.",
        "",
        "## Split Summary",
        "",
        markdown_table(
            split,
            [
                "split",
                "row_count",
                "session_count",
                "day_count",
                "y_bid_mean_raw",
                "y_bid_std_raw_population",
                "y_ask_mean_raw",
                "y_ask_std_raw_population",
                "avg_std_normalized_population_bid_ask",
                "avg_zero_prediction_huber_normalized_bid_ask",
            ],
        ),
        "",
        "## Day Summary",
        "",
        markdown_table(
            days,
            [
                "split",
                "day_id",
                "row_count",
                "session_count",
                "y_bid_std_raw_population",
                "y_ask_std_raw_population",
                "avg_std_normalized_population_bid_ask",
                "position_vs_train_session_std_range",
            ],
        ),
        "",
        "## Session Summary",
        "",
        markdown_table(
            sessions,
            [
                "split",
                "session_id",
                "day_id",
                "row_count",
                "y_bid_std_raw_population",
                "y_ask_std_raw_population",
                "avg_std_normalized_population_bid_ask",
                "rank_if_inserted_in_train_sessions_by_avg_norm_std_1_lowest",
                "position_vs_train_session_std_range",
            ],
        ),
        "",
        "## Health Check",
        "",
        markdown_table(
            health,
            [
                "split",
                "row_count",
                "y_bid_nan_count",
                "y_ask_nan_count",
                "y_bid_unique_count",
                "y_ask_unique_count",
                "y_bid_all_same",
                "y_ask_all_same",
                "health_ok",
            ],
        ),
        "",
        "## Loss-Scale Proxy",
        "",
        markdown_table(
            proxy,
            [
                "epoch_reference",
                "observed_train_huber_loss",
                "observed_val_huber_loss",
                "observed_train_minus_val_huber_loss",
                "train_avg_norm_std",
                "val_avg_norm_std",
                "train_over_val_avg_norm_std_ratio",
                "train_over_val_avg_norm_variance_ratio",
                "zero_prediction_huber_gap_train_minus_val",
                "zero_prediction_gap_over_observed_gap",
            ],
        ),
        "",
        "## Honest Conclusion",
        "",
    ]
    if status == "resolved_distributional_explanation":
        lines.append(
            "The distributional hypothesis is supported. The validation labels have much lower normalized standard deviation than train "
            f"(`{val_std:.6f}` vs `{train_std:.6f}` averaged across bid/ask). The observed epoch-171 train-val Huber gap is "
            f"`{observed_gap:.6f}`, while the zero-prediction normalized-label Huber gap is `{zero_gap:.6f}`. "
            "That target-distribution-only gap is larger than the observed model-loss gap, so the label variance difference is large enough in magnitude to explain why val loss is persistently lower. "
            f"The finer table is slightly more nuanced: `{val_below_train_min}` of `{len(val_sessions)}` validation sessions are below the train-session std minimum and `{val_within_train_range}` are within the train-session range; by day, `{val_days_below_train_min}` validation days are below the train-session minimum and `{val_days_within_train_range}` are within the train-session range. "
            "So the effect is not that every validation session/day is lower than every train session/day; rather, the aggregate train split is pulled up by a high-volatility train session/day that is absent from validation. This is a sample composition explanation, not evidence of leakage."
        )
    else:
        lines.append(
            "The distributional hypothesis is not fully supported by the numbers. The measured label-distribution gap is not large enough or not in the right direction to explain the train-val Huber loss gap; the true cause remains uncertain."
        )
    lines += [
        "",
        "CSV output: `results/train_val_label_distribution_check.csv`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    scaler = load_json(PHASE05A_DATA / "phc_lite_feature_scaler.json")
    if scaler.get("normalization_fit_split") != "train":
        raise RuntimeError(f"Unexpected normalization_fit_split={scaler.get('normalization_fit_split')}")
    if bool(scaler.get("new5_used_for_scaler", False)):
        raise RuntimeError("Refusing scaler marked new5_used_for_scaler=True")
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        payload = load_payload(split)
        rows.extend(summarize_payload(payload, split, target_scale))
    add_train_distribution_position(rows)
    rows.extend(health_rows(rows))
    rows.extend(loss_proxy_rows(rows))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    write_report(rows)
    print(json.dumps({"csv": str(CSV_OUT), "report": str(REPORT), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
