from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import _common
from new_era.models_playaround_15.data_loading import load_new_tw_data
from src.data.tw_replay import infer_tick_size_from_lob


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
OUT_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / EXPERIMENT_ID
    / "eval_outputs"
    / "diagnosis_v3"
)

NEW_DATA_DIR = PROJECT_ROOT / "data" / "NEW_ALL"
NEW3_DATA_DIR = PROJECT_ROOT / "data" / "NEW3"
NEW_SYMBOL = "TXFB6"
NEW3_SYMBOL = "TXFA6"

FILL_WINDOWS = [1, 3, 5]
HORIZONS = [5, 10, 20]
SAMPLE_STEP = 10
MIN_SIZE_DECREASE = 1.0
TOXIC_CUTOFF = -0.5
GOOD_CUTOFF = 0.5


def split_ordered_session_ids(
    session_ids: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(session_ids)
    total = len(ordered)
    if total == 1:
        return ordered, [], []
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def _format_schema(df: pd.DataFrame) -> str:
    lines = []
    lines.append("Columns:")
    lines.append(", ".join(df.columns.tolist()))
    lines.append("")
    lines.append("Dtypes:")
    lines.append(df.dtypes.to_string())
    return "\n".join(lines)


def write_fill_evidence_audit(
    out_dir: Path,
    new_lob: pd.DataFrame,
    new_trade: pd.DataFrame,
    new3_lob: pd.DataFrame,
    new3_trade: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Fill Evidence Audit")
    lines.append("")
    lines.append("## Available LOB Columns")
    lines.append("")
    lines.append("### NEW_ALL / TXFB6")
    lines.append(_format_schema(new_lob))
    lines.append("")
    lines.append("### NEW3 / TXFA6")
    lines.append(_format_schema(new3_lob))
    lines.append("")
    lines.append("## Available Trade Columns")
    lines.append("")
    lines.append("### NEW_ALL / TXFB6")
    lines.append(_format_schema(new_trade))
    lines.append("")
    lines.append("### NEW3 / TXFA6")
    lines.append(_format_schema(new3_trade))
    lines.append("")
    lines.append("## Fill Evidence Availability")
    lines.append("")
    lines.append("- TRADE_TAPE_AVAILABLE: YES")
    lines.append("- AGGRESSOR_SIDE_AVAILABLE: NO")
    lines.append("- TRADE_DIRECTION_AVAILABLE: NO")
    lines.append("- SIZE_DEPLETION_AVAILABLE: YES")
    lines.append("- MULTI_LEVEL_DEPTH_AVAILABLE: YES")
    lines.append("- ONLY_TOP_OF_BOOK_AVAILABLE: NO")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("True exchange-level fill direction is not available because aggressor_side is missing.")
    lines.append("This task uses LOB-based fill proxies, not true fills.")
    lines.append("")
    lines.append("## Usable Proxies")
    lines.append("")
    lines.append("Use in this audit:")
    lines.append("1. queue_depletion_proxy")
    lines.append("2. price_level_removed_proxy")
    lines.append("3. price_through_proxy")
    lines.append("")
    lines.append("Not used for fill truth:")
    lines.append("- trade_tape_fill: not reliable due to missing aggressor_side")
    lines.append("- touch_or_move_proxy: risk proxy only, not fill proxy")

    (out_dir / "fill_evidence_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _concat_split_frames(lob: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    if not sessions:
        return lob.iloc[0:0].copy()
    session_set = set(sessions)
    return lob[lob["session_id"].isin(session_set)].copy()


def _build_sample_indices(length: int, sample_step: int, max_lookahead: int) -> np.ndarray:
    if length <= max_lookahead:
        return np.array([], dtype=np.int64)
    return np.arange(0, length - max_lookahead, sample_step, dtype=np.int64)


def _conditional_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    fill_count = int(mask.sum())
    if fill_count <= 0:
        return {
            "fill_count": 0,
            "toxic_count": 0,
            "good_count": 0,
            "neutral_count": 0,
            "toxic_rate_conditional": np.nan,
            "good_rate_conditional": np.nan,
            "neutral_rate_conditional": np.nan,
            "mean_markout_conditional": np.nan,
            "median_markout_conditional": np.nan,
            "p10_markout_conditional": np.nan,
            "p90_markout_conditional": np.nan,
        }
    v = values[mask]
    toxic = v < TOXIC_CUTOFF
    good = v > GOOD_CUTOFF
    neutral = (~toxic) & (~good)
    return {
        "fill_count": fill_count,
        "toxic_count": int(toxic.sum()),
        "good_count": int(good.sum()),
        "neutral_count": int(neutral.sum()),
        "toxic_rate_conditional": float(toxic.mean()),
        "good_rate_conditional": float(good.mean()),
        "neutral_rate_conditional": float(neutral.mean()),
        "mean_markout_conditional": float(np.mean(v)),
        "median_markout_conditional": float(np.median(v)),
        "p10_markout_conditional": float(np.percentile(v, 10)),
        "p90_markout_conditional": float(np.percentile(v, 90)),
    }


def _compute_proxy_masks(
    idx: np.ndarray,
    fill_window: int,
    tick_size: float,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_vol: np.ndarray,
    ask_vol: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = len(idx)
    cur_bid = bid_px[idx]
    cur_ask = ask_px[idx]
    cur_bid_vol = bid_vol[idx]
    cur_ask_vol = ask_vol[idx]

    bid_b = np.zeros(n, dtype=bool)  # queue_depletion_proxy
    ask_b = np.zeros(n, dtype=bool)
    bid_c = np.zeros(n, dtype=bool)  # price_level_removed_proxy
    ask_c = np.zeros(n, dtype=bool)
    bid_d = np.zeros(n, dtype=bool)  # price_through_proxy
    ask_d = np.zeros(n, dtype=bool)

    atol = max(1e-9, tick_size * 0.1)
    for off in range(1, fill_window + 1):
        j = idx + off
        fb = bid_px[j]
        fa = ask_px[j]
        fbv = bid_vol[j]
        fav = ask_vol[j]

        same_bid = np.abs(fb - cur_bid) <= atol
        same_ask = np.abs(fa - cur_ask) <= atol

        bid_b |= same_bid & (fbv < (cur_bid_vol - MIN_SIZE_DECREASE))
        ask_b |= same_ask & (fav < (cur_ask_vol - MIN_SIZE_DECREASE))

        bid_c |= fb < cur_bid
        ask_c |= fa > cur_ask

        bid_d |= fa <= cur_bid
        ask_d |= fb >= cur_ask

    return {
        "queue_depletion_proxy": (bid_b, ask_b),
        "price_level_removed_proxy": (bid_c, ask_c),
        "price_through_proxy": (bid_d, ask_d),
    }


def compute_fill_conditional_distribution(
    *,
    dataset: str,
    split: str,
    frame: pd.DataFrame,
    tick_size: float,
    sample_step: int = SAMPLE_STEP,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_events = len(frame)
    max_lookahead = max(max(FILL_WINDOWS), max(HORIZONS))
    idx = _build_sample_indices(n_events, sample_step=sample_step, max_lookahead=max_lookahead)

    if idx.size == 0:
        row = {
            "dataset": dataset,
            "split": split,
            "proxy_type": "NA",
            "fill_window": np.nan,
            "horizon": np.nan,
            "sample_step": sample_step,
            "bid_fill_count": 0,
            "bid_fill_rate": np.nan,
            "bid_toxic_count": 0,
            "bid_good_count": 0,
            "bid_neutral_count": 0,
            "bid_toxic_rate_conditional": np.nan,
            "bid_good_rate_conditional": np.nan,
            "bid_neutral_rate_conditional": np.nan,
            "bid_mean_markout_conditional": np.nan,
            "bid_median_markout_conditional": np.nan,
            "bid_p10_markout_conditional": np.nan,
            "bid_p90_markout_conditional": np.nan,
            "ask_fill_count": 0,
            "ask_fill_rate": np.nan,
            "ask_toxic_count": 0,
            "ask_good_count": 0,
            "ask_neutral_count": 0,
            "ask_toxic_rate_conditional": np.nan,
            "ask_good_rate_conditional": np.nan,
            "ask_neutral_rate_conditional": np.nan,
            "ask_mean_markout_conditional": np.nan,
            "ask_median_markout_conditional": np.nan,
            "ask_p10_markout_conditional": np.nan,
            "ask_p90_markout_conditional": np.nan,
            "notes": f"SPLIT_TOO_SMALL;n_events={n_events}",
        }
        return pd.DataFrame([row])

    bid_px = frame["bid_px_1"].to_numpy(dtype=np.float64)
    ask_px = frame["ask_px_1"].to_numpy(dtype=np.float64)
    bid_vol = frame["bid_vol_1"].to_numpy(dtype=np.float64)
    ask_vol = frame["ask_vol_1"].to_numpy(dtype=np.float64)
    mid = (bid_px + ask_px) / 2.0

    proxy_masks_by_window: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for fill_window in FILL_WINDOWS:
        proxy_masks_by_window[fill_window] = _compute_proxy_masks(
            idx=idx,
            fill_window=fill_window,
            tick_size=tick_size,
            bid_px=bid_px,
            ask_px=ask_px,
            bid_vol=bid_vol,
            ask_vol=ask_vol,
        )

    base_notes = []
    if n_events < 10000:
        base_notes.append("SMALL_SPLIT_EVENTS")

    n_samples = len(idx)
    for horizon in HORIZONS:
        future_mid = mid[idx + horizon]
        bid_markout = (future_mid - bid_px[idx]) / max(tick_size, 1e-9)
        ask_markout = (ask_px[idx] - future_mid) / max(tick_size, 1e-9)

        for fill_window in FILL_WINDOWS:
            masks = proxy_masks_by_window[fill_window]
            for proxy_type, (bid_fill, ask_fill) in masks.items():
                bid_stats = _conditional_stats(bid_markout, bid_fill)
                ask_stats = _conditional_stats(ask_markout, ask_fill)

                bid_fill_count = int(bid_stats["fill_count"])
                ask_fill_count = int(ask_stats["fill_count"])
                bid_fill_rate = bid_fill_count / max(n_samples, 1)
                ask_fill_rate = ask_fill_count / max(n_samples, 1)

                notes = list(base_notes)
                if bid_fill_count < 100:
                    notes.append("BID_LOW_SAMPLE")
                if ask_fill_count < 100:
                    notes.append("ASK_LOW_SAMPLE")
                if bid_fill_rate < 0.001:
                    notes.append("BID_VERY_RARE_PROXY")
                if ask_fill_rate < 0.001:
                    notes.append("ASK_VERY_RARE_PROXY")

                row = {
                    "dataset": dataset,
                    "split": split,
                    "proxy_type": proxy_type,
                    "fill_window": int(fill_window),
                    "horizon": int(horizon),
                    "sample_step": int(sample_step),
                    "bid_fill_count": bid_fill_count,
                    "bid_fill_rate": float(bid_fill_rate),
                    "bid_toxic_count": int(bid_stats["toxic_count"]),
                    "bid_good_count": int(bid_stats["good_count"]),
                    "bid_neutral_count": int(bid_stats["neutral_count"]),
                    "bid_toxic_rate_conditional": bid_stats["toxic_rate_conditional"],
                    "bid_good_rate_conditional": bid_stats["good_rate_conditional"],
                    "bid_neutral_rate_conditional": bid_stats["neutral_rate_conditional"],
                    "bid_mean_markout_conditional": bid_stats["mean_markout_conditional"],
                    "bid_median_markout_conditional": bid_stats["median_markout_conditional"],
                    "bid_p10_markout_conditional": bid_stats["p10_markout_conditional"],
                    "bid_p90_markout_conditional": bid_stats["p90_markout_conditional"],
                    "ask_fill_count": ask_fill_count,
                    "ask_fill_rate": float(ask_fill_rate),
                    "ask_toxic_count": int(ask_stats["toxic_count"]),
                    "ask_good_count": int(ask_stats["good_count"]),
                    "ask_neutral_count": int(ask_stats["neutral_count"]),
                    "ask_toxic_rate_conditional": ask_stats["toxic_rate_conditional"],
                    "ask_good_rate_conditional": ask_stats["good_rate_conditional"],
                    "ask_neutral_rate_conditional": ask_stats["neutral_rate_conditional"],
                    "ask_mean_markout_conditional": ask_stats["mean_markout_conditional"],
                    "ask_median_markout_conditional": ask_stats["median_markout_conditional"],
                    "ask_p10_markout_conditional": ask_stats["p10_markout_conditional"],
                    "ask_p90_markout_conditional": ask_stats["p90_markout_conditional"],
                    "notes": ";".join(notes),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a):
        return np.nan
    return float(a) / max(float(b), 1e-9)


def _build_unconditional_vs_conditional(
    unconditional_path: Path,
    conditional_df: pd.DataFrame,
) -> pd.DataFrame:
    uncond = pd.read_csv(unconditional_path).copy()
    merged = conditional_df.merge(
        uncond[["split", "horizon", "bid_toxic_rate", "ask_toxic_rate"]],
        on=["split", "horizon"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "bid_toxic_rate": "unconditional_bid_toxic_rate",
            "ask_toxic_rate": "unconditional_ask_toxic_rate",
            "bid_toxic_rate_conditional": "conditional_bid_toxic_rate",
            "ask_toxic_rate_conditional": "conditional_ask_toxic_rate",
        }
    )
    merged["bid_toxic_rate_lift"] = [
        _safe_div(c, u if np.isfinite(u) else 1e-9)
        for c, u in zip(merged["conditional_bid_toxic_rate"], merged["unconditional_bid_toxic_rate"])
    ]
    merged["ask_toxic_rate_lift"] = [
        _safe_div(c, u if np.isfinite(u) else 1e-9)
        for c, u in zip(merged["conditional_ask_toxic_rate"], merged["unconditional_ask_toxic_rate"])
    ]
    out = merged[
        [
            "dataset",
            "split",
            "horizon",
            "proxy_type",
            "fill_window",
            "unconditional_bid_toxic_rate",
            "conditional_bid_toxic_rate",
            "bid_toxic_rate_lift",
            "unconditional_ask_toxic_rate",
            "conditional_ask_toxic_rate",
            "ask_toxic_rate_lift",
            "bid_fill_count",
            "ask_fill_count",
            "notes",
        ]
    ].copy()
    return out


def _build_h10_summary(
    conditional_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> pd.DataFrame:
    c10 = conditional_df[conditional_df["horizon"] == 10].copy()
    cmp10 = compare_df[compare_df["horizon"] == 10].copy()
    merged = c10.merge(
        cmp10[
            [
                "dataset",
                "split",
                "proxy_type",
                "fill_window",
                "bid_toxic_rate_lift",
                "ask_toxic_rate_lift",
            ]
        ],
        on=["dataset", "split", "proxy_type", "fill_window"],
        how="left",
    )
    rec_notes: list[str] = []
    for _, row in merged.iterrows():
        notes = []
        if row["proxy_type"] == "price_level_removed_proxy":
            notes.append("PRIMARY_CANDIDATE_PROXY_C")
        if row["bid_fill_count"] < 100 or row["ask_fill_count"] < 100:
            notes.append("LOW_SAMPLE")
        if row["bid_fill_rate"] < 0.001 or row["ask_fill_rate"] < 0.001:
            notes.append("VERY_RARE_PROXY")
        if np.isfinite(row["bid_toxic_rate_lift"]) and row["bid_toxic_rate_lift"] > 1.0:
            notes.append("BID_LIFT>1")
        if np.isfinite(row["ask_toxic_rate_lift"]) and row["ask_toxic_rate_lift"] > 1.0:
            notes.append("ASK_LIFT>1")
        rec_notes.append(";".join(notes))
    merged["recommendation_note"] = rec_notes
    out = merged[
        [
            "dataset",
            "split",
            "proxy_type",
            "fill_window",
            "bid_fill_count",
            "bid_fill_rate",
            "bid_toxic_rate_conditional",
            "bid_mean_markout_conditional",
            "ask_fill_count",
            "ask_fill_rate",
            "ask_toxic_rate_conditional",
            "ask_mean_markout_conditional",
            "bid_toxic_rate_lift",
            "ask_toxic_rate_lift",
            "recommendation_note",
        ]
    ].copy()
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                if np.isnan(val):
                    vals.append("NaN")
                else:
                    vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _pick_best_setting(h10: pd.DataFrame) -> tuple[pd.Series | None, str]:
    candidate = h10.copy()
    candidate = candidate[
        (candidate["bid_fill_count"] >= 100)
        & (candidate["ask_fill_count"] >= 100)
        & (candidate["bid_fill_rate"] >= 0.001)
        & (candidate["ask_fill_rate"] >= 0.001)
    ].copy()
    if candidate.empty:
        return None, "No candidate has sufficient fill_count/fill_rate."

    candidate["min_toxic"] = np.minimum(candidate["bid_toxic_rate_conditional"], candidate["ask_toxic_rate_conditional"])
    candidate["max_toxic"] = np.maximum(candidate["bid_toxic_rate_conditional"], candidate["ask_toxic_rate_conditional"])
    candidate = candidate[(candidate["max_toxic"] <= 0.80)].copy()
    if candidate.empty:
        return None, "All candidates are either sparse or have extreme toxicity >80%."

    candidate["proxy_priority"] = np.where(candidate["proxy_type"] == "price_level_removed_proxy", 1, 0)
    candidate["lift_score"] = np.nan_to_num(candidate["bid_toxic_rate_lift"], nan=0.0) + np.nan_to_num(candidate["ask_toxic_rate_lift"], nan=0.0)
    candidate["toxic_score"] = candidate["min_toxic"]
    candidate = candidate.sort_values(
        by=["proxy_priority", "toxic_score", "lift_score", "bid_fill_count", "ask_fill_count"],
        ascending=[False, False, False, False, False],
    )
    best = candidate.iloc[0]
    reason = (
        f"Selected with proxy_priority={int(best['proxy_priority'])}, "
        f"min_toxic={best['min_toxic']:.4f}, lift_score={best['lift_score']:.4f}, "
        f"fill_counts=({int(best['bid_fill_count'])},{int(best['ask_fill_count'])})."
    )
    return best, reason


def write_fill_conditional_report(
    out_dir: Path,
    conditional_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    h10_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# 1. Executive Summary")
    lines.append("")
    lines.append(
        "Original unconditional markout toxic rates were too low because they were not fill-conditional; "
        "this audit uses LOB-based fill proxies to estimate conditional toxicity."
    )
    lines.append("")
    lines.append("# 2. Fill Evidence Audit")
    lines.append("")
    lines.append("- Trade tape exists.")
    lines.append("- aggressor_side is missing.")
    lines.append("- true exchange-level fill direction is unavailable.")
    lines.append("- LOB 5-level depth is available.")
    lines.append("- all labels in this report are proxy-based, not true fills.")
    lines.append("")
    lines.append("# 3. Proxy Definitions and Limitations")
    lines.append("")
    lines.append("## queue_depletion_proxy")
    lines.append("- Advantage: uses volume changes and reflects queue dynamics.")
    lines.append("- Limitation: volume drop can be fill or cancel.")
    lines.append("")
    lines.append("## price_level_removed_proxy")
    lines.append("- Advantage: simple and stable from LOB only.")
    lines.append("- Limitation: removed level can be fill or cancel.")
    lines.append("- Note: treated as primary candidate proxy, not true exchange fill.")
    lines.append("")
    lines.append("## price_through_proxy")
    lines.append("- Advantage: conservative and less likely to over-call fills.")
    lines.append("- Limitation: can severely under-count fills (rare proxy).")
    lines.append("")

    lines.append("# 4. Conditional Toxic Rate Results")
    lines.append("")
    h10_show = h10_df[
        [
            "dataset",
            "split",
            "proxy_type",
            "fill_window",
            "bid_fill_count",
            "bid_toxic_rate_conditional",
            "ask_fill_count",
            "ask_toxic_rate_conditional",
            "bid_toxic_rate_lift",
            "ask_toxic_rate_lift",
        ]
    ].copy()
    lines.append(_markdown_table(h10_show))
    lines.append("")

    lines.append("# 5. NEW vs NEW3 Comparison")
    lines.append("")
    new3_h10 = h10_df[(h10_df["dataset"] == "NEW3")].copy()
    new_h10 = h10_df[(h10_df["dataset"] == "NEW")].copy()
    if not new3_h10.empty and not new_h10.empty:
        bid_new3 = float(new3_h10["bid_toxic_rate_conditional"].mean())
        ask_new3 = float(new3_h10["ask_toxic_rate_conditional"].mean())
        bid_new = float(new_h10["bid_toxic_rate_conditional"].mean())
        ask_new = float(new_h10["ask_toxic_rate_conditional"].mean())
        lines.append(f"- Mean H=10 conditional bid_toxic_rate: NEW={bid_new:.4f}, NEW3={bid_new3:.4f}")
        lines.append(f"- Mean H=10 conditional ask_toxic_rate: NEW={ask_new:.4f}, NEW3={ask_new3:.4f}")
        if bid_new3 > bid_new:
            lines.append("- NEW3 shows higher bid-side toxicity than NEW in this proxy audit.")
        else:
            lines.append("- NEW3 does not show consistently higher bid-side toxicity than NEW across all proxies.")
        if ask_new3 > ask_new:
            lines.append("- NEW3 also shows elevated ask-side toxicity in this proxy audit.")
        else:
            lines.append("- NEW3 ask-side toxicity is not consistently above NEW.")
    lines.append("- These comparisons are proxy-level evidence and should be interpreted with missing aggressor-side limitations.")
    lines.append("")

    lines.append("# 6. Best Proxy / Horizon Recommendation")
    lines.append("")
    best, reason = _pick_best_setting(h10_df)
    if best is None:
        lines.append("- No robust candidate passed minimum fill-quality checks.")
        lines.append(f"- Reason: {reason}")
    else:
        lines.append(
            f"- Selected: proxy_type={best['proxy_type']}, fill_window={int(best['fill_window'])}, horizon=10"
        )
        lines.append(
            f"- fill_count(bid/ask)=({int(best['bid_fill_count'])}/{int(best['ask_fill_count'])}), "
            f"toxic_rate(bid/ask)=({best['bid_toxic_rate_conditional']:.4f}/{best['ask_toxic_rate_conditional']:.4f})"
        )
        lines.append(f"- Reason: {reason}")
        lines.append("- Limitation: still proxy-based, not true exchange-level fill labels.")
    lines.append("")

    lines.append("# 7. Label Recommendation")
    lines.append("")
    all_toxic = pd.concat(
        [
            conditional_df["bid_toxic_rate_conditional"].dropna(),
            conditional_df["ask_toxic_rate_conditional"].dropna(),
        ],
        ignore_index=True,
    )
    max_toxic = float(all_toxic.max()) if not all_toxic.empty else np.nan
    has_any_gt2 = bool((all_toxic > 0.02).any()) if not all_toxic.empty else False
    low_sample_heavy = bool(
        conditional_df["notes"].fillna("").str.contains("LOW_SAMPLE").mean() > 0.4
    )

    if has_any_gt2 and not low_sample_heavy:
        choice = "A. Use fill-conditional bid_toxic / ask_toxic labels for next model."
        reason = (
            "Proxy-conditional toxicity is meaningfully lifted vs unconditional and sample counts are usable. "
            "Still keep explicit proxy uncertainty in evaluation."
        )
    elif not has_any_gt2:
        choice = "D. Data is insufficient for reliable fill-conditional labels."
        reason = (
            "All proxy-based toxic rates remain below 2%, so current LOB proxies still do not capture fills well enough."
        )
    else:
        choice = "B. Use side-adjusted markout regression instead of classification."
        reason = (
            "Conditional labels are noisy/sparse; continuous markout targets may retain more signal under proxy uncertainty."
        )

    lines.append(f"- Selected: {choice}")
    lines.append(f"- Why: {reason}")
    if not has_any_gt2:
        lines.append(
            "- current LOB-based proxies still do not capture actual fills well; "
            "need trade tape with aggressor side or order-level queue reconstruction."
        )
    lines.append("")

    lines.append("# 8. Next Coding Task")
    lines.append("")
    if choice.startswith("A"):
        lines.append("- Build bid-side and ask-side binary datasets separately.")
        lines.append("- Fit/select on train+val only, then test once.")
        lines.append("- Keep model architecture small and compare against current TCN risk score.")
    elif choice.startswith("B"):
        lines.append("- Build regression targets: bid_markout_ticks and ask_markout_ticks.")
        lines.append("- Use robust loss (Huber/quantile) and evaluate with net_cancel_value.")
    else:
        lines.append("- Prioritize data improvements:")
        lines.append("  - aggressor_side in trade tape")
        lines.append("  - order-level queue reconstruction")
        lines.append("  - tighter trade/LOB timestamp alignment")

    (out_dir / "fill_conditional_side_label_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _common.build_cfg()
    new_cfg = replace(
        cfg,
        data=replace(cfg.data, data_dir=NEW_DATA_DIR, symbols=(NEW_SYMBOL,)),
        backtest=replace(cfg.backtest, symbol=NEW_SYMBOL),
    )
    new3_cfg = replace(
        cfg,
        data=replace(cfg.data, data_dir=NEW3_DATA_DIR, symbols=(NEW3_SYMBOL,)),
        backtest=replace(cfg.backtest, symbol=NEW3_SYMBOL),
    )

    new_lob_frames, new_trade_frames = load_new_tw_data(new_cfg.data.data_dir, [NEW_SYMBOL])
    new3_lob_frames, new3_trade_frames = load_new_tw_data(new3_cfg.data.data_dir, [NEW3_SYMBOL])
    if NEW_SYMBOL not in new_lob_frames or NEW_SYMBOL not in new_trade_frames:
        raise RuntimeError(f"Missing NEW symbol data: {NEW_SYMBOL}")
    if NEW3_SYMBOL not in new_lob_frames and NEW3_SYMBOL not in new3_lob_frames:
        pass
    if NEW3_SYMBOL not in new3_lob_frames or NEW3_SYMBOL not in new3_trade_frames:
        raise RuntimeError(f"Missing NEW3 symbol data: {NEW3_SYMBOL}")

    new_lob = new_lob_frames[NEW_SYMBOL].copy()
    new_trade = new_trade_frames[NEW_SYMBOL].copy()
    new3_lob = new3_lob_frames[NEW3_SYMBOL].copy()
    new3_trade = new3_trade_frames[NEW3_SYMBOL].copy()

    write_fill_evidence_audit(
        OUT_DIR,
        new_lob=new_lob,
        new_trade=new_trade,
        new3_lob=new3_lob,
        new3_trade=new3_trade,
    )

    tick_new = float(infer_tick_size_from_lob(new_lob))
    tick_new3 = float(infer_tick_size_from_lob(new3_lob))

    new_sessions = sorted(new_lob["session_id"].unique().tolist())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(new_sessions, train_frac=0.70, val_frac=0.15)
    new_tv_frame = _concat_split_frames(new_lob, train_sessions + val_sessions)
    new_test_frame = _concat_split_frames(new_lob, test_sessions)
    new3_full_frame = new3_lob.copy()

    distributions = []
    distributions.append(
        compute_fill_conditional_distribution(
            dataset="NEW",
            split="NEW_train_val",
            frame=new_tv_frame,
            tick_size=tick_new,
            sample_step=SAMPLE_STEP,
        )
    )
    distributions.append(
        compute_fill_conditional_distribution(
            dataset="NEW",
            split="NEW_test",
            frame=new_test_frame,
            tick_size=tick_new,
            sample_step=SAMPLE_STEP,
        )
    )
    distributions.append(
        compute_fill_conditional_distribution(
            dataset="NEW3",
            split="NEW3_full",
            frame=new3_full_frame,
            tick_size=tick_new3,
            sample_step=SAMPLE_STEP,
        )
    )
    conditional_df = pd.concat(distributions, ignore_index=True)
    conditional_path = OUT_DIR / "side_label_distribution_fill_conditional.csv"
    conditional_df.to_csv(conditional_path, index=False)

    uncond_path = OUT_DIR / "side_label_distribution.csv"
    compare_df = _build_unconditional_vs_conditional(uncond_path, conditional_df)
    compare_path = OUT_DIR / "side_label_unconditional_vs_fill_conditional.csv"
    compare_df.to_csv(compare_path, index=False)

    h10_df = _build_h10_summary(conditional_df, compare_df)
    h10_path = OUT_DIR / "side_label_h10_summary.csv"
    h10_df.to_csv(h10_path, index=False)

    write_fill_conditional_report(OUT_DIR, conditional_df, compare_df, h10_df)

    print("Fill-conditional side toxic label audit completed.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Rows: conditional={len(conditional_df)}, compare={len(compare_df)}, h10={len(h10_df)}")


if __name__ == "__main__":
    main()
