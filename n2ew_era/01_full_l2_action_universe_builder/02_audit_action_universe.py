from __future__ import annotations

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from common_phase01 import (
    ACTION_COLUMNS,
    CAUSAL_FEATURE_COLUMNS,
    FEATURE_DRIFT_COLUMNS,
    FULL_DATASET_DIR,
    FUTURE_LABEL_COLUMNS,
    HORIZONS,
    IDENTITY_COLUMNS,
    RESULTS_DIR,
    ensure_dirs,
    split_dataset_dir,
)


def source_count_audit() -> pd.DataFrame:
    build_counts = pd.read_csv(RESULTS_DIR / "build_action_counts_by_split.csv")
    split_report = pd.read_csv(RESULTS_DIR / "source_filter_counts_by_split.csv")
    rows = []
    for r in split_report.itertuples(index=False):
        raw = int(r.raw_rows)
        valid_book = int(r.valid_book_rows)
        eligible = int(r.quote_eligible_rows)
        b = build_counts[build_counts["split"].astype(str) == str(r.split)].iloc[0]
        action_rows = int(b.action_rows)
        rows.append(
            {
                "split": str(r.split),
                "dataset": str(r.dataset),
                "symbol": str(r.symbol),
                "source_path": str(r.source_path),
                "raw_rows": raw,
                "valid_book_rows": valid_book,
                "quote_eligible_rows": eligible,
                "candidate_action_rows": action_rows,
                "expected_candidate_actions_quote_eligible_x3": eligible * 3,
                "candidate_action_match": action_rows == eligible * 3,
                "bid_actions": int(b.bid_actions),
                "ask_actions": int(b.ask_actions),
                "no_quote_actions": int(b.no_quote_actions),
                "session_count": int(b.session_count),
                "valid_book_retention_from_raw": valid_book / raw if raw else 0.0,
                "quote_eligible_retention_from_raw": eligible / raw if raw else 0.0,
                "action_multiplier_vs_quote_eligible": action_rows / eligible if eligible else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
    out.to_csv(RESULTS_DIR / "action_universe_by_split.csv", index=False)
    return out


def label_audit() -> pd.DataFrame:
    rows = []
    split_counts = pd.read_csv(RESULTS_DIR / "build_action_counts_by_split.csv")
    for split in split_counts["split"].astype(str).tolist():
        path = split_dataset_dir(split)
        for horizon in HORIZONS:
            cols = ["action_side", f"pnl_H{horizon}", f"markout_ticks_H{horizon}", f"negative_pnl_H{horizon}", f"severe_negative_pnl_H{horizon}"]
            df = pd.read_parquet(path, columns=cols)
            for side, group in df.groupby("action_side", observed=False):
                pnl = group[f"pnl_H{horizon}"]
                markout = group[f"markout_ticks_H{horizon}"]
                rows.append(
                    {
                        "split": split,
                        "action_side": str(side),
                        "horizon": horizon,
                        "row_count": int(len(group)),
                        "pnl_count": int(pnl.notna().sum()),
                        "pnl_mean": float(pnl.mean(skipna=True)) if pnl.notna().any() else None,
                        "pnl_median": float(pnl.median(skipna=True)) if pnl.notna().any() else None,
                        "pnl_positive_rate": float((pnl > 0.0).mean()) if len(group) else None,
                        "pnl_negative_rate": float(group[f"negative_pnl_H{horizon}"].astype(bool).mean()) if len(group) else None,
                        "pnl_severe_negative_rate": float(group[f"severe_negative_pnl_H{horizon}"].astype(bool).mean()) if len(group) else None,
                        "markout_count": int(markout.notna().sum()),
                        "markout_ticks_mean": float(markout.mean(skipna=True)) if markout.notna().any() else None,
                        "markout_ticks_median": float(markout.median(skipna=True)) if markout.notna().any() else None,
                        "markout_ticks_positive_rate": float((markout > 0.0).mean()) if len(group) else None,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sequential_proxy_label_summary.csv", index=False)
    return out


def classify_column(col: str) -> dict[str, str]:
    future_prefixes = ("future_mid_H", "pnl_H", "markout_ticks_H", "positive_pnl_H", "negative_pnl_H", "severe_negative_pnl_H")
    if col.startswith(future_prefixes):
        return {
            "category": "future_outcome_label",
            "causal_or_future": "future_label",
            "allowed_as_feature": "no",
            "audit_status": "pass",
            "reason": "Future outcome label; prohibited as causal feature.",
        }
    if col in ACTION_COLUMNS or col in {"action_side", "quote_price"}:
        return {
            "category": "action_context",
            "causal_or_future": "causal",
            "allowed_as_feature": "yes",
            "audit_status": "pass",
            "reason": "Known at quote decision time as the candidate action context.",
        }
    if col in IDENTITY_COLUMNS or col in {"timestamp"}:
        return {
            "category": "identity",
            "causal_or_future": "causal_metadata",
            "allowed_as_feature": "no",
            "audit_status": "pass",
            "reason": "Identifier or ordering metadata; not a model feature.",
        }
    if col in CAUSAL_FEATURE_COLUMNS or col in {"mid_price", "tick_size", "quote_eligible"}:
        return {
            "category": "causal_feature",
            "causal_or_future": "causal",
            "allowed_as_feature": "yes",
            "audit_status": "pass",
            "reason": "Current or past-only L2/book feature.",
        }
    return {
        "category": "unknown_review_required",
        "causal_or_future": "unknown",
        "allowed_as_feature": "no",
        "audit_status": "review",
        "reason": "Column not recognized by Phase 01 contract; keep out of feature sets until reviewed.",
    }


def causality_audit() -> pd.DataFrame:
    sample_part = next(FULL_DATASET_DIR.glob("split=*/session_id=*/*.parquet"))
    schema = pq.read_schema(sample_part)
    rows = []
    for col in schema.names:
        row = {"column_name": col, **classify_column(col)}
        rows.append(row)
    out = pd.DataFrame(rows)
    if (out["causal_or_future"].eq("future_label") & out["allowed_as_feature"].eq("yes")).any():
        raise RuntimeError("Future label column marked as allowed feature.")
    missing_future = sorted(set(FUTURE_LABEL_COLUMNS) - set(out["column_name"]))
    if missing_future:
        raise RuntimeError(f"Missing expected future label columns: {missing_future[:10]}")
    out.to_csv(RESULTS_DIR / "causality_audit.csv", index=False)
    return out


def feature_drift_summary() -> pd.DataFrame:
    split_counts = pd.read_csv(RESULTS_DIR / "build_action_counts_by_split.csv")
    rows = []
    for split in split_counts["split"].astype(str).tolist():
        cols = ["action_slot", *FEATURE_DRIFT_COLUMNS]
        df = pd.read_parquet(split_dataset_dir(split), columns=cols)
        event_rows = df[df["action_slot"].astype(int) == 0].copy()
        for col in FEATURE_DRIFT_COLUMNS:
            values = pd.to_numeric(event_rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                rows.append(
                    {
                        "split": split,
                        "feature": col,
                        "count": 0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "p01": None,
                        "p05": None,
                        "p25": None,
                        "p50": None,
                        "p75": None,
                        "p95": None,
                        "p99": None,
                        "max": None,
                    }
                )
                continue
            quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            rows.append(
                {
                    "split": split,
                    "feature": col,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "p01": float(quantiles.loc[0.01]),
                    "p05": float(quantiles.loc[0.05]),
                    "p25": float(quantiles.loc[0.25]),
                    "p50": float(quantiles.loc[0.50]),
                    "p75": float(quantiles.loc[0.75]),
                    "p95": float(quantiles.loc[0.95]),
                    "p99": float(quantiles.loc[0.99]),
                    "max": float(values.max()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "feature_drift_train_val_test.csv", index=False)
    return out


def main() -> None:
    ensure_dirs()
    count_df = source_count_audit()
    label_df = label_audit()
    causality_df = causality_audit()
    drift_df = feature_drift_summary()
    print("Action universe by split:")
    print(count_df.to_string(index=False))
    print("Label audit rows:", len(label_df))
    print("Feature drift summary rows:", len(drift_df))
    print("Causality audit statuses:")
    print(causality_df["audit_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
