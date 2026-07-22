from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03a5_pre_ml_freeze_audit"
RESULTS = ROOT / "results"

PHASE01_DATA = PROJECT_ROOT / "n2ew_era" / "01_full_l2_action_universe_builder" / "data"
PHASE02E_DATA = PROJECT_ROOT / "n2ew_era" / "02e_final_non_ml_feasibility_decision" / "data"
PHASE02F = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy"
EXCESS_DATA = PHASE02F / "data" / "full_action_universe_with_excess_markout.parquet"
PHASE03A = PROJECT_ROOT / "n2ew_era" / "03a_ml_readiness_excess_markout"

SPLITS = ["train", "val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
HORIZONS = [1, 5, 10, 20, 50, 100]
POINT_VALUE = 200.0
TOLERANCE = 1e-9


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def raw_split_dir(split: str) -> Path:
    if split == "new3":
        return PHASE02E_DATA / "full_l2_action_universe_new3.parquet"
    return PHASE01_DATA / f"full_l2_action_universe_{split}.parquet"


def excess_split_dir(split: str) -> Path:
    return EXCESS_DATA / f"split={split}"


def part_pairs(split: str) -> list[tuple[Path, Path]]:
    raw_parts = {p.name: p for p in raw_split_dir(split).glob("*.parquet")}
    excess_parts = {p.name: p for p in excess_split_dir(split).glob("*.parquet")}
    common = sorted(set(raw_parts) & set(excess_parts))
    return [(raw_parts[name], excess_parts[name]) for name in common]


def audit_excess_formula() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for raw_path, excess_path in part_pairs(split):
            session_id = raw_path.stem.replace("part-", "")
            raw_cols = ["action_slot", "mid_price", "best_bid", "best_ask", *[f"pnl_H{h}" for h in HORIZONS]]
            ex_cols = ["action_slot", *[f"excess_bid_pnl_H{h}" for h in HORIZONS], *[f"excess_ask_pnl_H{h}" for h in HORIZONS]]
            raw = pd.read_parquet(raw_path, columns=raw_cols)
            excess = pd.read_parquet(excess_path, columns=ex_cols)
            if len(raw) != len(excess):
                raise RuntimeError(f"Row count mismatch: {raw_path} vs {excess_path}")
            bid_mask = raw["action_slot"].astype(int).to_numpy() == 0
            ask_mask = raw["action_slot"].astype(int).to_numpy() == 1
            mid = raw["mid_price"].astype(float).to_numpy()
            best_bid = raw["best_bid"].astype(float).to_numpy()
            best_ask = raw["best_ask"].astype(float).to_numpy()
            half_bid = (mid - best_bid) * POINT_VALUE
            half_ask = (best_ask - mid) * POINT_VALUE
            for horizon in HORIZONS:
                raw_pnl = raw[f"pnl_H{horizon}"].astype(float).to_numpy()
                bid_recon = half_bid + excess[f"excess_bid_pnl_H{horizon}"].astype(float).to_numpy()
                ask_recon = half_ask + excess[f"excess_ask_pnl_H{horizon}"].astype(float).to_numpy()
                bid_err = np.abs(raw_pnl[bid_mask] - bid_recon[bid_mask])
                ask_err = np.abs(raw_pnl[ask_mask] - ask_recon[ask_mask])
                rows.append(
                    {
                        "split": split,
                        "session_id": session_id,
                        "horizon": int(horizon),
                        "bid_rows": int(bid_mask.sum()),
                        "ask_rows": int(ask_mask.sum()),
                        "max_abs_bid_reconstruction_error": float(np.nanmax(bid_err)) if len(bid_err) else np.nan,
                        "max_abs_ask_reconstruction_error": float(np.nanmax(ask_err)) if len(ask_err) else np.nan,
                        "mean_abs_bid_reconstruction_error": float(np.nanmean(bid_err)) if len(bid_err) else np.nan,
                        "mean_abs_ask_reconstruction_error": float(np.nanmean(ask_err)) if len(ask_err) else np.nan,
                        "tolerance": TOLERANCE,
                        "formula_pass": bool(
                            (len(bid_err) == 0 or np.nanmax(bid_err) <= TOLERANCE)
                            and (len(ask_err) == 0 or np.nanmax(ask_err) <= TOLERANCE)
                        ),
                    }
                )
        print(f"formula audit complete split={split}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "excess_formula_audit.csv", index=False)
    lines = [
        "# Excess Label Formula Audit",
        "",
        "Hard rule: for every horizon, raw bid/ask markout must reconstruct exactly from half-spread plus excess markout.",
        "",
        "```text",
        "raw_bid_pnl_H = (mid_t - best_bid) * point_value + excess_bid_pnl_H",
        "raw_ask_pnl_H = (best_ask - mid_t) * point_value + excess_ask_pnl_H",
        "```",
        "",
        f"Numerical tolerance: `{TOLERANCE}`.",
        "",
        out.groupby("horizon", as_index=False).agg(
            max_abs_bid_reconstruction_error=("max_abs_bid_reconstruction_error", "max"),
            max_abs_ask_reconstruction_error=("max_abs_ask_reconstruction_error", "max"),
            formula_pass=("formula_pass", "all"),
        ).to_markdown(index=False),
    ]
    write_markdown(ROOT / "EXCESS_LABEL_FORMULA_AUDIT.md", lines)
    return out


def schema_columns() -> list[str]:
    for split in SPLITS:
        paths = sorted(excess_split_dir(split).glob("*.parquet"))
        if paths:
            return pd.read_parquet(paths[0]).columns.tolist()
    return []


def feature_causality_audit() -> pd.DataFrame:
    columns = schema_columns()
    allowed_feature_names = {
        "action_side",
        "action_slot",
        "quote_price",
        "best_bid",
        "best_ask",
        "mid_price",
        "tick_size",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "volatility_past_50",
    }
    identity_cols = {"split", "session_id", "symbol", "event_index", "timestamp"}
    rows = []
    for col in columns:
        is_future = (
            col.startswith("future_mid_H")
            or col.startswith("excess_pnl_H")
            or col.startswith("excess_bid_pnl_H")
            or col.startswith("excess_ask_pnl_H")
            or col.startswith("positive_excess_pnl_H")
            or col.startswith("negative_excess_pnl_H")
            or col.startswith("pnl_H")
            or col.startswith("positive_pnl_H")
            or col.startswith("negative_pnl_H")
        )
        allowed = col in allowed_feature_names
        if col in identity_cols:
            category = "identity_not_model_feature"
        elif is_future:
            category = "future_or_target_label"
        elif allowed:
            category = "allowed_causal_feature"
        else:
            category = "not_in_frozen_feature_schema"
        rows.append(
            {
                "column": col,
                "category": category,
                "uses_future_data": bool(is_future),
                "allowed_as_feature": bool(allowed),
                "hard_rule_pass": bool(not (allowed and is_future)),
                "notes": "current/past-only by construction" if allowed else "",
            }
        )
    requested_missing = ["event_rate_past_50", "spread_change_past", "imbalance_change_past"]
    for col in requested_missing:
        if col not in columns:
            rows.append(
                {
                    "column": col,
                    "category": "requested_but_missing",
                    "uses_future_data": False,
                    "allowed_as_feature": False,
                    "hard_rule_pass": True,
                    "notes": "Missing from Phase 02F parquet; do not use until rebuilt causally.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "feature_causality_audit.csv", index=False)
    violations = out[(out["allowed_as_feature"].astype(bool)) & (out["uses_future_data"].astype(bool))]
    lines = [
        "# Feature Causality Audit",
        "",
        "Hard rule: no feature with `uses_future_data = true` may be allowed for ML.",
        "",
        f"Violation count: `{len(violations)}`.",
        "",
        "Allowed features are current-row or past-only causal fields. Identity columns are not future-leaking but are not model features.",
        "",
        out.to_markdown(index=False),
    ]
    write_markdown(ROOT / "FEATURE_CAUSALITY_AUDIT.md", lines)
    return out


def hash_frame(frame: pd.DataFrame, cols: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def split_duplicate_audit() -> pd.DataFrame:
    session_rows = []
    for split in SPLITS:
        for path in sorted(excess_split_dir(split).glob("*.parquet")):
            cols = ["session_id", "event_index", "timestamp", "action_slot", "best_bid", "best_ask", "mid_price"]
            df = pd.read_parquet(path, columns=cols)
            bid = df[df["action_slot"].astype(int) == 0].sort_values("event_index").reset_index(drop=True)
            session_id = str(bid["session_id"].iloc[0]) if len(bid) else path.stem.replace("part-", "")
            session_rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "row_count": int(len(bid)),
                    "timestamp_min": int(bid["timestamp"].min()) if len(bid) else np.nan,
                    "timestamp_max": int(bid["timestamp"].max()) if len(bid) else np.nan,
                    "timestamp_sequence_hash": hash_frame(bid, ["timestamp"]),
                    "price_sequence_hash": hash_frame(bid, ["best_bid", "best_ask", "mid_price"]),
                    "event_index_sequence_hash": hash_frame(bid, ["event_index"]),
                }
            )
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(RESULTS / "split_session_signatures.csv", index=False)

    rows = []
    for a, b in combinations(SPLITS, 2):
        left = sessions[sessions["split"] == a]
        right = sessions[sessions["split"] == b]
        session_overlap = set(left["session_id"].astype(str)) & set(right["session_id"].astype(str))
        timestamp_overlap = set(left["timestamp_sequence_hash"].astype(str)) & set(right["timestamp_sequence_hash"].astype(str))
        price_overlap = set(left["price_sequence_hash"].astype(str)) & set(right["price_sequence_hash"].astype(str))
        rows.append(
            {
                "split_a": a,
                "split_b": b,
                "session_id_overlap_count": int(len(session_overlap)),
                "timestamp_sequence_hash_overlap_count": int(len(timestamp_overlap)),
                "price_sequence_hash_overlap_count": int(len(price_overlap)),
                "session_id_overlap_values": ";".join(sorted(session_overlap)),
                "independence_pass": bool(len(session_overlap) == 0 and len(timestamp_overlap) == 0 and len(price_overlap) == 0),
                "classification": "independent" if len(session_overlap) == 0 and len(timestamp_overlap) == 0 and len(price_overlap) == 0 else "possible_duplicate_or_overlap",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "split_duplicate_audit.csv", index=False)
    lines = [
        "# Split Duplicate Audit",
        "",
        "Hard rule: train, val, test, NEW3, and NEW4 must not overlap by session_id, timestamp sequence hash, or price sequence hash.",
        "",
        out.to_markdown(index=False),
        "",
        "Session-level signatures are saved to `results/split_session_signatures.csv`.",
    ]
    write_markdown(ROOT / "SPLIT_DUPLICATE_AUDIT.md", lines)
    return out


def schema_freeze(causality: pd.DataFrame) -> tuple[list[str], list[str]]:
    allowed = causality[causality["allowed_as_feature"].astype(bool)]["column"].astype(str).tolist()
    forbidden = causality[
        causality["uses_future_data"].astype(bool)
        | causality["category"].astype(str).isin(["identity_not_model_feature", "requested_but_missing"])
    ]["column"].astype(str).tolist()
    extra_forbidden_patterns = [
        "raw_pnl_H*",
        "pnl_H*",
        "future_mid_H*",
        "excess_pnl_H*",
        "excess_bid_pnl_H*",
        "excess_ask_pnl_H*",
        "positive_excess_pnl_H*",
        "negative_excess_pnl_H*",
        "target_*",
    ]
    write_json(ROOT / "allowed_features.json", {"allowed_features": allowed, "selection_split": "train/val only", "test_new3_new4_used_for_feature_selection": False})
    write_json(ROOT / "forbidden_features.json", {"forbidden_columns": forbidden, "forbidden_patterns": extra_forbidden_patterns})
    lines = [
        "# Feature Schema For ML",
        "",
        "This file freezes the pre-ML schema. Future labels and targets are forbidden as features.",
        "",
        "## Allowed Features",
        "",
        pd.DataFrame({"allowed_feature": allowed}).to_markdown(index=False),
        "",
        "## Forbidden Columns / Patterns",
        "",
        pd.DataFrame({"forbidden_column_or_pattern": [*forbidden, *extra_forbidden_patterns]}).to_markdown(index=False),
        "",
        "Hard rule: no feature in `allowed_features.json` may have `uses_future_data = true` in `results/feature_causality_audit.csv`.",
    ]
    write_markdown(ROOT / "FEATURE_SCHEMA_FOR_ML.md", lines)
    return allowed, [*forbidden, *extra_forbidden_patterns]


def baseline_hurdles() -> pd.DataFrame:
    eval_path = PHASE02F / "results" / "excess_policy_evaluation.csv"
    seq_path = PHASE02F / "results" / "sequential_excess_inventory_backtest.csv"
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    seq_df = pd.read_csv(seq_path) if seq_path.exists() else pd.DataFrame()
    rows = []
    if not eval_df.empty:
        h20 = eval_df[eval_df["horizon"].astype(int) == 20]
        for split in ["val", *FROZEN_SPLITS]:
            for policy_id in ["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"]:
                r = h20[(h20["split"].astype(str) == split) & (h20["policy_id"].astype(str) == policy_id)]
                if not r.empty:
                    rows.append(
                        {
                            "split": split,
                            "hurdle_name": policy_id,
                            "hurdle_type": "excess_markout_H20",
                            "metric": "total_excess_pnl",
                            "hurdle_value": float(r["total_excess_pnl"].iloc[0]),
                            "ml_requirement": "ML policy total_excess_pnl must exceed this value on the same split after validation-only selection.",
                            "selection_allowed_on_split": bool(split == "val"),
                        }
                    )
            for base_policy in ["old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"]:
                r = h20[
                    (h20["split"].astype(str) == split)
                    & (h20["policy_id"].astype(str) == base_policy)
                ]
                if not r.empty and "matched_random_95p" in r:
                    rows.append(
                        {
                            "split": split,
                            "hurdle_name": f"matched_random_95p_for_{base_policy}",
                            "hurdle_type": "matched_random_excess_markout_H20",
                            "metric": "matched_random_95p",
                            "hurdle_value": float(r["matched_random_95p"].iloc[0]),
                            "ml_requirement": "ML policy must beat the matched-random 95p hurdle under the same matched universe.",
                            "selection_allowed_on_split": bool(split == "val"),
                        }
                    )
    if not seq_df.empty:
        focus = seq_df[
            (seq_df["fill_model"].astype(str) == "conservative_l2_stress")
            & (seq_df["inventory_cap"].astype(int) == 1)
            & (seq_df["fee_multiplier"].astype(float) == 1.0)
        ]
        for row in focus.itertuples(index=False):
            rows.append(
                {
                    "split": str(row.split),
                    "hurdle_name": str(row.policy_id),
                    "hurdle_type": "sequential_excess_causal_fill_cap1_fee1_H20",
                    "metric": "sequential_excess_pnl",
                    "hurdle_value": float(row.sequential_excess_pnl),
                    "ml_requirement": "ML policy must remain positive and exceed frozen baselines after validation-only selection.",
                    "selection_allowed_on_split": bool(str(row.split) == "val"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "frozen_baseline_hurdles.csv", index=False)
    lines = [
        "# Baseline Hurdle Contract",
        "",
        "Any Phase 03B ML model must be selected only on validation and must be evaluated frozen on test, NEW3, and NEW4.",
        "",
        "ML must be compared against:",
        "",
        "- `no_quote`",
        "- `old_bid_only_spread_ge_1` under excess markout",
        "- `side_by_queue_imbalance_abs_0p1` under excess markout",
        "- matched random 95p under side/session/spread/count matching",
        "- sequential excess proxy with `conservative_l2_stress`, `inventory_cap=1`, `fee_multiplier=1`",
        "",
        out.to_markdown(index=False) if not out.empty else "Baseline hurdles are missing.",
    ]
    write_markdown(ROOT / "BASELINE_HURDLE_CONTRACT.md", lines)
    return out


def evaluation_contract() -> None:
    lines = [
        "# ML Evaluation Contract",
        "",
        "Primary targets:",
        "",
        "- Regression: `excess_pnl_H20`",
        "- Binary classification: `positive_excess_H20 = excess_pnl_H20 > 0`",
        "",
        "Validation split is strictly for model family selection, hyperparameter selection, calibration, and threshold/action-policy selection.",
        "",
        "Test, NEW3, and NEW4 are strictly frozen out-of-sample. They must not be used for feature selection, target choice, threshold selection, model selection, hyperparameter tuning, or early stopping.",
        "",
        "Future labels are forbidden as model inputs. Raw PnL is not the main target because raw markout contains mechanical half-spread capture.",
        "",
        "Required frozen evaluations:",
        "",
        "- Excess markout H20 PnL",
        "- Positive excess classification metrics",
        "- Matched random 95p comparison",
        "- Sequential excess inventory proxy with causal fill",
        "- Split-by-side and split-by-session reporting",
    ]
    write_markdown(ROOT / "ML_EVALUATION_CONTRACT.md", lines)


def final_report(formula: pd.DataFrame, causality: pd.DataFrame, dup: pd.DataFrame, allowed: list[str], forbidden: list[str], hurdles: pd.DataFrame) -> None:
    formula_pass = bool(formula["formula_pass"].astype(bool).all()) if not formula.empty else False
    causality_pass = bool((~(causality["allowed_as_feature"].astype(bool) & causality["uses_future_data"].astype(bool))).all()) if not causality.empty else False
    duplicate_pass = bool(dup["independence_pass"].astype(bool).all()) if not dup.empty else False
    schema_pass = bool(len(allowed) > 0 and any("future_mid_H*" == f for f in forbidden) and any("excess_pnl_H*" == f for f in forbidden))
    baseline_pass = bool(not hurdles.empty)
    eval_contract_pass = (ROOT / "ML_EVALUATION_CONTRACT.md").exists()
    hard_gates = pd.DataFrame(
        [
            {"gate": "excess_label_formula", "pass": formula_pass},
            {"gate": "feature_causality", "pass": causality_pass},
            {"gate": "split_duplicate_independence", "pass": duplicate_pass},
            {"gate": "feature_schema_freeze", "pass": schema_pass},
            {"gate": "baseline_hurdle_freeze", "pass": baseline_pass},
            {"gate": "evaluation_contract_freeze", "pass": eval_contract_pass},
        ]
    )
    hard_gates.to_csv(RESULTS / "pre_ml_freeze_hard_gates.csv", index=False)
    missing_requested = causality[causality["category"].astype(str) == "requested_but_missing"] if not causality.empty else pd.DataFrame()
    if not hard_gates["pass"].astype(bool).all():
        status = "freeze_fail_fix_required"
    elif not missing_requested.empty:
        status = "freeze_pass_with_minor_caveats"
    else:
        status = "freeze_pass_ready_for_phase03b"
    lines = [
        "# Pre-ML Freeze Audit Report",
        "",
        f"Final freeze status: `{status}`.",
        "",
        "No ML was trained. No strategy or threshold was tuned. This phase freezes labels, features, splits, baselines, and evaluation rules.",
        "",
        "## Hard Gates",
        "",
        hard_gates.to_markdown(index=False),
        "",
        "## Gate 1: Excess Label Formula",
        "",
        formula.groupby("horizon", as_index=False).agg(
            max_abs_bid_reconstruction_error=("max_abs_bid_reconstruction_error", "max"),
            max_abs_ask_reconstruction_error=("max_abs_ask_reconstruction_error", "max"),
            formula_pass=("formula_pass", "all"),
        ).to_markdown(index=False) if not formula.empty else "Formula audit missing.",
        "",
        "## Gate 2: Feature Causality",
        "",
        f"Allowed feature count: `{len(allowed)}`. Future-leaking allowed feature violations: `{int((causality['allowed_as_feature'].astype(bool) & causality['uses_future_data'].astype(bool)).sum()) if not causality.empty else 'missing'}`.",
        "",
        "## Gate 3: Split / Duplicate Independence",
        "",
        dup.to_markdown(index=False) if not dup.empty else "Duplicate audit missing.",
        "",
        "## Gate 4: ML Feature Schema",
        "",
        "Schema frozen in `allowed_features.json` and `forbidden_features.json`.",
        "",
        "Minor caveat: requested features missing from Phase 02F parquet are not allowed until rebuilt causally:",
        "",
        missing_requested[["column", "notes"]].to_markdown(index=False) if not missing_requested.empty else "No requested causal features are missing.",
        "",
        "## Gate 5: Baseline Hurdles",
        "",
        "Frozen hurdles are saved to `results/frozen_baseline_hurdles.csv` and described in `BASELINE_HURDLE_CONTRACT.md`.",
        "",
        "## Gate 6: Evaluation Contract",
        "",
        "`ML_EVALUATION_CONTRACT.md` freezes primary targets, validation-only selection, and frozen test/NEW3/NEW4 evaluation.",
        "",
        "## Final Decision",
        "",
        f"`{status}`.",
    ]
    write_markdown(ROOT / "PRE_ML_FREEZE_AUDIT_REPORT.md", lines)
    print(f"Final freeze status: {status}")
    print(hard_gates.to_string(index=False))


def main() -> None:
    ensure_dirs()
    formula = audit_excess_formula()
    causality = feature_causality_audit()
    dup = split_duplicate_audit()
    allowed, forbidden = schema_freeze(causality)
    hurdles = baseline_hurdles()
    evaluation_contract()
    final_report(formula, causality, dup, allowed, forbidden, hurdles)


if __name__ == "__main__":
    main()

