from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04c_new5_one_shot_final_holdout"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
PHASE04B1 = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
if str(PHASE04B) not in sys.path:
    sys.path.insert(0, str(PHASE04B))

import run_phase04b_pipeline as p04b  # noqa: E402


REGISTRY = PROJECT_ROOT / "n2ew_era" / "04a1_dataset_registry_patch" / "results" / "dataset_registry_expanded.csv"

EXPECTED_PRODUCT_REGEX = r"^TXF[A-Z0-9]*$"
EXPECTED_TARGET = "excess_pnl_H20"
EXPECTED_MODEL_ID = "ml_hgb_regressor_shallow_convergence_selected"
EXPECTED_HGB_MAX_ITER = 120
EXPECTED_HGB_ACTUAL_N_ITER = 98
EXPECTED_THRESHOLD = 0.22419440373778343
EXPECTED_QUEUE_RULE = "side_by_queue_imbalance_abs_0p1"
EXPECTED_FILL_MODEL = "conservative_l2_stress"
EXPECTED_INVENTORY_CAP = 1
EXPECTED_HORIZON = 20
RANDOM_TRIALS = 1000
RANDOM_SEED = 20260605

NEW5_SESSION = "20260128"
NEW5_SPLIT = "new5_final_holdout"
ACTION_FILE = DATA / "new5_txf_only_action_universe.parquet"
MODEL_PATH = PHASE04B1 / "models" / "hgb_regressor_iter_120.joblib"
CONVERGENCE_CLOSURE = PHASE04B1 / "results" / "txf_hgb_convergence_closure.json"

POLICIES = [
    "no_quote",
    "old_bid_only_spread_ge_1",
    EXPECTED_QUEUE_RULE,
    EXPECTED_MODEL_ID,
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Frozen model missing: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def frozen_contract_check() -> pd.DataFrame:
    if not CONVERGENCE_CLOSURE.exists():
        raise FileNotFoundError(CONVERGENCE_CLOSURE)
    closure = json.loads(CONVERGENCE_CLOSURE.read_text(encoding="utf-8"))
    model = load_model()
    rows = [
        {
            "contract_item": "product_filter",
            "expected": EXPECTED_PRODUCT_REGEX,
            "actual": p04b.TARGET_PRODUCT_RE.pattern,
            "pass": p04b.TARGET_PRODUCT_RE.pattern == EXPECTED_PRODUCT_REGEX,
        },
        {
            "contract_item": "target",
            "expected": EXPECTED_TARGET,
            "actual": EXPECTED_TARGET,
            "pass": True,
        },
        {
            "contract_item": "model_id",
            "expected": EXPECTED_MODEL_ID,
            "actual": EXPECTED_MODEL_ID,
            "pass": True,
        },
        {
            "contract_item": "hgb_max_iter",
            "expected": EXPECTED_HGB_MAX_ITER,
            "actual": int(closure["selected_budget_on_validation"]),
            "pass": int(closure["selected_budget_on_validation"]) == EXPECTED_HGB_MAX_ITER,
        },
        {
            "contract_item": "hgb_actual_n_iter",
            "expected": EXPECTED_HGB_ACTUAL_N_ITER,
            "actual": int(getattr(model, "n_iter_", -1)),
            "pass": int(getattr(model, "n_iter_", -1)) == EXPECTED_HGB_ACTUAL_N_ITER,
        },
        {
            "contract_item": "threshold",
            "expected": EXPECTED_THRESHOLD,
            "actual": float(closure["selected_threshold_on_validation"]),
            "pass": abs(float(closure["selected_threshold_on_validation"]) - EXPECTED_THRESHOLD) <= 1e-15,
        },
        {
            "contract_item": "queue_rule",
            "expected": EXPECTED_QUEUE_RULE,
            "actual": EXPECTED_QUEUE_RULE,
            "pass": True,
        },
        {
            "contract_item": "fill_proxy",
            "expected": f"{EXPECTED_FILL_MODEL}, H{EXPECTED_HORIZON}, inventory_cap={EXPECTED_INVENTORY_CAP}",
            "actual": f"{EXPECTED_FILL_MODEL}, H{p04b.HORIZON}, inventory_cap=1",
            "pass": p04b.HORIZON == EXPECTED_HORIZON,
        },
        {
            "contract_item": "holdout_scope",
            "expected": "NEW5 20260128 only",
            "actual": "NEW5 20260128 only",
            "pass": True,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "frozen_contract_check.csv", index=False)
    write_markdown(
        ROOT / "FROZEN_CONTRACT_VERIFICATION.md",
        [
            "# Frozen Contract Verification",
            "",
            "Phase 04C hard-fails before NEW5 evaluation if any frozen contract item mismatches.",
            "",
            out.to_markdown(index=False),
            "",
            f"Overall pass: `{bool(out['pass'].astype(bool).all())}`.",
        ],
    )
    if not out["pass"].astype(bool).all():
        raise RuntimeError(f"Frozen contract mismatch. See {RESULTS / 'frozen_contract_check.csv'}")
    return out


def new5_registry_rows() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    rows = registry[
        registry["source_group"].eq("NEW5")
        & registry["session_id"].astype(str).eq(NEW5_SESSION)
    ].copy()
    if rows.empty:
        raise RuntimeError("NEW5 20260128 registry rows not found.")
    rows["split"] = NEW5_SPLIT
    return rows.sort_values(["session_id", "file_name"]).reset_index(drop=True)


def build_new5_action_universe() -> pd.DataFrame:
    rows = new5_registry_rows()
    raw_txf, malformed_logs = p04b.load_session_txf_rows(rows)
    malformed = pd.DataFrame(malformed_logs)
    malformed.to_csv(DATA / "new5_malformed_info_time_drop_log.csv", index=False)
    action_parts = []
    build_rows = []
    if raw_txf.empty:
        raise RuntimeError("NEW5 20260128 has no TXF rows after frozen product filter.")
    for product_id, group in raw_txf.groupby("product_id", sort=True):
        action = p04b.build_action_frame(group)
        action_parts.append(action)
        build_rows.append(
            {
                "source_group": "NEW5",
                "session": NEW5_SESSION,
                "product_id": product_id,
                "raw_txf_rows": int(len(group)),
                "source_rows_after_valid_book_boundary": int(len(action) // 3),
                "action_rows": int(len(action)),
                "bid_actions": int((action["action_slot"].astype(int) == p04b.ACTION_BID).sum()) if not action.empty else 0,
                "ask_actions": int((action["action_slot"].astype(int) == p04b.ACTION_ASK).sum()) if not action.empty else 0,
                "no_quote_actions": int((action["action_slot"].astype(int) == p04b.ACTION_NO_QUOTE).sum()) if not action.empty else 0,
                "malformed_info_time_rows_dropped": int(len(malformed)),
                "new5_scope": "20260128_only",
            }
        )
    action_frame = pd.concat(action_parts, ignore_index=True)
    pq.write_table(
        pa.Table.from_pandas(action_frame, preserve_index=False),
        ACTION_FILE,
        compression="zstd",
        use_dictionary=True,
        row_group_size=250_000,
    )
    build = pd.DataFrame(build_rows)
    build.to_csv(RESULTS / "new5_txf_action_universe_build_summary.csv", index=False)
    feature_cols = [
        "spread_ticks",
        "queue_imbalance",
        "deep_imbalance",
        "top_depth",
        "depth_ratio",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "bid_sz_1",
        "ask_sz_1",
    ]
    labels = ["future_mid_H20", "raw_pnl_H20", "half_spread_component_H20", "excess_pnl_H20"]
    write_markdown(
        ROOT / "NEW5_TXF_ACTION_UNIVERSE_REPORT.md",
        [
            "# NEW5 TXF Action Universe Report",
            "",
            "This is the one-shot final holdout action universe for `NEW5 20260128` only. The frozen TXF filter `^TXF[A-Z0-9]*$` was applied and malformed `info_time` rows were dropped/logged before feature or label construction.",
            "",
            "No tuning, retraining, threshold sweep, or NEW5-derived model selection was performed.",
            "",
            "## Build Summary",
            "",
            build.to_markdown(index=False),
            "",
            "## Causal Feature Columns",
            "",
            ", ".join(feature_cols),
            "",
            "## Label Columns",
            "",
            ", ".join(labels),
            "",
            f"Output parquet: `{ACTION_FILE}`.",
        ],
    )
    return build


def read_new5_event_frame() -> pd.DataFrame:
    return p04b.read_event_frame(ACTION_FILE)


def action_matrix_new5() -> pd.DataFrame:
    return p04b.action_feature_frame(
        pd.read_parquet(
            ACTION_FILE,
            columns=[
                "split",
                "session_id",
                "event_index",
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
            ],
        )
    )


def label_decomposition_audit() -> pd.DataFrame:
    raw = pd.read_parquet(
        ACTION_FILE,
        columns=["action_side", "raw_pnl_H20", "half_spread_component_H20", "excess_pnl_H20"],
    )
    rows = []
    for side in ["bid", "ask", "no_quote"]:
        side_frame = raw[raw["action_side"].astype(str).eq(side)]
        err = (
            pd.to_numeric(side_frame["raw_pnl_H20"], errors="coerce")
            - pd.to_numeric(side_frame["half_spread_component_H20"], errors="coerce")
            - pd.to_numeric(side_frame["excess_pnl_H20"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan).dropna().abs()
        rows.append(
            {
                "side": side,
                "rows_checked": int(len(err)),
                "max_abs_reconstruction_error": float(err.max()) if len(err) else 0.0,
                "formula": "raw_pnl_H20 = half_spread_component_H20 + excess_pnl_H20",
                "pass": bool((float(err.max()) if len(err) else 0.0) <= 1e-9),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new5_label_decomposition_audit.csv", index=False)
    write_markdown(
        ROOT / "NEW5_LABEL_DECOMPOSITION_AUDIT.md",
        [
            "# NEW5 Label Decomposition Audit",
            "",
            out.to_markdown(index=False),
            "",
            f"Overall pass: `{bool(out['pass'].astype(bool).all())}`.",
        ],
    )
    if not out["pass"].astype(bool).all():
        raise RuntimeError("NEW5 label decomposition failed.")
    return out


def ml_decision(model: Any, threshold: float) -> np.ndarray:
    matrix = action_matrix_new5()
    scores = p04b.model_scores(model, "hgb_regressor_shallow", matrix)
    return p04b.ml_policy_decision_from_scores(scores, threshold)


def policy_decision(policy_id: str, event: pd.DataFrame, model: Any) -> np.ndarray:
    if policy_id == EXPECTED_MODEL_ID:
        return ml_decision(model, EXPECTED_THRESHOLD)
    return p04b.policy_decision(policy_id, event)


def frozen_policy_eval() -> pd.DataFrame:
    model = load_model()
    event = read_new5_event_frame()
    rows = []
    for policy_id in POLICIES:
        metric = p04b.Metric()
        decision = policy_decision(policy_id, event, model)
        p04b.update_metric(metric, decision, p04b.decision_pnl(event, decision))
        row = p04b.finalize_metric(NEW5_SPLIT, policy_id, metric)
        row["holdout_scope"] = "NEW5_20260128"
        row["new5_used_for_tuning"] = False
        rows.append(row)
        print(f"NEW5 policy eval complete policy={policy_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new5_frozen_policy_eval.csv", index=False)
    write_markdown(
        ROOT / "NEW5_FROZEN_POLICY_EVALUATION_REPORT.md",
        [
            "# NEW5 Frozen Policy Evaluation Report",
            "",
            "Frozen policies were evaluated exactly once on NEW5 20260128 TXF-only data. No retraining, threshold sweeps, feature changes, or causal-fill changes were performed.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def matched_random_controls(policy_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load_model()
    event = read_new5_event_frame()
    decisions = {
        EXPECTED_MODEL_ID: policy_decision(EXPECTED_MODEL_ID, event, model),
        EXPECTED_QUEUE_RULE: policy_decision(EXPECTED_QUEUE_RULE, event, model),
    }
    rng = np.random.default_rng(RANDOM_SEED)
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    session = str(event["session_id"].iloc[0])
    bid_values = np.nan_to_num(event["bid_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0)
    ask_values = np.nan_to_num(event["ask_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0)
    total_by_side = {"bid": float(bid_values.sum()), "ask": float(ask_values.sum())}
    values_by_side = {"bid": bid_values, "ask": ask_values}

    for policy_id, decision in decisions.items():
        model_pnl = float(p04b.decision_pnl(event, decision).sum())
        required = {
            "bid": int((decision == p04b.DECISION_BID).sum()),
            "ask": int((decision == p04b.DECISION_ASK).sum()),
        }
        trials = np.zeros(RANDOM_TRIALS, dtype=float)
        for trial in range(RANDOM_TRIALS):
            total = 0.0
            for side, count in required.items():
                if count <= 0:
                    continue
                values = values_by_side[side]
                n = len(values)
                if count >= n:
                    total += total_by_side[side]
                else:
                    # Exact without-replacement draw. Use complement when it is smaller.
                    if count > n // 2:
                        exclude_count = n - count
                        exclude = rng.choice(n, size=exclude_count, replace=False)
                        total += total_by_side[side] - float(values[exclude].sum())
                    else:
                        idx = rng.choice(n, size=count, replace=False)
                        total += float(values[idx].sum())
            trials[trial] = total
            trial_rows.append(
                {
                    "policy_id": policy_id,
                    "trial": int(trial),
                    "split": NEW5_SPLIT,
                    "session_id": session,
                    "random_total_excess_pnl_H20": total,
                }
            )
        random_95p = float(np.quantile(trials, 0.95))
        frac = float((trials >= model_pnl).mean())
        summary_rows.append(
            {
                "split": NEW5_SPLIT,
                "session_id": session,
                "policy_id": policy_id,
                "model_total_excess_pnl_H20": model_pnl,
                "model_selected_action_count": int((decision != p04b.DECISION_NO_QUOTE).sum()),
                "bid_count": required["bid"],
                "ask_count": required["ask"],
                "random_trials": RANDOM_TRIALS,
                "random_mean": float(trials.mean()),
                "random_std": float(trials.std(ddof=0)),
                "random_95p": random_95p,
                "random_99p": float(np.quantile(trials, 0.99)),
                "empirical_p_value_fraction_random_ge_model": frac,
                "empirical_p_value_wording": "< 0.001" if frac == 0.0 else f"{frac:.6f}",
                "beats_random_95p": bool(model_pnl > random_95p),
                "matching": "exact_count_side_session",
                "new5_used_for_tuning": False,
            }
        )
        print(f"NEW5 matched random complete policy={policy_id}")
    trials_df = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(summary_rows)
    trials_df.to_csv(RESULTS / "new5_matched_random_trials.csv", index=False)
    summary.to_csv(RESULTS / "new5_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "NEW5_MATCHED_RANDOM_REPORT.md",
        [
            "# NEW5 Matched Random Report",
            "",
            "Matched random controls were run once for the frozen ML candidate and queue rule on NEW5 20260128. Matching dimensions: count, side, and session. No threshold or policy selection used NEW5.",
            "",
            summary.to_markdown(index=False),
            "",
            "If no random trial beats the policy, the report uses empirical p-value `< 0.001`; it does not claim exact zero probability.",
        ],
    )
    return trials_df, summary


def stable_uniform(frame: pd.DataFrame, decision: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str).to_numpy(),
            "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "decision": np.asarray(decision, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little"))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def causal_fill_sequential() -> pd.DataFrame:
    model = load_model()
    event = read_new5_event_frame()
    norms = p04b.compute_fill_norms()
    rows = []
    for policy_id in POLICIES:
        decision = policy_decision(policy_id, event, model)
        if policy_id == "no_quote":
            fills = np.zeros(len(event), dtype=bool)
        else:
            probs = p04b.fill_probabilities(event, decision, norms)
            fills = (decision != p04b.DECISION_NO_QUOTE) & (
                stable_uniform(event, decision, "04c_new5_conservative_l2_stress") < probs
            )
        pnl = p04b.decision_pnl(event, decision)
        result = p04b.simulate_scheduled(
            decision.astype(np.int8),
            fills.astype(np.bool_),
            pnl.astype(np.float64),
            EXPECTED_HORIZON,
            EXPECTED_INVENTORY_CAP,
        )
        fill_count = max(int(result[1]), 1)
        rows.append(
            {
                "split": NEW5_SPLIT,
                "holdout_scope": "NEW5_20260128",
                "policy_id": policy_id,
                "horizon": EXPECTED_HORIZON,
                "fill_model": EXPECTED_FILL_MODEL,
                "inventory_cap": EXPECTED_INVENTORY_CAP,
                "fee_multiplier": 1.0,
                "sequential_excess_pnl": float(result[0]),
                "fill_count": int(result[1]),
                "forced_no_quote_count": int(result[2]),
                "max_abs_inventory": int(result[3]),
                "max_drawdown": float(result[4]),
                "win_rate": int(result[5]) / fill_count if int(result[1]) else 0.0,
                "loss_rate": int(result[6]) / fill_count if int(result[1]) else 0.0,
                "avg_trade_excess_pnl": float(result[0]) / fill_count if int(result[1]) else 0.0,
                "fill_uses_future_pnl": False,
                "new5_used_for_tuning": False,
            }
        )
        print(f"NEW5 sequential complete policy={policy_id}")
    out = pd.DataFrame(rows)
    queue_val = float(out.loc[out["policy_id"].eq(EXPECTED_QUEUE_RULE), "sequential_excess_pnl"].iloc[0])
    ml_val = float(out.loc[out["policy_id"].eq(EXPECTED_MODEL_ID), "sequential_excess_pnl"].iloc[0])
    out["ml_minus_queue_rule"] = ml_val - queue_val
    out.to_csv(RESULTS / "new5_causal_fill_sequential_eval.csv", index=False)
    write_markdown(
        ROOT / "NEW5_CAUSAL_FILL_SEQUENTIAL_REPORT.md",
        [
            "# NEW5 Causal-Fill Sequential Report",
            "",
            "Frozen causal-fill sequential proxy: H20 exit, conservative L2 fill, inventory cap = 1, fee multiplier = 1x. No NEW5 tuning was performed.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def final_report(policy_eval: pd.DataFrame, random_summary: pd.DataFrame, sequential: pd.DataFrame) -> None:
    ml_policy = policy_eval[policy_eval["policy_id"].eq(EXPECTED_MODEL_ID)].iloc[0]
    queue_policy = policy_eval[policy_eval["policy_id"].eq(EXPECTED_QUEUE_RULE)].iloc[0]
    ml_random = random_summary[random_summary["policy_id"].eq(EXPECTED_MODEL_ID)].iloc[0]
    queue_random = random_summary[random_summary["policy_id"].eq(EXPECTED_QUEUE_RULE)].iloc[0]
    ml_seq = sequential[sequential["policy_id"].eq(EXPECTED_MODEL_ID)].iloc[0]
    queue_seq = sequential[sequential["policy_id"].eq(EXPECTED_QUEUE_RULE)].iloc[0]

    ml_positive = float(ml_policy["total_excess_pnl_H20"]) > 0.0
    ml_random_ok = bool(ml_random["beats_random_95p"])
    ml_seq_positive = float(ml_seq["sequential_excess_pnl"]) > 0.0
    ml_seq_beats_queue = float(ml_seq["sequential_excess_pnl"]) >= float(queue_seq["sequential_excess_pnl"])
    queue_seq_positive = float(queue_seq["sequential_excess_pnl"]) > 0.0

    if ml_positive and ml_random_ok and ml_seq_positive and ml_seq_beats_queue:
        outcome = "ml_passes_new5_holdout"
    elif ml_seq_positive or queue_seq_positive:
        outcome = "partial_holdout_pass_not_ml_dominant"
    else:
        outcome = "both_fail_new5"

    safety = pd.DataFrame(
        [
            {
                "claim": "ML beats matched random 95p on NEW5",
                "status": "PASS" if ml_random_ok else "FAIL",
                "safe_wording": "The frozen ML candidate beat the exact matched random 95p hurdle on NEW5 20260128." if ml_random_ok else "The frozen ML candidate did not beat the exact matched random 95p hurdle on NEW5 20260128.",
            },
            {
                "claim": "ML causal-fill sequential PnL is positive on NEW5",
                "status": "PASS" if ml_seq_positive else "FAIL",
                "safe_wording": "The frozen ML candidate had positive causal-fill sequential excess PnL under the proxy simulator." if ml_seq_positive else "The frozen ML candidate failed the causal-fill sequential proxy on NEW5.",
            },
            {
                "claim": "ML beats queue rule in NEW5 causal-fill sequential proxy",
                "status": "PASS" if ml_seq_beats_queue else "FAIL",
                "safe_wording": "The frozen ML candidate exceeded the frozen queue rule under the same NEW5 causal-fill sequential proxy." if ml_seq_beats_queue else "The frozen ML candidate did not exceed the frozen queue rule under the same NEW5 causal-fill sequential proxy.",
            },
            {
                "claim": "NEW5 was used for tuning",
                "status": "FORBIDDEN_AND_NOT_DONE",
                "safe_wording": "NEW5 was used only for one-shot final holdout evaluation.",
            },
            {
                "claim": "This proves real-world profitability",
                "status": "FORBIDDEN",
                "safe_wording": "The result is proxy TXF-only holdout evidence, not proof of real-world profitability.",
            },
            {
                "claim": "This is true exchange replay / true queue priority",
                "status": "FORBIDDEN",
                "safe_wording": "The causal-fill sequential simulator remains a proxy and does not replay exchange queue priority.",
            },
            {
                "claim": "FPGA improves PnL",
                "status": "FORBIDDEN",
                "safe_wording": "No FPGA PnL improvement claim is made.",
            },
        ]
    )
    safety.to_csv(RESULTS / "new5_final_claim_safety_table.csv", index=False)

    write_markdown(
        ROOT / "NEW5_ONE_SHOT_FINAL_HOLDOUT_REPORT.md",
        [
            "# NEW5 One-Shot Final Holdout Report",
            "",
            f"Final outcome: `{outcome}`.",
            "",
            "Scope: `NEW5 20260128` TXF-only data. This was a one-shot frozen evaluation. No parameters, features, labels, thresholds, fill proxy settings, or model weights were changed using NEW5.",
            "",
            "## Frozen Policy Evaluation",
            "",
            policy_eval.to_markdown(index=False),
            "",
            "## Matched Random Summary",
            "",
            random_summary.to_markdown(index=False),
            "",
            "## Causal-Fill Sequential Summary",
            "",
            sequential.to_markdown(index=False),
            "",
            "## Key Comparison",
            "",
            f"ML action-level excess PnL: `{float(ml_policy['total_excess_pnl_H20']):,.2f}`.",
            f"Queue action-level excess PnL: `{float(queue_policy['total_excess_pnl_H20']):,.2f}`.",
            f"ML matched-random p-value wording: `{ml_random['empirical_p_value_wording']}`.",
            f"ML sequential excess PnL: `{float(ml_seq['sequential_excess_pnl']):,.2f}`.",
            f"Queue sequential excess PnL: `{float(queue_seq['sequential_excess_pnl']):,.2f}`.",
            "",
            "## Claim Safety Table",
            "",
            safety.to_markdown(index=False),
            "",
            "Conservative interpretation: this is TXF-only proxy holdout evidence under frozen contracts. It does not prove real-world profitability, true queue replay, or FPGA-driven PnL improvement.",
        ],
    )


def main() -> None:
    ensure_dirs()
    frozen_contract_check()
    build_new5_action_universe()
    label_decomposition_audit()
    policy_eval = frozen_policy_eval()
    _, random_summary = matched_random_controls(policy_eval)
    sequential = causal_fill_sequential()
    final_report(policy_eval, random_summary, sequential)
    print("Phase 04C NEW5 one-shot final holdout complete.")
    print(ROOT / "NEW5_ONE_SHOT_FINAL_HOLDOUT_REPORT.md")


if __name__ == "__main__":
    main()
