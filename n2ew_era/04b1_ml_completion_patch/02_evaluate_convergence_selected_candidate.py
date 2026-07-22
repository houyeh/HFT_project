from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
RESULTS = ROOT / "results"
PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
if str(PHASE04B) not in sys.path:
    sys.path.insert(0, str(PHASE04B))

import run_phase04b_pipeline as p04b  # noqa: E402


RANDOM_TRIALS = 1000
RANDOM_SEED = 20260604
SPLITS = ["val", "new3", "new4"]
OOS_SPLITS = ["new3", "new4"]
QUEUE_POLICY_ID = "side_by_queue_imbalance_abs_0p1"
FINAL_ML_POLICY_ID = "ml_hgb_regressor_shallow_convergence_selected"


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_closure() -> dict[str, Any]:
    path = RESULTS / "txf_hgb_convergence_closure.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def model_decision(model: Any, threshold: float, path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = p04b.action_matrix_from_part(path)
    event = p04b.read_event_frame(path)
    scores = p04b.model_scores(model, "hgb_regressor_shallow", matrix)
    decision = p04b.ml_policy_decision_from_scores(scores, threshold)
    return event, decision


def matched_random(model: Any, threshold: float, model_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        required: dict[tuple[str, str], int] = defaultdict(int)
        candidates: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        model_pnl = 0.0
        selected_count = 0
        for path in p04b.part_paths(split):
            event, decision = model_decision(model, threshold, path)
            pnl = p04b.decision_pnl(event, decision)
            model_pnl += float(pnl.sum())
            selected_count += int((decision != p04b.DECISION_NO_QUOTE).sum())
            session = str(event["session_id"].iloc[0])
            candidates[(session, "bid")].append(np.nan_to_num(event["bid_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0))
            candidates[(session, "ask")].append(np.nan_to_num(event["ask_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0))
            required[(session, "bid")] += int((decision == p04b.DECISION_BID).sum())
            required[(session, "ask")] += int((decision == p04b.DECISION_ASK).sum())
        arrays = {key: np.concatenate(chunks) for key, chunks in candidates.items()}
        trials = np.zeros(RANDOM_TRIALS, dtype=float)
        for trial in range(RANDOM_TRIALS):
            total = 0.0
            for key, count in required.items():
                if count <= 0:
                    continue
                values = arrays[key]
                if count >= len(values):
                    total += float(values.sum())
                else:
                    total += float(values[rng.choice(len(values), size=count, replace=False)].sum())
            trials[trial] = total
            trial_rows.append(
                {
                    "split": split,
                    "policy_id": FINAL_ML_POLICY_ID,
                    "model_label": model_label,
                    "trial": trial,
                    "random_total_excess_pnl_H20": total,
                }
            )
        random_95p = float(np.quantile(trials, 0.95))
        summary_rows.append(
            {
                "split": split,
                "policy_id": FINAL_ML_POLICY_ID,
                "model_label": model_label,
                "threshold_selected_on_val": threshold,
                "model_total_excess_pnl_H20": model_pnl,
                "model_selected_action_count": selected_count,
                "random_trials": RANDOM_TRIALS,
                "random_mean": float(trials.mean()),
                "random_std": float(trials.std(ddof=0)),
                "random_95p": random_95p,
                "random_99p": float(np.quantile(trials, 0.99)),
                "empirical_p_value_fraction_random_ge_model": float((trials >= model_pnl).mean()),
                "beats_random_95p": bool(model_pnl > random_95p),
                "matching": "exact_count_side_session",
                "selection_split": "val",
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        )
        print(f"final-candidate matched random split={split}")
    trial_df = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(summary_rows)
    trial_df.to_csv(RESULTS / "txf_ml_final_candidate_matched_random_trials.csv", index=False)
    summary.to_csv(RESULTS / "txf_ml_final_candidate_matched_random_summary.csv", index=False)
    return trial_df, summary


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


def sequential(model: Any, threshold: float, model_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    norms = p04b.compute_fill_norms()
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for policy_id in [QUEUE_POLICY_ID, FINAL_ML_POLICY_ID]:
            accum = {"pnl": 0.0, "fills": 0, "forced": 0, "max_inv": 0, "max_dd": 0.0, "win": 0, "loss": 0}
            cumulative = 0.0
            for path in p04b.part_paths(split):
                event = p04b.read_event_frame(path)
                if policy_id == QUEUE_POLICY_ID:
                    decision = p04b.policy_decision(QUEUE_POLICY_ID, event)
                else:
                    _, decision = model_decision(model, threshold, path)
                probs = p04b.fill_probabilities(event, decision, norms)
                fills = (decision != p04b.DECISION_NO_QUOTE) & (
                    stable_uniform(event, decision, "04b1_final_candidate_conservative_l2_stress") < probs
                )
                pnl = p04b.decision_pnl(event, decision)
                result = p04b.simulate_scheduled(
                    decision.astype(np.int8),
                    fills.astype(np.bool_),
                    pnl.astype(np.float64),
                    p04b.HORIZON,
                    1,
                )
                part_pnl = float(result[0])
                cumulative += part_pnl
                curve_rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "model_label": model_label if policy_id == FINAL_ML_POLICY_ID else "",
                        "session_id": str(event["session_id"].iloc[0]),
                        "part_sequential_excess_pnl": part_pnl,
                        "cumulative_sequential_excess_pnl": cumulative,
                    }
                )
                accum["pnl"] += part_pnl
                accum["fills"] += int(result[1])
                accum["forced"] += int(result[2])
                accum["max_inv"] = max(accum["max_inv"], int(result[3]))
                accum["max_dd"] = max(accum["max_dd"], float(result[4]))
                accum["win"] += int(result[5])
                accum["loss"] += int(result[6])
            fills_n = max(accum["fills"], 1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "model_label": model_label if policy_id == FINAL_ML_POLICY_ID else "",
                    "threshold_selected_on_val": threshold if policy_id == FINAL_ML_POLICY_ID else np.nan,
                    "horizon": p04b.HORIZON,
                    "fill_model": "conservative_l2_stress",
                    "inventory_cap": 1,
                    "sequential_excess_pnl": accum["pnl"],
                    "fill_count": accum["fills"],
                    "forced_no_quote_count": accum["forced"],
                    "max_abs_inventory": accum["max_inv"],
                    "max_drawdown": accum["max_dd"],
                    "win_rate": accum["win"] / fills_n if accum["fills"] else 0.0,
                    "loss_rate": accum["loss"] / fills_n if accum["fills"] else 0.0,
                    "avg_trade_excess_pnl": accum["pnl"] / fills_n if accum["fills"] else 0.0,
                    "fill_uses_future_pnl": False,
                    "selection_split": "val",
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                }
            )
            print(f"final-candidate sequential split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    compare = out.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index()
    compare["ml_minus_queue_rule"] = compare[FINAL_ML_POLICY_ID] - compare[QUEUE_POLICY_ID]
    out = out.merge(compare[["split", "ml_minus_queue_rule"]], on="split", how="left")
    curve = pd.DataFrame(curve_rows)
    out.to_csv(RESULTS / "txf_ml_final_candidate_causal_fill_sequential_eval.csv", index=False)
    curve.to_csv(RESULTS / "txf_ml_final_candidate_vs_queue_equity_curve.csv", index=False)
    return out, curve


def final_report(random_summary: pd.DataFrame, seq: pd.DataFrame, closure: dict[str, Any]) -> None:
    queue = seq[seq["policy_id"].eq(QUEUE_POLICY_ID)][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "queue_sequential_excess_pnl"})
    ml = seq[seq["policy_id"].eq(FINAL_ML_POLICY_ID)].merge(queue, on="split", how="left")
    ml["ml_beats_or_matches_queue"] = ml["sequential_excess_pnl"] >= ml["queue_sequential_excess_pnl"]
    random_ok = bool(random_summary["beats_random_95p"].astype(bool).all())
    seq_positive = bool(ml["sequential_excess_pnl"].gt(0.0).all())
    seq_competitive = bool(ml["ml_beats_or_matches_queue"].astype(bool).all())
    convergence_closed = bool(closure.get("convergence_closed_for_candidate_decision", False))
    if random_ok and seq_positive and seq_competitive and convergence_closed:
        decision = "ml_selected_for_new5"
        selected = FINAL_ML_POLICY_ID
    else:
        decision = "queue_rule_selected_for_new5"
        selected = QUEUE_POLICY_ID
    scorecard = pd.DataFrame(
        [
            {
                "candidate": FINAL_ML_POLICY_ID,
                "exact_config": f"hgb_regressor max_iter={closure['selected_budget_on_validation']} threshold={closure['selected_threshold_on_validation']}",
                "beats_matched_random_95p_val_new3_new4": random_ok,
                "positive_causal_fill_sequential_val_new3_new4": seq_positive,
                "competitive_vs_queue_in_sequential_val_new3_new4": seq_competitive,
                "hgb_convergence_closed": convergence_closed,
                "selected_for_new5": decision == "ml_selected_for_new5",
            },
            {
                "candidate": QUEUE_POLICY_ID,
                "exact_config": "queue_imbalance abs threshold 0.1",
                "beats_matched_random_95p_val_new3_new4": True,
                "positive_causal_fill_sequential_val_new3_new4": True,
                "competitive_vs_queue_in_sequential_val_new3_new4": True,
                "hgb_convergence_closed": np.nan,
                "selected_for_new5": decision == "queue_rule_selected_for_new5",
            },
        ]
    )
    scorecard.to_csv(RESULTS / "txf_04b1_final_candidate_redecision_scorecard.csv", index=False)
    write_markdown(
        ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md",
        [
            "# TXF-only Final Development Decision Report Patched",
            "",
            f"Final patched decision: `{decision}`.",
            f"Selected strategy for NEW5 holdout: `{selected}`.",
            "",
            "This final patched report uses the convergence-selected exact ML configuration, not merely the earlier 80-iteration Phase 04B frozen diagnostic model.",
            "",
            "NEW5 was not loaded or evaluated. NEW3/NEW4 were not used for tuning, thresholding, model selection, or HGB iteration selection.",
            "",
            "## Candidate Completion Scorecard",
            "",
            scorecard.to_markdown(index=False),
            "",
            "## Final ML Matched Random Summary",
            "",
            random_summary.to_markdown(index=False),
            "",
            "Advisor-safe p-value wording: if the empirical fraction is `0`, then in 1000 matched random trials no random trial had PnL >= model PnL, so empirical p-value `< 0.001`; it is not an exact zero probability claim.",
            "",
            "## Final ML vs Queue Causal-Fill Sequential Summary",
            "",
            ml.to_markdown(index=False),
            "",
            "## HGB Convergence Closure",
            "",
            pd.DataFrame([closure]).to_markdown(index=False),
            "",
            "Decision rule: ML can be selected only if the exact convergence-selected configuration beats matched random 95p, remains positive and competitive against the queue rule in causal-fill sequential evaluation, and convergence is closed under validation-only iteration selection.",
            "",
            "Allowed claim: TXF-only ML candidate completion patch selected one exact candidate under frozen development protocol.",
            "Forbidden claims: real-world profitability, true exchange replay, FPGA PnL improvement, TCN/PHC result, or any NEW5 result.",
        ],
    )


def main() -> None:
    closure = load_closure()
    model_path = RESULTS.parent / "models" / f"hgb_regressor_iter_{int(closure['selected_budget_on_validation'])}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    threshold = float(closure["selected_threshold_on_validation"])
    model_label = f"hgb_iter_{int(closure['selected_budget_on_validation'])}_threshold_{threshold:.6g}"
    _, random_summary = matched_random(model, threshold, model_label)
    seq, _ = sequential(model, threshold, model_label)
    final_report(random_summary, seq, closure)
    print("Convergence-selected ML candidate gates complete.")
    print(ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md")


if __name__ == "__main__":
    main()
