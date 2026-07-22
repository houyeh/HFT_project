from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05b_phc_lite_new5_posthoc_diagnostic"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
PHASE04C = PROJECT_ROOT / "n2ew_era" / "04c_new5_one_shot_final_holdout"
ACTION_FILE = PHASE04C / "data" / "new5_txf_only_action_universe.parquet"
for path in [PHASE05A]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_05a as p05a  # noqa: E402


NEW5_SCOPE = "post_hoc_diagnostic_only_not_fresh_holdout"
PHC_POLICY_ID = "phc_lite_posthoc_new5"
N_RANDOM = 1000


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_new5_event_frame() -> pd.DataFrame:
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
    raw = pd.read_parquet(ACTION_FILE, columns=cols)
    bid = raw[raw["action_slot"].astype(int).eq(p05a.ACTION_BID)].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(p05a.ACTION_ASK)].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int).eq(p05a.ACTION_NO_QUOTE)].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError("NEW5 action triplet mismatch.")
    for col in ["session_id", "symbol", "event_index"]:
        if not bid[col].reset_index(drop=True).equals(ask[col].reset_index(drop=True)):
            raise RuntimeError(f"NEW5 bid/ask {col} mismatch.")
    out = bid.drop(columns=["action_slot", "excess_pnl_H20"]).copy()
    out["bid_excess_pnl_H20"] = pd.to_numeric(bid["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["ask_excess_pnl_H20"] = pd.to_numeric(ask["excess_pnl_H20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def contract_check() -> pd.DataFrame:
    selected = read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    scaler = read_json(PHASE05A / "data" / "phc_lite_feature_scaler.json")
    training_curve = pd.read_csv(PHASE05A / "results" / "phc_lite_training_curve.csv")
    rows = [
        {"item": "phc_checkpoint_exists", "expected": True, "actual": (PHASE05A / "models" / "phc_lite_best_val.pt").exists(), "pass": (PHASE05A / "models" / "phc_lite_best_val.pt").exists()},
        {"item": "threshold_source", "expected": "Phase05A validation-selected policy", "actual": selected.get("selection_split"), "pass": selected.get("selection_split") == "val"},
        {"item": "threshold", "expected": selected.get("threshold"), "actual": selected.get("threshold"), "pass": True},
        {"item": "scaler_fit_split", "expected": "train", "actual": scaler.get("normalization_fit_split"), "pass": scaler.get("normalization_fit_split") == "train"},
        {"item": "feature_list_match", "expected": p05a.PHC_FEATURES, "actual": scaler.get("feature_names"), "pass": scaler.get("feature_names") == p05a.PHC_FEATURES},
        {"item": "seq_len", "expected": 128, "actual": int(selected.get("seq_len", -1)), "pass": int(selected.get("seq_len", -1)) == 128},
        {"item": "new5_used_in_threshold_search", "expected": False, "actual": bool(selected.get("new5_used_for_selection")), "pass": not bool(selected.get("new5_used_for_selection"))},
        {"item": "new5_used_in_training_curve", "expected": False, "actual": "new5" in ",".join(training_curve.columns).lower(), "pass": "new5" not in ",".join(training_curve.columns).lower()},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_new5_posthoc_contract_check.csv", index=False)
    write_markdown(
        ROOT / "PHC_NEW5_POSTHOC_CONTRACT_CHECK.md",
        [
            "# PHC NEW5 Post-hoc Contract Check",
            "",
            "This phase is post-hoc diagnostic only. NEW5 is already burned by Phase 04C and is not a fresh holdout for PHC-lite.",
            "",
            out.to_markdown(index=False),
            "",
            f"Overall pass: `{bool(out['pass'].astype(bool).all())}`.",
        ],
    )
    if not out["pass"].astype(bool).all():
        raise RuntimeError("PHC NEW5 posthoc contract check failed.")
    return out


def build_sequence_dataset() -> dict[str, Any]:
    scaler = read_json(PHASE05A / "data" / "phc_lite_feature_scaler.json")
    frame = read_new5_event_frame()
    built = p05a.build_windows_for_frame(frame, scaler, p05a.SEQ_LEN, p05a.STRIDE)
    if built is None:
        raise RuntimeError("NEW5 sequence dataset could not be built.")
    payload = {
        "x": torch.from_numpy(built["x"]),
        "y_bid": torch.from_numpy(built["y_bid"].astype(np.float32)),
        "y_ask": torch.from_numpy(built["y_ask"].astype(np.float32)),
        "event_index": torch.from_numpy(built["event_index"].astype(np.int64)),
        "timestamp_us": torch.from_numpy(built["timestamp_us"].astype(np.int64)),
        "target_raw": torch.from_numpy(built["target_raw"].astype(np.float32)),
        "session_code": torch.from_numpy(built["session_code"].astype(np.int32)),
        "session_labels": built["session_labels"],
        "feature_names": p05a.PHC_FEATURES,
        "target_raw_columns": p05a.TARGET_RAW_COLUMNS,
        "seq_len": p05a.SEQ_LEN,
        "stride": p05a.STRIDE,
        "target_horizon": p05a.HORIZON,
        "split": NEW5_SCOPE,
        "post_hoc_diagnostic_only": True,
        "new5_used_for_training": False,
        "new5_used_for_selection": False,
    }
    torch.save(payload, DATA / "phc_new5_posthoc_sequences.pt")
    summary = pd.DataFrame(
        [
            {
                "split": NEW5_SCOPE,
                "source": "Phase04C NEW5 20260128 TXF-only action universe",
                "event_rows": len(frame),
                "window_rows": len(payload["y_bid"]),
                "session_count": len(payload["session_labels"]),
                "seq_len": p05a.SEQ_LEN,
                "stride": p05a.STRIDE,
                "target_horizon": p05a.HORIZON,
                "bid_positive_rate": float((payload["y_bid"].numpy() > 0).mean()),
                "ask_positive_rate": float((payload["y_ask"].numpy() > 0).mean()),
            }
        ]
    )
    summary.to_csv(RESULTS / "phc_new5_sequence_dataset_summary.csv", index=False)
    return payload


def load_payload() -> dict[str, Any]:
    path = DATA / "phc_new5_posthoc_sequences.pt"
    if not path.exists():
        return build_sequence_dataset()
    return torch.load(path, map_location="cpu", weights_only=False)


def phc_decision(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    selected = read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    scores = p05a.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, float(selected["threshold"])), scores


def evaluate_policies(payload: dict[str, Any]) -> pd.DataFrame:
    phc_dec, _ = phc_decision(payload)
    hgb_dec, _ = p05a.hgb_decision(payload)
    decisions = {
        "no_quote": p05a.no_quote_decision(payload),
        "old_bid_only_spread_ge_1": p05a.old_bid_only_decision(payload),
        "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
        "ml_hgb_regressor_shallow_convergence_selected": hgb_dec,
        PHC_POLICY_ID: phc_dec,
    }
    rows = []
    for policy_id, decision in decisions.items():
        row = p05a.metric_from_decision(NEW5_SCOPE, policy_id, payload, decision)
        row["post_hoc_diagnostic_only"] = True
        row["new5_used_for_selection"] = False
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_new5_posthoc_policy_eval.csv", index=False)
    write_markdown(
        ROOT / "PHC_NEW5_POSTHOC_POLICY_EVALUATION_REPORT.md",
        [
            "# PHC NEW5 Post-hoc Policy Evaluation Report",
            "",
            "This is post-hoc diagnostic only. No PHC training, epoch selection, threshold selection, architecture change, feature change, or seq_len change used NEW5.",
            "",
            "Comparison caveat: the frozen Phase 04B.1 HGB artifact emitted a local sklearn `InconsistentVersionWarning` when loaded. PHC-lite evaluation does not depend on that pickle; the caveat applies only to the HGB comparison row.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def matched_random(payload: dict[str, Any], decision: np.ndarray) -> pd.DataFrame:
    model_pnl = float(p05a.decision_pnl_from_payload(payload, decision).sum())
    session = payload["session_code"].numpy()
    spread_bucket = p05a.spread_bucket_from_payload(payload)
    y_bid = payload["y_bid"].numpy()
    y_ask = payload["y_ask"].numpy()
    selected_groups = []
    for side in [p05a.DECISION_BID, p05a.DECISION_ASK]:
        mask_side = decision == side
        for sess in np.unique(session[mask_side]):
            mask_sess = mask_side & (session == sess)
            for bucket in np.unique(spread_bucket[mask_sess]):
                count = int((mask_sess & (spread_bucket == bucket)).sum())
                if count:
                    selected_groups.append((side, int(sess), str(bucket), count))
    rng = np.random.default_rng(20260606)
    values = []
    for _ in range(N_RANDOM):
        total = 0.0
        for side, sess, bucket, count in selected_groups:
            pool = (session == sess) & (spread_bucket == bucket)
            pnl_pool = y_bid[pool] if side == p05a.DECISION_BID else y_ask[pool]
            if len(pnl_pool) < count:
                pool = session == sess
                pnl_pool = y_bid[pool] if side == p05a.DECISION_BID else y_ask[pool]
            idx = rng.choice(len(pnl_pool), size=count, replace=len(pnl_pool) < count)
            total += float(np.asarray(pnl_pool)[idx].sum())
        values.append(total)
    arr = np.asarray(values, dtype=float)
    p_value = float((arr >= model_pnl).mean())
    summary = pd.DataFrame(
        [
            {
                "split": NEW5_SCOPE,
                "policy_id": PHC_POLICY_ID,
                "model_total_excess_pnl_H20": model_pnl,
                "model_selected_action_count": int((decision != p05a.DECISION_NO_QUOTE).sum()),
                "random_trials": N_RANDOM,
                "random_mean": float(arr.mean()),
                "random_std": float(arr.std(ddof=0)),
                "random_95p": float(np.quantile(arr, 0.95)),
                "random_99p": float(np.quantile(arr, 0.99)),
                "empirical_p_value_fraction_random_ge_model": p_value,
                "empirical_p_value_wording": "< 0.001" if p_value == 0.0 else f"{p_value:.6f}",
                "beats_random_95p": bool(model_pnl > np.quantile(arr, 0.95)),
                "matching": "count_session_side_spread_bucket_with_session_side_fallback",
                "post_hoc_diagnostic_only": True,
            }
        ]
    )
    summary.to_csv(RESULTS / "phc_new5_posthoc_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "PHC_NEW5_POSTHOC_MATCHED_RANDOM_REPORT.md",
        [
            "# PHC NEW5 Post-hoc Matched Random Report",
            "",
            "Matched random is run for PHC-lite only, post-hoc diagnostic. Empirical p-value is the fraction of matched random trials whose PnL is greater than or equal to PHC-lite PnL.",
            "",
            summary.to_markdown(index=False),
        ],
    )
    return summary


def causal_fill(payload: dict[str, Any], policy_eval: pd.DataFrame) -> pd.DataFrame:
    phc_dec, _ = phc_decision(payload)
    hgb_dec, _ = p05a.hgb_decision(payload)
    decisions = {
        "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
        "ml_hgb_regressor_shallow_convergence_selected": hgb_dec,
        PHC_POLICY_ID: phc_dec,
    }
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for policy_id, decision in decisions.items():
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, f"05b_{policy_id}") < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
        rows.append(
            {
                "split": NEW5_SCOPE,
                "policy_id": policy_id,
                "horizon": p05a.HORIZON,
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
                "post_hoc_diagnostic_only": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_new5_posthoc_causal_fill_eval.csv", index=False)
    write_markdown(
        ROOT / "PHC_NEW5_POSTHOC_CAUSAL_FILL_REPORT.md",
        [
            "# PHC NEW5 Post-hoc Causal-Fill Report",
            "",
            "Same proxy family as Phase 05A: H20 exit, conservative L2 fill proxy, inventory cap 1. It is not true queue replay.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def final_report(policy_eval: pd.DataFrame, random_summary: pd.DataFrame, causal_df: pd.DataFrame) -> None:
    phc = policy_eval[policy_eval["policy_id"].eq(PHC_POLICY_ID)].iloc[0]
    hgb = policy_eval[policy_eval["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].iloc[0]
    phc_c = causal_df[causal_df["policy_id"].eq(PHC_POLICY_ID)].iloc[0]
    hgb_c = causal_df[causal_df["policy_id"].eq("ml_hgb_regressor_shallow_convergence_selected")].iloc[0]
    if float(phc["total_excess_pnl_H20"]) > float(hgb["total_excess_pnl_H20"]) and float(phc_c["sequential_excess_pnl"]) > float(hgb_c["sequential_excess_pnl"]):
        status = "phc_posthoc_beats_hgb_on_new5"
    elif float(phc["total_excess_pnl_H20"]) > 0.0:
        status = "phc_posthoc_positive_but_loses_to_hgb"
    else:
        status = "phc_posthoc_fails_new5"
    caveat = "NEW5 was already used in Phase 04C as the final HGB holdout. Therefore this PHC-lite NEW5 evaluation is post-hoc diagnostic only, not an independent fresh final holdout. A new fresh holdout such as NEW6 is required for a formal PHC final claim."
    write_markdown(
        ROOT / "PHC_NEW5_POSTHOC_DIAGNOSTIC_REPORT.md",
        [
            "# PHC NEW5 Post-hoc Diagnostic Report",
            "",
            f"Final status: `{status}`.",
            "",
            caveat,
            "",
            "No tuning on NEW5. No architecture change. No threshold change. No fresh holdout claim.",
            "",
            "Comparison caveat: the frozen Phase 04B.1 HGB artifact emitted a local sklearn `InconsistentVersionWarning` when loaded. PHC-lite evaluation does not depend on that pickle; the caveat applies only to the HGB comparison row.",
            "",
            "## Policy Evaluation",
            "",
            policy_eval.to_markdown(index=False),
            "",
            "## Matched Random",
            "",
            random_summary.to_markdown(index=False),
            "",
            "## Causal-Fill Sequential Proxy",
            "",
            causal_df.to_markdown(index=False),
            "",
            "This does not prove real-world profitability and is not true exchange replay.",
        ],
    )


def main() -> None:
    ensure_dirs()
    contract_check()
    payload = build_sequence_dataset()
    policy_eval = evaluate_policies(payload)
    phc_dec, _ = phc_decision(payload)
    random_summary = matched_random(payload, phc_dec)
    causal_df = causal_fill(payload, policy_eval)
    final_report(policy_eval, random_summary, causal_df)
    print(ROOT / "PHC_NEW5_POSTHOC_DIAGNOSTIC_REPORT.md")


if __name__ == "__main__":
    main()
