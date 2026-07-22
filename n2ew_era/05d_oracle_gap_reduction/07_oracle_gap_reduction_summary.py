from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common_05d as c

PHASE05A = c.PHASE05A
PHASE05C4 = c.PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def v1_rows() -> pd.DataFrame:
    action = load_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    causal = load_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    if action.empty:
        return pd.DataFrame()
    action = action[action["policy_id"].eq("phc_lite")].copy()
    causal = causal[causal["policy_id"].eq("phc_lite")].copy() if not causal.empty else pd.DataFrame()
    out = action[["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "action_pnl"})
    out["model"] = "phc_lite_v1"
    if not causal.empty:
        out = out.merge(causal[["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "causal_fill_pnl"}), on="split", how="left")
    v1_capture_path = c.PHASE05C2 / "results" / "phc_stp_window_oracle_capture.csv"
    if v1_capture_path.exists():
        cap = pd.read_csv(v1_capture_path)
        cap = cap[cap["policy_id"].eq("phc_lite")][["split", "capture_ratio"]].rename(columns={"capture_ratio": "oracle_capture"})
        out = out.merge(cap, on="split", how="left")
    return out


def model_rows(model: str, action_path: Path, causal_path: Path, oracle_path: Path | None = None, status: str = "diagnostic") -> pd.DataFrame:
    action = load_csv(action_path)
    if action.empty:
        return pd.DataFrame()
    causal = load_csv(causal_path)
    out = action[["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "action_pnl"})
    out["model"] = model
    if not causal.empty and "sequential_excess_pnl" in causal.columns:
        out = out.merge(causal[["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "causal_fill_pnl"}), on="split", how="left")
    if oracle_path and oracle_path.exists():
        oracle = load_csv(oracle_path)
        if "policy_id" in oracle.columns and model in set(oracle["policy_id"].astype(str)):
            oracle = oracle[oracle["policy_id"].astype(str).eq(model)].copy()
        col = "capture_ratio" if "capture_ratio" in oracle.columns else "oracle_capture_ratio"
        if col in oracle.columns:
            out = out.merge(oracle[["split", col]].rename(columns={col: "oracle_capture"}), on="split", how="left")
    elif "oracle_capture_ratio" in action.columns:
        out["oracle_capture"] = action["oracle_capture_ratio"].to_numpy()
    out["adopted_status"] = status
    return out


def add_error_rates(summary: pd.DataFrame, error_path: Path, model: str) -> pd.DataFrame:
    if not error_path.exists() or summary.empty:
        return summary
    errors = pd.read_csv(error_path)
    if errors.empty:
        return summary
    pivot = errors.pivot_table(index="split", columns="error_type", values="error_rate", aggfunc="first").reset_index()
    rename = {"false_positive": "false_positive_rate", "missed_opportunity": "missed_opportunity_rate", "side_error": "side_error_rate"}
    pivot = pivot.rename(columns=rename)
    for col in rename.values():
        if col not in summary.columns:
            summary[col] = np.nan
    for _, row in pivot.iterrows():
        mask = summary["model"].eq(model) & summary["split"].eq(row["split"])
        for col in rename.values():
            if col in row.index:
                summary.loc[mask, col] = row[col]
    return summary


def infer_status(model: str, rows: pd.DataFrame, v1: pd.DataFrame) -> str:
    if rows.empty:
        return "missing_not_evaluated"
    if model == "phc_streaming_dense":
        return "streaming_dense_hls_feasibility_diagnostic_not_adoption_candidate"
    if model == "phc_stp_120epoch":
        report = c.ROOT / "PHC_STP_120EPOCH_CLOSURE_REPORT.md"
        if report.exists():
            text = report.read_text(encoding="utf-8")
            for status in [
                "stp_adopted_as_phc_v2",
                "stp_not_adopted_convergence_incomplete",
                "stp_not_adopted_causal_fill_regression",
                "stp_not_adopted_overfits_validation",
                "stp_neutral_future_work",
            ]:
                if status in text:
                    return status
    if model == "phc_stp_v2_cosine_lr":
        gates_path = PHASE05C4 / "results" / "phc_stp_v2_adoption_gates.csv"
        if not gates_path.exists():
            return "phc_stp_v2_cosine_lr_not_adopted_missing_gate_audit"
        gates = pd.read_csv(gates_path)
        if gates.empty or "pass" not in gates.columns:
            return "phc_stp_v2_cosine_lr_not_adopted_invalid_gate_audit"
        all_pass = bool(gates["pass"].astype(str).str.lower().eq("true").all())
        if all_pass:
            return "stp_v2_adopted_as_best_sequence_model"
        return "phc_stp_v2_cosine_lr_not_adopted_gate_failure"
    merged = rows.merge(v1[["split", "action_pnl", "causal_fill_pnl"]].rename(columns={"action_pnl": "v1_action", "causal_fill_pnl": "v1_causal"}), on="split", how="left")
    action_ok = bool((merged["action_pnl"] >= merged["v1_action"]).all())
    causal_ok = bool((merged["causal_fill_pnl"] >= merged["v1_causal"]).all()) if "causal_fill_pnl" in merged else False
    if model == "phc_rank_head":
        seed_path = c.RESULTS / "phc_rank_seed_stability.csv"
        if seed_path.exists():
            seed_df = pd.read_csv(seed_path)
            seed_cmp = seed_df.merge(
                v1[["split", "action_pnl", "causal_fill_pnl"]].rename(columns={"action_pnl": "v1_action", "causal_fill_pnl": "v1_causal"}),
                on="split",
                how="left",
            )
            stable = bool((seed_cmp["action_pnl"] >= seed_cmp["v1_action"]).all() and (seed_cmp["causal_fill_pnl"] >= seed_cmp["v1_causal"]).all())
            if not stable:
                return "phc_rank_head_promising_not_adopted_seed_stability_caveat"
    if action_ok and causal_ok:
        return f"{model}_candidate_improves_v1"
    return f"{model}_not_adopted"


def main() -> None:
    c.ensure_dirs()
    v1 = v1_rows()
    rows = []
    if not v1.empty:
        if "oracle_capture" not in v1.columns:
            v1["oracle_capture"] = np.nan
        v1["adopted_status"] = "frozen_reference"
        rows.append(v1)
    configs = [
        (
            "phc_stp_v2_cosine_lr",
            PHASE05C4 / "results" / "phc_stp_v2_oos_eval.csv",
            PHASE05C4 / "results" / "phc_stp_v2_causal_fill_eval.csv",
            PHASE05C4 / "results" / "phc_stp_v2_window_oracle_capture.csv",
            PHASE05C4 / "results" / "phc_stp_v2_vs_phc_lite_v1_summary.csv",
        ),
        (
            "phc_stp_120epoch",
            c.RESULTS / "phc_stp_120epoch_oos_eval.csv",
            c.RESULTS / "phc_stp_120epoch_causal_fill.csv",
            c.RESULTS / "phc_stp_120epoch_oracle_capture.csv",
            c.RESULTS / "phc_stp_120epoch_vs_v1_summary.csv",
        ),
        (
            "phc_rank_head",
            c.RESULTS / "phc_rank_oos_eval.csv",
            c.RESULTS / "phc_rank_causal_fill.csv",
            c.RESULTS / "phc_rank_oracle_capture.csv",
            c.RESULTS / "phc_rank_vs_v1_summary.csv",
        ),
        (
            "phc_uncertainty_gating",
            c.RESULTS / "phc_uncertainty_oos_eval.csv",
            c.RESULTS / "phc_uncertainty_causal_fill.csv",
            None,
            c.RESULTS / "phc_uncertainty_vs_v1_summary.csv",
        ),
        (
            "phc_streaming_dense",
            c.RESULTS / "phc_streaming_dense_eval.csv",
            c.RESULTS / "phc_streaming_dense_causal_fill.csv",
            None,
            None,
        ),
    ]
    missing = []
    for model, action_path, causal_path, oracle_path, _ in configs:
        model_df = model_rows(model, action_path, causal_path, oracle_path)
        if model_df.empty:
            missing.append({"model": model, "missing_action_path": str(action_path), "missing_causal_path": str(causal_path)})
            continue
        model_df["adopted_status"] = infer_status(model, model_df, v1) if not v1.empty else "evaluated_no_v1_reference"
        rows.append(model_df)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not summary.empty:
        summary = add_error_rates(summary, c.RESULTS / "phc_rank_oracle_error.csv", "phc_rank_head")
        summary = add_error_rates(summary, c.RESULTS / "phc_uncertainty_oracle_error.csv", "phc_uncertainty_gating")
        summary.to_csv(c.RESULTS / "phc_oracle_gap_reduction_summary.csv", index=False)
    if missing:
        pd.DataFrame(missing).to_csv(c.RESULTS / "phc_oracle_gap_reduction_missing_outputs.csv", index=False)
    if not summary.empty:
        source = load_csv(PHASE05C4 / "results" / "phc_stp_v2_vs_phc_lite_v1_summary.csv")
        ingested = summary[summary["model"].eq("phc_stp_v2_cosine_lr")].copy()
        if not source.empty and not ingested.empty:
            compare = ingested[["split", "action_pnl", "causal_fill_pnl", "oracle_capture"]].merge(
                source[["split", "action_pnl", "causal_fill_pnl", "oracle_capture"]].rename(
                    columns={
                        "action_pnl": "report_action_pnl",
                        "causal_fill_pnl": "report_causal_fill_pnl",
                        "oracle_capture": "report_oracle_capture",
                    }
                ),
                on="split",
                how="left",
            )
            for col in ["action_pnl", "causal_fill_pnl", "oracle_capture"]:
                compare[f"abs_diff_{col}"] = (compare[col] - compare[f"report_{col}"]).abs()
            compare.to_csv(c.RESULTS / "phc_stp_v2_cosine_lr_ingestion_check.csv", index=False)
    adopted = []
    if not summary.empty:
        mask = summary["adopted_status"].str.contains("candidate_improves_v1|adopted_as_phc_v2|adopted_as_best_sequence_model", na=False)
        adopted = sorted(set(summary.loc[mask, "model"].tolist()))
    strict_adopted = []
    if not summary.empty:
        strict_mask = summary["adopted_status"].str.contains("adopted_as_phc_v2|adopted_as_best_sequence_model", na=False)
        strict_adopted = sorted(set(summary.loc[strict_mask, "model"].tolist()))
    if strict_adopted:
        recommendation = f"adopted_sequence_model: {', '.join(strict_adopted)}"
    elif adopted:
        recommendation = f"candidate_requires_pm_review: {', '.join(adopted)}"
    else:
        recommendation = "keep_phc_lite_v1"
    lines = [
        "# PHC Oracle Gap Reduction Summary Report",
        "",
        "This report summarizes controlled Phase 05D attempts to reduce the PHC-window Oracle gap. NEW5 is not used for model selection or threshold selection in this phase.",
        "",
        "## Summary Table",
        summary.to_markdown(index=False) if not summary.empty else "No model summaries were available.",
        "",
        "## Missing Outputs",
        pd.DataFrame(missing).to_markdown(index=False) if missing else "No missing outputs.",
        "",
        "## Final Recommendation",
        f"`{recommendation}`",
        "",
        "No model is adopted based only on validation. Oracle metrics remain non-causal diagnostics and cannot be used as a tradable strategy.",
    ]
    c.write_markdown(c.ROOT / "PHC_ORACLE_GAP_REDUCTION_SUMMARY_REPORT.md", lines)
    print(f"wrote summary recommendation={recommendation}")


if __name__ == "__main__":
    main()
