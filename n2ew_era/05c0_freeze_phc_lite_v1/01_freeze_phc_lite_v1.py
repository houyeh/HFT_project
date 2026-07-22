from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05c0_freeze_phc_lite_v1"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05B = PROJECT_ROOT / "n2ew_era" / "05b_phc_lite_new5_posthoc_diagnostic"
PHASE05B1 = PROJECT_ROOT / "n2ew_era" / "05b1_phc_window_oracle_headroom_analysis"


REQUIRED_ARTIFACTS = {
    "selected_policy": PHASE05A / "results" / "phc_lite_selected_policy.json",
    "model_checkpoint": PHASE05A / "models" / "phc_lite_best_val.pt",
    "feature_scaler": PHASE05A / "data" / "phc_lite_feature_scaler.json",
    "training_curve": PHASE05A / "results" / "phc_lite_training_curve.csv",
    "oos_eval": PHASE05A / "results" / "phc_lite_oos_policy_eval.csv",
    "matched_random": PHASE05A / "results" / "phc_lite_matched_random_summary.csv",
    "causal_fill": PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv",
    "seed_stability": PHASE05A / "results" / "phc_lite_seed_stability.csv",
    "parameter_count": PHASE05A / "results" / "phc_lite_parameter_count.csv",
    "new5_posthoc_policy": PHASE05B / "results" / "phc_new5_posthoc_policy_eval.csv",
    "new5_posthoc_matched_random": PHASE05B / "results" / "phc_new5_posthoc_matched_random_summary.csv",
    "new5_posthoc_causal_fill": PHASE05B / "results" / "phc_new5_posthoc_causal_fill_eval.csv",
    "phc_window_full_oracle": PHASE05B1 / "results" / "phc_new5_window_full_oracle_summary.csv",
    "phc_window_matched_count_oracle": PHASE05B1 / "results" / "phc_new5_window_matched_count_oracle.csv",
    "phc_window_matched_count_side_oracle": PHASE05B1 / "results" / "phc_new5_window_matched_count_side_oracle.csv",
}


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_manifest() -> pd.DataFrame:
    rows = []
    for name, path in REQUIRED_ARTIFACTS.items():
        rows.append(
            {
                "artifact_name": name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "phase_source": path.parts[path.parts.index("n2ew_era") + 1] if "n2ew_era" in path.parts else "",
                "required_for_freeze": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "phc_lite_v1_artifact_manifest.csv", index=False)
    return out


def build_config(manifest: pd.DataFrame) -> dict[str, Any]:
    selected = read_json(REQUIRED_ARTIFACTS["selected_policy"])
    scaler = read_json(REQUIRED_ARTIFACTS["feature_scaler"])
    ckpt = torch.load(REQUIRED_ARTIFACTS["model_checkpoint"], map_location="cpu", weights_only=False)
    parameter = pd.read_csv(REQUIRED_ARTIFACTS["parameter_count"]).iloc[0].to_dict()
    config = {
        "model_name": "PHC-lite v1",
        "model_type": "PHC-inspired recurrent sequence model",
        "hidden_dim": int(selected["hidden_dim"]),
        "mtl_steps": int(selected["mtl_steps"]),
        "seq_len": int(selected["seq_len"]),
        "stride": 32,
        "target": "excess_pnl_H20",
        "loss": "Huber bid + Huber ask",
        "parameter_count": int(parameter["parameter_count"]),
        "selected_epoch": int(selected["best_epoch"]),
        "selected_threshold": float(selected["threshold"]),
        "selection_split": "val",
        "feature_names": scaler["feature_names"],
        "target_scale": float(selected["target_scale"]),
        "checkpoint_seed": int(ckpt["seed"]),
        "architecture": [
            "Input causal L2 features",
            "encoder",
            "recurrent temporal state update with learned decay and gate",
            "within-timestep multi-transmission loop",
            "bid / ask action-value readout",
        ],
        "not_full_phcssm": True,
        "not_mamba": True,
        "not_alif_spiking_model": True,
        "not_stdp_model": True,
        "not_dale_constrained_yet": True,
        "not_true_exchange_replay": True,
        "not_real_world_profitability_proof": True,
        "data_protocol": {
            "product_filter": "^TXF[A-Z0-9]*$",
            "development": "NEW1 + NEW2 via Phase 05A train/val split",
            "diagnostic_oos": "NEW3 + NEW4",
            "new5": "not used for PHC training/selection; post-hoc diagnostic only",
        },
        "freeze_status": "phc_lite_v1_frozen_reference_model" if manifest["exists"].astype(bool).all() else "phc_lite_v1_freeze_incomplete_missing_artifacts",
    }
    write_json(ROOT / "phc_lite_v1_config.json", config)
    return config


def summary_tables(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    oos = pd.read_csv(REQUIRED_ARTIFACTS["oos_eval"])
    for _, row in oos[oos["policy_id"].isin(["phc_lite", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].iterrows():
        rows.append(
            {
                "section": "val_new3_new4_action_level",
                "split": row["split"],
                "policy_id": row["policy_id"],
                "metric": "total_excess_pnl_H20",
                "value": float(row["total_excess_pnl_H20"]),
            }
        )
    matched = pd.read_csv(REQUIRED_ARTIFACTS["matched_random"])
    for _, row in matched.iterrows():
        rows.append({"section": "matched_random", "split": row["split"], "policy_id": "phc_lite", "metric": "beats_random_95p", "value": bool(row["beats_random_95p"])})
        rows.append({"section": "matched_random", "split": row["split"], "policy_id": "phc_lite", "metric": "random_95p", "value": float(row["random_95p"])})
    causal = pd.read_csv(REQUIRED_ARTIFACTS["causal_fill"])
    for _, row in causal[causal["policy_id"].isin(["phc_lite", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].iterrows():
        rows.append({"section": "causal_fill", "split": row["split"], "policy_id": row["policy_id"], "metric": "sequential_excess_pnl", "value": float(row["sequential_excess_pnl"])})
    seed = pd.read_csv(REQUIRED_ARTIFACTS["seed_stability"])
    for _, row in seed.iterrows():
        rows.append({"section": "seed_stability", "split": row["split"], "policy_id": f"phc_lite_seed_{int(row['seed'])}", "metric": "pnl", "value": float(row["pnl"])})
        rows.append({"section": "seed_stability", "split": row["split"], "policy_id": f"phc_lite_seed_{int(row['seed'])}", "metric": "causal_fill_pnl", "value": float(row["causal_fill_pnl"])})
    oracle = pd.read_csv(REQUIRED_ARTIFACTS["phc_window_matched_count_oracle"])
    for _, row in oracle.iterrows():
        rows.append({"section": "phc_window_oracle", "split": "new5_posthoc", "policy_id": row["policy_id"], "metric": "capture_ratio", "value": float(row["capture_ratio"])})
        rows.append({"section": "phc_window_oracle", "split": "new5_posthoc", "policy_id": row["policy_id"], "metric": "opportunity_gap", "value": float(row["opportunity_gap"])})
    new5 = pd.read_csv(REQUIRED_ARTIFACTS["new5_posthoc_policy"])
    for _, row in new5[new5["policy_id"].isin(["phc_lite_posthoc_new5", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].iterrows():
        rows.append({"section": "new5_posthoc_action_level", "split": "new5_posthoc", "policy_id": row["policy_id"], "metric": "total_excess_pnl_H20", "value": float(row["total_excess_pnl_H20"])})
    new5_causal = pd.read_csv(REQUIRED_ARTIFACTS["new5_posthoc_causal_fill"])
    for _, row in new5_causal.iterrows():
        rows.append({"section": "new5_posthoc_causal_fill", "split": "new5_posthoc", "policy_id": row["policy_id"], "metric": "sequential_excess_pnl", "value": float(row["sequential_excess_pnl"])})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "phc_lite_v1_results_summary.csv", index=False)
    return out


def claim_boundary() -> None:
    write_markdown(
        ROOT / "phc_lite_v1_claim_boundary.md",
        [
            "# PHC-lite v1 Claim Boundary",
            "",
            "## Allowed Claims",
            "",
            "- PHC-lite v1 improves over the queue rule and HGB on development and diagnostic OOS under the TXF-only proxy evaluation.",
            "- PHC-lite v1 shows a stronger sequence-model signal than the tabular HGB baseline in Phase 05A diagnostics.",
            "- PHC-lite v1 has a better PHC-window oracle capture ratio than HGB and queue on NEW5 post-hoc diagnostic windows.",
            "- PHC-lite v1 is small and potentially HLS-friendly.",
            "",
            "## Forbidden Claims",
            "",
            "- PHC-lite v1 is full PHCSSM.",
            "- PHC-lite v1 passed an independent fresh final holdout.",
            "- PHC-lite v1 proves real-world profitability.",
            "- PHC-lite v1 is true exchange replay.",
            "- PHC-lite v1 proves FPGA improves PnL.",
            "- PHC-lite v1 is Dale-constrained, ALIF, STP, STDP, or Mamba.",
        ],
    )


def report(config: dict[str, Any], manifest: pd.DataFrame, summary: pd.DataFrame) -> None:
    train_curve = pd.read_csv(REQUIRED_ARTIFACTS["training_curve"])
    best = train_curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    oos = pd.read_csv(REQUIRED_ARTIFACTS["oos_eval"])
    causal = pd.read_csv(REQUIRED_ARTIFACTS["causal_fill"])
    matched = pd.read_csv(REQUIRED_ARTIFACTS["matched_random"])
    seed = pd.read_csv(REQUIRED_ARTIFACTS["seed_stability"])
    oracle = pd.read_csv(REQUIRED_ARTIFACTS["phc_window_matched_count_oracle"])
    new5 = pd.read_csv(REQUIRED_ARTIFACTS["new5_posthoc_policy"])
    lines = [
        "# PHC-lite v1 Freeze Report",
        "",
        f"Final status: `{config['freeze_status']}`.",
        "",
        "## 1. Data Protocol",
        "",
        "TXF-only product filter: `^TXF[A-Z0-9]*$`. Development uses NEW1+NEW2 via Phase 05A train/val split. NEW3+NEW4 are diagnostic OOS. NEW5 is not used for PHC training or selection; NEW5 is post-hoc diagnostic only.",
        "",
        "## 2. Label / Target",
        "",
        "Main target: `excess_pnl_H20`. Raw PnL is diagnostic only. Causal-fill sequential proxy is the strategy evaluation. Oracle is non-causal headroom diagnostic only.",
        "",
        "## 3. Model Architecture",
        "",
        f"Model: `{config['model_name']}`. Hidden dim `{config['hidden_dim']}`, MTL steps `{config['mtl_steps']}`, seq_len `{config['seq_len']}`, stride `{config['stride']}`, parameter count `{config['parameter_count']}`.",
        "",
        "Input causal L2 features -> encoder -> recurrent temporal state update with learned decay/gating -> within-timestep multi-transmission loop -> bid/ask action-value readout.",
        "",
        "This is not full PHCSSM, not Mamba, not ALIF, not STDP, not Dale-constrained yet, not true exchange replay, and not real-world profitability proof.",
        "",
        "## 4. Training Evidence",
        "",
        f"Best epoch: `{int(best['epoch'])}`. Train loss `{float(best['train_loss']):.6f}`, val loss `{float(best['val_loss']):.6f}`, validation policy PnL `{float(best['val_policy_excess_pnl']):,.2f}`, val Spearman bid/ask `{float(best['val_spearman_bid']):.4f}` / `{float(best['val_spearman_ask']):.4f}`, val PR-AUC bid/ask `{float(best['val_pr_auc_bid']):.4f}` / `{float(best['val_pr_auc_ask']):.4f}`, grad norm `{float(best['grad_norm']):.4f}`.",
        "Convergence caveat: seed 0 ran to max epoch 50 and validation policy PnL was still near its high-water mark; this is acceptable for frozen v1 reference but should be monitored in future ablations.",
        "",
        "## 5. Performance Evidence",
        "",
        "### PHC vs Queue vs HGB on val / NEW3 / NEW4",
        "",
        oos[oos["policy_id"].isin(["phc_lite", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].to_markdown(index=False),
        "",
        "### Matched Random",
        "",
        matched.to_markdown(index=False),
        "",
        "### Causal-fill Sequential",
        "",
        causal[causal["policy_id"].isin(["phc_lite", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].to_markdown(index=False),
        "",
        "### Seed Stability",
        "",
        seed.to_markdown(index=False),
        "",
        "### PHC-window Oracle Capture Ratio",
        "",
        oracle.to_markdown(index=False),
        "",
        "### NEW5 Post-hoc Diagnostic",
        "",
        new5[new5["policy_id"].isin(["phc_lite_posthoc_new5", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected"])].to_markdown(index=False),
        "",
        "## 6. Claim Boundary",
        "",
        "See `phc_lite_v1_claim_boundary.md`.",
        "",
        "## Artifact Manifest",
        "",
        manifest.to_markdown(index=False),
    ]
    write_markdown(ROOT / "PHC_LITE_V1_FREEZE_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    manifest = artifact_manifest()
    if not manifest["exists"].astype(bool).all():
        config = {"freeze_status": "phc_lite_v1_freeze_incomplete_missing_artifacts"}
        write_json(ROOT / "phc_lite_v1_config.json", config)
        write_markdown(
            ROOT / "PHC_LITE_V1_FREEZE_REPORT.md",
            [
                "# PHC-lite v1 Freeze Report",
                "",
                "Final status: `phc_lite_v1_freeze_incomplete_missing_artifacts`.",
                "",
                manifest.to_markdown(index=False),
            ],
        )
        return
    config = build_config(manifest)
    summary = summary_tables(config)
    claim_boundary()
    report(config, manifest, summary)
    print(config["freeze_status"])
    print(ROOT / "PHC_LITE_V1_FREEZE_REPORT.md")


if __name__ == "__main__":
    main()
