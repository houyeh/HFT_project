from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

import common_05d as c

PHASE05A = c.PHASE05A
if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402


def load_rank_module():
    path = c.ROOT / "03_train_ranking_head.py"
    spec = importlib.util.spec_from_file_location("phase05d_rank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    c.ensure_dirs()
    rank = load_rank_module()
    selected = c.read_json(c.RESULTS / "phc_rank_selected_policy.json")
    threshold_search = pd.read_csv(c.RESULTS / "phc_rank_threshold_search.csv")
    target = threshold_search[
        threshold_search["variant"].eq(selected["model"])
        & threshold_search["lambda_rank"].astype(float).eq(float(selected["lambda_rank"]))
        & threshold_search["lambda_gate"].astype(float).eq(float(selected["lambda_gate"]))
    ].copy()
    seed_best = (
        target.sort_values(["seed", "total_excess_pnl_H20", "oracle_capture_ratio"], ascending=[True, False, False])
        .groupby("seed")
        .head(1)
        .reset_index(drop=True)
    )
    rows = []
    for _, cfg in seed_best.iterrows():
        run_id = str(cfg["run_id"])
        model, ckpt = rank.load_rank_model(run_id)
        threshold = float(cfg["threshold"])
        for split in c.SPLITS:
            payload = p05a.load_sequences(split)
            scores, _ = rank.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
            decision = p05a.policy_from_scores(scores, threshold)
            metric = p05a.metric_from_decision(split, str(cfg["variant"]), payload, decision)
            causal = c.causal_fill_for_decision(split, str(cfg["variant"]), payload, decision, f"05d_rank_seed_stability_{run_id}")
            oracle = c.oracle_capture_for_decision(payload, decision)
            rows.append(
                {
                    "seed": int(cfg["seed"]),
                    "run_id": run_id,
                    "split": split,
                    "threshold_selected_on_val_for_seed": threshold,
                    "best_epoch": int(cfg["best_epoch"]),
                    "action_pnl": metric["total_excess_pnl_H20"],
                    "selected_action_count": metric["selected_action_count"],
                    "causal_fill_pnl": causal["sequential_excess_pnl"],
                    "oracle_capture_ratio": oracle["capture_ratio"],
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "phc_rank_seed_stability.csv", index=False)
    summary = out.groupby("split").agg(
        action_pnl_mean=("action_pnl", "mean"),
        action_pnl_min=("action_pnl", "min"),
        action_pnl_max=("action_pnl", "max"),
        causal_fill_pnl_mean=("causal_fill_pnl", "mean"),
        causal_fill_pnl_min=("causal_fill_pnl", "min"),
        oracle_capture_mean=("oracle_capture_ratio", "mean"),
        seed_count=("seed", "nunique"),
    ).reset_index()
    summary.to_csv(c.RESULTS / "phc_rank_seed_stability_summary.csv", index=False)
    c.write_markdown(
        c.ROOT / "PHC_RANKING_HEAD_SEED_STABILITY.md",
        [
            "# PHC Ranking Head Seed Stability",
            "",
            "Each seed uses the same validation-selected variant/lambda family and a per-seed threshold selected on validation only. NEW3/NEW4/NEW5 are not used for seed or threshold selection.",
            "",
            "## Summary",
            summary.to_markdown(index=False),
            "",
            "## Per-Seed Detail",
            out.to_markdown(index=False),
        ],
    )
    print("wrote PHC ranking seed stability")


if __name__ == "__main__":
    main()
