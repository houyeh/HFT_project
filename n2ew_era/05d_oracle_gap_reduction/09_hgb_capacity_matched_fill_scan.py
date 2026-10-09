from __future__ import annotations

import numpy as np
import pandas as pd

import common_matched_fill_diagnostics as d


CAPACITY_GRID = [40, 80, 120, 160, 240, 320]
POLICY_FILL_SALT = "05a_conservative_l2_stress"
ORACLE_FILL_SALT = "05d_val_capacity_matched_count_side_oracle"


def run_capacity_point(payload: dict, max_iter: int) -> tuple[dict, pd.DataFrame]:
    threshold_lookup = d.load_04b1_validation_thresholds()
    if int(max_iter) not in threshold_lookup:
        raise RuntimeError(f"Missing 04B1 validation threshold for max_iter={max_iter}")
    threshold_info = threshold_lookup[int(max_iter)]
    model = d.load_hgb_capacity_model(max_iter)
    decision_at_zero, scores = d.hgb_decision_scores(payload, model=model, threshold=0.0)
    threshold = float(threshold_info["threshold_selected_on_val"])
    decision = d.p05a.policy_from_scores(scores, threshold)
    policy_accepted, policy_summary = d.causal_fill_for_decision(payload, decision, salt=POLICY_FILL_SALT)
    oracle_decision = d.matched_count_side_oracle_for_policy(payload, decision)
    oracle_accepted, oracle_summary = d.causal_fill_for_decision(payload, oracle_decision, salt=ORACLE_FILL_SALT)
    dense_policy_pnl = float(d.p05a.decision_pnl_from_payload(payload, decision).sum())
    dense_oracle_pnl = float(d.p05a.decision_pnl_from_payload(payload, oracle_decision).sum())
    structural = d.hgb_structural_work(model)
    row = {
        "split": d.VAL_SPLIT,
        "capacity_id": f"hgb_iter_{int(max_iter)}",
        "model_path": str(d.PHASE04B1 / "models" / f"hgb_regressor_iter_{int(max_iter)}.joblib"),
        "max_iter": int(max_iter),
        **structural,
        "threshold_selected_on_val": float(threshold),
        "threshold_selection_metric": "04b1_validation_policy_pnl_on_txf_action_universe",
        "threshold_grid_source": "frozen_04b1_txf_hgb_iteration_sweep_validation_rows",
        **threshold_info,
        "zero_threshold_selected_count": int((decision_at_zero != d.p05a.DECISION_NO_QUOTE).sum()),
        "policy_selected_count": int((decision != d.p05a.DECISION_NO_QUOTE).sum()),
        "policy_bid_count": int((decision == d.p05a.DECISION_BID).sum()),
        "policy_ask_count": int((decision == d.p05a.DECISION_ASK).sum()),
        "oracle_selected_count": int((oracle_decision != d.p05a.DECISION_NO_QUOTE).sum()),
        "oracle_bid_count": int((oracle_decision == d.p05a.DECISION_BID).sum()),
        "oracle_ask_count": int((oracle_decision == d.p05a.DECISION_ASK).sum()),
        "policy_dense_excess_pnl": dense_policy_pnl,
        "oracle_dense_excess_pnl": dense_oracle_pnl,
        "policy_sequential_excess_pnl": float(policy_summary["sequential_excess_pnl"]),
        "oracle_sequential_excess_pnl": float(oracle_summary["sequential_excess_pnl"]),
        "matched_fill_capture_ratio": float(policy_summary["sequential_excess_pnl"] / oracle_summary["sequential_excess_pnl"])
        if oracle_summary["sequential_excess_pnl"]
        else np.nan,
        "policy_fill_count": int(policy_summary["fill_count"]),
        "oracle_fill_count": int(oracle_summary["fill_count"]),
        "policy_fill_attempt_count": int(policy_summary["fill_attempt_count"]),
        "oracle_fill_attempt_count": int(oracle_summary["fill_attempt_count"]),
        "policy_forced_no_quote_count": int(policy_summary["forced_no_quote_count"]),
        "oracle_forced_no_quote_count": int(oracle_summary["forced_no_quote_count"]),
        "fill_model": "conservative_l2_stress",
        "inventory_cap": 1,
        "horizon": d.p05a.HORIZON,
        "oracle_type": "matched_count_side_future_label_oracle_with_causal_fill",
        "oracle_uses_future_labels": True,
        "policy_fill_accepted_state_count": int(policy_accepted.sum()),
        "oracle_fill_accepted_state_count": int(oracle_accepted.sum()),
        "new3_used": False,
        "new4_used": False,
        "new5_used": False,
    }
    threshold_grid = pd.DataFrame(
        [
            {
                "split": d.VAL_SPLIT,
                "capacity_id": row["capacity_id"],
                "max_iter": int(max_iter),
                **threshold_info,
            }
        ]
    )
    return row, threshold_grid


def plot_capacity(out: pd.DataFrame) -> None:
    fig, ax = d.plt.subplots(figsize=(7, 4.5))
    ordered = out.sort_values("structural_work_units")
    ax.plot(ordered["structural_work_units"], ordered["matched_fill_capture_ratio"], marker="o")
    labels = (
        ordered.groupby(["structural_work_units", "matched_fill_capture_ratio"], as_index=False)["max_iter"]
        .apply(lambda s: "/".join(str(int(v)) for v in s))
        .reset_index(drop=True)
    )
    for _, row in labels.iterrows():
        ax.annotate(str(row["max_iter"]), (row["structural_work_units"], row["matched_fill_capture_ratio"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("structural work units")
    ax.set_ylabel("matched-fill capture ratio")
    ax.set_title("HGB capacity vs matched-fill capture ratio")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(d.FIGURES / "hgb_capacity_vs_capture_ratio.png", dpi=180)
    d.plt.close(fig)


def main() -> None:
    d.ensure_dirs()
    payload = d.load_val_payload()
    rows = []
    grids = []
    for max_iter in CAPACITY_GRID:
        row, grid = run_capacity_point(payload, max_iter)
        rows.append(row)
        grids.append(grid)
        print(
            f"diagnostic_b max_iter={max_iter} work={row['structural_work_units']:.2f} "
            f"capture={row['matched_fill_capture_ratio']:.6f}"
        )
    out = pd.DataFrame(rows)
    thresholds = pd.concat(grids, ignore_index=True)
    out.to_csv(d.RESULTS / "hgb_capacity_matched_fill_capture.csv", index=False)
    thresholds.to_csv(d.RESULTS / "hgb_capacity_threshold_grids.csv", index=False)
    plot_capacity(out)
    d.write_combined_report()
    print(d.ROOT / "MATCHED_FILL_SEPARABILITY_CAPACITY_DIAGNOSTICS_REPORT.md")


if __name__ == "__main__":
    main()
