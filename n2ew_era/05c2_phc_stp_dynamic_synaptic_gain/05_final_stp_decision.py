from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    code = pd.read_csv(RESULTS / "phc_stp_code_audit_checklist.csv") if (RESULTS / "phc_stp_code_audit_checklist.csv").exists() else pd.DataFrame()
    original_oos = pd.read_csv(RESULTS / "phc_stp_vs_v1_hgb_queue.csv") if (RESULTS / "phc_stp_vs_v1_hgb_queue.csv").exists() else pd.DataFrame()
    original_causal = pd.read_csv(RESULTS / "phc_stp_causal_fill_vs_v1.csv") if (RESULTS / "phc_stp_causal_fill_vs_v1.csv").exists() else pd.DataFrame()
    original_oracle = pd.read_csv(RESULTS / "phc_stp_window_oracle_capture.csv") if (RESULTS / "phc_stp_window_oracle_capture.csv").exists() else pd.DataFrame()
    extended = pd.read_csv(RESULTS / "phc_stp_extended_epoch_sanity.csv") if (RESULTS / "phc_stp_extended_epoch_sanity.csv").exists() else pd.DataFrame()
    seed_summary = pd.read_csv(RESULTS / "phc_stp_extended_seed_summary.csv") if (RESULTS / "phc_stp_extended_seed_summary.csv").exists() else pd.DataFrame()
    gain = pd.read_csv(RESULTS / "phc_stp_gain_statistics.csv") if (RESULTS / "phc_stp_gain_statistics.csv").exists() else pd.DataFrame()
    paired = pd.read_csv(RESULTS / "phc_stp_vs_v1_session_paired.csv") if (RESULTS / "phc_stp_vs_v1_session_paired.csv").exists() else pd.DataFrame()

    code_ok = (not code.empty) and bool((code["status"].eq("PASS")).all())
    if not code_ok:
        status = "stp_code_audit_failed"
    elif extended.empty:
        status = "stp_inconclusive_due_to_epoch_cap"
    else:
        val_ok = bool((extended[extended["split"].eq("val")]["action_delta_vs_v1"] >= 0).all())
        oos_ok = bool((extended[extended["split"].isin(["new3", "new4"])]["action_delta_vs_v1"] >= 0).all())
        causal_ok = bool((extended["causal_delta_vs_v1"] >= 0).all())
        oracle_ok = bool((extended["oracle_capture_delta_vs_v1"] >= 0).all())
        max_epoch_hit = bool((not seed_summary.empty) and seed_summary["max_epoch_hit"].astype(bool).all())
        if val_ok and oos_ok and causal_ok and oracle_ok:
            status = "stp_improves_phc_lite_v1"
        elif val_ok and oos_ok and causal_ok:
            status = "stp_neutral_regularizer_not_adopted"
        elif max_epoch_hit and val_ok and oos_ok and oracle_ok and not causal_ok:
            status = "stp_inconclusive_due_to_epoch_cap"
        else:
            status = "stp_hurts_performance_not_adopted"

    original_compare = pd.DataFrame()
    if not original_oos.empty:
        original_compare = original_oos[original_oos["policy_id"].isin(["phc_lite", "phc_stp"])][
            ["split", "policy_id", "total_excess_pnl_H20", "selected_action_count", "win_rate", "max_drawdown"]
        ]
    original_causal_compare = pd.DataFrame()
    if not original_causal.empty:
        original_causal_compare = original_causal[original_causal["policy_id"].isin(["phc_lite", "phc_stp"])][
            ["split", "policy_id", "sequential_excess_pnl", "fill_count", "win_rate", "max_drawdown"]
        ]
    original_oracle_compare = pd.DataFrame()
    if not original_oracle.empty:
        original_oracle_compare = original_oracle[original_oracle["policy_id"].isin(["phc_lite", "phc_stp"])][
            ["split", "policy_id", "policy_pnl", "capture_ratio", "opportunity_gap"]
        ]
    session_summary = pd.DataFrame()
    if not paired.empty:
        session_summary = paired.groupby(["variant", "split"]).agg(
            sessions=("session_id", "count"),
            stp_winning_sessions=("stp_wins_session", "sum"),
            total_delta_stp_minus_v1=("pnl_delta_stp_minus_v1", "sum"),
            mean_delta_stp_minus_v1=("pnl_delta_stp_minus_v1", "mean"),
        ).reset_index()
    gain_overall = pd.DataFrame()
    if not gain.empty:
        gain_overall = gain[gain["group_type"].isin(["overall_state_mean_gain", "selected_status", "selected_outcome"])]

    write_markdown(
        ROOT / "PHC_STP_FINAL_DECISION_REPORT.md",
        [
            "# PHC STP-like Final Decision Report",
            "",
            f"Final status: `{status}`.",
            "",
            "Decision rule: STP is not adopted unless it matches or beats frozen PHC-lite v1 on validation, NEW3/NEW4 action-level PnL, causal-fill sequential PnL, and oracle capture. NEW3/NEW4/NEW5 are not used for selection. Because the extended run improves action-level/OOS/oracle metrics but still misses causal-fill on NEW4 and hits max_epoch for all seeds, the model remains not adopted.",
            "",
            "## Code Audit",
            "",
            code.to_markdown(index=False) if not code.empty else "Code audit missing.",
            "",
            "## Original 50 Epoch Action-level Comparison",
            "",
            original_compare.to_markdown(index=False) if not original_compare.empty else "Original OOS comparison missing.",
            "",
            "## Original 50 Epoch Causal-fill Comparison",
            "",
            original_causal_compare.to_markdown(index=False) if not original_causal_compare.empty else "Original causal-fill comparison missing.",
            "",
            "## Original 50 Epoch Oracle Capture",
            "",
            original_oracle_compare.to_markdown(index=False) if not original_oracle_compare.empty else "Original oracle capture missing.",
            "",
            "## Extended Epoch Seed Summary",
            "",
            seed_summary.to_markdown(index=False) if not seed_summary.empty else "Extended epoch sanity missing.",
            "",
            "## Extended Epoch Selected Evaluation",
            "",
            extended.to_markdown(index=False) if not extended.empty else "Extended epoch selected evaluation missing.",
            "",
            "## Gain Behavior Highlights",
            "",
            gain_overall.head(30).to_markdown(index=False) if not gain_overall.empty else "Gain behavior audit missing.",
            "",
            "## Session Paired Summary",
            "",
            session_summary.to_markdown(index=False) if not session_summary.empty else "Session paired comparison missing.",
            "",
            "## Claim Boundary",
            "",
            "STP-like dynamic gain is not full biological STP. This phase does not overwrite PHC-lite v1, does not adopt STP, does not use NEW5 for selection, and does not claim real-world profitability, true exchange replay, or FPGA PnL improvement.",
        ],
    )
    print(status)


if __name__ == "__main__":
    main()
