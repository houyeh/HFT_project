from __future__ import annotations

import pandas as pd

from root_cause_utils import RESULTS_DIR, alignment_metrics, ensure_dirs, load_context, selected_quotes_for_policy, write_markdown


POLICIES = ["015", "016_v2", "020", "021", "spread_only"]


def run() -> None:
    ensure_dirs()
    frame, actions = load_context("test")
    side_rows = []
    session_rows = []
    spread_rows = []
    joint_rows = []
    overall_rows = []
    for policy in POLICIES:
        selected = selected_quotes_for_policy(frame, actions, policy)
        side_rows.append(alignment_metrics(actions, selected, policy, ["side"]))
        session_rows.append(alignment_metrics(actions, selected, policy, ["session_id"]))
        spread_rows.append(alignment_metrics(actions, selected, policy, ["spread_bucket"]))
        joint_rows.append(alignment_metrics(actions, selected, policy, ["session_id", "side", "spread_bucket"]))
        overall_rows.append(alignment_metrics(actions, selected, policy, []))

    by_side = pd.concat(side_rows, ignore_index=True)
    by_side.to_csv(RESULTS_DIR / "model_oracle_alignment_by_side.csv", index=False)
    pd.concat(session_rows, ignore_index=True).to_csv(RESULTS_DIR / "model_oracle_alignment_by_session.csv", index=False)
    pd.concat(spread_rows, ignore_index=True).to_csv(RESULTS_DIR / "model_oracle_alignment_by_spread_bucket.csv", index=False)
    pd.concat(joint_rows, ignore_index=True).to_csv(RESULTS_DIR / "model_oracle_alignment_by_session_side_spread.csv", index=False)
    overall = pd.concat(overall_rows, ignore_index=True)
    overall.to_csv(RESULTS_DIR / "model_oracle_alignment_overall.csv", index=False)

    best_recall = overall.sort_values("oracle_recall", ascending=False).iloc[0]
    best_pnl = overall.sort_values("selected_quote_pnl", ascending=False).iloc[0]
    write_markdown(
        RESULTS_DIR / "model_oracle_alignment_report.md",
        [
            "# Model-Oracle Alignment Report",
            "",
            "Comparison uses action-level selected quotes versus `oracle_positive_quote` under moderate stress.",
            "For cancel/block policies, `selected_quote` is the action that survives the cancel mask.",
            "`spread_only` is implemented as a validation-selected raw `spread_ticks` gate because the 022 sklearn pickle is version-incompatible in this runtime; test is not used for that threshold.",
            "",
            f"- Best oracle recall: `{best_recall['policy']}` at `{best_recall['oracle_recall']:.4f}`.",
            f"- Best selected quote PnL: `{best_pnl['policy']}` at `{best_pnl['selected_quote_pnl']:.3f}`.",
            "",
            "Required fields are included in every split table: oracle recall, oracle precision, captured oracle PnL, missed oracle PnL, false positive loss, avoided bad PnL, and missed good PnL.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote model-oracle alignment outputs to {RESULTS_DIR}")
