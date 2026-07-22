from __future__ import annotations

import pandas as pd

import common_05d as c
import common_05a as p05a


def build_error_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    c.ensure_dirs()
    row_chunks = []
    summary_rows = []
    session_rows = []
    for split in c.SPLITS:
        payload = p05a.load_sequences(split)
        oracle_action, oracle_pnl, oracle_value = c.oracle_decision_and_pnl(payload)
        sessions = c.session_ids(payload)
        event_index = payload["event_index"].numpy()
        decisions = c.model_decisions(payload)
        for model_id, (decision, scores) in decisions.items():
            err = c.error_type(decision, oracle_action)
            pnl = c.model_pnl(payload, decision)
            selected_score = c.selected_scores(scores, decision)
            chunk = pd.DataFrame(
                {
                    "split": split,
                    "model_id": model_id,
                    "state_id": range(len(decision)),
                    "session_id": sessions,
                    "event_index": event_index,
                    "oracle_action": c.action_name(oracle_action),
                    "model_action": c.action_name(decision),
                    "error_type": err,
                    "oracle_pnl": oracle_pnl,
                    "model_pnl": pnl,
                    "oracle_value": oracle_value,
                    "model_selected_score": selected_score,
                    "new5_used": False,
                }
            )
            row_chunks.append(chunk)
            grouped = chunk.groupby("error_type").agg(
                state_count=("state_id", "count"),
                oracle_pnl=("oracle_pnl", "sum"),
                model_pnl=("model_pnl", "sum"),
                avg_oracle_value=("oracle_value", "mean"),
                avg_model_score=("model_selected_score", "mean"),
            ).reset_index()
            total_states = len(chunk)
            total_oracle = float(chunk["oracle_pnl"].sum())
            total_model = float(chunk["model_pnl"].sum())
            for _, row in grouped.iterrows():
                summary_rows.append(
                    {
                        "split": split,
                        "model_id": model_id,
                        "error_type": row["error_type"],
                        "state_count": int(row["state_count"]),
                        "state_rate": float(row["state_count"] / max(total_states, 1)),
                        "oracle_pnl": float(row["oracle_pnl"]),
                        "model_pnl": float(row["model_pnl"]),
                        "total_oracle_pnl": total_oracle,
                        "total_model_pnl": total_model,
                        "capture_ratio": total_model / total_oracle if total_oracle else 0.0,
                        "avg_oracle_value": float(row["avg_oracle_value"]),
                        "avg_model_score": float(row["avg_model_score"]) if pd.notna(row["avg_model_score"]) else None,
                        "new5_used": False,
                    }
                )
            sess = chunk.groupby(["session_id", "error_type"]).agg(
                state_count=("state_id", "count"),
                oracle_pnl=("oracle_pnl", "sum"),
                model_pnl=("model_pnl", "sum"),
            ).reset_index()
            sess["split"] = split
            sess["model_id"] = model_id
            session_rows.append(sess)
    detail = pd.concat(row_chunks, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    session = pd.concat(session_rows, ignore_index=True)
    detail.to_csv(c.RESULTS / "oracle_error_decomposition.csv", index=False)
    summary.to_csv(c.RESULTS / "oracle_error_by_split_model.csv", index=False)
    session.to_csv(c.RESULTS / "oracle_error_by_session.csv", index=False)
    return detail, summary


def report(summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(index=["split", "model_id"], columns="error_type", values="state_count", fill_value=0, aggfunc="sum").reset_index()
    rates = pivot.copy()
    error_cols = [col for col in rates.columns if col not in ["split", "model_id"]]
    denom = rates[error_cols].sum(axis=1).replace(0, 1)
    for col in error_cols:
        rates[col + "_rate"] = rates[col] / denom
    capture = summary.groupby(["split", "model_id"]).agg(
        total_model_pnl=("total_model_pnl", "first"),
        total_oracle_pnl=("total_oracle_pnl", "first"),
        capture_ratio=("capture_ratio", "first"),
    ).reset_index()
    phc = rates[rates["model_id"].eq("phc_lite_v1")]
    hgb = rates[rates["model_id"].eq("ml_hgb_regressor_shallow_convergence_selected")]
    merged = phc.merge(hgb, on="split", suffixes=("_phc", "_hgb"))
    answers = []
    for split in c.SPLITS:
        row = merged[merged["split"].eq(split)]
        if row.empty:
            continue
        r = row.iloc[0]
        for err in ["missed_opportunity", "false_positive", "side_error"]:
            answers.append(
                {
                    "split": split,
                    "error_type": err,
                    "phc_rate": float(r.get(err + "_rate_phc", 0.0)),
                    "hgb_rate": float(r.get(err + "_rate_hgb", 0.0)),
                    "phc_reduces_vs_hgb": bool(float(r.get(err + "_rate_phc", 0.0)) < float(r.get(err + "_rate_hgb", 0.0))),
                }
            )
    answer_df = pd.DataFrame(answers)
    answer_df.to_csv(c.RESULTS / "oracle_error_phc_vs_hgb_answers.csv", index=False)
    c.write_markdown(
        c.ROOT / "ORACLE_ERROR_DECOMPOSITION_REPORT.md",
        [
            "# Oracle Error Decomposition Report",
            "",
            "Universe: PHC sequence-window terminal states on val/NEW3/NEW4. NEW5 is not used.",
            "Oracle uses future `excess_pnl_H20` labels and is diagnostic only; it is not a tradable strategy and is not used for model selection.",
            "",
            "## Capture Ratios",
            "",
            capture.to_markdown(index=False),
            "",
            "## Error Counts by Split / Model",
            "",
            pivot.to_markdown(index=False),
            "",
            "## PHC vs HGB Error Rate Comparison",
            "",
            answer_df.to_markdown(index=False),
            "",
            "Required interpretation: dominant PHC gap types should be read from `oracle_error_by_split_model.csv`; session concentration is in `oracle_error_by_session.csv`.",
        ],
    )


def main() -> None:
    _, summary = build_error_rows()
    report(summary)
    print("oracle_error_decomposition_complete")


if __name__ == "__main__":
    main()
