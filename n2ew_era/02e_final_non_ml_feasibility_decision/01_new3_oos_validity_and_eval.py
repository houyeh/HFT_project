from __future__ import annotations

import pandas as pd

from common_phase02e import (
    NEW3_RAW_DIR,
    PHASE01_RESULTS,
    RESULTS,
    ROOT,
    SELECTED_HORIZON,
    build_new3_action_universe,
    ensure_dirs,
    hash_int64,
    iter_event_frames,
    matched_random_trial_pnls,
    part_paths,
    policy_set,
    summarize_pnl,
    write_markdown,
)


def existing_session_signatures() -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test", "new4"]:
        for frame in iter_event_frames(split, horizons=[SELECTED_HORIZON]):
            rows.append(
                {
                    "split": split,
                    "session_id": str(frame["session_id"].iloc[0]),
                    "row_count": int(len(frame)),
                    "timestamp_min": int(frame["timestamp"].min()),
                    "timestamp_max": int(frame["timestamp"].max()),
                    "timestamp_hash": hash_int64(frame["timestamp"].to_numpy(dtype="int64")),
                }
            )
    return pd.DataFrame(rows)


def build_oos_validity_report(new3_report: pd.DataFrame) -> pd.DataFrame:
    existing = existing_session_signatures()
    new3 = new3_report.rename(columns={"eligible_timestamp_hash": "timestamp_hash"}).copy()
    new3_sig = new3[["split", "session_id", "quote_eligible_rows", "timestamp_hash"]].rename(
        columns={"quote_eligible_rows": "row_count"}
    )
    session_overlap = set(new3_sig["session_id"].astype(str)) & set(existing["session_id"].astype(str))
    hash_overlap = set(new3_sig["timestamp_hash"].astype(str)) & set(existing["timestamp_hash"].astype(str))
    rows = []
    for row in new3_sig.itertuples(index=False):
        rows.append(
            {
                "new3_session_id": row.session_id,
                "new3_quote_eligible_rows": int(row.row_count),
                "session_id_overlaps_existing": str(row.session_id) in session_overlap,
                "timestamp_hash_overlaps_existing": str(row.timestamp_hash) in hash_overlap,
                "classification": "valid_frozen_oos"
                if str(row.session_id) not in session_overlap and str(row.timestamp_hash) not in hash_overlap
                else "possible_duplicate",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new3_dedup_audit.csv", index=False)
    existing.to_csv(RESULTS / "reference_session_signatures.csv", index=False)
    return out


def evaluate_new3() -> pd.DataFrame:
    rows = []
    policies = policy_set()
    for policy_id in ["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"]:
        rows.append(summarize_pnl(policies[policy_id], "new3", horizon=SELECTED_HORIZON))
    out = pd.DataFrame(rows)

    random_rows = []
    for policy_id in ["old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"]:
        trials = matched_random_trial_pnls("new3", policies[policy_id], SELECTED_HORIZON, trials=1000)
        random_rows.append(
            {
                "policy_id": policy_id,
                "matched_random_mean": float(trials["random_pnl"].mean()),
                "matched_random_95p": float(trials["random_pnl"].quantile(0.95)),
                "matched_random_99p": float(trials["random_pnl"].quantile(0.99)),
                "empirical_p_value_random_ge_model": float((trials["random_pnl"] >= trials["model_pnl"].iloc[0]).mean()),
                "random_trials": int(len(trials)),
            }
        )
    random = pd.DataFrame(random_rows)
    out = out.merge(random, on="policy_id", how="left")
    out["beats_matched_random_95p"] = out["total_pnl"].astype(float) > out["matched_random_95p"].astype(float)
    out["selection_split"] = "val"
    out["new3_used_for_selection"] = False
    out.to_csv(RESULTS / "new3_frozen_oos_results.csv", index=False)
    return out


def display_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "empirical_p_value_random_ge_model" not in df:
        return df
    out = df.copy()
    out["empirical_p_value_random_ge_model"] = out["empirical_p_value_random_ge_model"].map(
        lambda value: "<0.001" if pd.notna(value) and float(value) == 0.0 else value
    )
    return out


def main() -> None:
    ensure_dirs()
    if not NEW3_RAW_DIR.exists():
        write_markdown(
            ROOT / "NEW3_OOS_VALIDITY_REPORT.md",
            [
                "# NEW3 OOS Validity Report",
                "",
                f"NEW3 raw directory was not found: `{NEW3_RAW_DIR}`.",
                "",
                "NEW3 was therefore not evaluated.",
            ],
        )
        pd.DataFrame().to_csv(RESULTS / "new3_frozen_oos_results.csv", index=False)
        return

    new3_report = build_new3_action_universe()
    audit = build_oos_validity_report(new3_report)
    valid = bool((audit["classification"] == "valid_frozen_oos").all()) if not audit.empty else False
    eval_df = evaluate_new3() if valid else pd.DataFrame()

    phase01_counts = pd.read_csv(PHASE01_RESULTS / "action_universe_by_split.csv")
    lines = [
        "# NEW3 OOS Validity Report",
        "",
        "NEW3 was loaded from raw L2 CSV files and converted with the same Phase 01 full-action construction: every quote-eligible row emits bid, ask, and no_quote actions. NEW3 was not used for validation selection or any parameter tuning.",
        "",
        "## NEW3 Source Filter Counts",
        "",
        new3_report.to_markdown(index=False),
        "",
        "## Dedup Audit",
        "",
        "Deduplication checks compare `session_id` and SHA-256 hashes of the eligible timestamp sequence against train/val/test/NEW4 action-universe sessions.",
        "",
        audit.to_markdown(index=False),
        "",
        f"NEW3 frozen OOS classification: `{'valid_frozen_oos' if valid else 'possible_duplicate_or_invalid'}`.",
        "",
        "## Existing Phase 01 Split Counts",
        "",
        phase01_counts.to_markdown(index=False),
        "",
        "## Frozen NEW3 Policy Evaluation",
        "",
        display_pvalues(eval_df).to_markdown(index=False) if not eval_df.empty else "NEW3 evaluation skipped because the OOS validity audit did not pass.",
    ]
    write_markdown(ROOT / "NEW3_OOS_VALIDITY_REPORT.md", lines)
    print("NEW3 audit and frozen evaluation complete.")
    if not eval_df.empty:
        print(eval_df[["policy_id", "total_pnl", "selected_action_count", "matched_random_95p", "beats_matched_random_95p"]].to_string(index=False))


if __name__ == "__main__":
    main()
