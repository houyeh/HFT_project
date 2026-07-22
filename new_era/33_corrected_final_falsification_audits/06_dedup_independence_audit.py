from __future__ import annotations

from itertools import combinations

import pandas as pd

from common_033 import RESULTS, ensure_dirs, load_unused_oos_actions, row_price_sequence_hash


def classify_pair(common_sessions: int, identical_hashes: int, row_overlap_rate: float) -> str:
    if common_sessions > 0 and common_sessions == identical_hashes:
        return "likely_duplicate"
    if row_overlap_rate >= 0.90:
        return "likely_duplicate"
    if common_sessions == 0 and row_overlap_rate <= 0.01:
        return "independent"
    return "uncertain"


def main() -> None:
    ensure_dirs()
    actions = load_unused_oos_actions()
    session_hashes = row_price_sequence_hash(actions)
    session_hashes.to_csv(RESULTS / "unused_oos_price_sequence_hashes.csv", index=False)
    dirs = sorted(session_hashes["data_dir"].astype(str).unique())
    rows = []
    for left, right in combinations(dirs, 2):
        a = session_hashes[session_hashes["data_dir"].astype(str) == left].copy()
        b = session_hashes[session_hashes["data_dir"].astype(str) == right].copy()
        merged = a.merge(
            b,
            on=["symbol", "session_id"],
            suffixes=("_left", "_right"),
            how="inner",
        )
        common_sessions = int(len(merged))
        identical_hashes = int((merged["price_sequence_hash_left"] == merged["price_sequence_hash_right"]).sum()) if common_sessions else 0
        a_rows = set(
            actions.loc[actions["data_dir"].astype(str) == left, ["symbol", "session_id", "local_row_id"]]
            .drop_duplicates()
            .apply(lambda r: (r["symbol"], r["session_id"], int(r["local_row_id"])), axis=1)
            .tolist()
        )
        b_rows = set(
            actions.loc[actions["data_dir"].astype(str) == right, ["symbol", "session_id", "local_row_id"]]
            .drop_duplicates()
            .apply(lambda r: (r["symbol"], r["session_id"], int(r["local_row_id"])), axis=1)
            .tolist()
        )
        union = len(a_rows | b_rows)
        inter = len(a_rows & b_rows)
        row_overlap_rate = float(inter / union) if union else 0.0
        rows.append(
            {
                "data_dir_left": left,
                "data_dir_right": right,
                "left_session_count": int(len(a)),
                "right_session_count": int(len(b)),
                "common_symbol_session_count": common_sessions,
                "identical_price_sequence_hash_count": identical_hashes,
                "price_sequence_hash_match_rate_on_common_sessions": float(identical_hashes / common_sessions) if common_sessions else 0.0,
                "row_key_overlap_count": inter,
                "row_key_union_count": union,
                "row_key_overlap_rate": row_overlap_rate,
                "classification": classify_pair(common_sessions, identical_hashes, row_overlap_rate),
                "interpretation": "Do not sum PnL as independent OOS when classification is likely_duplicate.",
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "dedup_independence_audit.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote dedup independence audit to {RESULTS}")
