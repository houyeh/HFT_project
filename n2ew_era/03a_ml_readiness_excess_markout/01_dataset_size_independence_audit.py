from __future__ import annotations

import numpy as np
import pandas as pd

from common_03a import ACTION_SIDES, RESULTS, SPLITS, TARGET, autocorr, effective_sample_size, ensure_dirs, part_paths, read_part


def main() -> None:
    ensure_dirs()
    size_rows = []
    indep_rows = []
    cols = ["split", "session_id", "event_index", "action_side", TARGET]
    for split in SPLITS:
        for path in part_paths(split):
            df = read_part(path, cols)
            if df.empty:
                continue
            session_id = str(df["session_id"].iloc[0])
            for side in ACTION_SIDES:
                side_df = df[df["action_side"].astype(str) == side].sort_values("event_index")
                values = pd.to_numeric(side_df[TARGET], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(values)
                n = int(len(values))
                positive = int((values[finite] > 0.0).sum())
                negative = int((values[finite] < 0.0).sum())
                zero = int((values[finite] == 0.0).sum())
                size_rows.append(
                    {
                        "split": split,
                        "session_id": session_id,
                        "action_side": side,
                        "row_count": n,
                        "finite_label_count": int(finite.sum()),
                        "positive_count": positive,
                        "negative_count": negative,
                        "zero_count": zero,
                        "positive_rate": positive / max(int(finite.sum()), 1),
                        "negative_rate": negative / max(int(finite.sum()), 1),
                        "zero_rate": zero / max(int(finite.sum()), 1),
                    }
                )
                r1 = autocorr(values, 1)
                r5 = autocorr(values, 5)
                r20 = autocorr(values, 20)
                indep_rows.append(
                    {
                        "split": split,
                        "session_id": session_id,
                        "action_side": side,
                        "label": TARGET,
                        "row_count": n,
                        "lag1_autocorr": r1,
                        "lag5_autocorr": r5,
                        "lag20_autocorr": r20,
                        "effective_sample_size_lag1": effective_sample_size(n, r1),
                        "effective_sample_ratio_lag1": effective_sample_size(n, r1) / n if n else np.nan,
                        "high_overlap_warning": bool(np.isfinite(r20) and r20 > 0.5),
                    }
                )
        print(f"dataset size and independence audited split={split}")

    size = pd.DataFrame(size_rows)
    split_summary = (
        size.groupby(["split", "action_side"], as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            finite_label_count=("finite_label_count", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            zero_count=("zero_count", "sum"),
            unique_sessions=("session_id", "nunique"),
        )
    )
    split_summary["positive_rate"] = split_summary["positive_count"] / split_summary["finite_label_count"].replace(0, np.nan)
    split_summary["negative_rate"] = split_summary["negative_count"] / split_summary["finite_label_count"].replace(0, np.nan)
    split_summary["zero_rate"] = split_summary["zero_count"] / split_summary["finite_label_count"].replace(0, np.nan)
    split_summary.to_csv(RESULTS / "ml_dataset_size_by_split.csv", index=False)
    size.to_csv(RESULTS / "ml_dataset_size_by_session.csv", index=False)
    pd.DataFrame(indep_rows).to_csv(RESULTS / "session_independence_audit.csv", index=False)
    print("Wrote dataset size and independence outputs.")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()

