from __future__ import annotations

import numpy as np
import pandas as pd

from common_03a import (
    ACTION_SIDES,
    DRIFT_FEATURES,
    RESULTS,
    SPLITS,
    TARGET,
    available_columns,
    collect_sample,
    ensure_dirs,
    ks_stat,
    psi,
    stats_for_series,
)


def stat_rows(samples: dict[str, pd.DataFrame], columns: list[str], kind: str) -> pd.DataFrame:
    rows = []
    train = samples.get("train", pd.DataFrame())
    for col in columns:
        if col not in train:
            rows.append(
                {
                    "kind": kind,
                    "feature_or_label": col,
                    "split": "__ALL__",
                    "status": "missing_from_phase02f_dataset",
                }
            )
            continue
        train_values = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy(dtype=float)
        for split, sample in samples.items():
            if col not in sample:
                rows.append({"kind": kind, "feature_or_label": col, "split": split, "status": "missing_from_phase02f_dataset"})
                continue
            values = pd.to_numeric(sample[col], errors="coerce").dropna().to_numpy(dtype=float)
            stats = stats_for_series(sample[col])
            rows.append(
                {
                    "kind": kind,
                    "feature_or_label": col,
                    "split": split,
                    "status": "ok",
                    **stats,
                    "ks_vs_train": 0.0 if split == "train" else ks_stat(train_values, values),
                    "psi_vs_train": 0.0 if split == "train" else psi(train_values, values),
                    "sample_based": True,
                    "sample_note": "Stats/KS/PSI are computed on deterministic per-part samples to avoid loading the full action universe into RAM.",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    available_features = available_columns(DRIFT_FEATURES)
    feature_cols = ["split", "session_id", "action_side", "action_slot", *available_features]
    feature_samples = {}
    for split in SPLITS:
        sample = collect_sample(split, feature_cols, per_part=80_000, action_sides=["bid"])
        feature_samples[split] = sample
        print(f"feature drift sample split={split} rows={len(sample):,}")
    feature_out = stat_rows(feature_samples, DRIFT_FEATURES, kind="causal_feature")
    feature_out.to_csv(RESULTS / "feature_drift_for_ml.csv", index=False)

    label_rows = []
    label_cols = ["split", "session_id", "action_side", TARGET]
    for side in ACTION_SIDES:
        label_samples = {}
        for split in SPLITS:
            sample = collect_sample(split, label_cols, per_part=60_000, action_sides=[side])
            label_samples[split] = sample
        side_stats = stat_rows(label_samples, [TARGET], kind=f"label_{side}")
        side_stats["action_side"] = side
        label_rows.append(side_stats)
        print(f"label drift sampled side={side}")
    label_out = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    label_out.to_csv(RESULTS / "label_drift_for_ml.csv", index=False)
    print("Feature and label drift outputs complete.")
    print(feature_out[feature_out["split"].astype(str).isin(["test", "new4", "new3"])][["feature_or_label", "split", "ks_vs_train", "psi_vs_train", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

