from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common_phase03b import (
    DATA,
    RESULTS,
    SPLITS,
    TRAIN_SAMPLE_ACTIONS,
    build_feature_frame,
    ensure_dirs,
    feature_matrix_columns,
    part_paths,
    save_feature_matrix,
    train_matrix_path,
    val_matrix_path,
    write_json,
)


def deterministic_train_sample() -> pd.DataFrame:
    paths = part_paths("train")
    if not paths:
        raise RuntimeError("No train parts found.")
    per_part = int(math.ceil(TRAIN_SAMPLE_ACTIONS / len(paths)))
    frames = []
    rows = []
    for idx, path in enumerate(paths):
        frame = build_feature_frame(path, include_targets=True)
        sampled_parts = []
        for action_side, group in frame.groupby("action_side", sort=False):
            take = min(len(group), max(1, per_part // 3))
            sampled_parts.append(group.sample(n=take, random_state=20260601 + idx + int(group["action_slot"].iloc[0])))
        sample = pd.concat(sampled_parts, ignore_index=True)
        frames.append(sample)
        rows.append(
            {
                "split": "train",
                "part": path.name,
                "source_rows": len(frame),
                "sample_rows": len(sample),
                "positive_rate_sample": float(sample["positive_excess_pnl_H20"].mean()),
                "mean_target_sample": float(sample["excess_pnl_H20"].mean()),
            }
        )
        print(f"sampled train part={path.name} rows={len(sample)}")
    out = pd.concat(frames, ignore_index=True)
    if len(out) > TRAIN_SAMPLE_ACTIONS:
        out = out.sample(n=TRAIN_SAMPLE_ACTIONS, random_state=20260601).reset_index(drop=True)
    pd.DataFrame(rows).to_csv(RESULTS / "train_sampling_manifest.csv", index=False)
    return out


def full_split_matrix(split: str) -> pd.DataFrame:
    frames = []
    for path in part_paths(split):
        frame = build_feature_frame(path, include_targets=True)
        frames.append(frame)
        print(f"loaded {split} part={path.name} rows={len(frame)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ensure_dirs()
    features = feature_matrix_columns()
    train = deterministic_train_sample()
    val = full_split_matrix("val")
    save_feature_matrix(train, train_matrix_path())
    save_feature_matrix(val, val_matrix_path())
    manifest = pd.DataFrame(
        [
            {
                "matrix": "train",
                "path": str(train_matrix_path()),
                "rows": len(train),
                "feature_count": len(features),
                "target": "excess_pnl_H20",
                "positive_rate": float(train["positive_excess_pnl_H20"].mean()),
                "target_mean": float(train["excess_pnl_H20"].mean()),
                "source": "deterministic_train_only_sample",
            },
            {
                "matrix": "val",
                "path": str(val_matrix_path()),
                "rows": len(val),
                "feature_count": len(features),
                "target": "excess_pnl_H20",
                "positive_rate": float(val["positive_excess_pnl_H20"].mean()),
                "target_mean": float(val["excess_pnl_H20"].mean()),
                "source": "full_validation_split",
            },
        ]
    )
    manifest.to_csv(RESULTS / "feature_matrix_manifest.csv", index=False)
    write_json(
        DATA / "feature_matrix_config.json",
        {
            "train_sample_actions": TRAIN_SAMPLE_ACTIONS,
            "train_sampling": "deterministic_stratified_by_action_side_within_train_only",
            "validation": "full_validation_split",
            "feature_columns": features,
            "test_new3_new4_used_for_training_or_selection": False,
        },
    )
    print("Feature matrix build complete.")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()

