from __future__ import annotations

import pandas as pd

from common_030 import EXPORT, FEATURE_COLUMNS, ensure_dirs, load_028_split, prepare_export_frame, write_markdown


SEQUENCE_LENGTH = 64


def make_sequence_index(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for split, df in split_frames.items():
        for i, row in df.reset_index(drop=True).iterrows():
            target = int(row["target_row_id"])
            start = max(0, target - SEQUENCE_LENGTH + 1)
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "session_id": str(row["session_id"]),
                    "start_row_id": start,
                    "end_row_id": target,
                    "target_row_id": target,
                    "sequence_length_requested": SEQUENCE_LENGTH,
                    "sequence_length_available": int(target - start + 1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    exported: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test", "new4"]:
        df = prepare_export_frame(load_028_split(split), split)
        exported[split] = df
        df.to_parquet(EXPORT / f"{split}.parquet", index=False)
    make_sequence_index(exported).to_parquet(EXPORT / "sequence_index.parquet", index=False)

    schema_rows = []
    for split, df in exported.items():
        schema_rows.append(
            {
                "split": split,
                "rows": int(len(df)),
                "negative_ev_rate": float(df["negative_ev_moderate"].mean()) if len(df) else 0.0,
                "severe_negative_ev_rate_le_minus_5000": float(df["severe_negative_ev_moderate"].mean()) if len(df) else 0.0,
                "quote_ev_sum": float(df["quote_ev_moderate"].sum()),
            }
        )
    pd.DataFrame(schema_rows).to_csv(EXPORT / "split_summary.csv", index=False)

    readme = [
        "# GPU Training Pack for Phase 030",
        "",
        "Purpose: train deeper residual Quote-EV veto models outside this workstation, for example on an RTX 4050 Laptop GPU.",
        "",
        "Files:",
        "",
        "- `train.parquet`, `val.parquet`, `test.parquet`, `new4.parquet`: baseline-universe quote samples selected by `bid_only_spread_ge_1`.",
        "- `sequence_index.parquet`: past-only sequence pointers. It stores row ranges only and does not expand full sequences into RAM.",
        "- `split_summary.csv`: counts and label rates.",
        "",
        "Targets:",
        "",
        "- Classification: `negative_ev_moderate`.",
        "- Regression: `quote_ev_moderate`.",
        "- Optional severe-loss target: `severe_negative_ev_moderate` where `quote_ev_moderate <= -5000`.",
        "",
        "Feature columns used by the local tabular audit:",
        "",
        "```text",
        ", ".join(FEATURE_COLUMNS),
        "```",
        "",
        "Critical split rule:",
        "",
        "- Train models on `train.parquet`.",
        "- Select thresholds and hyperparameters on `val.parquet` only.",
        "- Evaluate `test.parquet` and `new4.parquet` only once per frozen policy.",
        "- NEW4 is external OOS evidence only and must not be used for tuning.",
        "",
        "Sequence note:",
        "",
        "`sequence_index.parquet` contains `[sample_id, start_row_id, end_row_id, target_row_id]` using the original per-session `local_row_id`. A sequence loader should reconstruct past rows from the raw feature frame and must never include rows after `target_row_id`.",
    ]
    write_markdown(EXPORT / "README_GPU_TRAINING.md", readme)

    env = [
        "name: hft-ev-veto-gpu",
        "channels:",
        "  - pytorch",
        "  - nvidia",
        "  - conda-forge",
        "dependencies:",
        "  - python=3.11",
        "  - pandas",
        "  - pyarrow",
        "  - numpy",
        "  - scikit-learn",
        "  - pytorch",
        "  - pytorch-cuda=12.1",
        "  - matplotlib",
        "  - tqdm",
        "  - pip",
    ]
    (EXPORT / "environment.yml").write_text("\n".join(env) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    print(f"Exported GPU training pack to {EXPORT}")

