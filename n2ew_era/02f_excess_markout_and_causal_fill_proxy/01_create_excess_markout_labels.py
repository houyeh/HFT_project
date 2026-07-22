from __future__ import annotations

import pandas as pd

from common_phase02f import (
    ALL_SPLITS,
    BASE_COLUMNS,
    HORIZONS,
    RESULTS,
    add_excess_columns,
    ensure_dirs,
    reset_excess_dataset,
    source_part_paths,
    write_excess_part,
    write_json,
)


def summarize_part(frame: pd.DataFrame, split: str, session_id: str) -> list[dict]:
    rows = []
    for horizon in HORIZONS:
        for side in ["bid", "ask", "no_quote"]:
            side_frame = frame[frame["action_side"].astype(str) == side]
            values = pd.to_numeric(side_frame[f"excess_pnl_H{horizon}"], errors="coerce")
            finite = values.dropna()
            rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "action_side": side,
                    "horizon": int(horizon),
                    "row_count": int(len(side_frame)),
                    "finite_label_count": int(len(finite)),
                    "mean_excess_pnl": float(finite.mean()) if len(finite) else 0.0,
                    "median_excess_pnl": float(finite.median()) if len(finite) else 0.0,
                    "positive_rate": float((finite > 0.0).mean()) if len(finite) else 0.0,
                    "negative_rate": float((finite < 0.0).mean()) if len(finite) else 0.0,
                }
            )
    return rows


def main() -> None:
    ensure_dirs()
    reset_excess_dataset()
    summary_rows = []
    manifest_rows = []
    read_cols = [*BASE_COLUMNS, *[f"future_mid_H{h}" for h in HORIZONS]]
    for split in ALL_SPLITS:
        paths = source_part_paths(split)
        for path in paths:
            raw = pd.read_parquet(path, columns=read_cols)
            session_id = str(raw["session_id"].iloc[0]) if len(raw) else path.stem.replace("part-", "")
            out = add_excess_columns(raw)
            target = write_excess_part(out, split=split, session_id=session_id)
            summary_rows.extend(summarize_part(out, split, session_id))
            manifest_rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "source_path": str(path),
                    "target_path": str(target),
                    "action_rows": int(len(out)),
                }
            )
            print(f"excess labels built split={split} session={session_id} rows={len(out):,}")
    summary = pd.DataFrame(summary_rows)
    split_summary = (
        summary.groupby(["split", "action_side", "horizon"], as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            finite_label_count=("finite_label_count", "sum"),
            mean_excess_pnl=("mean_excess_pnl", "mean"),
            median_excess_pnl=("median_excess_pnl", "mean"),
            positive_rate=("positive_rate", "mean"),
            negative_rate=("negative_rate", "mean"),
        )
    )
    summary.to_csv(RESULTS / "excess_label_summary_by_session.csv", index=False)
    split_summary.to_csv(RESULTS / "excess_label_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(RESULTS / "excess_label_build_manifest.csv", index=False)
    write_json(
        RESULTS / "excess_label_definition.json",
        {
            "formula_bid": "excess_bid_pnl_H = (future_mid_H - mid_t) * point_value - maker_fee",
            "formula_ask": "excess_ask_pnl_H = (mid_t - future_mid_H) * point_value - maker_fee",
            "formula_no_quote": "excess_pnl_H = 0",
            "point_value": 200.0,
            "maker_fee": 15.0,
            "mechanical_half_spread_removed": True,
            "future_mid_used_only_as_label": True,
        },
    )
    print("Excess markout label build complete.")
    print(split_summary[split_summary["horizon"].astype(int) == 20].to_string(index=False))


if __name__ == "__main__":
    main()

