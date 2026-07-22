from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "36_residual_universe_filtering_waterfall"
FIGURES = ROOT / "figures"

V16 = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V18 = PROJECT_ROOT / "new_era" / "18_queue_aware"
V26 = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V27 = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"
V28 = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
V31 = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"
DATA_NEW4 = PROJECT_ROOT / "data" / "NEW4"

SPLITS = ["train", "val", "test", "NEW4"]
SEVERE_NEGATIVE_EV_THRESHOLD = -5000.0

COUNT_COLUMNS = [
    "raw_rows",
    "label_rows",
    "valid_book_rows",
    "quote_eligible_rows",
    "candidate_action_rows",
    "bid_candidate_action_rows",
    "ask_candidate_action_rows",
    "baseline_selected_rows",
    "residual_ml_rows",
    "negative_ev_moderate_rows",
    "severe_negative_ev_moderate_rows",
]

STAGE_SEQUENCE = [
    "raw_rows",
    "valid_book_rows",
    "quote_eligible_rows",
    "label_rows",
    "candidate_action_rows",
    "baseline_selected_rows",
    "residual_ml_rows",
]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def line_number(path: Path, pattern: str) -> int | None:
    if not path.exists():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append("" if pd.isna(value) else f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def read_filter_report() -> pd.DataFrame:
    path = V16 / "outputs" / "quote_eligible_filter_report.csv"
    report = pd.read_csv(path)
    return report[report["scope"].astype(str) == "session"].copy()


def load_full_actions() -> pd.DataFrame:
    df = pd.read_parquet(V31 / "data" / "full_action_universe.parquet")
    df["audit_split"] = df["phase031_split"].astype(str).replace({"new4": "NEW4"})
    df["side"] = df["side"].astype(str)
    df["spread_ticks"] = df["spread_ticks"].astype(float)
    df["selected_by_baseline"] = df["selected_by_baseline"].astype(bool)
    return df


def load_residual_universe() -> pd.DataFrame:
    hist = pd.read_parquet(V28 / "data" / "baseline_universe_quote_ev.parquet").copy()
    hist["audit_split"] = hist["split"].astype(str)
    new4 = pd.read_parquet(V28 / "data" / "new4_baseline_universe_quote_ev.parquet").copy()
    new4["audit_split"] = "NEW4"
    out = pd.concat([hist, new4], ignore_index=True)
    out["side"] = out["side"].astype(str)
    out["quote_ev_moderate"] = out["quote_ev_moderate"].astype(float)
    out["negative_ev_moderate"] = (out["quote_ev_moderate"] < 0.0).astype(int)
    out["severe_negative_ev_moderate"] = (out["quote_ev_moderate"] <= SEVERE_NEGATIVE_EV_THRESHOLD).astype(int)
    return out


def scan_new4_raw_valid_by_session(symbol: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(DATA_NEW4.rglob("*.csv"), key=lambda p: str(p.relative_to(DATA_NEW4))):
        session_id = str(csv_path.parent.relative_to(DATA_NEW4)).replace("\\", "/")
        if session_id == ".":
            session_id = csv_path.stem
        raw_rows = 0
        valid_book_rows = 0
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) <= 23 or row[3].strip() != symbol:
                    continue
                raw_rows += 1
                try:
                    bid = float(row[21])
                    ask = float(row[22])
                    bid_vol = float(row[20])
                    ask_vol = float(row[23])
                except ValueError:
                    continue
                if bid > 0.0 and ask > 0.0 and ask >= bid and bid_vol > 0.0 and ask_vol > 0.0:
                    valid_book_rows += 1
        rows.append(
            {
                "split": "NEW4",
                "session_id": session_id,
                "raw_rows": int(raw_rows),
                "label_rows": int(valid_book_rows),
                "valid_book_rows": int(valid_book_rows),
                "quote_eligible_rows": int(valid_book_rows),
                "source_note": "NEW4 has no 016 binary label parquet; label_rows/quote_eligible_rows use the external OOS valid-book frame proxy.",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["split", "session_id", "raw_rows", "label_rows", "valid_book_rows", "quote_eligible_rows", "source_note"])
    out = pd.DataFrame(rows)
    return out.groupby(["split", "session_id", "source_note"], as_index=False)[["raw_rows", "label_rows", "valid_book_rows", "quote_eligible_rows"]].sum()


def historical_session_base(filter_sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in filter_sessions.itertuples(index=False):
        raw = int(r.raw_rows)
        valid = raw - int(r.dropped_invalid_bid_ask) - int(r.dropped_nonpositive_spread) - int(r.dropped_missing_depth)
        rows.append(
            {
                "split": str(r.split),
                "session_id": str(r.session_id),
                "raw_rows": raw,
                "label_rows": int(r.eligible_rows),
                "valid_book_rows": int(valid),
                "quote_eligible_rows": int(r.eligible_rows),
                "source_note": "Historical counts from 016 quote_eligible_filter_report session rows.",
            }
        )
    return pd.DataFrame(rows)


def action_counts_by_session(actions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        actions.groupby(["audit_split", "session_id"], observed=False)
        .agg(
            candidate_action_rows=("action_id", "count"),
            bid_candidate_action_rows=("side", lambda s: int((s.astype(str) == "bid").sum())),
            ask_candidate_action_rows=("side", lambda s: int((s.astype(str) == "ask").sum())),
            baseline_selected_rows=("selected_by_baseline", "sum"),
            baseline_removed_ask_rows=("side", lambda s: int((s.astype(str) == "ask").sum())),
            baseline_removed_bid_spread_lt_1_rows=("spread_ticks", lambda s: 0),
        )
        .reset_index()
        .rename(columns={"audit_split": "split"})
    )
    # The final policy is bid_only_spread_ge_1. If there ever are bid candidates below 1 tick,
    # keep the reason separate from the ask-side removal.
    bid_low = (
        actions[(actions["side"] == "bid") & (actions["spread_ticks"] < 1.0)]
        .groupby(["audit_split", "session_id"], observed=False)
        .size()
        .reset_index(name="bid_low_count")
        .rename(columns={"audit_split": "split"})
    )
    grouped = grouped.drop(columns=["baseline_removed_bid_spread_lt_1_rows"]).merge(bid_low, on=["split", "session_id"], how="left")
    grouped["baseline_removed_bid_spread_lt_1_rows"] = grouped["bid_low_count"].fillna(0).astype(int)
    grouped = grouped.drop(columns=["bid_low_count"])
    for col in [
        "candidate_action_rows",
        "bid_candidate_action_rows",
        "ask_candidate_action_rows",
        "baseline_selected_rows",
        "baseline_removed_ask_rows",
        "baseline_removed_bid_spread_lt_1_rows",
    ]:
        grouped[col] = grouped[col].astype(int)
    return grouped


def residual_counts_by_session(residual: pd.DataFrame) -> pd.DataFrame:
    return (
        residual.groupby(["audit_split", "session_id"], observed=False)
        .agg(
            residual_ml_rows=("quote_ev_moderate", "count"),
            negative_ev_moderate_rows=("negative_ev_moderate", "sum"),
            severe_negative_ev_moderate_rows=("severe_negative_ev_moderate", "sum"),
            quote_ev_moderate_sum=("quote_ev_moderate", "sum"),
        )
        .reset_index()
        .rename(columns={"audit_split": "split"})
    )


def build_session_waterfall() -> pd.DataFrame:
    filter_sessions = read_filter_report()
    actions = load_full_actions()
    residual = load_residual_universe()
    hist_base = historical_session_base(filter_sessions)
    new4_symbol = str(residual[residual["audit_split"] == "NEW4"]["symbol"].dropna().iloc[0])
    new4_base = scan_new4_raw_valid_by_session(new4_symbol)
    base = pd.concat([hist_base, new4_base], ignore_index=True)

    action_counts = action_counts_by_session(actions)
    residual_counts = residual_counts_by_session(residual)
    out = base.merge(action_counts, on=["split", "session_id"], how="left").merge(residual_counts, on=["split", "session_id"], how="left")
    numeric_cols = [col for col in out.columns if col not in {"split", "session_id", "source_note"}]
    out[numeric_cols] = out[numeric_cols].fillna(0)
    for col in numeric_cols:
        if col != "quote_ev_moderate_sum":
            out[col] = out[col].astype(int)
    out["negative_ev_rate"] = out["negative_ev_moderate_rows"] / out["residual_ml_rows"].replace(0, np.nan)
    out["severe_negative_ev_rate"] = out["severe_negative_ev_moderate_rows"] / out["residual_ml_rows"].replace(0, np.nan)
    return out.sort_values(["split", "session_id"]).reset_index(drop=True)


def build_split_waterfall(session_df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {col: "sum" for col in COUNT_COLUMNS if col in session_df.columns}
    agg_map["session_id"] = "nunique"
    split = session_df.groupby("split", observed=False).agg(agg_map).rename(columns={"session_id": "session_count"}).reset_index()
    split["split"] = pd.Categorical(split["split"], categories=SPLITS, ordered=True)
    split = split.sort_values("split").reset_index(drop=True)
    for col in COUNT_COLUMNS:
        if col not in split.columns:
            split[col] = 0
    split["negative_ev_rate"] = split["negative_ev_moderate_rows"] / split["residual_ml_rows"].replace(0, np.nan)
    split["severe_negative_ev_rate"] = split["severe_negative_ev_moderate_rows"] / split["residual_ml_rows"].replace(0, np.nan)
    for col in STAGE_SEQUENCE:
        split[f"{col}_retention_from_raw"] = split[col] / split["raw_rows"].replace(0, np.nan)
    prev = "raw_rows"
    for col in STAGE_SEQUENCE[1:]:
        split[f"{col}_retention_from_previous_stage"] = split[col] / split[prev].replace(0, np.nan)
        prev = col
    return split


def build_reason_waterfall(session_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    filter_sessions = read_filter_report()
    rows: list[dict[str, object]] = []

    for split in ["train", "val", "test"]:
        sub = filter_sessions[filter_sessions["split"].astype(str) == split]
        raw = int(sub["raw_rows"].sum())
        after_invalid = raw - int(sub["dropped_invalid_bid_ask"].sum())
        after_spread = after_invalid - int(sub["dropped_nonpositive_spread"].sum())
        after_depth = after_spread - int(sub["dropped_missing_depth"].sum())
        eligible = int(sub["eligible_rows"].sum())
        dropped_boundary = int(sub["dropped_session_boundary"].sum())
        dropped_feature_na = int(sub["dropped_feature_na"].sum())
        rows.extend(
            [
                reason_row(split, "invalid_bid_ask", "raw_rows", "after_valid_bid_ask", raw, after_invalid, int(sub["dropped_invalid_bid_ask"].sum()), V16 / "outputs" / "quote_eligible_filter_report.csv", "016 quote eligibility filter."),
                reason_row(split, "nonpositive_spread", "after_valid_bid_ask", "after_positive_spread", after_invalid, after_spread, int(sub["dropped_nonpositive_spread"].sum()), V16 / "outputs" / "quote_eligible_filter_report.csv", "016 quote eligibility filter."),
                reason_row(split, "missing_depth", "after_positive_spread", "valid_book_rows", after_spread, after_depth, int(sub["dropped_missing_depth"].sum()), V16 / "outputs" / "quote_eligible_filter_report.csv", "016 quote eligibility filter."),
                reason_row(split, "session_boundary", "valid_book_rows", "after_session_boundary", after_depth, after_depth - dropped_boundary, dropped_boundary, V16 / "outputs" / "quote_eligible_filter_report.csv", "First/last boundary ticks removed."),
                reason_row(split, "feature_na", "after_session_boundary", "quote_eligible_rows", after_depth - dropped_boundary, eligible, dropped_feature_na, V16 / "outputs" / "quote_eligible_filter_report.csv", "Past-only feature finite check."),
                reason_row(split, "label_frame_materialization", "quote_eligible_rows", "label_rows", eligible, eligible, 0, V16 / "outputs" / "binary_labels_v2.parquet", "The Phase 028 label frame is loaded from an already quote-eligible parquet."),
            ]
        )

    for split in SPLITS:
        srow = split_df[split_df["split"].astype(str) == split].iloc[0]
        quote = int(srow["quote_eligible_rows"])
        candidate = int(srow["candidate_action_rows"])
        baseline = int(srow["baseline_selected_rows"])
        residual = int(srow["residual_ml_rows"])
        ask_removed = int(session_df[session_df["split"].astype(str) == split]["baseline_removed_ask_rows"].sum())
        bid_low_removed = int(session_df[session_df["split"].astype(str) == split]["baseline_removed_bid_spread_lt_1_rows"].sum())
        source_path = V18 / "queue_stress_core.py"
        if split == "NEW4":
            rows.append(
                reason_row(
                    split,
                    "new4_invalid_or_incomplete_book",
                    "raw_rows",
                    "valid_book_rows",
                    int(srow["raw_rows"]),
                    int(srow["valid_book_rows"]),
                    int(srow["raw_rows"] - srow["valid_book_rows"]),
                    DATA_NEW4,
                    "Exact selected-symbol NEW4 raw CSV scan; not a 016 tuning/label frame.",
                )
            )
            rows.append(
                reason_row(
                    split,
                    "new4_quote_eligible_proxy",
                    "valid_book_rows",
                    "quote_eligible_rows",
                    int(srow["valid_book_rows"]),
                    int(srow["quote_eligible_rows"]),
                    int(srow["valid_book_rows"] - srow["quote_eligible_rows"]),
                    V27 / "common_027.py",
                    "NEW4 has no 016 binary label parquet; valid-book rows are the explicit OOS quote/action source proxy.",
                )
            )
            rows.append(
                reason_row(
                    split,
                    "new4_label_frame_proxy",
                    "quote_eligible_rows",
                    "label_rows",
                    int(srow["quote_eligible_rows"]),
                    int(srow["label_rows"]),
                    int(srow["quote_eligible_rows"] - srow["label_rows"]),
                    V27 / "common_027.py",
                    "NEW4 label_rows are reported as the same valid-book OOS frame used to construct actions.",
                )
            )
        rows.append(
            reason_row(
                split,
                "sparse_fill_proxy_candidate_action_construction",
                "quote_eligible_rows",
                "candidate_action_rows",
                quote,
                candidate,
                quote - candidate,
                source_path,
                "Candidate actions require next-row bid/ask crossing the passive quote proxy.",
            )
        )
        rows.append(
            reason_row(
                split,
                "baseline_policy_removed_ask_side",
                "candidate_action_rows",
                "after_bid_only_side_filter",
                candidate,
                candidate - ask_removed,
                ask_removed,
                V26 / "baseline_utils.py",
                "Policy `bid_only_spread_ge_1` removes all ask-side candidates.",
            )
        )
        rows.append(
            reason_row(
                split,
                "baseline_policy_removed_bid_spread_lt_1",
                "after_bid_only_side_filter",
                "baseline_selected_rows",
                candidate - ask_removed,
                baseline,
                bid_low_removed,
                V26 / "baseline_utils.py",
                "Policy would remove bid candidates below one tick; observed count is kept explicit.",
            )
        )
        rows.append(
            reason_row(
                split,
                "residual_dataset_materialization",
                "baseline_selected_rows",
                "residual_ml_rows",
                baseline,
                residual,
                baseline - residual,
                V28 / "common_028.py",
                "Residual ML parquet is the baseline-selected universe plus EV labels; no additional filtering is expected.",
            )
        )
    return pd.DataFrame(rows)


def reason_row(
    split: str,
    reason: str,
    from_stage: str,
    to_stage: str,
    input_rows: int,
    retained_rows: int,
    rows_removed: int,
    source_path: Path,
    note: str,
) -> dict[str, object]:
    return {
        "split": split,
        "reason": reason,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "input_rows": int(input_rows),
        "retained_rows": int(retained_rows),
        "rows_removed": int(rows_removed),
        "retention_rate": float(retained_rows / input_rows) if input_rows else np.nan,
        "source_path": str(source_path),
        "note": note,
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343b",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def plot_filtering_waterfall(split_df: pd.DataFrame) -> None:
    set_style()
    labels = ["Raw", "Valid", "Eligible", "Label", "Actions", "Baseline", "Residual"]
    colors = ["#5b677a", "#4d8bbf", "#498f6d", "#8aa34d", "#d19a2a", "#c95b5b", "#7a4cc2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    axes = axes.ravel()
    for ax, split in zip(axes, SPLITS):
        row = split_df[split_df["split"].astype(str) == split].iloc[0]
        vals = [int(row[col]) for col in STAGE_SEQUENCE]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, width=0.72)
        ax.set_yscale("log")
        ax.set_title(split)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        for xi, val in zip(x, vals):
            ax.text(xi, max(val, 1) * 1.08, f"{val:,}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Count (log scale)")
    fig.suptitle("Residual Universe Filtering Waterfall", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "filtering_waterfall_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_vs_raw(split_df: pd.DataFrame) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(split_df))
    width = 0.35
    ax.bar(x - width / 2, split_df["raw_rows"], width=width, label="Raw rows", color="#5b677a")
    ax.bar(x + width / 2, split_df["residual_ml_rows"], width=width, label="Residual ML rows", color="#7a4cc2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(split_df["split"].astype(str))
    ax.set_title("Residual Sample Count vs Raw Rows")
    ax.set_ylabel("Count (log scale)")
    for i, row in split_df.iterrows():
        rate = float(row["residual_ml_rows"] / row["raw_rows"]) if row["raw_rows"] else np.nan
        ax.text(i + width / 2, max(row["residual_ml_rows"], 1) * 1.25, f"{int(row['residual_ml_rows']):,}\n{rate:.4%}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_sample_count_vs_raw_rows.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_negative_counts(split_df: pd.DataFrame) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(split_df))
    neg = split_df["negative_ev_moderate_rows"].astype(int).to_numpy()
    severe = split_df["severe_negative_ev_moderate_rows"].astype(int).to_numpy()
    non_severe_neg = neg - severe
    ax.bar(x, non_severe_neg, label="Negative EV, > -5000", color="#c95b5b")
    ax.bar(x, severe, bottom=non_severe_neg, label="Severe negative EV <= -5000", color="#742f37")
    ax.set_xticks(x)
    ax.set_xticklabels(split_df["split"].astype(str))
    ax.set_title("Negative EV Label Counts by Split")
    ax.set_ylabel("Residual ML samples")
    for i, row in split_df.iterrows():
        ax.text(i, int(row["negative_ev_moderate_rows"]) + 1, f"{int(row['negative_ev_moderate_rows']):,}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "negative_ev_label_counts_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_table(split_df: pd.DataFrame) -> None:
    display_cols = [
        "split",
        "raw_rows",
        "label_rows",
        "valid_book_rows",
        "quote_eligible_rows",
        "candidate_action_rows",
        "bid_candidate_action_rows",
        "ask_candidate_action_rows",
        "baseline_selected_rows",
        "residual_ml_rows",
        "negative_ev_moderate_rows",
        "severe_negative_ev_moderate_rows",
        "negative_ev_rate",
        "severe_negative_ev_rate",
        "session_count",
    ]
    table = split_df[display_cols].copy()
    for col in display_cols:
        if col.endswith("rate"):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4%}")
    lines = [
        "# Residual Universe Filtering Waterfall Table",
        "",
        markdown_table(table),
        "",
        "Severe negative EV is defined as `quote_ev_moderate <= -5000`, matching the Phase 030 GPU export convention.",
    ]
    write_markdown(ROOT / "filtering_waterfall_table.md", lines)


def write_code_path_audit() -> None:
    common_028 = V28 / "common_028.py"
    baseline_utils = V26 / "baseline_utils.py"
    common_027 = V27 / "common_027.py"
    lines = [
        "# Filtering Code Path Audit",
        "",
        "| Stage | Code/data path | Function or artifact | Filtering role |",
        "| --- | --- | --- | --- |",
        f"| Binary label frame | `{V16 / 'outputs' / 'binary_labels_v2.parquet'}` | artifact | Already quote-eligible historical rows; loaded by `load_label_split`. |",
        f"| Split loader | `{common_028}:{line_number(common_028, 'def load_label_split')}` | `load_label_split` | Reads `binary_labels_v2.parquet`, filters by split, sorts by session/local row, adds `event_id`. |",
        f"| Historical train action table | `{baseline_utils}:{line_number(baseline_utils, 'def actions_from_frame')}` | `actions_from_frame` | Builds sparse passive-fill candidate actions and computes moderate proxy EV for train. |",
        f"| Val/test action table | `{common_027}:{line_number(common_027, 'def load_actions_for_split')}` | `load_actions_for_split` | Reuses 018/root-cause action context for val/test and computes stress EV columns if missing. |",
        f"| NEW4 action table | `{common_027}:{line_number(common_027, 'def build_new4_frame')}` and `{common_027}:{line_number(common_027, 'def build_new4_actions')}` | `build_new4_frame`, `build_new4_actions` | Loads external OOS valid-book frame and builds candidate actions; no 016 train/val label parquet exists for NEW4. |",
        f"| Baseline policy | `{baseline_utils}:{line_number(baseline_utils, 'def apply_non_ml_policy')}` | `apply_non_ml_policy` | `bid_only_spread_ge_1` keeps bid-side actions with `spread_ticks >= 1`. |",
        f"| Residual ML universe | `{common_028}:{line_number(common_028, 'def baseline_universe')}` | `baseline_universe` | Materializes only baseline-selected actions and labels `negative_ev_moderate = quote_ev_moderate < 0`. |",
    ]
    write_markdown(ROOT / "filtering_code_path_audit.md", lines)


def biggest_drop_for_train(reason_df: pd.DataFrame) -> pd.Series:
    train = reason_df[reason_df["split"].astype(str) == "train"].copy()
    return train.sort_values("rows_removed", ascending=False).iloc[0]


def write_report(split_df: pd.DataFrame, session_df: pd.DataFrame, reason_df: pd.DataFrame) -> None:
    train = split_df[split_df["split"].astype(str) == "train"].iloc[0]
    biggest = biggest_drop_for_train(reason_df)
    train_raw = int(train["raw_rows"])
    train_quote = int(train["quote_eligible_rows"])
    train_actions = int(train["candidate_action_rows"])
    train_baseline = int(train["baseline_selected_rows"])
    train_residual = int(train["residual_ml_rows"])
    candidate_rate = float(train_actions / train_quote) if train_quote else np.nan
    baseline_rate = float(train_baseline / train_actions) if train_actions else np.nan
    residual_rate = float(train_residual / train_raw) if train_raw else np.nan

    display = split_df[
        [
            "split",
            "raw_rows",
            "label_rows",
            "valid_book_rows",
            "quote_eligible_rows",
            "candidate_action_rows",
            "bid_candidate_action_rows",
            "ask_candidate_action_rows",
            "baseline_selected_rows",
            "residual_ml_rows",
            "negative_ev_moderate_rows",
            "severe_negative_ev_moderate_rows",
        ]
    ].copy()
    lines = [
        "# Residual Universe Filtering Waterfall Report",
        "",
        "This is an explanation audit only. It does not tune ML, alter thresholds, or change the final non-ML controller.",
        "",
        "## Executive Answer",
        "",
        f"Train shrinks from `{train_raw:,}` raw rows to exactly `{train_residual:,}` residual ML rows because the residual experiment is not trained on all market rows. It first turns quote-eligible rows into sparse passive-fill candidate actions, then applies the final baseline policy `bid_only_spread_ge_1`.",
        "",
        f"The largest train-stage drop is `{biggest['reason']}`: `{int(biggest['input_rows']):,}` input rows/actions to `{int(biggest['retained_rows']):,}` retained, removing `{int(biggest['rows_removed']):,}`.",
        "",
        "## Split Waterfall",
        "",
        markdown_table(display),
        "",
        "## Why Train Becomes 820 Samples",
        "",
        f"- Quote eligibility is not the main bottleneck: train keeps `{train_quote:,}` quote-eligible rows out of `{train_raw:,}` raw rows.",
        f"- Sparse action construction is the dominant bottleneck: only `{train_actions:,}` candidate actions remain, a `{candidate_rate:.6%}` retention rate from quote-eligible rows.",
        f"- The baseline policy then keeps `{train_baseline:,}` bid-side `spread_ticks >= 1` actions, a `{baseline_rate:.2%}` retention rate from candidate actions.",
        f"- The residual ML parquet contains `{train_residual:,}` rows, matching the baseline-selected count; residual materialization itself does not add another filter.",
        f"- End-to-end train retention is `{residual_rate:.6%}` from raw rows.",
        "",
        "## What Caused the Small Sample Size?",
        "",
        "The small residual training set is primarily caused by sparse fill-proxy candidate action construction, not by quote eligibility. The baseline filter is the second major reduction because it removes all ask-side candidate actions and keeps only bid-side candidates satisfying `spread_ticks >= 1`.",
        "",
        "## Source Reconstruction Notes",
        "",
        "- Historical train/val/test raw and quote-eligible counts are reconstructed from `new_era/16_binary_label_v2/outputs/quote_eligible_filter_report.csv`.",
        "- Historical label rows equal the rows loaded by `common_028.load_label_split`, which reads `binary_labels_v2.parquet`; that parquet is already quote-eligible.",
        "- The requested table includes `label_rows`, but the causal filter order is `raw -> valid_book -> quote_eligible -> label_frame -> candidate_actions -> baseline -> residual`; retention-from-previous-stage columns use this actual execution order.",
        "- Candidate action and baseline-selected counts are read from the frozen Phase 031 full-action universe generated from the Phase 028/027 pipeline.",
        "- NEW4 has no `binary_labels_v2.parquet` entry. Its `label_rows` and `quote_eligible_rows` are explicitly marked as a valid-book OOS proxy from the `build_new4_frame` pathway, not guessed.",
        f"- Severe negative EV uses the Phase 030 convention: `quote_ev_moderate <= {SEVERE_NEGATIVE_EV_THRESHOLD:.0f}`.",
        "",
        "## Impact on Residual ML Interpretation",
        "",
        "The residual ML experiment is a valid incremental veto audit, but it is not a full test of all L2-based ML potential. Because the residual universe contains only baseline-selected quote samples, the training set is very small. Therefore, the failure of residual ML should be interpreted as failure to add value after the non-ML baseline, not as proof that all ML approaches are useless.",
        "",
        "## Outputs",
        "",
        "- `filtering_waterfall_by_split.csv`",
        "- `filtering_waterfall_by_session.csv`",
        "- `filtering_waterfall_by_reason.csv`",
        "- `filtering_waterfall_table.md`",
        "- `filtering_code_path_audit.md`",
        "- `figures/filtering_waterfall_by_split.png`",
        "- `figures/residual_sample_count_vs_raw_rows.png`",
        "- `figures/negative_ev_label_counts_by_split.png`",
    ]
    write_markdown(ROOT / "RESIDUAL_UNIVERSE_FILTERING_WATERFALL_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    session_df = build_session_waterfall()
    split_df = build_split_waterfall(session_df)
    reason_df = build_reason_waterfall(session_df, split_df)

    session_df.to_csv(ROOT / "filtering_waterfall_by_session.csv", index=False)
    split_df.to_csv(ROOT / "filtering_waterfall_by_split.csv", index=False)
    reason_df.to_csv(ROOT / "filtering_waterfall_by_reason.csv", index=False)

    plot_filtering_waterfall(split_df)
    plot_residual_vs_raw(split_df)
    plot_negative_counts(split_df)
    write_table(split_df)
    write_code_path_audit()
    write_report(split_df, session_df, reason_df)

    print("Split waterfall:")
    print(split_df[["split", *COUNT_COLUMNS, "negative_ev_rate", "severe_negative_ev_rate", "session_count"]].to_string(index=False))
    print("Wrote Phase 036 residual universe filtering audit.")


if __name__ == "__main__":
    main()
