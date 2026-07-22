from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_034 import FIGURES, PROJECT_ROOT, RESULTS, ROOT, ensure_dirs, set_style, write_markdown


V16 = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V27 = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"
V28 = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
V31 = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"
V18 = PROJECT_ROOT / "new_era" / "18_queue_aware"
V26 = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"

DATA_NEW4 = PROJECT_ROOT / "data" / "NEW4"

SPLIT_ORDER = ["train", "val", "test", "NEW4"]
STAGES = [
    "raw_rows",
    "valid_book_rows",
    "quote_eligible_rows",
    "candidate_actions",
    "baseline_selected_actions",
    "residual_ml_samples",
]


def _line_number(path: Path, pattern: str) -> int | None:
    if not path.exists():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def _count_new4_raw_and_valid(symbol: str) -> tuple[int, int, list[dict[str, object]]]:
    raw_rows = 0
    valid_book_rows = 0
    per_file: list[dict[str, object]] = []
    for csv_path in sorted(DATA_NEW4.rglob("*.csv"), key=lambda p: str(p.relative_to(DATA_NEW4))):
        file_raw = 0
        file_valid = 0
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) <= 23 or row[3].strip() != symbol:
                    continue
                file_raw += 1
                try:
                    bid = float(row[21])
                    ask = float(row[22])
                    bid_vol = float(row[20])
                    ask_vol = float(row[23])
                except ValueError:
                    continue
                if bid > 0.0 and ask > 0.0 and ask >= bid and bid_vol > 0.0 and ask_vol > 0.0:
                    file_valid += 1
        raw_rows += file_raw
        valid_book_rows += file_valid
        per_file.append(
            {
                "file": str(csv_path.relative_to(DATA_NEW4)),
                "symbol": symbol,
                "raw_rows": int(file_raw),
                "valid_book_rows": int(file_valid),
            }
        )
    return int(raw_rows), int(valid_book_rows), per_file


def _load_residual_baseline_universe() -> pd.DataFrame:
    hist = pd.read_parquet(V28 / "data" / "baseline_universe_quote_ev.parquet").copy()
    hist["waterfall_split"] = hist["split"].astype(str)
    new4 = pd.read_parquet(V28 / "data" / "new4_baseline_universe_quote_ev.parquet").copy()
    new4["waterfall_split"] = "NEW4"
    out = pd.concat([hist, new4], ignore_index=True)
    out["quote_ev_moderate"] = out["quote_ev_moderate"].astype(float)
    out["negative_ev_moderate"] = (out["quote_ev_moderate"] < 0.0).astype(int)
    return out


def build_waterfall() -> pd.DataFrame:
    filter_report = pd.read_csv(V16 / "results" / "filter_waterfall_by_split.csv")
    full_actions = pd.read_parquet(V31 / "data" / "full_action_universe.parquet")
    residual = _load_residual_baseline_universe()

    full_actions["waterfall_split"] = full_actions["phase031_split"].astype(str).replace({"new4": "NEW4"})

    rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        filt = filter_report[filter_report["split"].astype(str) == split].iloc[0]
        acts = full_actions[full_actions["waterfall_split"] == split]
        res = residual[residual["waterfall_split"] == split]
        rows.append(
            {
                "split": split,
                "raw_rows": int(filt["raw_rows"]),
                "valid_book_rows": int(filt["after_complete_depth"]),
                "quote_eligible_rows": int(filt["quote_eligible_rows"]),
                "candidate_actions": int(len(acts)),
                "baseline_selected_actions": int(acts["selected_by_baseline"].astype(bool).sum()),
                "residual_ml_samples": int(len(res)),
                "negative_ev_labels": int(res["negative_ev_moderate"].sum()),
                "nonnegative_ev_labels": int((res["negative_ev_moderate"] == 0).sum()),
                "negative_ev_rate": float(res["negative_ev_moderate"].mean()) if len(res) else np.nan,
                "source_note": "Historical split from 016 quote-eligible universe, then Phase 031 full-action table.",
            }
        )

    new4_res = residual[residual["waterfall_split"] == "NEW4"]
    new4_acts = full_actions[full_actions["waterfall_split"] == "NEW4"]
    new4_symbol = str(new4_res["symbol"].dropna().iloc[0]) if len(new4_res) and "symbol" in new4_res.columns else "TXFA6"
    raw_rows, valid_book_rows, per_file = _count_new4_raw_and_valid(new4_symbol)
    pd.DataFrame(per_file).to_csv(RESULTS / "new4_raw_valid_book_counts.csv", index=False)
    rows.append(
        {
            "split": "NEW4",
            "raw_rows": raw_rows,
            "valid_book_rows": valid_book_rows,
            "quote_eligible_rows": valid_book_rows,
            "candidate_actions": int(len(new4_acts)),
            "baseline_selected_actions": int(new4_acts["selected_by_baseline"].astype(bool).sum()),
            "residual_ml_samples": int(len(new4_res)),
            "negative_ev_labels": int(new4_res["negative_ev_moderate"].sum()),
            "nonnegative_ev_labels": int((new4_res["negative_ev_moderate"] == 0).sum()),
            "negative_ev_rate": float(new4_res["negative_ev_moderate"].mean()) if len(new4_res) else np.nan,
            "source_note": (
                "External OOS NEW4 used the valid-book controller/action-source flow; no train/val "
                "quote-eligible refit or boundary tuning was applied."
            ),
        }
    )
    out = pd.DataFrame(rows)
    out["baseline_to_candidate_rate"] = out["baseline_selected_actions"] / out["candidate_actions"].replace(0, np.nan)
    out["residual_to_baseline_rate"] = out["residual_ml_samples"] / out["baseline_selected_actions"].replace(0, np.nan)
    return out


def plot_waterfall(waterfall: pd.DataFrame) -> None:
    set_style()
    stage_labels = [
        "Raw rows",
        "Valid book",
        "Quote-eligible",
        "Candidate actions",
        "Baseline-selected",
        "Residual ML",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharey=False)
    axes = axes.ravel()
    colors = ["#5b677a", "#4d8bbf", "#3a9d74", "#d19a2a", "#c95b5b", "#7a4cc2"]
    for ax, split in zip(axes, SPLIT_ORDER):
        row = waterfall[waterfall["split"] == split].iloc[0]
        vals = [float(row[col]) for col in STAGES]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, width=0.72)
        ax.set_title(split)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, rotation=28, ha="right")
        ax.set_ylabel("Rows / actions (log scale)")
        for xi, val in zip(x, vals):
            ax.text(xi, val * 1.08 if val > 0 else 1, f"{int(val):,}", ha="center", va="bottom", fontsize=8)
        neg = int(row["negative_ev_labels"])
        nonneg = int(row["nonnegative_ev_labels"])
        ax.text(
            0.02,
            0.95,
            f"Labels: neg_EV={neg:,}, nonneg_EV={nonneg:,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#d6dae1"},
        )
    fig.suptitle("ML Filtering Waterfall by Split", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "ml_filtering_waterfall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_waterfall_report(waterfall: pd.DataFrame) -> None:
    lines = [
        "# ML Filtering Waterfall",
        "",
        "This table traces the data flow used by the residual Quote-EV ML veto diagnostics.",
        "Counts are split-specific and are read from frozen artifacts; no model is retrained here.",
        "",
        "Pipeline meaning:",
        "",
        "- `raw_rows`: selected-symbol input rows before the local filter stage.",
        "- `valid_book_rows`: rows with valid top-of-book/depth after the book-validity filter.",
        "- `quote_eligible_rows`: rows admitted into the quote/action source universe. For NEW4, this equals the external OOS valid-book action source because NEW4 was not used for training or validation selection.",
        "- `candidate_actions`: passive bid/ask candidate fills generated by the fill-proxy action table.",
        "- `baseline_selected_actions`: candidate actions selected by `bid_only_spread_ge_1`.",
        "- `residual_ml_samples`: the residual EV-veto dataset restricted to the baseline-selected universe.",
        "- `negative_ev_labels` / `nonnegative_ev_labels`: final residual label counts using `quote_ev_moderate < 0`.",
        "",
        "![ML Filtering Waterfall](figures/ml_filtering_waterfall.png)",
        "",
        "## Counts",
        "",
        waterfall.to_markdown(index=False),
        "",
        "## Source Artifacts",
        "",
        f"- Historical raw/valid/quote-eligible counts: `{V16 / 'results' / 'filter_waterfall_by_split.csv'}`.",
        f"- Full candidate action universe: `{V31 / 'data' / 'full_action_universe.parquet'}`.",
        f"- Residual baseline universe labels: `{V28 / 'data' / 'baseline_universe_quote_ev.parquet'}` and `{V28 / 'data' / 'new4_baseline_universe_quote_ev.parquet'}`.",
        f"- NEW4 exact selected-symbol raw/valid scan: `{RESULTS / 'new4_raw_valid_book_counts.csv'}`.",
    ]
    write_markdown(ROOT / "ml_filtering_waterfall.md", lines)


def write_table_column_definitions() -> None:
    lines = [
        "# Table Column Definitions and Blank Cell Policy",
        "",
        "This project intentionally distinguishes missingness from zero-valued metrics.",
        "A blank cell or `NaN` should never be read as a hidden zero.",
        "",
        "| Cell representation | Meaning | Typical reason | Interpretation rule |",
        "| --- | --- | --- | --- |",
        "| Empty string / blank cell | Not computed in that artifact | Metric did not belong to the table or the upstream script did not produce it | Do not compare numerically; inspect the table note/source artifact |",
        "| `NaN` | Computation was attempted but undefined | No samples in a subgroup, one-class label vector, or division by zero denominator | Treat as undefined, not as pass/fail by itself |",
        "| `NA` / `not_applicable` | Metric is structurally not applicable | Example: regression metric for a pure classification-only plot | Exclude from aggregate comparisons |",
        "| `source_missing` | Required upstream source artifact or source column was absent | Placeholder reports or compatibility audits | The claim is unsupported until the source is regenerated |",
        "| `0` or `0.0` | Computed numeric zero | Count, PnL, loss, or rate genuinely evaluated to zero | It is a real value and can be compared numerically |",
        "",
        "For final advisor-facing tables, any ambiguous blank from older phases should be read alongside the nearest `status`, `note`, `source_note`, or report paragraph.",
        "New Phase 034 follow-up tables use explicit note/status columns where a source was not available or a metric is not applicable.",
    ]
    write_markdown(ROOT / "table_column_definitions.md", lines)


def write_quote_ev_formula_audit() -> None:
    baseline_utils = V26 / "baseline_utils.py"
    common_027 = V27 / "common_027.py"
    common_028 = V28 / "common_028.py"
    queue_core = V18 / "queue_stress_core.py"
    lines = [
        "# Quote EV Formula Audit",
        "",
        "`quote_ev_moderate` is a simulator-derived proxy label. It is not a true exchange queue replay label, and it does not use real queue position, order acknowledgements, or exchange fill IDs.",
        "",
        "## Code Locations",
        "",
        f"- Historical train action EV generation: `{baseline_utils}` around line {_line_number(baseline_utils, 'def actions_from_frame')} (`actions_from_frame`).",
        f"- Historical val/test and NEW4 stress formula: `{common_027}` around line {_line_number(common_027, 'def quote_value_for_stress')} (`quote_value_for_stress`).",
        f"- Residual ML dataset construction: `{common_028}` around line {_line_number(common_028, 'def baseline_universe')} (`baseline_universe`).",
        f"- Candidate action/fill-proxy trigger: `{queue_core}` around line {_line_number(queue_core, 'def build_action_table')} (`build_action_table`).",
        "",
        "## Horizon and Fill Proxy",
        "",
        "- Horizon `H = 10` ticks.",
        "- Historical labels store `label_horizon_ticks = 10` and `future_mid_return_H_ticks` in `new_era/16_binary_label_v2/outputs/binary_labels_v2.parquet`.",
        "- NEW4 construction also uses `build_new4_frame(horizon_ticks=10)`.",
        "- Bid fill-proxy candidate: next row ask price crosses the current quote, `ask[t+1] <= bid[t]`.",
        "- Ask fill-proxy candidate: next row bid price crosses the current quote, `bid[t+1] >= ask[t]`.",
        "",
        "## Quote Price Assumption",
        "",
        "- Bid action quote price is the current best bid, `price = bid_px_1[t]`.",
        "- Ask action quote price is the current best ask, `price = ask_px_1[t]`.",
        "- The final mark-to-market price is the session final mid price, `final_mid = last(mid_price)` within the same session.",
        "",
        "## Base Formula",
        "",
        "For an action with side-specific future markout:",
        "",
        "```text",
        "side_markout = future_mid_return_H_ticks                for bid",
        "side_markout = -future_mid_return_H_ticks               for ask",
        "adverse_ticks = max(0, -side_markout)",
        "terminal_ticks = final_mid - quote_price                for bid",
        "terminal_ticks = quote_price - final_mid                for ask",
        "eligible = neutral_u <= neutral_fill_haircut",
        "quote_ev = terminal_ticks * POINT_VALUE - MAKER_FEE - max(toxic_multiplier - 1, 0) * adverse_ticks * POINT_VALUE",
        "quote_ev_stressed = quote_ev if eligible else 0",
        "```",
        "",
        "Constants used by the current proxy implementation:",
        "",
        "- `POINT_VALUE = 200.0`.",
        "- `MAKER_FEE = 15.0`.",
        "- `neutral_u` is a deterministic stable random mask keyed by event/action and side; it is fixed across stress settings for comparability.",
        "",
        "## Stress Parameters",
        "",
        "| Stress level | neutral_fill_haircut | toxic_multiplier | cancel_delay_ticks |",
        "| --- | ---: | ---: | ---: |",
        "| base | 1.00 | 1.00 | 0 |",
        "| mild | 0.75 | 1.25 | 0 |",
        "| moderate | 0.50 | 1.50 | 0 |",
        "",
        "The residual ML label uses the moderate setting: `quote_ev_moderate = quote_ev_stressed` under `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, and `delay=0`.",
        "",
        "## Explicit Limitation",
        "",
        "This formula is a proxy quote-EV label. It is useful for controlled falsification and stress tests, but it should not be described as realized exchange PnL or true queue-aware fill toxicity.",
    ]
    write_markdown(ROOT / "quote_ev_formula_audit.md", lines)


def write_baseline_oracle_gap() -> pd.DataFrame:
    residual = _load_residual_baseline_universe()
    rows: list[dict[str, object]] = []
    for split in SPLIT_ORDER:
        df = residual[residual["waterfall_split"] == split].copy()
        values = df["quote_ev_moderate"].astype(float)
        pos = values > 0.0
        neg = values < 0.0
        baseline_pnl = float(values.sum())
        oracle_pnl = float(values[pos].sum())
        rows.append(
            {
                "split": split,
                "universe": "bid_only_spread_ge_1_selected_rows",
                "baseline_selected_count": int(len(df)),
                "baseline_pnl": baseline_pnl,
                "oracle_positive_ev_filter_pnl": oracle_pnl,
                "oracle_gap_vs_baseline": float(oracle_pnl - baseline_pnl),
                "negative_ev_loss_removed_by_oracle": float(-values[neg].sum()),
                "positive_ev_count": int(pos.sum()),
                "negative_ev_count": int(neg.sum()),
                "zero_ev_count": int((values == 0.0).sum()),
                "positive_ev_rate": float(pos.mean()) if len(df) else np.nan,
                "negative_ev_rate": float(neg.mean()) if len(df) else np.nan,
                "test_used_for_selection": False,
                "note": "Oracle uses future quote_ev_moderate and is non-tradable; it is an upper bound inside the final baseline universe only.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "baseline_universe_oracle_gap.csv", index=False)
    return out


def main() -> None:
    ensure_dirs()
    waterfall = build_waterfall()
    waterfall.to_csv(RESULTS / "ml_filtering_waterfall.csv", index=False)
    plot_waterfall(waterfall)
    write_waterfall_report(waterfall)
    write_table_column_definitions()
    write_quote_ev_formula_audit()
    oracle_gap = write_baseline_oracle_gap()
    print("Waterfall counts:")
    print(waterfall[["split", *STAGES, "negative_ev_labels", "nonnegative_ev_labels"]].to_string(index=False))
    print("Baseline universe oracle gap:")
    print(oracle_gap[["split", "baseline_pnl", "oracle_positive_ev_filter_pnl", "oracle_gap_vs_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
