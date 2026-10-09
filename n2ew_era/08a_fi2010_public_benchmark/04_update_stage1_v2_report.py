from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "REPORT_STAGE1.md"
INTERRUPTION_DIAGNOSTIC = ROOT / "STAGE1_V2_INTERRUPTION_DIAGNOSTIC.md"
HORIZONS = [10, 20, 30, 50, 100]
SECTION_TITLE = "## Stage 1 收斂重驗(v2)"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(x: object) -> float:
    return 100.0 * float(x)


def fmt_pct(mean: object, std: object) -> str:
    return f"{pct(mean):.2f} +/- {pct(std):.2f}"


def require_v2_complete(primary_v2: list[dict[str, str]]) -> None:
    missing: list[str] = []
    for horizon in HORIZONS:
        rows = [
            r
            for r in primary_v2
            if r["split"] == "test" and int(r["horizon"]) == horizon and int(r["n_seeds"]) == 3
        ]
        if not rows:
            missing.append(f"k={horizon}")
    if missing:
        raise RuntimeError(
            "Stage 1 v2 primary metrics are incomplete; missing 3-seed test rows for "
            + ", ".join(missing)
            + ". Finish 02_train_phc_lite_stage1.py before updating REPORT_STAGE1.md."
        )


def build_delta_rows(primary_v1: list[dict[str, str]], primary_v2: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        v1 = next(r for r in primary_v1 if r["split"] == "test" and int(r["horizon"]) == horizon)
        v2 = next(r for r in primary_v2 if r["split"] == "test" and int(r["horizon"]) == horizon)
        rows.append(
            {
                "horizon": horizon,
                "v1_primary_loss_mode": v1["primary_loss_mode"],
                "v2_primary_loss_mode": v2["primary_loss_mode"],
                "v1_test_macro_f1_pct": pct(v1["macro_f1_mean"]),
                "v2_test_macro_f1_pct": pct(v2["macro_f1_mean"]),
                "delta_f1_pct_points": pct(v2["macro_f1_mean"]) - pct(v1["macro_f1_mean"]),
                "v1_test_accuracy_pct": pct(v1["accuracy_mean"]),
                "v2_test_accuracy_pct": pct(v2["accuracy_mean"]),
                "delta_accuracy_pct_points": pct(v2["accuracy_mean"]) - pct(v1["accuracy_mean"]),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_report_section(
    primary_v1: list[dict[str, str]],
    primary_v2: list[dict[str, str]],
    convergence_v2: list[dict[str, str]],
    comparison_v2: list[dict[str, str]],
    delta_rows: list[dict[str, object]],
) -> str:
    conv_rows = []
    for row in convergence_v2:
        conv_rows.append(
            [
                row["horizon"],
                row["seed"],
                row["loss_mode"],
                row["best_epoch"],
                row["final_epoch"],
                row["early_stop"],
                row["max_epoch_hit"],
                row["best_epoch_distance_to_max"],
                row["convergence_closed"],
            ]
        )

    primary_rows = []
    for horizon in HORIZONS:
        v1 = next(r for r in primary_v1 if r["split"] == "test" and int(r["horizon"]) == horizon)
        v2 = next(r for r in primary_v2 if r["split"] == "test" and int(r["horizon"]) == horizon)
        delta = next(r for r in delta_rows if int(r["horizon"]) == horizon)
        primary_rows.append(
            [
                horizon,
                v1["primary_loss_mode"],
                fmt_pct(v1["macro_f1_mean"], v1["macro_f1_std"]),
                v2["primary_loss_mode"],
                fmt_pct(v2["macro_f1_mean"], v2["macro_f1_std"]),
                f"{float(delta['delta_f1_pct_points']):+.2f}",
            ]
        )

    published_rows = []
    for row in comparison_v2:
        published_rows.append(
            [
                row["model"],
                row["parameters"],
                row["verification_level"],
                row.get("f1_k10", ""),
                row.get("f1_k20", ""),
                row.get("f1_k50", ""),
            ]
        )

    closed = [r for r in convergence_v2 if str(r["convergence_closed"]).lower() == "true"]
    boundary = [r for r in convergence_v2 if str(r["max_epoch_hit"]).lower() == "true"]
    f1_deltas = [float(r["delta_f1_pct_points"]) for r in delta_rows]
    max_delta = max(f1_deltas)
    min_delta = min(f1_deltas)
    if boundary:
        conclusion = (
            "v2 still has at least one run with `max_epoch_hit=True`, so convergence is not fully closed. "
            "Stage 2 should remain blocked until this is reviewed."
        )
    elif max_delta > 2.0:
        conclusion = (
            "v2 improves the Stage 1 F1 enough to confirm the 4-epoch run understated this small model. "
            "The convergence gate is now closed, so Stage 1 no longer blocks the separately reviewed Stage 2 HLS feasibility step."
        )
    elif min_delta < -2.0:
        conclusion = (
            "v2 does not rescue the Stage 1 result; longer training lowered at least one horizon materially, "
            "so the 4-epoch concern was not the dominant explanation."
        )
    else:
        conclusion = (
            "v2 is close to v1; the longer run mainly closes the convergence audit gap rather than changing "
            "the headline FI-2010 accuracy/F1 story."
        )

    return "\n\n".join(
        [
            SECTION_TITLE,
            "status: `stage1_v2_convergence_revalidation_complete`",
            (
                "Interruption diagnostic: the prior v2 attempt stopped during `k=10/seed=0/weighted` "
                "before final evaluation/writeout, after the runner timeout. The checkpoint existed but "
                "no weighted metrics row could be written because `train_one()` had not returned. See "
                "`STAGE1_V2_INTERRUPTION_DIAGNOSTIC.md` for artifact-level evidence."
            ),
            (
                "Part A reran training only. `fi2010_phc_lite.py`, `01_prepare_fi2010_stage1.py`, "
                "the cached FI-2010 Setup 2 split, and all Stage 2/HLS artifacts were left untouched. "
                "No project-internal NEW1-NEW5 data were read."
            ),
            (
                "v2 protocol: max_epochs=60, patience=10, early stopping on validation macro F1, "
                "same raw-first/class-weighted fallback rule as v1, seeds 0/1/2, horizons k=10/20/30/50/100. "
                "The single boundary case `k=30/seed=2/raw` was then rerun with max_epochs=100 and the same "
                "patience=10, closing naturally at final_epoch=61 without changing any other split, seed, model, "
                "or threshold. "
                "The run used the script's verified `--fast-exact-final-step` path: because this FI-2010 "
                "classifier has zero `decay` and zero `alpha_scalar` buffers, full-window logits and "
                "final-step logits are mathematically identical; the script checks this before training."
            ),
            markdown_table(
                [
                    "k",
                    "v1 primary",
                    "v1 test macro F1",
                    "v2 primary",
                    "v2 test macro F1",
                    "delta pct-pt",
                ],
                primary_rows,
            ),
            "Convergence gate: `convergence_closed=True` means `early_stop=True` and `best_epoch < max_epochs - patience`.",
            markdown_table(
                [
                    "k",
                    "seed",
                    "loss",
                    "best_epoch",
                    "final_epoch",
                    "early_stop",
                    "max_epoch_hit",
                    "distance_to_max",
                    "closed",
                ],
                conv_rows,
            ),
            f"Closed runs: {len(closed)} / {len(convergence_v2)}. Max-epoch boundary runs: {len(boundary)}.",
            markdown_table(
                ["model", "params", "verification", "k=10 F1", "k=20 F1", "k=50 F1"],
                published_rows,
            ),
            f"Conclusion: {conclusion}",
            (
                "Stage 1 v2 outputs: `results/training_curves_v2.csv`, `results/stage1_metrics_v2.csv`, "
                "`results/stage1_summary_v2.csv`, `results/stage1_primary_metrics_v2.csv`, "
                "`results/stage1_comparison_table_v2.csv`, `results/stage1_convergence_v2.csv`, and "
                "`results/stage1_v1_v2_delta.csv`."
            ),
            "Hard stop: Stage 2 HLS export/csim/csynth was not started in this run.",
        ]
    )


def replace_or_append_section(report_text: str, section: str) -> str:
    marker = SECTION_TITLE
    if marker not in report_text:
        return report_text.rstrip() + "\n\n" + section.rstrip() + "\n"
    prefix = report_text.split(marker, 1)[0].rstrip()
    return prefix + "\n\n" + section.rstrip() + "\n"


def main() -> None:
    primary_v1 = read_rows(RESULTS / "stage1_primary_metrics.csv")
    primary_v2 = read_rows(RESULTS / "stage1_primary_metrics_v2.csv")
    convergence_v2 = read_rows(RESULTS / "stage1_convergence_v2.csv")
    comparison_v2 = read_rows(RESULTS / "stage1_comparison_table_v2.csv")
    require_v2_complete(primary_v2)
    delta_rows = build_delta_rows(primary_v1, primary_v2)
    write_rows(RESULTS / "stage1_v1_v2_delta.csv", delta_rows)
    section = build_report_section(primary_v1, primary_v2, convergence_v2, comparison_v2, delta_rows)
    REPORT.write_text(replace_or_append_section(REPORT.read_text(encoding="utf-8"), section), encoding="utf-8")
    print(section)


if __name__ == "__main__":
    main()
