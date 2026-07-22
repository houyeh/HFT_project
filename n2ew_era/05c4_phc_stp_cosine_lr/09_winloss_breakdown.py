from __future__ import annotations

import math

import numpy as np
import pandas as pd

import common_05c4 as c


OUT_CSV = c.RESULTS / "phc_stp_v2_winloss_breakdown.csv"
OUT_MD = c.RESULTS / "phc_stp_v2_winloss_breakdown.md"
OOS_EVAL_CSV = c.RESULTS / "phc_stp_v2_oos_eval.csv"


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _relative_identity_error(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale


def compute_breakdown() -> pd.DataFrame:
    model, ckpt, selected = c.selected_model()
    threshold = float(selected["threshold"])
    oos = pd.read_csv(OOS_EVAL_CSV, dtype={"win_rate": "string"}).set_index("split")

    rows: list[dict[str, float | int | str | bool]] = []
    for split in c.SPLITS:
        payload = c.p05a.load_sequences(split)
        decision, _ = c.stp_decision_for_payload(payload, model, ckpt, threshold)
        pnl = c.p05a.decision_pnl_from_payload(payload, decision)
        selected_mask = decision != c.p05a.DECISION_NO_QUOTE
        selected_pnl = pnl[selected_mask]

        win_mask = selected_pnl > 0
        loss_mask = selected_pnl < 0
        zero_mask = selected_pnl == 0

        win_count = int(win_mask.sum())
        loss_count = int(loss_mask.sum())
        zero_count = int(zero_mask.sum())
        selected_count = int(len(selected_pnl))

        avg_win = _safe_mean(selected_pnl[win_mask])
        avg_loss = _safe_mean(selected_pnl[loss_mask])
        avg_loss_abs = abs(avg_loss) if math.isfinite(avg_loss) else float("nan")
        win_loss_ratio = avg_win / avg_loss_abs if avg_loss_abs and math.isfinite(avg_loss_abs) else float("nan")
        win_rate_check = float(win_mask.mean()) if selected_count else 0.0
        negative_rate_check = float(loss_mask.mean()) if selected_count else 0.0
        avg_selected_excess_check = float(selected_pnl.mean()) if selected_count else 0.0
        identity_lhs = win_rate_check * avg_win + negative_rate_check * avg_loss
        identity_rhs = avg_selected_excess_check
        identity_relative_error = _relative_identity_error(identity_lhs, identity_rhs)
        identity_pass = identity_relative_error <= 1e-9

        existing_win_rate_text = str(oos.loc[split, "win_rate"])
        computed_win_rate_text = str(win_rate_check)
        expected_win_rate = float(existing_win_rate_text)
        win_rate_exact_match = computed_win_rate_text == existing_win_rate_text
        if not win_rate_exact_match:
            raise RuntimeError(
                f"win_rate sanity check failed for {split}: "
                f"computed_text={computed_win_rate_text!r}, "
                f"existing_csv_text={existing_win_rate_text!r}"
            )
        if not identity_pass:
            raise RuntimeError(
                f"identity sanity check failed for {split}: "
                f"lhs={identity_lhs:.17g}, rhs={identity_rhs:.17g}, "
                f"relative_error={identity_relative_error:.17g}"
            )

        rows.append(
            {
                "split": split,
                "policy_id": c.POLICY_ID,
                "seed_selected_on_val": int(selected["seed"]),
                "threshold_selected_on_val": threshold,
                "selected_action_count": selected_count,
                "win_count": win_count,
                "loss_count": loss_count,
                "zero_count": zero_count,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_loss_abs": avg_loss_abs,
                "win_loss_ratio": win_loss_ratio,
                "win_rate_check": win_rate_check,
                "existing_win_rate": expected_win_rate,
                "computed_win_rate_text": computed_win_rate_text,
                "existing_win_rate_text": existing_win_rate_text,
                "win_rate_exact_match": win_rate_exact_match,
                "negative_rate_check": negative_rate_check,
                "avg_selected_excess_check": avg_selected_excess_check,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_relative_error": identity_relative_error,
                "identity_pass": identity_pass,
            }
        )

    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame) -> None:
    display = df[
        [
            "split",
            "selected_action_count",
            "win_count",
            "loss_count",
            "zero_count",
            "avg_win",
            "avg_loss",
            "avg_loss_abs",
            "win_loss_ratio",
            "win_rate_check",
            "negative_rate_check",
            "avg_selected_excess_check",
            "identity_lhs",
            "identity_rhs",
            "identity_relative_error",
        ]
    ].copy()
    numeric_cols = [
        "avg_win",
        "avg_loss",
        "avg_loss_abs",
        "win_loss_ratio",
        "win_rate_check",
        "negative_rate_check",
        "avg_selected_excess_check",
        "identity_lhs",
        "identity_rhs",
        "identity_relative_error",
    ]
    for col in numeric_cols:
        display[col] = display[col].map(lambda x: f"{float(x):.12g}")
    sanity_display = df[
        [
            "split",
            "win_rate_check",
            "existing_win_rate",
            "computed_win_rate_text",
            "existing_win_rate_text",
            "win_rate_exact_match",
            "identity_lhs",
            "identity_rhs",
            "identity_relative_error",
            "identity_pass",
        ]
    ].copy()
    for col in ["win_rate_check", "existing_win_rate", "identity_lhs", "identity_rhs", "identity_relative_error"]:
        sanity_display[col] = sanity_display[col].map(lambda x: f"{float(x):.17g}")
    ratio_min = float(df["win_loss_ratio"].min())
    ratio_max = float(df["win_loss_ratio"].max())
    if ratio_min > 1.0:
        interpretation = (
            "All three splits have `win_loss_ratio > 1`, so the average winning selected "
            "action is larger than the absolute average losing selected action. This is "
            "direct evidence of asymmetric payoff magnitude, explaining how the selected "
            "policy can remain positive even when the selected-action win rate is near "
            "or slightly below 50%."
        )
    else:
        interpretation = (
            "At least one split does not have `win_loss_ratio > 1`. The payoff magnitude "
            "asymmetry alone would not be enough to explain positive total excess in every "
            "split, so any positive result would need additional explanation."
        )

    lines = [
        "# PHC STP v2 Win/Loss Magnitude Breakdown",
        "",
        "Scope: post-hoc statistics for the frozen `phc_stp_v2_cosine_lr` selected checkpoint and validation-selected threshold.",
        "",
        "No training is run. The script reuses `selected_model()`, `stp_decision_for_payload()`, and `decision_pnl_from_payload()` from the existing 05C4/05A code path.",
        "",
        "Splits analyzed: `val`, `new3`, `new4`. No other split is loaded.",
        "",
        "Definitions:",
        "",
        "- `avg_win`: mean selected-action excess over selected rows where excess is strictly positive.",
        "- `avg_loss`: mean selected-action excess over selected rows where excess is strictly negative.",
        "- `avg_loss_abs`: absolute value of `avg_loss`.",
        "- `win_loss_ratio`: `avg_win / avg_loss_abs`.",
        "- zero excess rows are counted separately and excluded from both win and loss counts.",
        "",
        "## Breakdown",
        "",
        display.to_markdown(index=False, disable_numparse=True),
        "",
        "## Sanity Checks",
        "",
        "The `win_rate_check` value is compared against the existing `results/phc_stp_v2_oos_eval.csv` `win_rate` column. Exact equality is required; otherwise the script stops.",
        "",
        "The arithmetic identity checked is:",
        "",
        "`win_rate_check * avg_win + negative_rate_check * avg_loss ~= avg_selected_excess_check`",
        "",
        sanity_display.to_markdown(index=False, disable_numparse=True),
        "",
        "## Interpretation",
        "",
        f"Observed `win_loss_ratio` range: `{ratio_min:.6f}` to `{ratio_max:.6f}`.",
        "",
        interpretation,
        "",
        "This is a descriptive decomposition of already-selected actions. It does not change the model, threshold, or policy.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    c.ensure_dirs()
    df = compute_breakdown()
    df.to_csv(OUT_CSV, index=False)
    write_report(df)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
