from __future__ import annotations

import numpy as np
import pandas as pd

from root_cause_utils import RESULTS_DIR, alignment_metrics, ensure_dirs, eval_quote_selection, load_context, score_specs, write_markdown


def rolling_select(scores: np.ndarray, *, window: int, quantile: float, higher_is_better: bool) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    rolling = s.shift(1).rolling(window=int(window), min_periods=max(5, int(window // 5))).quantile(float(quantile))
    if higher_is_better:
        return (s >= rolling).fillna(False).to_numpy(dtype=bool)
    return (s <= rolling).fillna(False).to_numpy(dtype=bool)


def calibration_transfer(val_actions: pd.DataFrame, test_actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_name, spec in score_specs().items():
        col = spec["column"]
        higher = bool(spec["higher_is_better"])
        val_scores = val_actions[col].astype(float).to_numpy()
        test_scores = test_actions[col].astype(float).to_numpy()
        for q in [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]:
            thr = float(np.nanquantile(val_scores, q))
            if higher:
                val_mask = val_scores >= thr
                test_mask = test_scores >= thr
            else:
                val_mask = val_scores <= thr
                test_mask = test_scores <= thr
            rows.append(
                {
                    "score_name": score_name,
                    "validation_quantile": q,
                    "threshold_from_validation": thr,
                    "higher_is_better": higher,
                    "val_selected_rate": float(np.mean(val_mask)),
                    "test_diagnostic_selected_rate": float(np.mean(test_mask)),
                    "val_selected_pnl": float(val_actions["quote_value_moderate"].to_numpy(dtype=float)[val_mask].sum()),
                    "test_diagnostic_selected_pnl": float(test_actions["quote_value_moderate"].to_numpy(dtype=float)[test_mask].sum()),
                    "test_used_for_selection": False,
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    ensure_dirs()
    _, val_actions = load_context("val")
    _, test_actions = load_context("test")
    calibration_transfer(val_actions, test_actions).to_csv(RESULTS_DIR / "calibration_transfer.csv", index=False)

    candidates = []
    for score_name, spec in score_specs().items():
        col = spec["column"]
        higher = bool(spec["higher_is_better"])
        for window in [25, 50, 100, 200]:
            quantiles = [0.80, 0.90, 0.95] if higher else [0.05, 0.10, 0.20]
            for q in quantiles:
                sel = rolling_select(val_actions[col].to_numpy(dtype=float), window=window, quantile=q, higher_is_better=higher)
                metrics = eval_quote_selection(val_actions, sel, f"{score_name}_w{window}_q{q}")
                candidates.append(
                    {
                        "score_name": score_name,
                        "score_column": col,
                        "window": window,
                        "quantile": q,
                        "higher_is_better": higher,
                        "selection_split": "validation",
                        "used_for_model_selection": True,
                        **metrics,
                    }
                )
    val_grid = pd.DataFrame(candidates).sort_values("selected_quote_pnl", ascending=False).reset_index(drop=True)
    selected = val_grid.iloc[0].to_dict()
    val_grid.to_csv(RESULTS_DIR / "rolling_quantile_validation_grid.csv", index=False)

    test_rows = []
    detail_rows = []
    for _, row in val_grid.iterrows():
        col = row["score_column"]
        sel = rolling_select(
            test_actions[col].to_numpy(dtype=float),
            window=int(row["window"]),
            quantile=float(row["quantile"]),
            higher_is_better=bool(row["higher_is_better"]),
        )
        metrics = eval_quote_selection(test_actions, sel, str(row["policy"]))
        test_rows.append(
            {
                "score_name": row["score_name"],
                "score_column": col,
                "window": int(row["window"]),
                "quantile": float(row["quantile"]),
                "higher_is_better": bool(row["higher_is_better"]),
                "split": "test_diagnostic_burned",
                "DIAGNOSTIC_ONLY": True,
                "used_for_model_selection": False,
                "is_selected_validation_policy": bool(
                    row["score_name"] == selected["score_name"]
                    and int(row["window"]) == int(selected["window"])
                    and float(row["quantile"]) == float(selected["quantile"])
                ),
                **metrics,
            }
        )
        if bool(
            row["score_name"] == selected["score_name"]
            and int(row["window"]) == int(selected["window"])
            and float(row["quantile"]) == float(selected["quantile"])
        ):
            detail = alignment_metrics(test_actions, sel, str(row["policy"]), ["session_id", "side", "spread_bucket"])
            detail["split"] = "test_diagnostic_burned"
            detail["DIAGNOSTIC_ONLY"] = True
            detail["used_for_model_selection"] = False
            detail["score_name"] = row["score_name"]
            detail["window"] = int(row["window"])
            detail["quantile"] = float(row["quantile"])
            detail_rows.append(detail)
    pd.DataFrame(test_rows).to_csv(RESULTS_DIR / "rolling_quantile_test_diagnostic.csv", index=False)
    if detail_rows:
        pd.concat(detail_rows, ignore_index=True).to_csv(RESULTS_DIR / "rolling_quantile_selected_by_session_side_spread.csv", index=False)

    write_markdown(
        RESULTS_DIR / "calibration_transfer_and_rolling_quantile_report.md",
        [
            "# Calibration Transfer and Rolling Quantile",
            "",
            "Validation quantiles are mapped to the burned diagnostic test without retuning.",
            "Rolling quantile policy parameters are selected on validation only. Test sweeps are marked `DIAGNOSTIC_ONLY` and `used_for_model_selection=False`.",
            "",
            f"Selected validation rolling policy: `{selected['policy']}` with validation selected quote PnL `{selected['selected_quote_pnl']:.3f}`.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote calibration/rolling outputs to {RESULTS_DIR}")
