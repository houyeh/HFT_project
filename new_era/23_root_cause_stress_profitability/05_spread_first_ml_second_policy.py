from __future__ import annotations

import json

import numpy as np
import pandas as pd

from root_cause_utils import (
    RESULTS_DIR,
    alignment_metrics,
    eval_quote_selection,
    ensure_dirs,
    load_context,
    select_by_threshold,
    selected_quotes_for_policy,
    spread_only_higher_is_better,
    spread_only_selection_spec,
    spread_only_threshold,
    write_markdown,
)


def rolling_select(scores: np.ndarray, *, window: int, quantile: float, higher_is_better: bool) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    rolling = s.shift(1).rolling(window=int(window), min_periods=max(5, int(window // 5))).quantile(float(quantile))
    if higher_is_better:
        return (s >= rolling).fillna(False).to_numpy(dtype=bool)
    return (s <= rolling).fillna(False).to_numpy(dtype=bool)


def build_policies(frame: pd.DataFrame, actions: pd.DataFrame, rolling_spec: dict | None = None) -> dict[str, np.ndarray]:
    spread_first = select_by_threshold(
        actions["spread_only_score"].to_numpy(dtype=float),
        spread_only_threshold(),
        higher_is_better=spread_only_higher_is_better(),
    )
    sel015 = selected_quotes_for_policy(frame, actions, "015")
    sel016 = selected_quotes_for_policy(frame, actions, "016_v2")
    out = {
        "spread_first_only": spread_first,
        "spread_first_plus_015": spread_first & sel015,
        "spread_first_plus_016_v2": spread_first & sel016,
        "spread_first_plus_ensemble": spread_first & sel015 & sel016,
    }
    if rolling_spec is not None:
        rolling_sel = rolling_select(
            actions[str(rolling_spec["score_column"])].to_numpy(dtype=float),
            window=int(rolling_spec["window"]),
            quantile=float(rolling_spec["quantile"]),
            higher_is_better=bool(rolling_spec["higher_is_better"]),
        )
        out["spread_first_plus_rolling_ML"] = spread_first & rolling_sel
    return out


def run() -> None:
    ensure_dirs()
    _, val_actions = load_context("val")
    val_grid_path = RESULTS_DIR / "rolling_quantile_validation_grid.csv"
    rolling_spec = None
    if val_grid_path.exists():
        rolling_spec = pd.read_csv(val_grid_path).sort_values("selected_quote_pnl", ascending=False).iloc[0].to_dict()

    rows = []
    detail_rows = []
    selected_specs = {
        "spread_first_threshold_source": "validation-selected raw spread gate",
        **spread_only_selection_spec(),
    }
    for split in ["val", "test"]:
        frame, actions = load_context(split)
        policies = build_policies(frame, actions, rolling_spec)
        base_pnl = None
        for name, selected in policies.items():
            metrics = eval_quote_selection(actions, selected, name)
            if name == "spread_first_only":
                base_pnl = metrics["selected_quote_pnl"]
            rows.append(
                {
                    "split": "validation" if split == "val" else "test_diagnostic_burned",
                    "policy": name,
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    **metrics,
                    "incremental_pnl_over_spread_first_only": metrics["selected_quote_pnl"] - float(base_pnl if base_pnl is not None else 0.0),
                }
            )
            detail = alignment_metrics(actions, selected, name, ["session_id", "side", "spread_bucket"])
            detail["split"] = "validation" if split == "val" else "test_diagnostic_burned"
            detail["selection_split"] = "validation"
            detail["test_used_for_selection"] = False
            detail_rows.append(detail)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "spread_first_ml_second_policy.csv", index=False)
    if detail_rows:
        pd.concat(detail_rows, ignore_index=True).to_csv(RESULTS_DIR / "spread_first_ml_second_by_session_side_spread.csv", index=False)
    (RESULTS_DIR / "spread_first_policy_manifest.json").write_text(json.dumps(selected_specs, indent=2), encoding="utf-8")

    test = out[out["split"] == "test_diagnostic_burned"].copy()
    best = test.sort_values("selected_quote_pnl", ascending=False).iloc[0]
    lines = [
        "# Spread-First, ML-Second Policy",
        "",
        "Spread-first uses the 022 validation-selected `spread_only` Quote-EV policy. ML-second ablations only remove quotes after the spread-first gate.",
        "",
        f"Best burned diagnostic test action-level quote PnL: `{best['policy']}` at `{best['selected_quote_pnl']:.3f}`.",
        "",
        "Incremental value is reported relative to `spread_first_only` on the same split.",
    ]
    write_markdown(RESULTS_DIR / "spread_first_ml_second_policy_report.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote spread-first ML-second outputs to {RESULTS_DIR}")
