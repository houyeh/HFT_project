from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from common import MODEL_DIAG_DIR, POLICIES_DIR, ensure_output_dirs, get_logger


def _parse_gate_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pick_top(sub: pd.DataFrame) -> pd.Series | None:
    if sub.empty:
        return None
    return sub.sort_values(["net_pnl", "trade_count"], ascending=[False, False]).iloc[0]


def _row_from_gate(
    dataset: str,
    policy_name: str,
    gate_row: pd.Series,
    *,
    oracle_flag: bool,
    deployable_flag: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "policy_name": policy_name,
        "gate_type": str(gate_row["gate_type"]),
        "gate_params": str(gate_row["gate_params"]),
        "model_threshold": _safe_float(gate_row["model_threshold"]),
        "net_pnl": _safe_float(gate_row["net_pnl"]),
        "baseline_net_pnl": _safe_float(gate_row["baseline_net_pnl"]),
        "improvement": _safe_float(gate_row["improvement"]),
        "active_event_rate": _safe_float(gate_row["active_event_rate"]),
        "trade_count": int(_safe_float(gate_row["trade_count"], 0.0)),
        "max_drawdown": _safe_float(gate_row["max_drawdown"]),
        "oracle_flag": bool(oracle_flag),
        "deployable_flag": bool(deployable_flag),
        "notes": notes,
    }


def main() -> None:
    ensure_output_dirs()
    logger = get_logger("evaluate_static_gates")
    regime_path = MODEL_DIAG_DIR / "regime_gate_sweep.csv"
    baseline_summary_path = MODEL_DIAG_DIR / "baseline_summary_fixed.csv"
    baseline_detail_path = MODEL_DIAG_DIR / "baseline_comparison_fixed.csv"

    regime_df = pd.read_csv(regime_path)
    baseline_summary_df = pd.read_csv(baseline_summary_path)
    baseline_detail_df = pd.read_csv(baseline_detail_path)

    regime_df["gate_params_obj"] = regime_df["gate_params"].map(_parse_gate_params)
    rows: list[dict[str, Any]] = []

    for dataset in sorted(regime_df["dataset"].dropna().unique().tolist()):
        ds = regime_df[regime_df["dataset"] == dataset].copy()

        spread_only = ds[ds["gate_type"] == "spread_filter"].copy()
        spread_only["spread_ticks_min"] = spread_only["gate_params_obj"].map(
            lambda p: _safe_float(p.get("spread_ticks_min", float("nan")))
        )
        combined_only = ds[ds["gate_type"] == "combined_filter"].copy()

        no_gate_sub = spread_only[np.isfinite(spread_only["spread_ticks_min"]) & (spread_only["spread_ticks_min"] <= 1.0 + 1e-9)]
        if no_gate_sub.empty:
            no_gate_sub = spread_only[spread_only["active_event_rate"] >= 0.999]
        no_gate_row = _pick_top(no_gate_sub)
        if no_gate_row is not None:
            rows.append(
                _row_from_gate(
                    dataset,
                    "no_gate_tcn",
                    no_gate_row,
                    oracle_flag=False,
                    deployable_flag=True,
                    notes="No extra regime gate; equivalent to spread_filter with spread_ticks_min close to 1.0.",
                )
            )

        spread_row = _pick_top(spread_only)
        if spread_row is not None:
            rows.append(
                _row_from_gate(
                    dataset,
                    "always_spread_filter",
                    spread_row,
                    oracle_flag=False,
                    deployable_flag=True,
                    notes="Best static spread_filter from existing sweep.",
                )
            )

        combined_row = _pick_top(combined_only)
        if combined_row is not None:
            rows.append(
                _row_from_gate(
                    dataset,
                    "always_combined_filter",
                    combined_row,
                    oracle_flag=False,
                    deployable_flag=True,
                    notes="Best static combined_filter from existing sweep.",
                )
            )

        oracle_candidates = [r for r in [spread_row, combined_row] if r is not None]
        if oracle_candidates:
            oracle_row = sorted(
                oracle_candidates,
                key=lambda s: (_safe_float(s["net_pnl"]), _safe_float(s["trade_count"])),
                reverse=True,
            )[0]
            rows.append(
                _row_from_gate(
                    dataset,
                    "dataset_oracle_best_gate",
                    oracle_row,
                    oracle_flag=True,
                    deployable_flag=False,
                    notes="Oracle selected using dataset identity / hindsight; not deployable.",
                )
            )

        if dataset == "NEW":
            random_rows = baseline_summary_df[
                baseline_summary_df["experiment_group"].astype(str).str.startswith("Random_")
            ].copy()
            if not random_rows.empty:
                random_top = random_rows.sort_values("mean_net_pnl", ascending=False).iloc[0]
                random_group = str(random_top["experiment_group"])
                detail = baseline_detail_df[baseline_detail_df["experiment_group"] == random_group]
                mean_signal = _safe_float(detail["signal_rate"].mean()) if not detail.empty else float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "policy_name": "random_cancel_reference",
                        "gate_type": "random_cancel",
                        "gate_params": json.dumps({"group": random_group}, ensure_ascii=False),
                        "model_threshold": _safe_float(str(random_group).replace("Random_", "")),
                        "net_pnl": _safe_float(random_top["mean_net_pnl"]),
                        "baseline_net_pnl": _safe_float(
                            baseline_summary_df.loc[
                                baseline_summary_df["experiment_group"] == "Baseline_A_NoModel", "mean_net_pnl"
                            ].iloc[0]
                        ),
                        "improvement": _safe_float(
                            random_top["mean_net_pnl"]
                            - baseline_summary_df.loc[
                                baseline_summary_df["experiment_group"] == "Baseline_A_NoModel", "mean_net_pnl"
                            ].iloc[0]
                        ),
                        "active_event_rate": mean_signal,
                        "trade_count": int(_safe_float(random_top["mean_trade_count"], 0.0)),
                        "max_drawdown": _safe_float(random_top["mean_max_drawdown"]),
                        "oracle_flag": False,
                        "deployable_flag": False,
                        "notes": "Random cancel baseline from existing diagnosis output (NEW only).",
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(POLICIES_DIR / "static_gate_summary.csv", index=False)
    logger.info("Saved static gate summary with %s rows", len(out_df))


if __name__ == "__main__":
    main()

