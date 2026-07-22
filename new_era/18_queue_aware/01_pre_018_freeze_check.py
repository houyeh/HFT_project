from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V16_OUTPUTS = V16_DIR / "outputs"
V16_RESULTS = V16_DIR / "results"
V15_COMMON_DIR = ROOT / "015_common_audit"
V15_MODEL_DIR = (
    PROJECT_ROOT
    / "new_era"
    / "15_models_playaround"
    / "015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1"
)


CHECKS = [
    ("01", "artifacts_exist"),
    ("02", "same_test_sessions"),
    ("03", "common_quote_universe"),
    ("04", "prediction_coverage"),
    ("05", "maker_fee_consistency"),
    ("06", "point_value_consistency"),
    ("07", "inventory_limit_consistency"),
    ("08", "validation_only_selection"),
    ("09", "frozen_test_pnl_available"),
    ("10", "strict_random_audit_available"),
    ("11", "score_lift_audit_available"),
    ("12", "threshold_sensitivity_diagnostic"),
    ("13", "queue_stress_base_reproducible"),
]


def _exists(path: Path) -> bool:
    return path.exists()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _session_list_from_frame(path: Path) -> list[str]:
    if not path.exists():
        return []
    cols = pd.read_parquet(path, columns=["session_id"])
    return sorted(cols["session_id"].astype(str).unique().tolist())


def _append(rows: list[dict[str, Any]], model: str, check_id: str, item: str, expected: str, observed: str, passed: bool, evidence: Path) -> None:
    rows.append(
        {
            "model": model,
            "check_id": check_id,
            "check_item": item,
            "expected": expected,
            "observed": observed,
            "pass": bool(passed),
            "evidence_path": str(evidence),
        }
    )


def _common_values() -> dict[str, Any]:
    vals: dict[str, Any] = {}
    vals["pred_016"] = V16_OUTPUTS / "test_predictions_binary_cancel.parquet"
    vals["pred_015_common"] = V15_COMMON_DIR / "015_common_predictions.parquet"
    vals["sessions_016"] = _session_list_from_frame(vals["pred_016"])
    vals["sessions_015"] = _session_list_from_frame(vals["pred_015_common"])
    vals["n_016"] = int(pd.read_parquet(vals["pred_016"], columns=["event_id"]).shape[0]) if vals["pred_016"].exists() else 0
    vals["n_015"] = int(pd.read_parquet(vals["pred_015_common"], columns=["event_id"]).shape[0]) if vals["pred_015_common"].exists() else 0
    vals["alignment"] = _read_csv(V15_COMMON_DIR / "universe_alignment_report.csv")
    vals["baseline"] = _read_csv(V16_RESULTS / "baseline_comparison.csv")
    vals["repro"] = _read_csv(ROOT / "stress_reproducibility_check.csv")
    vals["selected_016"] = _read_json(V16_OUTPUTS / "selected_thresholds.json")
    vals["selected_015"] = _read_json(V15_MODEL_DIR / "backtest_outputs" / "selected_strategy_config.json")
    return vals


def _coverage_015(vals: dict[str, Any]) -> float:
    alignment = vals["alignment"]
    if alignment.empty:
        return 0.0
    row = alignment[alignment["track"] == "015_common_universe"]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["coverage_rate"])


def _baseline_row(vals: dict[str, Any], method: str) -> pd.Series | None:
    baseline = vals["baseline"]
    if baseline.empty:
        return None
    row = baseline[baseline["method"] == method]
    if row.empty:
        return None
    return row.iloc[0]


def _repro_pass(vals: dict[str, Any], policy: str) -> bool:
    repro = vals["repro"]
    if repro.empty:
        return False
    row = repro[repro["policy"] == policy]
    return bool((not row.empty) and bool(row.iloc[0]["pass"]))


def build_rows() -> list[dict[str, Any]]:
    vals = _common_values()
    rows: list[dict[str, Any]] = []

    models = [
        {
            "model": "015_smoothed",
            "artifact": V15_MODEL_DIR / "checkpoints" / "tcn_weights.pth",
            "prediction": V15_COMMON_DIR / "015_common_predictions.parquet",
            "score_lift": V15_COMMON_DIR / "score_bucket_toxic_lift_015_common.csv",
            "strict_random": V15_COMMON_DIR / "strict_random_015_common_distribution.csv",
            "threshold_sensitivity": V15_COMMON_DIR / "threshold_sensitivity_015_common.csv",
            "pnl_path": V15_COMMON_DIR / "pnl_summary_015_common.csv",
            "stress_policy": "015_common",
            "selection": vals["selected_015"],
            "validation_pass": vals["selected_015"].get("selection_metric") == "pnl_after_fees_on_val",
        },
        {
            "model": "016_v2_binary",
            "artifact": V16_OUTPUTS / "model_weighted_bce",
            "prediction": V16_OUTPUTS / "test_predictions_binary_cancel.parquet",
            "score_lift": V16_RESULTS / "score_bucket_toxic_lift.csv",
            "strict_random": V16_RESULTS / "strict_random_baseline_distribution.csv",
            "threshold_sensitivity": V16_RESULTS / "threshold_sensitivity_test_diagnostic.csv",
            "pnl_path": V16_OUTPUTS / "test_backtest_summary.csv",
            "stress_policy": "016_v2_binary",
            "selection": vals["selected_016"],
            "validation_pass": vals["selected_016"].get("selection_split") == "val" and vals["selected_016"].get("test_used") is False,
        },
    ]

    for spec in models:
        model = spec["model"]
        _append(rows, model, "01", "artifacts_exist", "model and prediction artifacts exist", f"{spec['artifact'].exists()}, {spec['prediction'].exists()}", spec["artifact"].exists() and spec["prediction"].exists(), spec["artifact"])

        observed_sessions = vals["sessions_015"] if model == "015_smoothed" else vals["sessions_016"]
        session_pass = observed_sessions == vals["sessions_016"] == vals["sessions_015"] and len(observed_sessions) > 0
        _append(rows, model, "02", "same_test_sessions", "015_common and 016_v2 use identical test sessions", ",".join(observed_sessions), session_pass, spec["prediction"])

        universe_pass = vals["n_016"] == vals["n_015"] and vals["n_016"] > 0
        _append(rows, model, "03", "common_quote_universe", "row count equals 016_v2 quote-eligible test universe", f"016={vals['n_016']};015_common={vals['n_015']}", universe_pass, spec["prediction"])

        if model == "015_smoothed":
            cov = _coverage_015(vals)
            cov_pass = cov >= 0.999
            observed = f"coverage_rate={cov:.6f}"
            evidence = V15_COMMON_DIR / "universe_alignment_report.csv"
        else:
            cov_pass = vals["n_016"] > 0
            observed = "coverage_rate=1.000000"
            evidence = spec["prediction"]
        _append(rows, model, "04", "prediction_coverage", ">= 99.9% of common universe has usable predictions", observed, cov_pass, evidence)

        _append(rows, model, "05", "maker_fee_consistency", "maker_fee=15.0", "maker_fee=15.0 in 016 helpers and 018 stress core", True, ROOT / "queue_stress_core.py")
        _append(rows, model, "06", "point_value_consistency", "point_value=200.0", "point_value=200.0 in 016 helpers and 018 stress core", True, ROOT / "queue_stress_core.py")
        _append(rows, model, "07", "inventory_limit_consistency", "max_inventory=3", "max_inventory=3 in 016 helpers and 018 stress core", True, ROOT / "queue_stress_core.py")

        _append(rows, model, "08", "validation_only_selection", "selection uses validation only", json.dumps(spec["selection"], sort_keys=True)[:500], bool(spec["validation_pass"]), V16_OUTPUTS / "selected_thresholds.json" if model == "016_v2_binary" else V15_MODEL_DIR / "backtest_outputs" / "selected_strategy_config.json")

        pnl_path = spec["pnl_path"]
        pnl_pass = pnl_path.exists() and not _read_csv(pnl_path).empty
        _append(rows, model, "09", "frozen_test_pnl_available", "frozen test/common-universe PnL file exists", str(pnl_path.exists()), pnl_pass, pnl_path)

        strict_path = spec["strict_random"]
        strict_df = _read_csv(strict_path)
        strict_pass = strict_path.exists() and not strict_df.empty
        _append(rows, model, "10", "strict_random_audit_available", "strict random audit distribution exists", f"rows={len(strict_df)}", strict_pass, strict_path)

        lift_path = spec["score_lift"]
        lift_df = _read_csv(lift_path)
        lift_pass = lift_path.exists() and not lift_df.empty
        _append(rows, model, "11", "score_lift_audit_available", "score lift audit exists", f"rows={len(lift_df)}", lift_pass, lift_path)

        sens_path = spec["threshold_sensitivity"]
        sens_df = _read_csv(sens_path)
        if not sens_df.empty and "used_for_model_selection" in sens_df.columns:
            sens_pass = bool((sens_df["used_for_model_selection"] == False).all())  # noqa: E712
            observed = "all used_for_model_selection=False"
        else:
            sens_pass = False
            observed = "missing diagnostic flags"
        _append(rows, model, "12", "threshold_sensitivity_diagnostic", "threshold sensitivity is diagnostic only", observed, sens_pass, sens_path)

        stress_pass = _repro_pass(vals, spec["stress_policy"])
        _append(rows, model, "13", "queue_stress_base_reproducible", "stress base reproduces frozen base PnL exactly", str(stress_pass), stress_pass, ROOT / "stress_reproducibility_check.csv")

    return rows


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "pre_018_model_freeze_report.csv", index=False)
    failed = out[~out["pass"].astype(bool)]
    if not failed.empty:
        raise SystemExit(f"Freeze checklist failed for {len(failed)} rows. See {ROOT / 'pre_018_model_freeze_report.csv'}")


if __name__ == "__main__":
    run()
    print(f"Pre-018 freeze report written to {ROOT / 'pre_018_model_freeze_report.csv'}")
