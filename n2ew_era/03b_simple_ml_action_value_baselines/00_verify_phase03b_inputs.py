from __future__ import annotations

import pandas as pd

from common_phase03b import (
    PHASE02F_DATA,
    PHASE03A6,
    RAW_ABSOLUTE_PRICE_FEATURES,
    REBUILT_DYNAMIC_COLUMNS,
    REBUILT_DYNAMICS,
    RESULTS,
    ROOT,
    SPLITS,
    TARGET,
    POSITIVE_TARGET,
    ensure_dirs,
    feature_matrix_columns,
    part_paths,
    schema_columns,
    dynamic_paths,
    validate_feature_schema,
    write_feature_schema,
    write_markdown,
)


def main() -> None:
    ensure_dirs()
    rows = []
    gates = pd.read_csv(PHASE03A6 / "results" / "root_cause_closure_gates.csv")
    closure_pass = bool(gates["pass"].astype(bool).all())
    rows.append({"audit_item": "phase03a6_closure_pass", "status": "PASS" if closure_pass else "FAIL", "details": str(closure_pass)})

    if not PHASE02F_DATA.exists():
        rows.append({"audit_item": "phase02f_source_exists", "status": "FAIL", "details": str(PHASE02F_DATA)})
    else:
        rows.append({"audit_item": "phase02f_source_exists", "status": "PASS", "details": str(PHASE02F_DATA)})

    feature_rows = validate_feature_schema(feature_matrix_columns())
    feature_audit = pd.DataFrame(feature_rows)
    feature_audit.to_csv(RESULTS / "feature_schema_audit.csv", index=False)
    no_forbidden_features = bool(feature_audit["allowed_for_phase03b"].astype(bool).all())
    rows.append({"audit_item": "forbidden_features_absent_from_ml_schema", "status": "PASS" if no_forbidden_features else "FAIL", "details": feature_audit.to_json(orient="records")})

    for split in SPLITS:
        paths = part_paths(split)
        rows.append({"audit_item": f"{split}_split_parts_present", "status": "PASS" if paths else "FAIL", "details": f"parts={len(paths)}"})
        dyn = dynamic_paths(split)
        rows.append({"audit_item": f"{split}_rebuilt_dynamic_parts_present", "status": "PASS" if dyn else "FAIL", "details": f"parts={len(dyn)}"})
        for path in paths[:1]:
            cols = set(schema_columns(path))
            required = {"split", "session_id", "event_index", "action_side", "action_slot", "spread_ticks", "queue_imbalance", "deep_imbalance", "volatility_past_50", TARGET, POSITIVE_TARGET}
            missing = sorted(required - cols)
            rows.append({"audit_item": f"{split}_source_required_columns", "status": "PASS" if not missing else "FAIL", "details": ",".join(missing)})
            source_has_abs_prices = sorted(set(RAW_ABSOLUTE_PRICE_FEATURES) & cols)
            rows.append({"audit_item": f"{split}_source_contains_forbidden_raw_prices_but_not_feature_schema", "status": "PASS", "details": ",".join(source_has_abs_prices)})
        for path in dyn[:1]:
            cols = set(schema_columns(path))
            missing = sorted(set(REBUILT_DYNAMIC_COLUMNS) - cols)
            rows.append({"audit_item": f"{split}_dynamic_required_columns", "status": "PASS" if not missing else "FAIL", "details": ",".join(missing)})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "input_verification.csv", index=False)
    write_feature_schema()
    write_markdown(
        ROOT / "FEATURE_SCHEMA_03B.md",
        [
            "# Phase 03B Feature Schema",
            "",
            "Phase 03B uses only verified causal L2 and rebuilt dynamic features. Raw absolute price columns exist in the source parquet for label construction lineage, but they are excluded from the ML feature schema by the Phase 03A.6 decision.",
            "",
            "## Allowed Feature Columns",
            "",
            pd.DataFrame({"feature": feature_matrix_columns()}).to_markdown(index=False),
            "",
            "## Forbidden Columns",
            "",
            "`best_bid`, `best_ask`, `mid_price`, `quote_price`, all `future_mid_*`, all raw/excess PnL label columns, and all target columns are forbidden as ML features.",
        ],
    )
    hard_fail = out[out["status"].eq("FAIL")]
    if not hard_fail.empty:
        write_markdown(
            ROOT / "ROOT_CAUSE_BLOCKER.md",
            [
                "# Phase 03B Input Verification Failed",
                "",
                hard_fail.to_markdown(index=False),
            ],
        )
        raise RuntimeError("Phase 03B input verification failed. See ROOT_CAUSE_BLOCKER.md")
    print("Phase 03B input verification PASS")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

