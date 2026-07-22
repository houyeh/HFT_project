from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04a2_product_filter_and_loader_contract_closure"
RESULTS = ROOT / "results"
PHASE04A1 = PROJECT_ROOT / "n2ew_era" / "04a1_dataset_registry_patch"
EXPANDED_REGISTRY = PHASE04A1 / "results" / "dataset_registry_expanded.csv"

CHUNKSIZE = 250_000

# Primary final protocol: TXF only.  MXF/TMF require separate point-value,
# tick-size, and fee normalization before pooled evaluation.
PRIMARY_TARGET_PRODUCT_RE = re.compile(r"^TXF[A-Z0-9]*$")
LEGACY_BROAD_TARGET_PRODUCT_RE = re.compile(r"^(TXF|MXF|TMF)[A-Z0-9]*$")
MIN_USABLE_TARGET_RATE = 0.50
MIN_USABLE_TARGET_ROWS = 50_000


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def parse_info_time(value: Any) -> tuple[float | None, bool]:
    parts = str(value).strip().split(":")
    if len(parts) != 5:
        return None, False
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2])
        millisecond = int(parts[3])
        fifth = int(parts[4])
    except ValueError:
        return None, False
    if fifth != 0:
        return None, False
    us = float(((hour * 60 + minute) * 60 + second) * 1_000_000 + millisecond * 1_000)
    return us, True


def is_primary_target(product_id: Any) -> bool:
    return bool(PRIMARY_TARGET_PRODUCT_RE.match(str(product_id).strip()))


def is_legacy_broad_target(product_id: Any) -> bool:
    return bool(LEGACY_BROAD_TARGET_PRODUCT_RE.match(str(product_id).strip()))


def product_family(product_id: Any) -> str:
    text = str(product_id).strip()
    if text.startswith("TXF"):
        return "TXF"
    if text.startswith("MXF"):
        return "MXF"
    if text.startswith("TMF"):
        return "TMF"
    return text[:3] if len(text) >= 3 else text


def summarize_counts(counter: Counter[str], max_items: int | None = None) -> str:
    items = counter.most_common(max_items)
    return " | ".join(f"{key}:{value}" for key, value in items)


def scan_file(row: pd.Series) -> dict[str, Any]:
    path = Path(row["physical_path"])
    raw_rows = 0
    valid_info_time_rows = 0
    malformed_info_time_rows = 0
    target_rows = 0
    target_valid_info_time_rows = 0
    broad_target_rows = 0
    target_time_min_us = np.inf
    target_time_max_us = -np.inf
    target_time_min = ""
    target_time_max = ""
    target_products: Counter[str] = Counter()
    excluded_products: Counter[str] = Counter()
    product_families: Counter[str] = Counter()
    malformed_examples: list[str] = []

    for chunk in pd.read_csv(
        path,
        usecols=lambda c: str(c).strip() in {"info_time", "product_id"},
        chunksize=CHUNKSIZE,
    ):
        chunk = clean_columns(chunk)
        chunk_start_row = raw_rows
        raw_rows += len(chunk)
        products = chunk["product_id"].fillna("").astype(str).str.strip()
        target_mask = products.map(is_primary_target)
        broad_mask = products.map(is_legacy_broad_target)
        target_rows += int(target_mask.sum())
        broad_target_rows += int(broad_mask.sum())
        target_products.update(products[target_mask].tolist())
        excluded_products.update(products[~target_mask].tolist())
        product_families.update(products.map(product_family).tolist())

        if "info_time" in chunk:
            time_values = chunk["info_time"].to_numpy()
            product_values = products.to_numpy()
            target_values = target_mask.to_numpy()
            for idx, (raw_time, product_id, is_target) in enumerate(zip(time_values, product_values, target_values, strict=False)):
                us, valid = parse_info_time(raw_time)
                if valid:
                    valid_info_time_rows += 1
                    if is_target:
                        target_valid_info_time_rows += 1
                        raw_time_str = str(raw_time).strip()
                        if us is not None and us < target_time_min_us:
                            target_time_min_us = us
                            target_time_min = raw_time_str
                        if us is not None and us > target_time_max_us:
                            target_time_max_us = us
                            target_time_max = raw_time_str
                else:
                    malformed_info_time_rows += 1
                    if len(malformed_examples) < 10:
                        malformed_examples.append(
                            f"global_row_index={chunk_start_row + idx}; product_id={product_id}; info_time={raw_time}"
                        )

    return {
        "source_group": row["source_group"],
        "assigned_role": row["assigned_role"],
        "session_id": row["session_id"],
        "file_name": row["file_name"],
        "physical_path": row["physical_path"],
        "sha256": row["sha256"],
        "raw_rows": int(raw_rows),
        "valid_info_time_rows": int(valid_info_time_rows),
        "malformed_info_time_rows": int(malformed_info_time_rows),
        "target_regex": PRIMARY_TARGET_PRODUCT_RE.pattern,
        "target_product_rows": int(target_rows),
        "target_valid_info_time_rows": int(target_valid_info_time_rows),
        "target_row_rate": float(target_rows / raw_rows) if raw_rows else 0.0,
        "target_valid_row_rate": float(target_valid_info_time_rows / raw_rows) if raw_rows else 0.0,
        "legacy_broad_txf_mxf_tmf_rows": int(broad_target_rows),
        "legacy_broad_txf_mxf_tmf_row_rate": float(broad_target_rows / raw_rows) if raw_rows else 0.0,
        "target_product_ids": summarize_counts(target_products),
        "excluded_product_ids": summarize_counts(excluded_products),
        "excluded_product_ids_top20": summarize_counts(excluded_products, 20),
        "product_family_counts": summarize_counts(product_families),
        "target_timestamp_min": target_time_min,
        "target_timestamp_max": target_time_max,
        "target_timestamp_min_us": None if not np.isfinite(target_time_min_us) else float(target_time_min_us),
        "target_timestamp_max_us": None if not np.isfinite(target_time_max_us) else float(target_time_max_us),
        "malformed_info_time_examples": " | ".join(malformed_examples),
    }


def product_universe_contract() -> pd.DataFrame:
    rows = [
        {
            "contract_item": "primary_final_product_universe",
            "decision": "TXF family only",
            "regex": PRIMARY_TARGET_PRODUCT_RE.pattern,
            "allowed_in_final_evaluation": True,
            "reason": "Inherited proxy point_value=200 and maker_fee=15 are TXF-compatible project assumptions; pooling MXF/TMF without normalization would mix contract scales.",
        },
        {
            "contract_item": "legacy_broad_target_universe",
            "decision": "TXF/MXF/TMF diagnostic only unless normalized",
            "regex": LEGACY_BROAD_TARGET_PRODUCT_RE.pattern,
            "allowed_in_final_evaluation": False,
            "reason": "MXF and TMF are not assumed comparable to TXF in point value, tick value, or fee. They require separate metrics or product-normalized PnL.",
        },
        {
            "contract_item": "raw_mixed_product_stream",
            "decision": "forbidden for pooled final evaluation",
            "regex": "all product_id values",
            "allowed_in_final_evaluation": False,
            "reason": "Raw files contain hundreds of product IDs; unfiltered evaluation would mix unrelated markets.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "product_universe_contract.csv", index=False)
    write_markdown(
        ROOT / "PRODUCT_UNIVERSE_CONTRACT.md",
        [
            "# Product Universe Contract",
            "",
            "Phase 04A.2 freezes the primary final product universe as **TXF-family only** using regex `^TXF[A-Z0-9]*$`.",
            "",
            "TXF, MXF, and TMF are not treated as directly comparable under the inherited proxy constants. The project has repeatedly used `POINT_VALUE = 200` and `MAKER_FEE = 15`; those assumptions are compatible with a TXF-style proxy contract, but they are not exchange-certified and should not be pooled with MXF/TMF without product-specific normalization.",
            "",
            "MXF/TMF may be studied only as a separate normalized diagnostic, with separate point value, tick size, fee, and reporting rows.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def scan_all_files(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in registry.iterrows():
        scanned = scan_file(row)
        rows.append(scanned)
        print(
            f"filtered {scanned['source_group']} {scanned['session_id']} {scanned['file_name']} "
            f"target={scanned['target_product_rows']}/{scanned['raw_rows']}"
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "target_product_filter_impact.csv", index=False)
    return out


def new5_usability(filter_impact: pd.DataFrame) -> pd.DataFrame:
    new5 = filter_impact[filter_impact["source_group"].eq("NEW5")].copy()
    session = new5.groupby(["source_group", "assigned_role", "session_id"], as_index=False).agg(
        files=("file_name", "count"),
        raw_rows=("raw_rows", "sum"),
        target_rows=("target_product_rows", "sum"),
        target_valid_info_time_rows=("target_valid_info_time_rows", "sum"),
        malformed_info_time_rows=("malformed_info_time_rows", "sum"),
        target_timestamp_min_us=("target_timestamp_min_us", "min"),
        target_timestamp_max_us=("target_timestamp_max_us", "max"),
    )
    session["target_row_rate"] = session["target_rows"] / session["raw_rows"]
    session["usability_threshold"] = (
        f"target_row_rate >= {MIN_USABLE_TARGET_RATE} and target_rows >= {MIN_USABLE_TARGET_ROWS}"
    )
    session["usable_for_final_holdout"] = (
        session["target_row_rate"].ge(MIN_USABLE_TARGET_RATE)
        & session["target_rows"].ge(MIN_USABLE_TARGET_ROWS)
        & session["malformed_info_time_rows"].eq(0)
    )
    session["decision"] = np.where(
        session["usable_for_final_holdout"],
        "usable_after_txf_filter_metadata_only_locked_until_final_one_shot",
        "not_usable_for_primary_txf_final_holdout_due_to_sparse_target_rows",
    )
    session.to_csv(RESULTS / "new5_holdout_usability.csv", index=False)
    write_markdown(
        ROOT / "NEW5_HOLDOUT_USABILITY_REPORT.md",
        [
            "# NEW5 Holdout Usability Report",
            "",
            "NEW5 was evaluated only with metadata and product counts. No PnL, score, policy, threshold, or model output was computed.",
            "",
            f"A NEW5 session is usable for the primary TXF final holdout only if `{session['usability_threshold'].iloc[0]}` and malformed `info_time` rows are zero.",
            "",
            session.to_markdown(index=False),
            "",
            "Decision: use only NEW5 sessions marked `usable_for_final_holdout = True` after applying the frozen TXF product filter. Sessions with sparse TXF rows remain locked but are not part of the primary TXF final holdout.",
        ],
    )
    return session


def loader_contract(filter_impact: pd.DataFrame) -> pd.DataFrame:
    malformed = filter_impact.loc[filter_impact["malformed_info_time_rows"].gt(0)].copy()
    rows = [
        {
            "rule_id": "loader_001",
            "rule": "Malformed info_time rows MUST be dropped before feature or label construction.",
            "implementation_requirement": "Log physical_path, session_id, row locator, raw info_time, and product_id when available.",
            "silent_coercion_allowed": False,
            "status": "FROZEN",
        },
        {
            "rule_id": "loader_002",
            "rule": "Rows must pass the frozen primary product filter before final target-universe evaluation.",
            "implementation_requirement": f"product_id must match {PRIMARY_TARGET_PRODUCT_RE.pattern}",
            "silent_coercion_allowed": False,
            "status": "FROZEN",
        },
        {
            "rule_id": "loader_003",
            "rule": "Valid info_time parser interprets hour:minute:second:millisecond:flag and requires flag == 0.",
            "implementation_requirement": "Non-zero or missing fifth segment is malformed until separately audited.",
            "silent_coercion_allowed": False,
            "status": "FROZEN",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "malformed_info_time_loader_contract.csv", index=False)
    write_markdown(
        ROOT / "RAW_LOADER_CONTRACT.md",
        [
            "# Raw Loader Contract",
            "",
            "Malformed `info_time` rows must be dropped before feature or label construction and explicitly logged. Silently coercing timestamps is strictly forbidden.",
            "",
            out.to_markdown(index=False),
            "",
            "## Malformed Rows Found During Closure Scan",
            "",
            malformed[
                [
                    "source_group",
                    "session_id",
                    "file_name",
                    "malformed_info_time_rows",
                    "malformed_info_time_examples",
                ]
            ].to_markdown(index=False)
            if not malformed.empty
            else "No malformed `info_time` rows were found.",
        ],
    )
    return out


def filtered_registry_summary(filter_impact: pd.DataFrame) -> pd.DataFrame:
    out = filter_impact[
        [
            "source_group",
            "assigned_role",
            "session_id",
            "file_name",
            "physical_path",
            "sha256",
            "raw_rows",
            "valid_info_time_rows",
            "malformed_info_time_rows",
            "target_product_rows",
            "target_valid_info_time_rows",
            "target_row_rate",
            "target_valid_row_rate",
            "target_product_ids",
            "target_timestamp_min",
            "target_timestamp_max",
            "target_timestamp_min_us",
            "target_timestamp_max_us",
        ]
    ].copy()
    out.to_csv(RESULTS / "filtered_target_product_registry_summary.csv", index=False)
    group_summary = out.groupby(["source_group", "assigned_role"], as_index=False).agg(
        files=("file_name", "count"),
        sessions=("session_id", "nunique"),
        raw_rows=("raw_rows", "sum"),
        target_valid_info_time_rows=("target_valid_info_time_rows", "sum"),
        malformed_info_time_rows=("malformed_info_time_rows", "sum"),
    )
    group_summary["target_valid_rate"] = group_summary["target_valid_info_time_rows"] / group_summary["raw_rows"]
    write_markdown(
        ROOT / "FILTERED_TARGET_PRODUCT_REGISTRY_REPORT.md",
        [
            "# Filtered Target Product Registry Report",
            "",
            "This registry represents the final metadata-only target-product universe after the frozen TXF product filter and strict `info_time` validity rule. It does not compute PnL, model scores, future labels, or thresholds.",
            "",
            group_summary.to_markdown(index=False),
            "",
            "Full file-level registry: `results/filtered_target_product_registry_summary.csv`.",
        ],
    )
    return out


def final_report(
    product_contract: pd.DataFrame,
    filter_impact: pd.DataFrame,
    new5: pd.DataFrame,
    loader: pd.DataFrame,
    filtered_registry: pd.DataFrame,
) -> None:
    group_summary = filter_impact.groupby(["source_group", "assigned_role"], as_index=False).agg(
        files=("file_name", "count"),
        sessions=("session_id", "nunique"),
        raw_rows=("raw_rows", "sum"),
        target_rows=("target_product_rows", "sum"),
        target_valid_rows=("target_valid_info_time_rows", "sum"),
        malformed_info_time_rows=("malformed_info_time_rows", "sum"),
    )
    group_summary["target_valid_rate"] = group_summary["target_valid_rows"] / group_summary["raw_rows"]
    new5_usable = int(new5["usable_for_final_holdout"].sum())
    new5_total = len(new5)
    write_markdown(
        ROOT / "PHASE04A2_PRODUCT_FILTER_AND_LOADER_CLOSURE_REPORT.md",
        [
            "# Phase 04A.2 Product Filter and Loader Contract Closure Report",
            "",
            "Final Status: `clean_protocol_freeze_ready`.",
            "",
            "No ML training, strategy tuning, PnL evaluation, model scoring, or threshold evaluation was performed on NEW5.",
            "",
            "## Product Universe Frozen",
            "",
            "Primary final product universe is TXF-family only: `^TXF[A-Z0-9]*$`. MXF/TMF are not pooled with TXF unless product-specific normalization is implemented and reported separately.",
            "",
            product_contract.to_markdown(index=False),
            "",
            "## Target Product Filter Impact",
            "",
            group_summary.to_markdown(index=False),
            "",
            "## NEW5 Final Usable State",
            "",
            f"Usable NEW5 sessions for primary TXF final holdout: `{new5_usable}/{new5_total}`.",
            "",
            new5.to_markdown(index=False),
            "",
            "NEW5 sessions marked unusable are not discarded from disk; they remain locked metadata-only files and are excluded from the primary TXF final holdout because their target TXF row rate is too sparse.",
            "",
            "## Raw Loader Contract Frozen",
            "",
            "Malformed `info_time` rows must be dropped before feature or label construction and logged. Silent timestamp coercion is forbidden.",
            "",
            loader.to_markdown(index=False),
            "",
            "## Closure Decision",
            "",
            "The product universe, target filter, NEW5 usability rule, and malformed-row loader behavior are now explicit and machine-readable. Product mismatch ambiguity is closed by narrowing the primary final universe to TXF. Parser ambiguity is closed by the strict drop-and-log loader contract.",
        ],
    )


def main() -> None:
    ensure_dirs()
    if not EXPANDED_REGISTRY.exists():
        raise FileNotFoundError(f"Missing Phase 04A.1 expanded registry: {EXPANDED_REGISTRY}")
    registry = pd.read_csv(EXPANDED_REGISTRY)
    product_contract = product_universe_contract()
    filter_impact = scan_all_files(registry)
    new5 = new5_usability(filter_impact)
    loader = loader_contract(filter_impact)
    filtered_registry = filtered_registry_summary(filter_impact)
    final_report(product_contract, filter_impact, new5, loader, filtered_registry)
    print("Phase 04A.2 product filter and loader closure complete.")
    print(ROOT / "PHASE04A2_PRODUCT_FILTER_AND_LOADER_CLOSURE_REPORT.md")


if __name__ == "__main__":
    main()
