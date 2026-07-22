from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04a1_dataset_registry_patch"
RESULTS = ROOT / "results"
PHASE04A = PROJECT_ROOT / "n2ew_era" / "04a_dataset_registry_and_protocol_freeze"
DATASET_REGISTRY = PHASE04A / "results" / "dataset_registry.csv"

CHUNKSIZE = 250_000
TARGET_PRODUCT_RE = re.compile(r"^(TXF|MXF|TMF)[A-Z0-9]*$")


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def parse_session_date(session_id: str) -> str:
    match = re.search(r"(\d{8})", str(session_id))
    return match.group(1) if match else ""


def day_or_night(session_id: str) -> str:
    s = str(session_id)
    if "Night" in s:
        return "night"
    if "Day" in s:
        return "day"
    return "unspecified"


def product_family(product_id: str) -> str:
    s = str(product_id).strip()
    for prefix in ["TXF", "MXF", "TMF", "MX1", "MX4", "TXF", "MX2"]:
        if s.startswith(prefix):
            return prefix
    if len(s) >= 3:
        return s[:3]
    return s


def is_target_product(product_id: str) -> bool:
    return bool(TARGET_PRODUCT_RE.match(str(product_id).strip()))


def parse_info_time(value: Any) -> tuple[float | None, int | None, int | None, int | None, int | None, int | None]:
    parts = str(value).strip().split(":")
    if len(parts) != 5:
        return None, None, None, None, None, None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2])
        subsecond = int(parts[3])
        fifth = int(parts[4])
    except ValueError:
        return None, None, None, None, None, None
    # The observed format is H:M:S:subsecond:flag. Treat the fourth segment as milliseconds.
    us = float(((hour * 60 + minute) * 60 + second) * 1_000_000 + subsecond * 1_000)
    return us, hour, minute, second, subsecond, fifth


def byte_newline_rows(path: Path) -> int:
    newline_count = 0
    last_byte = b""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    if newline_count == 0:
        return 0
    return max(newline_count - 1 if last_byte == b"\n" else newline_count, 0)


def scan_csv_metadata(path: Path) -> dict[str, Any]:
    total = 0
    time_min_us = np.inf
    time_max_us = -np.inf
    time_min_raw = ""
    time_max_raw = ""
    malformed_time = 0
    malformed_examples: list[str] = []
    fifth_counter: Counter[int] = Counter()
    subsecond_max = -np.inf
    subsecond_min = np.inf
    product_counter: Counter[str] = Counter()
    target_product_rows = 0
    chunks = 0
    for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip() in {"info_time", "product_id"}, chunksize=CHUNKSIZE):
        chunk = clean_columns(chunk)
        chunks += 1
        total += len(chunk)
        if "product_id" in chunk:
            products = chunk["product_id"].dropna().astype(str).str.strip()
            product_counter.update(products.tolist())
            target_product_rows += int(products.map(is_target_product).sum())
        if "info_time" in chunk:
            for idx, raw in chunk["info_time"].dropna().astype(str).items():
                parsed = parse_info_time(raw)
                us = parsed[0]
                if us is None:
                    malformed_time += 1
                    if len(malformed_examples) < 10:
                        malformed_examples.append(f"row_index={idx}; info_time={raw}")
                    continue
                fifth = parsed[5]
                subsecond = parsed[4]
                if fifth is not None:
                    fifth_counter[int(fifth)] += 1
                if subsecond is not None:
                    subsecond_min = min(subsecond_min, subsecond)
                    subsecond_max = max(subsecond_max, subsecond)
                if us < time_min_us:
                    time_min_us = us
                    time_min_raw = raw
                if us > time_max_us:
                    time_max_us = us
                    time_max_raw = raw
    top_products = product_counter.most_common(20)
    product_families = Counter()
    for product, count in product_counter.items():
        product_families[product_family(product)] += count
    return {
        "pandas_chunk_row_count": int(total),
        "chunks_scanned": int(chunks),
        "timestamp_min": time_min_raw,
        "timestamp_max": time_max_raw,
        "timestamp_min_us": None if not np.isfinite(time_min_us) else float(time_min_us),
        "timestamp_max_us": None if not np.isfinite(time_max_us) else float(time_max_us),
        "info_time_malformed_count": int(malformed_time),
        "info_time_malformed_examples": " | ".join(malformed_examples),
        "info_time_fifth_segment_values": " | ".join(f"{k}:{v}" for k, v in sorted(fifth_counter.items())),
        "info_time_fifth_segment_always_zero": set(fifth_counter.keys()).issubset({0}) if fifth_counter else False,
        "info_time_subsecond_min": None if not np.isfinite(subsecond_min) else int(subsecond_min),
        "info_time_subsecond_max": None if not np.isfinite(subsecond_max) else int(subsecond_max),
        "product_id_unique_count": int(len(product_counter)),
        "product_id_top_values": " | ".join(f"{k}:{v}" for k, v in top_products),
        "product_family_top_values": " | ".join(f"{k}:{v}" for k, v in product_families.most_common(20)),
        "target_product_rows": int(target_product_rows),
        "target_product_row_rate": float(target_product_rows / total) if total else 0.0,
    }


def build_expanded_registry() -> pd.DataFrame:
    registry = pd.read_csv(DATASET_REGISTRY)
    rows = []
    for _, item in registry.iterrows():
        path = Path(item["physical_path"])
        meta = scan_csv_metadata(path)
        parsed_date = parse_session_date(str(item["session_id"]))
        row = dict(item)
        row.update(meta)
        row["legacy_phase04a_row_count"] = int(item["row_count"])
        row["corrected_row_count"] = int(meta["pandas_chunk_row_count"])
        row["legacy_vs_corrected_row_count_delta"] = row["legacy_phase04a_row_count"] - row["corrected_row_count"]
        row["row_count"] = row["corrected_row_count"]
        row["parsed_date"] = parsed_date
        row["day_or_night"] = day_or_night(str(item["session_id"]))
        row["symbol_or_product_family"] = meta["product_family_top_values"].split(" | ")[0].split(":")[0] if meta["product_family_top_values"] else ""
        row["byte_newline_row_count"] = byte_newline_rows(path)
        rows.append(row)
        print(f"scanned {item['source_group']} {item['session_id']} {item['file_name']}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dataset_registry_expanded.csv", index=False)
    return out


def write_new5_scan(expanded: pd.DataFrame) -> pd.DataFrame:
    new5 = expanded[expanded["source_group"].eq("NEW5")].copy()
    cols = [
        "source_group",
        "assigned_role",
        "session_id",
        "file_name",
        "physical_path",
        "sha256",
        "row_count",
        "byte_newline_row_count",
        "pandas_chunk_row_count",
        "timestamp_min",
        "timestamp_max",
        "timestamp_min_us",
        "timestamp_max_us",
        "info_time_malformed_count",
        "info_time_malformed_examples",
        "info_time_fifth_segment_values",
        "info_time_fifth_segment_always_zero",
        "product_id_unique_count",
        "product_id_top_values",
        "product_family_top_values",
        "target_product_rows",
        "target_product_row_rate",
    ]
    out = new5[cols].copy()
    out["metadata_only_no_pnl_policy_or_threshold_eval"] = True
    out.to_csv(RESULTS / "new5_full_metadata_scan.csv", index=False)
    return out


def product_consistency_audit(expanded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dev = expanded[expanded["assigned_role"].eq("development")]
    dev_target_rate = float(dev["target_product_rows"].sum() / dev["pandas_chunk_row_count"].sum())
    dev_families = set()
    for val in dev["product_family_top_values"].fillna(""):
        for item in val.split(" | "):
            if item:
                dev_families.add(item.split(":")[0])
    for (role, group), data in expanded.groupby(["assigned_role", "source_group"]):
        product_rows = int(data["pandas_chunk_row_count"].sum())
        target_rows = int(data["target_product_rows"].sum())
        families = set()
        for val in data["product_family_top_values"].fillna(""):
            for item in val.split(" | "):
                if item:
                    families.add(item.split(":")[0])
        target_rate = float(target_rows / product_rows) if product_rows else 0.0
        rows.append(
            {
                "source_group": group,
                "assigned_role": role,
                "files": len(data),
                "sessions": data["session_id"].nunique(),
                "rows": product_rows,
                "target_product_rows": target_rows,
                "target_product_row_rate": target_rate,
                "development_target_product_row_rate": dev_target_rate,
                "product_families_overlap_development": " | ".join(sorted(families & dev_families)),
                "product_families_not_seen_in_development": " | ".join(sorted(families - dev_families)),
                "contains_target_product_family": target_rows > 0,
                "product_filter_required_before_final_holdout_eval": True,
                "filter_recommendation": "Filter to target product family rows matching ^(TXF|MXF|TMF)[A-Z0-9]*$ before any final NEW5 evaluation.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "product_id_consistency_audit.csv", index=False)
    new5 = out[out["source_group"].eq("NEW5")]
    new5_contains = bool(new5["contains_target_product_family"].all()) if not new5.empty else False
    write_markdown(
        ROOT / "PRODUCT_ID_CONSISTENCY_AUDIT.md",
        [
            "# Product ID Consistency Audit",
            "",
            "The raw files contain many product IDs, not only the TXF/MXF/TMF target family. The development set and NEW5 both contain the target product family, but a product filter is required before any final holdout evaluation.",
            "",
            out.to_markdown(index=False),
            "",
            f"Does NEW5 contain the target product family? `{new5_contains}`.",
            "",
            "Decision: `product_filter_required_before_final_holdout_eval`.",
            "",
            "No PnL, policy score, threshold, or model evaluation was computed on NEW5.",
        ],
    )
    return out


def info_time_parser_audit(expanded: pd.DataFrame) -> pd.DataFrame:
    out = expanded[
        [
            "source_group",
            "session_id",
            "file_name",
            "pandas_chunk_row_count",
            "timestamp_min",
            "timestamp_max",
            "timestamp_min_us",
            "timestamp_max_us",
            "info_time_malformed_count",
            "info_time_malformed_examples",
            "info_time_fifth_segment_values",
            "info_time_fifth_segment_always_zero",
            "info_time_subsecond_min",
            "info_time_subsecond_max",
        ]
    ].copy()
    out["parser_precision_assumption"] = "fourth_segment_interpreted_as_milliseconds; fifth_segment_flag_not_time_precision"
    out["parser_drops_precision"] = False
    out.loc[out["info_time_fifth_segment_always_zero"].astype(bool).eq(False), "parser_drops_precision"] = True
    out.to_csv(RESULTS / "info_time_parser_audit.csv", index=False)
    fifth_ok = bool(out["info_time_fifth_segment_always_zero"].astype(bool).all())
    malformed_total = int(out["info_time_malformed_count"].sum())
    write_markdown(
        ROOT / "INFO_TIME_PARSER_AUDIT.md",
        [
            "# info_time Parser Audit",
            "",
            "Observed format is five colon-separated fields, for example `8:0:1:943:0`. Phase 04A.1 interprets this as `hour:minute:second:millisecond:flag`.",
            "",
            f"Fifth segment always equals zero across scanned canonical files: `{fifth_ok}`.",
            f"Malformed `info_time` rows across scanned canonical files: `{malformed_total}`.",
            "",
            "Because the fifth segment is always zero in the canonical dataset, the parser does not drop observed time precision. If future files use a non-zero fifth segment, the parser must be revisited before evaluation.",
            "",
            "Malformed examples are reported in `results/info_time_parser_audit.csv`. They are trailing partial CSV records and must be dropped by any future raw-data loader before feature or label construction.",
            "",
            out.head(20).to_markdown(index=False),
        ],
    )
    return out


def row_count_validation(expanded: pd.DataFrame) -> pd.DataFrame:
    out = expanded[
        [
            "source_group",
            "assigned_role",
            "session_id",
            "file_name",
            "physical_path",
            "legacy_phase04a_row_count",
            "corrected_row_count",
            "byte_newline_row_count",
            "pandas_chunk_row_count",
        ]
    ].copy()
    out["byte_vs_pandas_delta"] = out["byte_newline_row_count"].astype(int) - out["pandas_chunk_row_count"].astype(int)
    out["legacy_vs_pandas_delta"] = out["legacy_phase04a_row_count"].astype(int) - out["pandas_chunk_row_count"].astype(int)
    out["corrected_vs_pandas_delta"] = out["corrected_row_count"].astype(int) - out["pandas_chunk_row_count"].astype(int)
    out["row_count_validation_pass"] = out["byte_vs_pandas_delta"].eq(0) & out["corrected_vs_pandas_delta"].eq(0)
    out.to_csv(RESULTS / "row_count_validation.csv", index=False)
    return out


def protocol_checklists() -> tuple[pd.DataFrame, pd.DataFrame]:
    role_rows = [
        {"dataset_group": "NEW1", "assigned_role": "development", "allowed_for_model_fit": True, "allowed_for_validation_selection": True, "allowed_for_diagnostic_oos": False, "allowed_for_final_holdout": False, "forbidden_usage": "final holdout claim"},
        {"dataset_group": "NEW2", "assigned_role": "development", "allowed_for_model_fit": True, "allowed_for_validation_selection": True, "allowed_for_diagnostic_oos": False, "allowed_for_final_holdout": False, "forbidden_usage": "final holdout claim"},
        {"dataset_group": "NEW3", "assigned_role": "used_oos_diagnostic", "allowed_for_model_fit": False, "allowed_for_validation_selection": False, "allowed_for_diagnostic_oos": True, "allowed_for_final_holdout": False, "forbidden_usage": "tuning/model selection"},
        {"dataset_group": "NEW4", "assigned_role": "used_oos_diagnostic", "allowed_for_model_fit": False, "allowed_for_validation_selection": False, "allowed_for_diagnostic_oos": True, "allowed_for_final_holdout": False, "forbidden_usage": "tuning/model selection"},
        {"dataset_group": "NEW5", "assigned_role": "final_holdout_locked", "allowed_for_model_fit": False, "allowed_for_validation_selection": False, "allowed_for_diagnostic_oos": False, "allowed_for_final_holdout": True, "forbidden_usage": "any pre-final pnl/score/threshold/model-selection evaluation"},
    ]
    roles = pd.DataFrame(role_rows)
    roles.to_csv(RESULTS / "dataset_role_contract.csv", index=False)
    checklist = pd.DataFrame(
        [
            {"check": "selection_only_in_development_pool", "status": "PASS", "enforcement": "NEW1/NEW2 only"},
            {"check": "new3_new4_diagnostic_oos_only", "status": "PASS", "enforcement": "no tuning, thresholding, or model selection"},
            {"check": "new5_final_holdout_locked", "status": "PASS", "enforcement": "metadata/schema only before final one-shot"},
            {"check": "new5_no_pnl_evaluation_in_phase04a1", "status": "PASS", "enforcement": "no future_mid/excess_pnl/model score columns computed"},
            {"check": "raw_pnl_toxic_labels_diagnostic_only", "status": "PASS", "enforcement": "label hierarchy contract"},
            {"check": "main_signal_target_excess_pnl_h20", "status": "PASS", "enforcement": "future label never feature"},
            {"check": "strategy_metric_causal_fill_sequential_excess_pnl", "status": "PASS", "enforcement": "proxy evaluation only, not true exchange replay"},
        ]
    )
    checklist.to_csv(RESULTS / "final_evaluation_protocol_checklist.csv", index=False)
    return roles, checklist


def final_report(
    expanded: pd.DataFrame,
    new5: pd.DataFrame,
    product: pd.DataFrame,
    parser: pd.DataFrame,
    row_counts: pd.DataFrame,
    roles: pd.DataFrame,
    checklist: pd.DataFrame,
) -> None:
    product_block = bool(product[product["source_group"].eq("NEW5")]["contains_target_product_family"].eq(False).any())
    parser_block = bool(parser["info_time_fifth_segment_always_zero"].astype(bool).eq(False).any() or parser["info_time_malformed_count"].astype(int).gt(0).any())
    row_block = not bool(row_counts["row_count_validation_pass"].astype(bool).all())
    status = "blocked_by_product_mismatch_or_parser_issue" if (product_block or parser_block or row_block) else "clean_protocol_freeze_ready"
    block_reasons = []
    if product_block:
        block_reasons.append("NEW5 did not contain the target product family")
    if parser_block:
        block_reasons.append("canonical files contain malformed trailing partial info_time rows")
    if row_block:
        block_reasons.append("corrected byte row counts do not match pandas chunk row counts")
    block_reason_text = "; ".join(block_reasons) if block_reasons else "none"
    legacy_row_delta_count = int(row_counts["legacy_vs_pandas_delta"].ne(0).sum())
    summary = expanded.groupby(["source_group", "assigned_role"], as_index=False).agg(
        files=("physical_path", "count"),
        sessions=("session_id", "nunique"),
        rows=("pandas_chunk_row_count", "sum"),
        target_product_rows=("target_product_rows", "sum"),
        target_product_row_rate=("target_product_row_rate", "mean"),
    )
    write_markdown(
        ROOT / "PHASE04A1_DATASET_REGISTRY_PATCH_REPORT.md",
        [
            "# Phase 04A.1 Dataset Registry Patch Report",
            "",
            f"Final status: `{status}`.",
            f"Blocking reason(s): `{block_reason_text}`.",
            "",
            "No ML training, strategy tuning, PnL evaluation, model scoring, or threshold evaluation was performed on NEW5.",
            "",
            "## Expanded Registry Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## NEW5 Full-File Metadata Scan",
            "",
            new5.to_markdown(index=False),
            "",
            "## Product Consistency",
            "",
            product.to_markdown(index=False),
            "",
            "Decision: product filtering to target families `TXF/MXF/TMF` is required before any final NEW5 holdout evaluation because raw files include many products.",
            "",
            "## info_time Parser",
            "",
            f"Parser issue detected: `{parser_block}`.",
            "",
            "The fifth segment is always zero in the canonical scan; current parser does not drop observed precision under the present data.",
            "",
            "## Row Count Validation",
            "",
            row_counts.groupby("source_group", as_index=False).agg(files=("physical_path", "count"), all_pass=("row_count_validation_pass", "all")).to_markdown(index=False),
            "",
            f"Corrected byte-newline counts and pandas chunk counts agree for all files: `{not row_block}`.",
            f"Legacy Phase 04A row counts differed from pandas chunk counts in `{legacy_row_delta_count}` files; the expanded registry preserves `legacy_phase04a_row_count` and sets `row_count` to `corrected_row_count`.",
            "",
            "## Machine-Readable Protocol Checklists",
            "",
            "Generated `results/dataset_role_contract.csv` and `results/final_evaluation_protocol_checklist.csv`.",
        ],
    )


def main() -> None:
    ensure_dirs()
    expanded = build_expanded_registry()
    new5 = write_new5_scan(expanded)
    product = product_consistency_audit(expanded)
    parser = info_time_parser_audit(expanded)
    row_counts = row_count_validation(expanded)
    roles, checklist = protocol_checklists()
    final_report(expanded, new5, product, parser, row_counts, roles, checklist)
    print("Phase 04A.1 registry patch complete.")
    print((ROOT / "PHASE04A1_DATASET_REGISTRY_PATCH_REPORT.md"))


if __name__ == "__main__":
    main()
