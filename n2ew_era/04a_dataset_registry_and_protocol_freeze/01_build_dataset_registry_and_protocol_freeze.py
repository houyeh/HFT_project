from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04a_dataset_registry_and_protocol_freeze"
RESULTS = ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"

PHYSICAL_GROUPS = ["NEW1", "NEW2", "NEW3", "NEW4", "NEW5"]
ALIAS_GROUPS = ["NEW", "NEW_ALL"]
ROLE_BY_GROUP = {
    "NEW1": "development",
    "NEW2": "development",
    "NEW3": "used_oos_diagnostic",
    "NEW4": "used_oos_diagnostic",
    "NEW5": "final_holdout_locked",
}

REQUIRED_L2_COLUMNS = [
    "info_time",
    "channel_id",
    "channel_seq",
    "product_id",
    "msg_type",
    "index",
    "last_trade_price",
    "last_trade_qty",
    "total_match_qty",
    "1th_buy_qty",
    "1th_buy_price",
    "1th_sell_price",
    "1th_sell_qty",
    "5th_buy_qty",
    "5th_buy_price",
    "5th_sell_price",
    "5th_sell_qty",
]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
    return max(count - 1, 0)


def clean_columns(cols: list[str]) -> list[str]:
    return [str(c).strip() for c in cols]


def parse_info_time_to_us(value: Any) -> float | None:
    parts = str(value).strip().split(":")
    if len(parts) < 4:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2])
        milli_or_micro = int(parts[3])
        return float(((hour * 60 + minute) * 60 + second) * 1_000_000 + milli_or_micro * 1000)
    except Exception:
        return None


def csv_paths(group: str) -> list[Path]:
    base = DATA_ROOT / group
    return sorted(base.glob("*/*.csv")) if base.exists() else []


def build_physical_inventory() -> pd.DataFrame:
    rows = []
    for group in PHYSICAL_GROUPS:
        for path in csv_paths(group):
            session_id = path.parent.name
            rows.append(
                {
                    "source_group": group,
                    "session_id": session_id,
                    "file_name": path.name,
                    "physical_path": str(path.resolve()),
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "file_size_bytes": path.stat().st_size,
                    "row_count": row_count(path),
                    "sha256": sha256_file(path),
                    "assigned_role": ROLE_BY_GROUP[group],
                }
            )
    return pd.DataFrame(rows)


def canonical_registry(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_rows = []
    canonical_rows = []
    precedence = {group: idx for idx, group in enumerate(PHYSICAL_GROUPS)}
    for sha, group in inventory.groupby("sha256", sort=False):
        g = group.copy()
        g["source_precedence"] = g["source_group"].map(precedence)
        selected = g.sort_values(["source_precedence", "physical_path"]).iloc[0].copy()
        duplicate_group_size = len(g)
        selected["duplicate_group_size"] = duplicate_group_size
        selected["excluded_duplicate_paths"] = " | ".join(sorted(set(g["physical_path"]) - {selected["physical_path"]}))
        canonical_rows.append(selected)
        if duplicate_group_size > 1:
            for _, row in g.iterrows():
                duplicate_rows.append(
                    {
                        "sha256": sha,
                        "source_group": row["source_group"],
                        "session_id": row["session_id"],
                        "physical_path": row["physical_path"],
                        "canonical_path": selected["physical_path"],
                        "is_canonical": row["physical_path"] == selected["physical_path"],
                    }
                )
    registry = pd.DataFrame(canonical_rows).drop(columns=["source_precedence"], errors="ignore")
    duplicates = pd.DataFrame(duplicate_rows)
    registry = registry.sort_values(["source_group", "session_id", "file_name"]).reset_index(drop=True)
    registry.insert(0, "dataset_id", [f"ds_{i:03d}" for i in range(1, len(registry) + 1)])
    return registry, duplicates


def alias_audit(canonical: pd.DataFrame) -> pd.DataFrame:
    canonical_hashes = set(canonical["sha256"])
    rows = []
    for group in ALIAS_GROUPS:
        paths = csv_paths(group)
        hashes = []
        for path in paths:
            hashes.append(sha256_file(path))
        unique_hashes = set(hashes)
        base = DATA_ROOT / group
        child_dirs = list(base.glob("*")) if base.exists() else []
        resolved_outside_self = 0
        for child in child_dirs:
            try:
                if child.is_dir() and child.resolve() != child.absolute():
                    resolved_outside_self += 1
            except OSError:
                pass
        rows.append(
            {
                "alias_group": group,
                "path_exists": base.exists(),
                "child_dirs_resolving_to_other_paths": resolved_outside_self,
                "alias_or_aggregate_interpretation": "junction_alias" if resolved_outside_self else "aggregate_duplicate_or_regular_dir",
                "csv_files_seen": len(paths),
                "unique_hashes_seen": len(unique_hashes),
                "new_unique_hashes_not_in_canonical_registry": len(unique_hashes - canonical_hashes),
                "alias_contributes_new_data": len(unique_hashes - canonical_hashes) > 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "alias_duplicate_audit.csv", index=False)
    return out


def new5_schema_check(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    new5 = registry[registry["source_group"].eq("NEW5")].copy()
    for _, item in new5.iterrows():
        path = Path(item["physical_path"])
        header = pd.read_csv(path, nrows=0)
        cols = clean_columns(list(header.columns))
        missing = sorted(set(REQUIRED_L2_COLUMNS) - set(cols))
        sample = pd.read_csv(path, usecols=lambda c: str(c).strip() in {"info_time", "product_id"}, nrows=20000)
        sample.columns = clean_columns(list(sample.columns))
        info = sample["info_time"].dropna().astype(str) if "info_time" in sample else pd.Series(dtype=str)
        parsed = info.map(parse_info_time_to_us).dropna()
        products = sorted({str(x).strip() for x in sample.get("product_id", pd.Series(dtype=str)).dropna().unique()})
        rows.append(
            {
                "source_group": item["source_group"],
                "assigned_role": item["assigned_role"],
                "session_id": item["session_id"],
                "file_name": item["file_name"],
                "physical_path": item["physical_path"],
                "row_count": int(item["row_count"]),
                "file_size_bytes": int(item["file_size_bytes"]),
                "sha256": item["sha256"],
                "column_count": len(cols),
                "schema_required_columns_present": not missing,
                "missing_required_columns": " | ".join(missing),
                "info_time_sample_min": info.min() if len(info) else "",
                "info_time_sample_max": info.max() if len(info) else "",
                "info_time_sample_min_us": float(parsed.min()) if len(parsed) else None,
                "info_time_sample_max_us": float(parsed.max()) if len(parsed) else None,
                "product_id_sample": " | ".join(products[:25]),
                "metadata_only_no_pnl_policy_or_threshold_eval": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new5_lock_schema_check.csv", index=False)
    return out


def write_contracts(registry: pd.DataFrame, alias: pd.DataFrame, duplicates: pd.DataFrame, new5: pd.DataFrame) -> None:
    summary = registry.groupby(["source_group", "assigned_role"], as_index=False).agg(
        unique_files=("physical_path", "count"),
        sessions=("session_id", "nunique"),
        row_count=("row_count", "sum"),
        file_size_bytes=("file_size_bytes", "sum"),
    )
    duplicate_summary = duplicates if not duplicates.empty else pd.DataFrame(columns=["sha256", "source_group", "session_id", "physical_path", "canonical_path", "is_canonical"])
    write_markdown(
        ROOT / "DEDUP_VALIDATION_REPORT.md",
        [
            "# Dedup Validation Report",
            "",
            "Deduplication was performed by exact SHA256 over physical CSV files in `NEW1`, `NEW2`, `NEW3`, `NEW4`, and `NEW5`. `NEW` is treated as an alias/junction layer and `NEW_ALL` as an aggregate duplicate, not as additional data.",
            "",
            "## Canonical Dataset Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Alias Audit",
            "",
            alias.to_markdown(index=False),
            "",
            "## Internal Duplicate Resolved",
            "",
            duplicate_summary.to_markdown(index=False),
            "",
            f"Final canonical unique CSV count: `{len(registry)}`.",
            f"Final canonical raw row count: `{int(registry['row_count'].sum())}`.",
        ],
    )
    write_markdown(
        ROOT / "DATASET_ROLE_CONTRACT.md",
        [
            "# Dataset Role Contract",
            "",
            "| Dataset group | Frozen role | Allowed usage | Forbidden usage |",
            "|---|---|---|---|",
            "| `NEW1` + `NEW2` | `development` | feature engineering, model fitting, validation selection inside development protocol | claiming final OOS robustness by itself |",
            "| `NEW3` + `NEW4` | `used_oos_diagnostic` | robust diagnostics and stress checks after development selection is frozen | tuning, thresholding, model selection |",
            "| `NEW5` | `final_holdout_locked` | one-shot final evaluation only after final development is complete | any tuning, thresholding, model selection, repeated probing, or PnL evaluation before final lock release |",
            "",
            "`NEW5` is locked as the final untouched holdout. Phase 04A performed metadata/schema checks only.",
        ],
    )
    write_markdown(
        ROOT / "LABEL_HIERARCHY_CONTRACT.md",
        [
            "# Label Hierarchy Contract",
            "",
            "| Level | Label / metric | Status | Notes |",
            "|---|---|---|---|",
            "| Diagnostic only | `raw_pnl_H` | diagnostic only | Contains mechanical spread capture and must not be the main target. |",
            "| Diagnostic only | toxic label | diagnostic only | Proxy toxicity label, not true aggressor-side exchange toxicity. |",
            "| Main signal target | `excess_pnl_H20` | primary ML signal target | Half-spread-adjusted future-mid markout target. Future label, never a feature. |",
            "| Strategy evaluation | causal-fill sequential excess PnL | strategy proxy metric | Uses causal fill proxy and sequential inventory cap; still not true exchange replay. |",
            "",
            "Future labels and PnL columns are never allowed as model features.",
        ],
    )
    write_markdown(
        ROOT / "FINAL_EVALUATION_PROTOCOL.md",
        [
            "# Final Evaluation Protocol",
            "",
            "1. Model and strategy selection are allowed only in the Development pool: `NEW1` + `NEW2`.",
            "2. `NEW3` + `NEW4` are used only as robust diagnostic OOS checks after the selected configuration is frozen.",
            "3. `NEW5` is a one-shot final holdout. It remains locked until final development is complete.",
            "4. No strategy may be tuned on `NEW3`, `NEW4`, or `NEW5`.",
            "5. No PnL, policy scoring, threshold sweep, or model comparison may be run on `NEW5` during development.",
            "6. Any NEW5 evaluation must be explicitly labeled as final one-shot holdout evaluation and logged with exact code/config.",
        ],
    )
    write_markdown(
        ROOT / "NEW5_HOLDOUT_LOCK_REPORT.md",
        [
            "# NEW5 Holdout Lock Report",
            "",
            "Phase 04A performed metadata and schema checks only on `NEW5`. No PnL labels, policy decisions, threshold sweeps, model scores, or strategy evaluation were computed on `NEW5`.",
            "",
            "## NEW5 Metadata",
            "",
            new5.to_markdown(index=False),
            "",
            "Lock status: `final_holdout_locked`.",
        ],
    )
    write_markdown(
        ROOT / "PHASE04A_DATASET_AND_PROTOCOL_FREEZE_REPORT.md",
        [
            "# Phase 04A Dataset and Protocol Freeze Report",
            "",
            "Final Status: `dataset_protocol_frozen_ready_for_final_development_rerun`.",
            "",
            "## 1. True Physical Dataset After Dedup",
            "",
            "The true TW L2 physical dataset excludes the `NEW` junction alias and the `NEW_ALL` aggregate duplicate. Exact SHA256 deduplication over `NEW1` through `NEW5` leaves 58 unique CSV files.",
            "",
            summary.to_markdown(index=False),
            "",
            f"Total unique CSV files: `{len(registry)}`.",
            f"Total raw rows: `{int(registry['row_count'].sum())}`.",
            "",
            "## 2. Frozen Roles",
            "",
            "- `NEW1` + `NEW2`: `development`.",
            "- `NEW3` + `NEW4`: `used_oos_diagnostic`.",
            "- `NEW5`: `final_holdout_locked`.",
            "",
            "## 3. Frozen Label Hierarchy",
            "",
            "Diagnostic-only labels are `raw_pnl_H` and toxic labels. The main signal target is `excess_pnl_H20`. Strategy evaluation uses causal-fill sequential excess PnL.",
            "",
            "## 4. Frozen Evaluation Protocol",
            "",
            "Selection is allowed only in the development pool. `NEW3`/`NEW4` are diagnostic OOS. `NEW5` is one-shot final holdout only after final development is complete.",
            "",
            "## 5. NEW5 Lock",
            "",
            "NEW5 was checked only for metadata, hashes, row counts, schema, product samples, and timestamp range. No PnL or policy scoring was run.",
        ],
    )


def main() -> None:
    ensure_dirs()
    inventory = build_physical_inventory()
    registry, duplicates = canonical_registry(inventory)
    registry.to_csv(RESULTS / "dataset_registry.csv", index=False)
    duplicates.to_csv(RESULTS / "resolved_internal_duplicates.csv", index=False)
    alias = alias_audit(registry)
    new5 = new5_schema_check(registry)
    write_contracts(registry, alias, duplicates, new5)
    print("Phase 04A dataset registry and protocol freeze complete.")
    print(registry.groupby(["source_group", "assigned_role"], as_index=False).agg(unique_files=("physical_path", "count"), sessions=("session_id", "nunique"), rows=("row_count", "sum")).to_string(index=False))


if __name__ == "__main__":
    main()
