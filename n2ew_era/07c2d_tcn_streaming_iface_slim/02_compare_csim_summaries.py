from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_RESULTS = ROOT.parent / "07c2c_tcn_streaming_cosim_diagnostic" / "results"
NEW_RESULTS = ROOT / "results"


CASES = [
    (
        "distilled_shared_u4",
        174,
        OLD_RESULTS / "stage2_cosim174_distilled_shared_u4_e174_tb_summary.csv",
        NEW_RESULTS / "stage2_iface_slim_csim_debug174_distilled_shared_u4_event174_tb_summary.csv",
    ),
    (
        "scratch_shared_u4",
        174,
        OLD_RESULTS / "stage2_cosim174_scratch_shared_u4_e174_tb_summary.csv",
        NEW_RESULTS / "stage2_iface_slim_csim_debug174_scratch_shared_u4_event174_tb_summary.csv",
    ),
    (
        "distilled_shared_u4",
        252,
        OLD_RESULTS / "stage2_cosim252_distilled_shared_u4_1window_tb_summary.csv",
        NEW_RESULTS / "stage2_iface_slim_csim_debug252_distilled_shared_u4_1window_tb_summary.csv",
    ),
    (
        "scratch_shared_u4",
        252,
        OLD_RESULTS / "stage2_cosim252_scratch_shared_u4_1window_tb_summary.csv",
        NEW_RESULTS / "stage2_iface_slim_csim_debug252_scratch_shared_u4_1window_tb_summary.csv",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compare_case(variant: str, event_limit: int, old_path: Path, new_path: Path) -> dict[str, object]:
    if not old_path.exists() or not new_path.exists():
        return {
            "variant": variant,
            "event_limit": event_limit,
            "old_summary": str(old_path),
            "new_summary": str(new_path),
            "old_exists": old_path.exists(),
            "new_exists": new_path.exists(),
            "rows_equal": False,
            "numeric_cells_compared": 0,
            "max_abs_numeric_delta": "",
            "exact_text_match": False,
        }

    old_text = old_path.read_text(encoding="utf-8")
    new_text = new_path.read_text(encoding="utf-8")
    old_rows = read_rows(old_path)
    new_rows = read_rows(new_path)
    max_delta = 0.0
    numeric_count = 0
    rows_equal = len(old_rows) == len(new_rows)

    for old_row, new_row in zip(old_rows, new_rows):
        keys = sorted(set(old_row) | set(new_row))
        for key in keys:
            old_value = old_row.get(key, "")
            new_value = new_row.get(key, "")
            old_float = as_float(old_value)
            new_float = as_float(new_value)
            if old_float is not None and new_float is not None:
                numeric_count += 1
                delta = abs(old_float - new_float)
                if delta > max_delta:
                    max_delta = delta
            elif old_value != new_value:
                rows_equal = False

    if numeric_count > 0 and max_delta != 0.0:
        rows_equal = False

    return {
        "variant": variant,
        "event_limit": event_limit,
        "old_summary": str(old_path),
        "new_summary": str(new_path),
        "old_exists": True,
        "new_exists": True,
        "rows_equal": rows_equal,
        "numeric_cells_compared": numeric_count,
        "max_abs_numeric_delta": max_delta,
        "exact_text_match": old_text == new_text,
    }


def main() -> None:
    rows = [compare_case(*case) for case in CASES]
    out = NEW_RESULTS / "iface_slim_csim_vs_07c2c_summary_compare.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['variant']} event{row['event_limit']}: rows_equal={row['rows_equal']} "
            f"max_abs_numeric_delta={row['max_abs_numeric_delta']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
