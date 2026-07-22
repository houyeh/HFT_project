from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_phase02d import PHASE02B_DIR, PHASE02C_DIR, RESULTS, ensure_dirs


REQUIRED_INPUTS = [
    ("02B", "strict_policy_grid", PHASE02B_DIR / "results" / "strict_policy_grid.csv"),
    ("02B", "random_summary_strict", PHASE02B_DIR / "results" / "random_summary_strict.csv"),
    ("02B", "random_summary_strict_exact_top_candidates", PHASE02B_DIR / "results" / "random_summary_strict_exact_top_candidates.csv"),
    ("02B", "fee_sensitivity", PHASE02B_DIR / "results" / "fee_sensitivity.csv"),
    ("02B", "strict_selected_candidates", PHASE02B_DIR / "results" / "strict_selected_candidates.json"),
    ("02C", "queue_stress_grid", PHASE02C_DIR / "results" / "queue_stress_grid.csv"),
    ("02C", "delay_sensitivity", PHASE02C_DIR / "results" / "delay_sensitivity.csv"),
    ("02C", "fill_keep_break_even", PHASE02C_DIR / "results" / "fill_keep_break_even.csv"),
]


def csv_shape(path: Path) -> tuple[int | None, int | None]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None, None
    df = pd.read_csv(path)
    return int(len(df)), int(len(df.columns))


def main() -> None:
    ensure_dirs()
    manifest_rows = []
    missing_rows = []
    for phase, name, path in REQUIRED_INPUTS:
        exists = path.exists()
        rows, cols = csv_shape(path) if exists else (None, None)
        manifest_rows.append(
            {
                "phase": phase,
                "input_name": name,
                "path": str(path),
                "exists": bool(exists),
                "row_count": rows,
                "column_count": cols,
                "bytes": int(path.stat().st_size) if exists else None,
            }
        )
        if not exists:
            missing_rows.append({"phase": phase, "input_name": name, "path": str(path), "reason": "missing_file"})
    manifest = pd.DataFrame(manifest_rows)
    missing = pd.DataFrame(missing_rows, columns=["phase", "input_name", "path", "reason"])
    manifest.to_csv(RESULTS / "phase02bc_input_manifest.csv", index=False)
    missing.to_csv(RESULTS / "missing_inputs.csv", index=False)
    print("Phase 02BC input collection complete.")
    print(manifest.to_string(index=False))
    if not missing.empty:
        print("Missing inputs:")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
