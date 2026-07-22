from __future__ import annotations

import shutil

import pandas as pd

from common_final32 import PROJECT_ROOT, RESULTS, ROOT, ensure_dirs


PHASE033_ROOT = PROJECT_ROOT / "new_era" / "33_corrected_final_falsification_audits"
PHASE033_RESULTS = PHASE033_ROOT / "results"


def annotate_matched_random_summary() -> None:
    path = RESULTS / "matched_random_quote_summary.csv"
    df = pd.read_csv(path)
    df["usable_for_quote_selection_skill_claim"] = False
    df["corrected_interpretation"] = (
        "Non-informative saturation diagnostic: policy selects entire exact candidate universe side=bid and spread>=1."
    )
    df["forbidden_claim"] = "Do not claim bid_only_spread_ge_1 beats exact-matched random quote selection."
    df.to_csv(path, index=False)


def annotate_delay_grid() -> None:
    path = RESULTS / "delay_sensitivity_grid.csv"
    df = pd.read_csv(path)
    df["corrected_stress_type"] = "lagged_book_timing_artifact_stress"
    df["recalculates_delayed_entry_ev"] = False
    df["usable_as_true_execution_delay_proxy"] = False
    df["corrected_interpretation"] = (
        "This Phase 032 table lags decision features only; use Phase 033 execution_delay_proxy_grid.csv for delayed-entry proxy EV."
    )
    df.to_csv(path, index=False)


def copy_corrected_artifacts() -> None:
    copies = {
        PHASE033_RESULTS / "execution_delay_proxy_grid.csv": RESULTS / "execution_delay_proxy_grid.csv",
        PHASE033_RESULTS / "dedup_independence_audit.csv": RESULTS / "dedup_independence_audit.csv",
        PHASE033_RESULTS / "regime_vs_random_baselines.csv": RESULTS / "regime_vs_random_baselines.csv",
        PHASE033_RESULTS / "corrected_claim_safety_table.csv": RESULTS / "corrected_claim_safety_table_from_phase033.csv",
        PHASE033_ROOT / "CORRECTED_FINAL_AUDIT_REPORT.md": ROOT / "CORRECTED_FINAL_AUDIT_REPORT.md",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copyfile(src, dst)


def main() -> None:
    ensure_dirs()
    annotate_matched_random_summary()
    annotate_delay_grid()
    copy_corrected_artifacts()


if __name__ == "__main__":
    main()
    print(f"Applied corrected Phase 032 overwrites in {ROOT}")
