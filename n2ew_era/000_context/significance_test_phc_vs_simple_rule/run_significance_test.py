"""
Formal paired significance test: PHC-lite vs simple rule (side_by_queue_imbalance_abs_0p1).

Data source: n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_session_pnl.csv
This file already contains per-session, per-policy session_excess_pnl_H20 for both
'phc_lite' and 'side_by_queue_imbalance_abs_0p1' across splits val/new3/new4.

This closes the previously-flagged gap: prior evidence for "PHC beats simple rule" was
only a 9/9-session empirical consistency count with no formal significance test attached.

Three tests are run per split and pooled OOS (new3+new4, the genuine frozen holdouts;
val was used during development/selection so is reported separately, not pooled into the
headline OOS claim):

1. Exact sign test (binomial, p=0.5 null) on the number of sessions where PHC's session PnL
   exceeds the simple rule's session PnL.
2. Wilcoxon signed-rank test (exact method, since n is small) on paired session PnL differences.
3. Exact paired sign-flip permutation test (full 2^n enumeration, not Monte Carlo) on the sum
   of session-level PnL differences -- this is an exact test, stronger than the project's
   existing 1000-trial Monte Carlo "matched_random" methodology used elsewhere.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "05a_phc_lite_microstructure_sequence_feasibility" / "results" / "phc_lite_session_pnl.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def exact_sign_flip_test(diffs: np.ndarray) -> dict:
    """Full enumeration (not Monte Carlo) two-sided sign-flip permutation test on sum(diffs)."""
    n = len(diffs)
    observed = float(diffs.sum())
    abs_diffs = np.abs(diffs)
    count_ge = 0
    total = 2 ** n
    # Full enumeration is tractable for n <= ~20.
    for signs in itertools.product([1, -1], repeat=n):
        s = float(np.dot(np.array(signs), abs_diffs))
        if abs(s) >= abs(observed) - 1e-9:
            count_ge += 1
    p_two_sided = count_ge / total
    return {
        "n": n,
        "observed_sum_diff": observed,
        "n_permutations_enumerated": total,
        "p_value_two_sided_exact": p_two_sided,
    }


def run_for_group(df: pd.DataFrame, label: str) -> dict:
    piv = df.pivot(index="session_id", columns="policy_id", values="session_excess_pnl_H20")
    phc = piv["phc_lite"].to_numpy(dtype=float)
    simple = piv["side_by_queue_imbalance_abs_0p1"].to_numpy(dtype=float)
    diffs = phc - simple
    n = len(diffs)
    n_wins = int((diffs > 0).sum())
    n_losses = int((diffs < 0).sum())
    n_ties = int((diffs == 0).sum())

    # 1. Exact sign test (binomial, two-sided) excluding ties.
    from scipy.stats import binomtest
    n_nonzero = n_wins + n_losses
    sign_test = binomtest(n_wins, n_nonzero, p=0.5, alternative="two-sided") if n_nonzero > 0 else None

    # 2. Wilcoxon signed-rank, exact method.
    try:
        wres = wilcoxon(phc, simple, alternative="greater", method="exact")
        wilcoxon_result = {"statistic": float(wres.statistic), "p_value_one_sided_greater": float(wres.pvalue)}
    except Exception as e:
        wilcoxon_result = {"error": str(e)}

    # 3. Exact paired sign-flip permutation test.
    perm = exact_sign_flip_test(diffs)

    return {
        "label": label,
        "n_sessions": n,
        "n_phc_wins": n_wins,
        "n_phc_losses": n_losses,
        "n_ties": n_ties,
        "session_ids": list(piv.index),
        "phc_session_pnl": phc.tolist(),
        "simple_rule_session_pnl": simple.tolist(),
        "diffs": diffs.tolist(),
        "total_phc_pnl": float(phc.sum()),
        "total_simple_pnl": float(simple.sum()),
        "total_advantage": float(diffs.sum()),
        "sign_test_binomial": {
            "n_wins": n_wins,
            "n_trials_nonzero": n_nonzero,
            "p_value_two_sided": float(sign_test.pvalue) if sign_test else None,
        },
        "wilcoxon_signed_rank_exact": wilcoxon_result,
        "exact_sign_flip_permutation_test": perm,
    }


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["policy_id"].isin(["phc_lite", "side_by_queue_imbalance_abs_0p1"])]

    results = {}
    for split in ["val", "new3", "new4"]:
        results[split] = run_for_group(df[df["split"] == split], split)

    pooled_oos = df[df["split"].isin(["new3", "new4"])]
    results["pooled_oos_new3_new4"] = run_for_group(pooled_oos, "pooled_oos_new3_new4")

    pooled_all = df.copy()
    results["pooled_all_val_new3_new4"] = run_for_group(pooled_all, "pooled_all_val_new3_new4")

    out_path = OUT_DIR / "phc_vs_simple_rule_significance_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    print("\n=== Summary ===")
    for key in ["val", "new3", "new4", "pooled_oos_new3_new4", "pooled_all_val_new3_new4"]:
        r = results[key]
        print(f"{key}: n={r['n_sessions']} wins={r['n_phc_wins']} losses={r['n_phc_losses']} "
              f"sign_p={r['sign_test_binomial']['p_value_two_sided']:.5g} "
              f"wilcoxon_p(greater)={r['wilcoxon_signed_rank_exact'].get('p_value_one_sided_greater', 'NA')} "
              f"permutation_p={r['exact_sign_flip_permutation_test']['p_value_two_sided_exact']:.5g} "
              f"total_advantage={r['total_advantage']:.0f}")


if __name__ == "__main__":
    main()
