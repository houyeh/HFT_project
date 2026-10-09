status: resolved_significant

# PHC-lite vs Simple Rule — Formal Paired Significance Test

## Scope

This closes a previously-flagged gap: the "PHC beats the simple rule" claim previously rested only
on an empirical session-win-count with no formal significance test attached. This audit runs three
independent, standard paired tests directly on already-existing per-session PnL data. No retraining,
no NEW5 access, no model changes. CPU/Python analysis only.

## Data Source

`n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_session_pnl.csv` —
an existing artifact containing per-session `session_excess_pnl_H20` for both `phc_lite` and
`side_by_queue_imbalance_abs_0p1` (the simple rule) across the `val`, `new3`, and `new4` splits.
`val` was used during development/policy selection; `new3` and `new4` are the frozen out-of-sample
holdouts used elsewhere in this project's methodology, so the pooled `new3+new4` result is the
headline OOS claim, not the pooled-with-`val` number.

## Method

Three tests per split and per pooled group, all on paired (same-session) PnL differences:

1. **Exact sign test** (binomial, p=0.5 null, two-sided) on the count of sessions where PHC's
   session PnL exceeds the simple rule's.
2. **Wilcoxon signed-rank test**, exact method (appropriate given small n; no normal approximation).
3. **Exact paired sign-flip permutation test** — full `2^n` enumeration of all sign-flip
   combinations on the session-level differences, not Monte Carlo sampling. This is a stronger
   guarantee than the project's existing 1000-trial Monte Carlo `matched_random` methodology used
   in `02b`/`07d`, because with n this small (4-16 sessions), full enumeration is tractable and
   gives an exact p-value rather than an approximation.

Script: `run_significance_test.py`. Full results: `phc_vs_simple_rule_significance_results.json`.

## Results

| group | n sessions | PHC wins | PHC losses | sign test p (two-sided) | Wilcoxon p (one-sided, PHC>simple) | exact permutation p (two-sided) | total PHC advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (dev split, not OOS) | 6 | 6 | 0 | 0.03125 | 0.015625 | 0.03125 | 690,025 |
| new3 (OOS) | 6 | 6 | 0 | 0.03125 | 0.015625 | 0.03125 | 2,066,745 |
| new4 (OOS) | 4 | 4 | 0 | 0.125 | 0.0625 | 0.125 | 758,585 |
| **pooled OOS (new3+new4) — headline claim** | **10** | **10** | **0** | **0.0020** | **0.00098** | **0.0020** | **2,825,330** |
| pooled all (val+new3+new4, reference only) | 16 | 16 | 0 | 0.0000305 | 0.0000153 | 0.0000305 | 3,515,355 |

## Honest Interpretation

PHC beats the simple rule in every session in every split with no exceptions (16/16 pooled across
all three splits). On the pooled genuine out-of-sample evidence (`new3`+`new4`, n=10), this is
significant at p ≈ 0.002 by all three tests, comfortably below conventional 0.05 and 0.01
thresholds. This is the number that should be cited as "PHC vs simple rule, formally
significant, OOS."

The `new4`-only result (p=0.125) does **not** reach conventional significance on its own — but this
is a sample-size floor, not a weak effect: with only 4 paired sessions, the best possible two-sided
sign-test/permutation p-value is 0.125 (`2 × 0.5^4`) even with a perfect 4/4 record, so `new4` alone
can never cross p<0.05 regardless of effect size. This should be disclosed explicitly if `new4`
alone is ever cited, and the pooled OOS number should be used as the headline claim instead.

This test compares session-aggregated PnL, not event-level decisions, and inherits whatever
assumptions are already baked into `session_excess_pnl_H20` (including the previously-disclosed
`POINT_VALUE=200` / `MAKER_FEE=15` proxy assumptions). The significance result says PHC's edge over
the simple rule is unlikely to be due to session-to-session chance under those assumptions; it does
not independently validate the assumptions themselves.

## Reproducibility

```bash
python3 n2ew_era/000_context/significance_test_phc_vs_simple_rule/run_significance_test.py
```

Reads only the existing `phc_lite_session_pnl.csv`; writes
`phc_vs_simple_rule_significance_results.json` with full per-session values for independent
re-verification.
