status: code_fixed_resynthesis_pending

# FI-2010 Stage 2 Checkpoint Selection — Test-Isolation Fix

## Bug (identified by red-team review, `VALIDATION_METHODOLOGY_SOUNDNESS_AUDIT.md` §3.x)

`05_prepare_stage2_hls.py::select_checkpoint()` ranked the 3 candidate seeds
(k=50, raw loss, Stage 1 v2) by `test_macro_f1` and took the median-ranked
seed. Using `test_macro_f1` — a held-out test-set metric — to make a
model-selection decision violates the test-isolation discipline that Stage 1
itself claims elsewhere (val used for model/checkpoint decisions, test used
for reporting only).

## Fix applied (2026-08-16)

`select_checkpoint()` now ranks candidates by `val_macro_f1` only.
`test_macro_f1` is still recorded per-row for reporting but is never used in
the sort/selection. The `06_collect_stage2_hls_results.py` report-generation
docstring was updated to describe the selection rule dynamically (pulling
seed/checkpoint from the manifest) instead of hardcoding the old seed=1
narrative.

## Effect on the selected checkpoint

Verified directly from `results/stage1_metrics_v2.csv` (k=50, raw, phase
`08a_stage1_v2`):

| seed | val_macro_f1 | test_macro_f1 | checkpoint |
| --- | --- | --- | --- |
| 0 | 0.36649 | 0.42444 | `models/phc_fi2010_h50_seed0_raw_v2.pt` |
| 1 | 0.37458 | 0.42439 | `models/phc_fi2010_h50_seed1_raw_v2.pt` |
| 2 | 0.30191 | 0.37274 | `models/phc_fi2010_h50_seed2_raw_v2.pt` |

- Old (buggy) rule, median by `test_macro_f1`: selects **seed=1**.
- Fixed rule, median by `val_macro_f1`: selects **seed=0**.

The representative checkpoint changes. This is not a cosmetic fix — the
concrete `.pt` file that should be exported to HLS is different under the
corrected rule.

## What still needs to be done (not run in this sandbox)

This sandbox has no `torch` install and no Vitis HLS toolchain, so only the
selection logic and the pure-CSV verification above could be done here.
Outstanding, to be run locally:

1. Re-run `05_prepare_stage2_hls.py` end-to-end with the fixed selection rule
   (weight export, fixed-point conversion, HLS source/testbench generation)
   against `phc_fi2010_h50_seed0_raw_v2.pt`.
2. Re-run csim and csynth in Vitis HLS against the regenerated sources.
3. Re-run `06_collect_stage2_hls_results.py` to regenerate
   `REPORT_STAGE2.md`, `results/stage2_checkpoint_selection.csv`, and
   `results/stage2_hls_manifest.json` from the new outputs.
4. Because the model architecture and fixed-point config (`ap_fixed<24,12>`)
   are unchanged, DSP/LUT/FF/BRAM utilization and cycle count are expected to
   stay close to the previously reported values (33 DSP, 20,582 LUT, 7,347
   cycles ≈ 73.470µs @ 100MHz) — but this must be confirmed by the actual
   re-run, not assumed. The number that is expected to change is the
   accuracy figure tied to this specific exported checkpoint (seed0's
   val/test macro-F1 differs from seed1's).
5. Until step 1–3 are done, `REPORT_STAGE2.md` carries a correction note at
   the top and all Stage 2 HLS numbers should be cited as provisional /
   pending re-synthesis.
