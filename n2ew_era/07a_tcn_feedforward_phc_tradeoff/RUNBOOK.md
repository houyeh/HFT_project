# Phase 07A Runbook

This phase evaluates whether a feedforward HLS-friendly TCN can replace the PHC-lite recurrent model on the same TXF L2 bid/ask action-value task.

## Protocol

- Data: Phase 05A sequence payloads only.
- Features: the same 16 causal PHC-lite features.
- Splits: `train` = NEW2, `val` = NEW1, OOS diagnostics = NEW3/NEW4.
- Target: `excess_pnl_H20`.
- Variants:
  - `tcn_prepad_scratch_16feat`
  - `tcn_prepad_distilled_16feat`
- Distillation teacher: frozen PHC-lite v1 scores from Phase 05A.
- NEW5 is not loaded.

## Execution Order

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\01_train_all_seeds.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\02_select_policy.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\03_evaluate_oos.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\04_matched_random.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\05_causal_fill.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\06_seed_stability.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\07_latency_estimate.py
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\08_final_report.py
```

or:

```powershell
.\.venv\Scripts\python.exe n2ew_era\07a_tcn_feedforward_phc_tradeoff\run_pipeline.py
```

## Runtime Note

The local environment checked during setup is CPU-only. The full run trains 6 models: 2 variants x 3 seeds. Based on prior 5-block TCN runs in this repository, this may require many hours on CPU.

## Key Outputs

- `results/tcn_prepad_training_curve_all_variants.csv`
- `results/tcn_prepad_validation_threshold_search.csv`
- `results/tcn_prepad_oos_eval.csv`
- `results/tcn_prepad_matched_random_summary.csv`
- `results/tcn_prepad_causal_fill_eval.csv`
- `results/tcn_prepad_seed_stability.csv`
- `results/tcn_prepad_latency_estimate.csv`
- `results/tcn_vs_phc_tradeoff_summary.csv`
- `TCN_FEEDFORWARD_PHC_TRADEOFF_REPORT.md`

## Claim Boundary

The latency estimate is MAC-rate based and pre-synthesis only. It is not csynth/cosim/P&R evidence.
