# Repository Map

- `data/`
  - purpose: Raw market data, LOB sequences, trade tapes.
  - important files: `NEW_ALL`, `NEW3`
  - safe to modify? NO
  - should archive? NO
  - notes: Never delete.

- `new_era/15_models_playaround/`
  - purpose: Source code for generating main baseline TCN models.
  - important files: `run_diagnosis_1.py`, `00_run_all_one_click.py`
  - safe to modify? NO
  - should archive? NO
  - notes: Core pipeline logic.

- `new_era/15_models_playaround/015_models_playaround_result/`
  - purpose: Primary model outputs and diagnosis sweeps.
  - important files: `experiment_config.json`, `train_history.csv`, `baseline_summary_fixed.csv`
  - safe to modify? NO
  - should archive? NO
  - notes: The single source of truth for the baseline model.

- `new_era/016_new_binary/`
  - purpose: Binary fill-conditional quote-risk experiments.
  - important files: `outputs/reports/new_binary_model_report.md`
  - safe to modify? YES (can write to `outputs_fix`)
  - should archive? NO
  - notes: Negative result but important PoC.

- `new_era/017_regime_auto_gate/`
  - purpose: Causal auto-gating experiments.
  - important files: `outputs/reports/regime_auto_gate_report.md`
  - safe to modify? YES (script additions)
  - should archive? NO
  - notes: Proof of regime dependency.

- `new_era/018_queue_aware_validation/`
  - purpose: Next priority validation scripts.
  - important files: `outputs/audit/queue_data_audit.md`
  - safe to modify? YES
  - should archive? NO
  - notes: Currently pending execution.

- `meeting/w13/`
  - purpose: Week 13 meeting presentation generation.
  - important files: `簡報大綱.md`, `__results/`
  - safe to modify? YES
  - should archive? NO
  - notes: Slide decks and generated result extractions.

- `scripts/`
  - purpose: Helper scripts.
  - safe to modify? YES
  - should archive? NO

- `src/`
  - purpose: Core libraries.
  - safe to modify? YES
  - should archive? NO
