# Project Control Checkpoint

## 1. Created Files
- `00_PROJECT_STATE.md`
- `01_AGENT_ROLES.md`
- `02_EXPERIMENT_REGISTRY.csv`
- `03_ALLOWED_CLAIMS.md`
- `04_FORBIDDEN_CLAIMS.md`
- `05_CURRENT_SLIDE_STORY.md`
- `06_NEXT_TASKS.md`
- `07_REPO_MAP.md`
- `08_DATA_SOURCES.md`
- `09_ENVIRONMENT_AUDIT.md`
- `10_CLEANUP_PLAN.md`
- `AGENTS.md`
- `README_FOR_NEW_AGENT.md`
- `pm_reviews/current_repo_pm_review.md`

## 2. File Purposes
Serve as a unified, non-destructive brain context for future LLM agents to synchronize instantly with the project's statistical facts, safe claims, and priorities.

## 3. Data Sources Used
- `015` baseline: `train_history.csv`, `classification_report_test.csv`, `baseline_summary_fixed.csv`, `threshold_sweep_enhanced.csv`, `validation_threshold_selection.csv`
- `015` diagnosis: `regime_gate_sweep.csv`, `new3_session_audit.csv`
- `016` binary: `binary_label_distribution.csv`, `new_binary_model_report.md`, `model_comparison_summary.csv`, `binary_vs_baselines.csv`, `final_checkpoint.md`
- `017` auto-gate: `final_checkpoint.md`, `regime_auto_gate_report.md`
- `018` queue: `queue_data_audit.md`

## 4. Missing Sources
- `016_new_binary/outputs_fix/` results are MISSING.
- `018_queue_aware_validation/outputs/` backtest and comparison CSVs are MISSING (experiment pending).

## 5. Requires Human Confirmation
- Approval to archive `015_models_playaround_result_expC_only/` and related duplicate folders.
- Review of the queue-aware proxy assumptions since exact tape matching is mathematically impossible without `aggressor_side`.

## 6. Cleanup Summary
The repo is mostly clean but suffers from heavy folder duplication for snapshots. No files have been deleted. A structured `10_CLEANUP_PLAN.md` is registered for human sign-off.

## 7. Next Step Recommendations
Proceed directly to 018 Queue-Aware Validation execution to establish conservative PnL bounds.
