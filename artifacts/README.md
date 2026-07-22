# Artifacts Layout

## Purpose

This folder stores saved experiment outputs, frozen checkpoints, reports, and
hardware handoff files.

The structure is organized by artifact purpose and by when you would usually
need the outputs.

## Folder Guide

### `01_project_plans/`

High-level planning documents and pivot memos.

- `first_principles_pivot_plan.md`

### `02_model_runs/`

Training runs, validation outputs, sweep results, and market-transfer results.

- `cme_eval_smoke/`
- `cme_eval_transfer/`
- future outputs such as `tw_baseline/`, `cme_baseline/`, `tw_sweep/`,
  `tw_improvements/`, `tw_execution_aware_valcal/`

### `03_latency_proof/`

Artifacts used to prove latency sensitivity, frozen winner quality, and
delay-dependent PnL differences.

- `latency_proof/`
- `latency_proof_A/`
- `latency_proof_C/`
- `latency_method_comparison.csv`

### `04_hardware_handoff/`

Everything related to INT8 export, latency estimation, and hls4ml feasibility.

- `int8_export/`
- `latency_estimate/`
- `hls4ml_attempt/`

## Why It Matters

When you need evidence for a teacher, reviewer, or teammate, `artifacts/` is
the fastest place to show that the work has already been executed and saved.
