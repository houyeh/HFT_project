# Scripts Layout

## Purpose

This folder contains the runnable entrypoints for the project.

The structure is organized by function and by the order you would usually use
the scripts in a real workflow.

## Folder Guide

### `01_data_prep/`

Use these first when you need mock data or an early-stage tensor pipeline.

- `generate_mock_data.py`
- `run_pipeline.py`

### `02_train_eval/`

Use these when you want to train a baseline model or directly evaluate one.

- `train_tw_baseline.py`
- `train_cme_baseline.py`
- `evaluate_cme_frozen_model.py`

### `03_experiments/`

Use these when you want batch sweeps over hyperparameters or strategy settings.

- `sweep_latency.py`
- `sweep_tw_baseline.py`
- `sweep_tw_improvements.py`

### `04_analysis/`

Use these after a run to inspect behavior, latency, gating, or quantization.

- `analyze_*`
- `simulate_int8_inference.py`
- `estimate_tcn_latency.py`

### `05_hardware_handoff/`

Use these when the model is chosen and you want to export, verify, or bridge to
HLS/C++.

- `export_tcn_int8_artifacts.py`
- `verify_cpp_python_bit_accurate.py`
- `verify_hls_equivalence.py`
- `attempt_hls4ml_conversion.py`

### `06_reporting/`

Use these after experiments are done and you want charts, reports, or slides.

- `generate_cme_w6_result_assets.py`
- `generate_meeting_charts.py`
- `generate_tradeoff_chart.py`
- `generate_failed_chart.py`
- `generate_professor_ppt.py`

## Suggested Entry Order

If someone asks what to open first, the most representative path is:

1. `scripts/02_train_eval/train_tw_baseline.py`
2. `scripts/03_experiments/sweep_latency.py`
3. `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`
4. `scripts/05_hardware_handoff/verify_cpp_python_bit_accurate.py`
