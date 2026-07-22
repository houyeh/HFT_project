# tests_explanation

## Purpose

This folder checks that the project pipeline still works after changes.

## Coverage Areas

- data parsing
- feature engineering
- sliding-window datasets
- TCN model behavior
- event-driven backtest
- hybrid market maker
- hardware profiling and quantization helpers

## Why Tests Matter For HLS

The HLS path adds many transformations:

- FP32 model
- HLS-friendly model
- INT8 export
- C++ reference
- Vitis HLS testbench

Without regression tests, it becomes hard to know where numerical drift or logic mismatch first appears.
