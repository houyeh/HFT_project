# training_explanation

## Purpose

This folder trains the TCN and produces prediction tables that the backtests consume.

## Files

- `datasets.py`
  - lazy sliding-window dataset that avoids materializing all tensors at once
- `trainer.py`
  - epoch loop, validation, prediction export, and early stopping
- `metrics.py`
  - small classification metric helpers

## Important Detail

The project preserves chronological order when splitting train, validation, and test data.
That is important because HFT research can otherwise leak future information.
