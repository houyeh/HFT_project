# models_explanation

## Purpose

This folder contains the neural-network definitions and model-side hardware helpers.

## Files

- `tcn.py`
  - the main TCN classifier plus an HLS-friendly PyTorch variant
- `hardware.py`
  - MAC counting and post-training quantization helpers
- `walk_forward_optimize.py`
  - chronological sweep tool for choosing trading controls on saved predictions

## Why `tcn.py` Is Important

This is the key bridge file between research and hardware:

- training model
- HLS-friendly model
- weight migration path between the two

That is why this file matters the most when explaining your HLS progress.
