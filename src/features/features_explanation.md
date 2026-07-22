# features_explanation

## Purpose

This folder turns normalized order-book data into model-ready signals.

## Files

- `dimensionless_features.py`
  - base scale-invariant signals such as spread, micro return, OBI, and weighted mid
- `execution_features.py`
  - trade-flow, queue, aggressor, and cancel-pressure features
- `sliding_window.py`
  - converts event rows into `(samples, sequence, features)` tensors and future-direction labels

## Current Project Role

The execution-aware feature set is the important one for the current HLS line.
It is what connects raw replay data to the TCN that later gets exported into INT8 and C++.
