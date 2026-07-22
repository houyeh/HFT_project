# utils_explanation

## Purpose

This folder holds small shared definitions that many parts of the project depend on.

## Current Files

- `constants.py`
  - canonical tick sizes, tensor shapes, class counts, and default TCN hyperparameters

## Why It Matters

Even though this folder is small, it keeps the training, backtest, and hardware-export paths consistent.
If these constants drift, the software and HLS branches stop matching.
