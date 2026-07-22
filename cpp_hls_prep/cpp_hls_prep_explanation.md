# cpp_hls_prep_explanation

## Purpose

This folder is the hardware-preparation bridge between the PyTorch project and Vitis HLS.

## Structure

- `include/int8_kernels.hpp`
  - declarations for the dynamic CPU-side INT8 reference path
- `include/fixed_tcn.hpp`
  - fixed-shape tensor and weight containers for the HLS-oriented model
- `include/fixed_tcn_modules.hpp`
  - declarations for fixed-point building blocks
- `include/fixed_tcn_top.hpp`
  - top-level HLS entrypoint declaration
- `src/int8_kernels.cpp`
  - dynamic INT8 inference using named tensors loaded from `.wts`
- `src/fixed_tcn.cpp`
  - fixed-shape TCN reference inference
- `src/fixed_tcn_modules.cpp`
  - fixed-point conv, residual, pooling, and softmax kernels
- `src/fixed_tcn_top.cpp`
  - Vitis HLS top wrapper
- `src/hls_testbench.cpp`
  - HLS C simulation testbench
- `src/main.cpp`
  - command-line harness for CPU inference and fixed-core comparison
- `hls/`
  - Vitis HLS scripts and generated project outputs

## What Has Been Achieved Here

- fixed model shape defined
- integer kernels written by hand
- generated headers connected to the C++ code
- HLS testbench prepared
- `csim` passed
- `csynth` completed

## Current Limitation

The current HLS design is functionally complete but too large for the selected `xc7z020-clg400-1` target.
So this folder now represents an optimization problem, not a missing-pipeline problem.
