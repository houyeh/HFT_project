# PHC-lite vs DeepLOB Efficiency / Resource Trade-off

status: analysis_complete_no_deeplob_hls_claim

## Scope

This is an analysis-only comparison. It does not implement DeepLOB in HLS, does not retrain either model, does not run new csim/csynth, and does not use project-internal NEW1-NEW5 data.

The goal is to put measured PHC-lite FI-2010 numbers and published DeepLOB numbers into one efficiency/resource framework while clearly labeling `measured` versus `estimated`.

## Sources And Verification Labels

| item | value |
| --- | --- |
| PHC-lite FI-2010 F1 source | `results/stage1_primary_metrics_v2.csv`, `REPORT_STAGE1.md` Stage 1 v2 |
| PHC-lite Stage2 HLS source | `results/stage2_csynth_summary.csv`, `REPORT_STAGE2.md` |
| PHC-lite trainable parameters | 2,115, measured locally |
| PHC-lite Stage2 csynth resource | DSP 33, LUT 20582, FF 11711, BRAM_18K 3, latency 7347 cycles / 73.470 us |
| DeepLOB F1 source | Zhang, Zohren, Roberts, arXiv:1808.03668, Table II, as already quoted in `REPORT_STAGE1.md` |
| DeepLOB paper parameter source | Zhang, Zohren, Roberts, Table III; printed as `60k`, not an exact integer |
| DeepLOB reference implementation parameter source | public `zcakhaa/DeepLOB...` TensorFlow notebook summary; exact count 142,435 for the inspected Keras implementation |

Important: because the DeepLOB paper table prints `60k` rather than an exact parameter integer, this report uses 60,000 as the paper-table value and explicitly labels it `paper-rounded`. That is favorable to DeepLOB in the F1-per-parameter calculation. The reference implementation exact count is shown separately, not silently substituted.

## Efficiency Table

F1 values are macro-F1 percentage points. `F1_pct_per_1M_MACs` uses nominal linear/conv/recurrent MACs and excludes nonlinearities, normalization, memory access, control, and softmax.

| model | k | macro_F1_pct | params_used | F1_pct_per_1k_params | MACs_used | F1_pct_per_1M_MACs | verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHC-lite FI-2010 v2 | 10 | 40.67 | 2115 measured | 19.23 | 1,968 nominal | 20666.26 | software_measured_F1 + local_MAC_formula |
| DeepLOB | 10 | 83.40 | 60,000 paper-rounded | 1.39 | 39,040,192 estimated | 2.14 | published_F1 + estimated_MACs |
| PHC-lite FI-2010 v2 | 20 | 39.80 | 2115 measured | 18.82 | 1,968 nominal | 20223.69 | software_measured_F1 + local_MAC_formula |
| DeepLOB | 20 | 72.82 | 60,000 paper-rounded | 1.21 | 39,040,192 estimated | 1.87 | published_F1 + estimated_MACs |
| PHC-lite FI-2010 v2 | 50 | 40.72 | 2115 measured | 19.25 | 1,968 nominal | 20690.74 | software_measured_F1 + local_MAC_formula |
| DeepLOB | 50 | 80.35 | 60,000 paper-rounded | 1.34 | 39,040,192 estimated | 2.06 | published_F1 + estimated_MACs |

## DeepLOB Parameter Count Sensitivity

The paper-table parameter count and the public reference implementation parameter count differ materially. This table makes that visible instead of blending the two.

| k | DeepLOB_F1_pct | paper_60k_F1_per_1k_params | ref_impl_142435_F1_per_1k_params |
| --- | --- | --- | --- |
| 10 | 83.40 | 1.39 | 0.59 |
| 20 | 72.82 | 1.21 | 0.51 |
| 50 | 80.35 | 1.34 | 0.56 |

## PHC-lite MAC Accounting

The FI-2010 v2 classifier has zero `decay` and zero `alpha_scalar` buffers across all surveyed v2 checkpoints (`results/decay_parameter_survey.csv`). Therefore gain/synapse branches exist in the source architecture but are inactive in the mathematical Stage2 path and may be pruned by HLS.

For the main efficiency/resource estimate, this report uses the conservative nominal source count of 1,968 MACs. The active pruned count is 1,200 MACs and would make PHC-lite look even more resource-efficient.

| block | formula | MACs | active_when_decay_alpha_zero |
| --- | --- | --- | --- |
| enc_linear | 40 * 16 | 640 | yes |
| gate | 16 * 16 | 256 | yes |
| candidate | 16 * 16 | 256 | yes |
| gain_from_encoded | 16 * 16 | 256 | no |
| gain_from_prev_state | 16 * 16 | 256 | no |
| synapse | 16 * 16 | 256 | no |
| readout | 16 * 3 | 48 | yes |
| total_nominal_source | 640 + 5*256 + 48 | 1968 | mixed |
| total_active_HLS_pruned | 640 + 2*256 + 48 | 1200 | yes |

## DeepLOB MAC Estimate

DeepLOB MACs are estimated from the public CNN + inception + LSTM architecture commonly used for FI-2010: input 100x40x1, convolutional blocks reducing the feature axis, inception branches, LSTM with input width 192 and hidden size 64, then a 3-way dense output.

| block | formula | MACs |
| --- | --- | --- |
| conv block 1 | 128k + 8.192M + 8.192M | 16512000 |
| conv block 2 | 2.048M + 4.096M + 4.096M | 10240000 |
| conv block 3 | 1.024M + 409.6k + 409.6k | 1843200 |
| inception | 204.8k + 1.2288M + 204.8k + 2.048M + 204.8k | 3891200 |
| LSTM | 100 * 4 * 64 * (192 + 64) | 6553600 |
| dense | 64 * 3 | 192 |
| total | sum above | 39040192 |

DeepLOB total estimated MACs per inference: **39,040,192**.

## Resource Extrapolation

PHC-lite Stage2 measured csynth gives 33 DSP and 20582 LUT for 1,968 nominal MACs. That corresponds to:

- DSP per 1,000 nominal PHC MACs: 16.768
- LUT per 1,000 nominal PHC MACs: 10458.333

**The DeepLOB resource rows below are estimate_only. They are a linear order-of-magnitude extrapolation, not a synthesis result. CNN convolution and LSTM gates may scale better or worse than this due to weight reuse, reuse factor, pipeline scheduling, memory banking, quantization, and hand optimization. These estimates must not be cited as real DeepLOB HLS utilization.**

| basis | verification | MACs | DSP | LUT | DSP_vs_Zynq7020 | LUT_vs_Zynq7020 |
| --- | --- | --- | --- | --- | --- | --- |
| PHC-lite Stage2 measured baseline | measured_csynth | 1,968 | 33 | 20582 | 0.15x | 0.39x |
| DeepLOB MAC-linear extrapolation from PHC Stage2 | estimate_only | 39,040,192 | 654,637 | 408,295,341 | 2975.6x | 7674.7x |
| DeepLOB parameter-linear lower-bound using paper 60k | estimate_only_sanity_check | not used | 936 | 583,887 | 4.3x | 11.0x |
| DeepLOB parameter-linear using reference implementation 142,435 | estimate_only_sanity_check | not used | 2,222 | 1,386,098 | 10.1x | 26.1x |

## Literature Check For DeepLOB FPGA/HLS Implementations

Searches performed on 2026-07-30:

- `DeepLOB FPGA`
- `DeepLOB HLS`
- `DeepLOB hls4ml`
- `DeepLOB Vitis FPGA`
- `FINN-GL ConvLSTM DeepLOB`

I found the DeepLOB paper and the public architecture notebooks. I also found FINN-GL, a 2025 FPGA/LSTM deployment paper that targets quantized ConvLSTM-style short-term stock prediction and mentions a DeepLOB-like setting. It is not a direct public HLS/Vitis implementation of the original DeepLOB CNN+inception+LSTM architecture on FI-2010 with comparable Table II F1 numbers. Therefore this report keeps Section `Resource Extrapolation` as `estimate_only` instead of replacing it with real DeepLOB FPGA numbers.

## Honest Conclusion

PHC-lite does **not** beat DeepLOB on FI-2010 accuracy. On the Stage 1 v2 public benchmark, PHC-lite reaches roughly 40-41 macro-F1 for k=10/20/50, while DeepLOB reports roughly 73-83 macro-F1. The gap is large and should not be softened.

PHC-lite's advantage is efficiency and low-resource feasibility. With 2,115 trainable parameters and a csynth-measured Stage2 design using 33 DSP and 20,582 LUT on Zynq-7020, it is a small model that can be synthesized under this project's manual HLS method. DeepLOB is much stronger statistically, but the published CNN+LSTM architecture is much larger. Under the requested MAC-linear extrapolation, DeepLOB would exceed Zynq-7020 by about **2975.6x DSP** and **7674.7x LUT**. Even the much softer parameter-linear sanity check using the paper-rounded 60k parameter count suggests about **4.3x DSP** and **11.0x LUT**.

The correct claim is therefore:

> PHC-lite is weaker than DeepLOB in FI-2010 predictive accuracy, but it has a large parameter/MAC/resource-efficiency advantage. A DeepLOB-class model may need compression, quantization, reuse-factor tuning, or a different FPGA target before fitting the same low-cost Zynq-7020 envelope. This is an efficiency/resource trade-off, not an overall performance win.

## Source Links

- DeepLOB arXiv page: https://arxiv.org/abs/1808.03668
- DeepLOB PDF: https://arxiv.org/pdf/1808.03668
- DeepLOB public GitHub notebook repository: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- FINN-GL related FPGA/LSTM paper: https://arxiv.org/abs/2506.20810
