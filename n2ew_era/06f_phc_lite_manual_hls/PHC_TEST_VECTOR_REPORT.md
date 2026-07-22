# PHC Step Test Vector Report

This phase generates float32 input/output reference vectors for a manual Vitis HLS C++ testbench. It reuses the frozen 06E step model and the Phase 05A validation sequence loader. No training, no threshold selection, and no NEW5 access are performed.

## Corpus Size

- Split: `val`.
- Sequences used: `200`.
- Timesteps per sequence: `128`.
- Total rows: `25600`.

The 200-sequence corpus gives 25,600 recurrent-step rows. This is large enough to cover many state transitions while staying practical for a recurrent HLS cosimulation with LayerNorm/sigmoid/tanh-style nonlinearities. It is a validation subsample, not the full split.

## Equivalence Cross-check

- Source checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt`.
- Step checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06e_phc_stp_step_hls4ml\models\phc_stp_step_hls_seed2.pt`.
- Source seed: `2`.
- Source best epoch: `171`.
- Max absolute final-readout diff: `3.87430191e-07`.
- Mean absolute final-readout diff: `7.95000261e-08`.
- Allclose at `atol=1e-4`: `True`.
- Step alpha source formula: `raw PHCLiteSTP.alpha, because source step uses z + self.alpha * message`.

## CSV Schema

- `row_id`, `sample_idx`, `timestep`
- `x_t_0..x_t_15`
- `state_prev_0..state_prev_15`
- `ref_readout_0`, `ref_readout_1`
- `ref_state_next_0..ref_state_next_15`

All floating-point values are written with `%.9g` formatting after conversion to float32, to avoid CSV round-trip noise dominating the testbench tolerance.

## Output

- CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\results\phc_step_test_vectors.csv`.
- CSV size: `16052080` bytes.
- Summary CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\results\phc_step_test_vector_summary.csv`.

This is a testbench reference artifact only. It does not claim FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.
