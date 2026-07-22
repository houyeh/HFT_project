# PHC Diagonal Step Test Vector Report

This file documents the float32 input/output vectors used for the Phase 06G manual HLS C++ testbench. It reuses the Phase 05A validation sequence loader and the Phase 05F selected diagonal checkpoint. No training, no threshold selection, and no NEW5 access are performed.

## Corpus Size

- Split: `val`.
- Sequences used: `200`.
- Timesteps per sequence: `128`.
- Total recurrent-step rows: `25600`.

The corpus intentionally mirrors Phase 06F: 200 validation sequences x 128 timesteps = 25,600 rows.

## Equivalence Cross-check

- Source checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05f_phc_diagonal_ssm\models\phc_diagonal_ssm_seed2_best_val.pt`.
- Selected seed: `2`.
- Selected best epoch: `157`.
- Max absolute final-readout diff: `0`.
- Mean absolute final-readout diff: `0`.
- Allclose at `atol=1e-06`: `True`.

## CSV Schema

- `row_id`, `sample_idx`, `timestep`
- `x_t_0..x_t_15`
- `state_prev_0..state_prev_15`
- `ref_readout_0`, `ref_readout_1`
- `ref_state_next_0..ref_state_next_15`

All floating-point values are written with `%.9g` formatting after conversion to float32.

## Output

- CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\phc_diag_step_test_vectors.csv`.
- CSV size: `15823949` bytes.
- Summary CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\phc_diag_step_test_vector_summary.csv`.

This is a testbench reference artifact only. It does not claim FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.
