# Phase 07B Runbook: TCN PrePad Manual HLS

This phase builds a manual Vitis HLS implementation for the Phase 07A `tcn_prepad_16feat` models.

## Scope

- Model class: `src/models/tcn_prepad.py::TCNPrePadRegressor`.
- Variants:
  - `scratch`: `tcn_prepad_scratch_16feat_seed2_best_val.pt`
  - `distilled`: `tcn_prepad_distilled_16feat_seed1_best_val.pt`
- Fixed-point scheme:
  - inputs / weights / outputs: `ap_fixed<24,12>`
  - accumulators: `ap_fixed<40,16>`
- Nonlinearity: ReLU only. No tanh/sigmoid LUT is needed.
- This phase does not run P&R.

## Latency Definition

This HLS top function measures a naive per-event recomputation path:

`new event -> provide latest normalized 128x16 feature window -> internally prepend 124 zero timesteps -> run full 5-block valid-conv TCN -> emit bid/ask readout`

It does **not** implement sliding-window activation caching. Therefore its latency is not the same object as PHC-lite's single recurrent step latency. The result is still useful as a first real HLS synthesis replacement for the earlier MAC-rate estimate.

## Commands

Run from repository root:

```bash
python n2ew_era/07b_tcn_prepad_manual_hls/00_export_tcn_weights.py
python n2ew_era/07b_tcn_prepad_manual_hls/01_generate_tcn_test_vectors.py
python n2ew_era/07b_tcn_prepad_manual_hls/02_prepare_hls_components.py
```

Then run Vitis manually.

Scratch:

```bash
cd n2ew_era/07b_tcn_prepad_manual_hls/hls_component_scratch
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --syn  --config hls_config.cfg --work_dir hls_component
```

Distilled:

```bash
cd ../hls_component_distilled
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --syn  --config hls_config.cfg --work_dir hls_component
```

Optional cosim only after csynth fits budget:

```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

Collect outputs:

```bash
cd ~/projects/01_HFT_projject
python n2ew_era/07b_tcn_prepad_manual_hls/03_collect_hls_results.py
```

## Expected Output Files

- `hls_src/tcn_weights_scratch.hpp`
- `hls_src/tcn_weights_distilled.hpp`
- `results/tcn_weight_export_audit.csv`
- `results/tcn_prepad_scratch_test_vectors.csv`
- `results/tcn_prepad_distilled_test_vectors.csv`
- `results/tcn_prepad_manual_hls_summary.csv`
- `TCN_PREPAD_MANUAL_HLS_SUMMARY_REPORT.md`

## Claim Boundary

- `32.656us` from Phase 07A is a pre-synthesis MAC-rate estimate.
- This phase upgrades the TCN latency evidence to C-sim/C-synthesis only after the user runs Vitis.
- PHC-lite `83.16us` remains a routed P&R timing-verified number, a stronger verification level.
- No NEW5 is used.
- No real-world profitability or FPGA PnL improvement claim.

