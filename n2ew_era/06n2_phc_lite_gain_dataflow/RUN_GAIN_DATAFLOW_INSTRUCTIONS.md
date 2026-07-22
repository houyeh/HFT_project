# PHC-lite Gain/Main DATAFLOW Experiment

This phase is copied from `n2ew_era/06n_phc_lite_branch_dataflow` and keeps 06N unchanged as the stage1 baseline.

## What Changed

- `phc_step()` still performs the front-end encoder path and final loop/readout path in the same order.
- The middle section is now wrapped by `step_dataflow_fn()`.
- `step_dataflow_fn()` contains exactly three same-level function calls:
  - `fork_inputs_for_main_gain(...)`
  - `main_chain_fn(...)`
  - `gain_chain_fn(...)`
- `main_chain_fn()` is `#pragma HLS INLINE off` and performs:
  - existing `gate_candidate_dataflow(...)`
  - `state_temporal`
  - `SYNAPSE_W`
  - `silu`
- `gain_chain_fn()` is `#pragma HLS INLINE off` and performs:
  - `GAIN_FROM_ENCODED_W`
  - `GAIN_FROM_PREV_STATE_W`
  - add
  - `sigmoid_16`
- `fork_inputs_for_main_gain()` is `#pragma HLS INLINE off` and duplicates both `encoded` and `state_prev` so the main and gain processes do not share input array reads directly.

## Required C-Sim Check

Run C-sim against the same test vectors, then compare the output CSV directly against the 06N stage1 C-sim CSV.
This experiment is intended to change scheduling only, so the CSV diff should be empty.

Recommended PowerShell commands from the repository root:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06n2_phc_lite_gain_dataflow\hls_component
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

Then compare:

```powershell
fc.exe /b `
  C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06n_phc_lite_branch_dataflow\results\phc_step_csim_out.csv `
  C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06n2_phc_lite_gain_dataflow\results\phc_step_csim_out.csv
```

Any difference means the refactor changed observable C-level output and should not proceed to synthesis.

## Result Collection

After C-sim, C-synth, or C/RTL co-sim artifacts exist, run:

```powershell
python C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06n2_phc_lite_gain_dataflow\01_collect_results.py
```

This writes:

- `results/hls_result_summary.csv`
- `results/hls_result_summary.md`

## Claim Boundary

This is only a scheduling experiment. It does not modify weights, retrain the model, use NEW5, prove real exchange behavior, or make any PnL claim.
