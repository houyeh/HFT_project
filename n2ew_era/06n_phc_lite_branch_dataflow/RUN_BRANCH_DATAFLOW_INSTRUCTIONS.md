# PHC-lite Branch DATAFLOW Experiment

This phase is an isolated experiment copied from `n2ew_era/06l_phc_lite_pipeline_budget_sweep`.
It does not modify the frozen 06F production baseline or the 06L sweep sources.

## What Changed

- Added `#pragma HLS DATAFLOW` only inside `gate_candidate_dataflow()`.
- Replaced the sequential gate/candidate section with two adjacent same-level calls:
  - `gate_branch(encoded_for_gate, gate_t)`
  - `candidate_branch(encoded_for_candidate, candidate_t)`
- `gate_branch()` and `candidate_branch()` are explicitly `#pragma HLS INLINE off`.
- `phc_step()` itself is not a DATAFLOW region. This keeps the broader recurrent step out of DATAFLOW canonical-form checks.
- `fork_encoded_for_gate_candidate()` duplicates the shared `encoded` vector into two branch-local buffers so Vitis does not need to merge the gate and candidate processes due to a shared read port.
- The branch internals preserve the same numerical path:
  - `GATE_W(encoded) -> sigmoid`
  - `CANDIDATE_W(encoded) -> tanh`
- All other matrix multiplies, `PIPELINE off` pragmas, weights, and testbench tolerance are unchanged.

## Run Commands

Run from:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06n_phc_lite_branch_dataflow\hls_component
```

C simulation:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

C synthesis:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --syn --config hls_config.cfg --work_dir hls_component
```

Co-simulation:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

## Required Checks

1. C-Sim must pass with `mismatches=0`.
2. For this branch-only refactor, the intended strict check is `max_abs_diff=0`.
3. Inspect synthesis logs for DATAFLOW acceptance:
   - If Vitis reports a DATAFLOW canonical form warning or violation, copy the warning text into the experiment notes.
   - Do not treat latency changes as valid evidence of branch parallelism unless DATAFLOW was accepted.
4. Compare synthesis output against the 06L factor-1 / 06F v8 endpoint:
   - Baseline latency: 8316 cycles.
   - Baseline DSP: 46.
   - Baseline LUT: 24836.
   - Baseline FF: 14518.
   - Baseline BRAM_18K: 6.

## Claim Boundary

This experiment only tests whether the independent gate and candidate branches can be scheduled with DATAFLOW.
It does not retrain the model, does not change weights, does not use NEW5, and does not make any FPGA PnL claim.
