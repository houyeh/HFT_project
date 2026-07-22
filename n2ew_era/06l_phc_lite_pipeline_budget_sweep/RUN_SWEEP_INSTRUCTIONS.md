# PHC-lite Pipeline Budget Sweep Instructions

This phase prepares synthesis variants only. It does not modify `n2ew_era/06f_phc_lite_manual_hls/`.

For each factor, run from the corresponding variant component directory:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\variants\factor_<N>\hls_component
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --csim --config hls_config.cfg --work_dir hls_component
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --syn  --config hls_config.cfg --work_dir hls_component
```

Only if DSP < 220, LUT < 53200, FF < 106400, and BRAM_18K < 280 should you run:

```powershell
& 'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat' --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

After running the desired factors, collect the table with:

```powershell
python n2ew_era\06l_phc_lite_pipeline_budget_sweep\01_collect_sweep_results.py
```

Factor 1 intentionally preserves the v8 semantics: no inner unroll and no weight partition. Factors 2/4/8/16 add `#pragma HLS UNROLL factor=<N>` to the inner `i` loops of the four linear helper functions and `#pragma HLS ARRAY_PARTITION variable=w cyclic factor=<N> dim=2` to the corresponding formal weight argument.

Do not raise `PHC_ABS_TOL`, do not edit the 06F baseline, and do not use these synthesis numbers as PnL claims.
