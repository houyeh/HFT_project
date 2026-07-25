status: open

# AUDIT: 06L unroll-16 P&R timing handoff

## Scope

This audit attempts to promote the 06L unroll-16 PHC-lite latency result
from HLS/cosim evidence to Vivado routed timing evidence.

Guardrails honored:

- Frozen 06F v8 artifacts were not modified.
- The 06L work was copied into an independent workspace:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/`.
- The 06L design uses the same top/interface/part/clock as 06F v8:
  `phc_step`, `xc7z020clg400-1`, `clock=10ns`.
- The only source-level design difference versus 06F v8 is the existing 06L
  unroll-16 linear-loop pragma set (`UNROLL factor=16` plus matching cyclic
  weight partition).
- No NEW5 data was loaded or used.

## Completed Verification

| Item | 06F v8 frozen baseline | 06L unroll-16 copied workspace |
|---|---|---|
| HLS latency | 8316 cycles / 83.160 us (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) | 2316 cycles / 23.160 us (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| HLS interval | 8317 cycles (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) | 2317 cycles (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| HLS estimated clock | 7.147 ns (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) | 8.223 ns (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| RTL cosim | Verilog PASS, existing 06F evidence (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_cosim.rpt`) | Verilog PASS; `total=25600 mismatches=0`, `max_abs_diff=0.00725275`, `mean_abs_diff=0.000372553`, `tolerance=1` (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_cosim.log`) |
| RTL cosim latency | 8316 cycles (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_cosim.rpt`) | 2316 cycles (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/reports/hls_cosim.rpt`) |

06L csynth was run from:

```powershell
subst P: "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz"
cd P:\hls_component
& "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --tcl run_csynth_only.tcl
subst P: /D
```

06L cosim was then run from:

```powershell
subst P: "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz"
cd P:\hls_component
& "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --cosim --config hls_config.cfg --work_dir hls_component
subst P: /D
```

## Post-Route Status

100 MHz implementation/P&R was attempted but did not reach routed timing.
Therefore the 06L unroll-16 result is not yet upgraded to a routed
post-implementation latency/timing claim.

| Metric | 06F v8 routed baseline | 06L unroll-16 100 MHz P&R attempt |
|---|---|---|
| WNS | +0.020 ns (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt`) | not produced; latest implementation stopped before project creation (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_impl.log`; `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/vivado.log`) |
| TNS | 0.000 ns (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt`) | not produced; latest implementation stopped before project creation (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_impl.log`; `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/vivado.log`) |
| LUT | 7555 (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt`) | not produced post-route; csynth estimate is 36792 (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| FF | 8217 (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt`) | not produced post-route; csynth estimate is 35774 (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| DSP | 47 (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt`) | not produced post-route; csynth estimate is 121 (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |
| BRAM | 6 (`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt`) | not produced post-route; csynth estimate is 0 (`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`) |

The 100 MHz implementation attempt passed the earlier HLS/IP packaging
blocker after setting a local Tcl package path, then failed inside Vivado's
synthesis run launcher:

```text
P:\hls_component\hls_component\hls\impl\verilog\project.runs\bd_0_hls_inst_0_synth_1\rundef.js(122, 5) JavaScript runtime error: Access denied
```

Evidence:

- Top-level implementation log:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vitis_hls_impl_unroll16_100mhz_pdrive_tcllibpath_full.log`
- Vitis HLS implementation log:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_impl.log`
- Vivado sub-run log:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/project.runs/bd_0_hls_inst_0_synth_1/runme.log`

This is an environment/tool execution failure, not a reported timing failure.
No 90 MHz rerun was started because there is no 100 MHz routed WNS<0 result.

### User-run update: 2026-07-22 23:59 to 2026-07-23 00:00

The pasted user-run log shows this command was launched from inside
`pnr_unroll16_100mhz/hls_component/hls_component`:

```powershell
& "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --impl --config C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component\..\hls_config.cfg --work_dir .
```

This run advanced past the previous IP packaging/TclStore blocker:

```text
INFO: Created IP C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/ip/component.xml
INFO: Created IP archive C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/ip/xilinx_com_hls_phc_step_1_0.zip
```

Vivado then started RTL evaluation and failed at the first project creation
step, before synth/place/route:

```text
# create_project $vivado_proj_name $vivado_proj_dir -part $target_device -force
ERROR: [Project 1-161] Failed to remove the directory 'C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/project.runs'. The directory might be in use by some other process.
```

Evidence is now in the project logs:

- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_impl.log`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/vivado.log`

This second failure is also an environment/workspace lock failure, not a
timing failure. It means the 06L IP package exists, but no 100 MHz routed
WNS/TNS or post-route utilization report has been produced yet.

## Handoff Commands For Next User-Run P&R

Run this outside the Codex sandbox, preferably from a normal local PowerShell
with permissions to execute Vivado-generated JavaScript run launchers. If
Windows Controlled Folder Access, antivirus, or corporate policy blocks
Vivado run scripts, run the shell as Administrator or allow-list:

- `C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat`
- `C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat`
- the workspace directory under
  `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz`

Before retrying, close Vitis IDE/Vivado windows that may be holding file
handles under `hls/impl/verilog`. Confirm there are no live Vitis/Vivado
subprocesses using:

```powershell
Get-Process | Where-Object { $_.ProcessName -match "vitis|vivado|xsim|xelab|xvlog|xsc|vitis-run|java" } | Select-Object Id,ProcessName,Path,StartTime
```

If the same workspace is reused, remove only generated Vivado implementation
project directories under the 06L copied workspace after all Vitis/Vivado
processes are closed. Do not run this against the frozen 06F directory.

```powershell
$impl = "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component\hls\impl\verilog"
foreach ($name in "project.runs","project.cache","project.gen","project.hw","project.ip_user_files","project.srcs",".Xil") {
    $p = Join-Path $impl $name
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Recurse -Force
    }
}
```

Then rerun the implementation from the same directory layout that successfully
parsed the config and packaged the IP:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component
& "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --impl --config C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component\..\hls_config.cfg --work_dir . *> ..\..\results\vitis_hls_impl_unroll16_100mhz_user_retry_clean.log
$LASTEXITCODE
```

The previous `subst P:` form is still available as an alternate route if long
paths or Windows path handling reappear:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project
subst P: "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz"
$env:TCLLIBPATH = "C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/support/appinit C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/tclapp/xilinx/xsim C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/tclapp/xilinx C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/tclapp C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/support"
cd P:\hls_component
& "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --impl --config hls_config.cfg --work_dir hls_component *> P:\results\vitis_hls_impl_unroll16_100mhz_user_run.log
$LASTEXITCODE
cd C:\Users\holbo\00VS_projjects\00_HFT_project
subst P: /D
```

If the run succeeds, collect these files:

- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/reports/hls_impl_pnr.rpt`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/report/phc_step_utilization_routed.rpt`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/report/phc_step_timing_routed.rpt`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/report/phc_step_timing_paths_routed.rpt`

Useful extraction commands after a successful run:

```powershell
Select-String -Path n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component\hls\impl\verilog\project.runs\impl_1\bd_0_wrapper_timing_summary_routed.rpt -Pattern "WNS\(ns\)|ap_clk|All user specified timing constraints"
Select-String -Path n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz\hls_component\hls_component\reports\hls_impl_pnr.rpt -Pattern "LUT:|FF:|DSP:|BRAM:|CP achieved post-implementation|Timing"
```

If 100 MHz completes with `WNS < 0`, do not change this 100 MHz run. Make a
separate 90 MHz copy and change only the HLS clock line to `clock=11.111ns`:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project
Copy-Item -Recurse n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_100mhz n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_90mhz
(Get-Content n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_90mhz\hls_component\hls_config.cfg) -replace "clock=10ns","clock=11.111ns" | Set-Content n2ew_era\06l_phc_lite_pipeline_budget_sweep\pnr_unroll16_90mhz\hls_component\hls_config.cfg
```

Then rerun the same implementation command with `subst P:` pointing at
`pnr_unroll16_90mhz`.

## Honest Conclusion

The 06L unroll-16 design is verified at csynth and RTL cosim level in the
independent workspace: latency is 2316 cycles / 23.160 us at 100 MHz, and
cosim reports `total=25600 mismatches=0`.

The requested routed 100 MHz timing evidence is still open. The run did not
fail timing. The first Codex-run attempt failed in a Vivado run-launch
permission error; the later user-run attempt packaged the IP successfully, then
failed when Vivado could not remove a locked `project.runs` directory. Until a
clean user-run P&R produces routed WNS/TNS and post-route utilization reports,
the 23.16 us number must be described as csynth/cosim latency, not as
post-route-equivalent evidence.
