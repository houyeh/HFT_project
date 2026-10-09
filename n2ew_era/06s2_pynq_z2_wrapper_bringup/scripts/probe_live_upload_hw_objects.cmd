@echo off
setlocal

for %%I in ("%~dp0..\..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat" -mode batch -source "n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\probe_live_upload_hw_objects.tcl"

endlocal
