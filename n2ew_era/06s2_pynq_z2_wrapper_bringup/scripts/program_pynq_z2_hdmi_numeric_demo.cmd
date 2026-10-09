@echo off
setlocal

set "PROJECT_ROOT=C:\Users\holbo\00VS_projjects\00_HFT_project"
set "VIVADO_BAT=C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
set "PROGRAM_TCL=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\program_pynq_z2_hdmi_demo.tcl"
set "PHC_06S2_PROGRAM_BIT_FILE=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper_hdmi_numeric.bit"
set "TIMING_RPT=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper_hdmi_numeric_timing_summary_routed.rpt"

if not exist "%VIVADO_BAT%" (
  echo ERROR: Vivado not found: "%VIVADO_BAT%"
  exit /b 1
)

if not exist "%PROGRAM_TCL%" (
  echo ERROR: Tcl script not found: "%PROGRAM_TCL%"
  exit /b 1
)

if not exist "%PHC_06S2_PROGRAM_BIT_FILE%" (
  echo ERROR: Bitstream not found: "%PHC_06S2_PROGRAM_BIT_FILE%"
  exit /b 1
)

if not exist "%TIMING_RPT%" (
  echo ERROR: Timing report not found: "%TIMING_RPT%"
  exit /b 1
)

findstr /C:"All user specified timing constraints are met." "%TIMING_RPT%" >nul
if errorlevel 1 (
  echo ERROR: Numeric HDMI timing report does not show timing met.
  echo        Refusing to program "%PHC_06S2_PROGRAM_BIT_FILE%".
  echo        Use scripts\program_pynq_z2_hdmi_demo.cmd for the verified solid-color HDMI fallback.
  exit /b 1
)

cd /d "%PROJECT_ROOT%"
"%VIVADO_BAT%" -mode batch -source "%PROGRAM_TCL%"
