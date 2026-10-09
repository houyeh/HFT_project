@echo off
setlocal

set "PROJECT_ROOT=C:\Users\holbo\00VS_projjects\00_HFT_project"
set "VIVADO_BAT=C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
set "PROGRAM_TCL=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\program_pynq_z2_hdmi_demo.tcl"
set "BIT_FILE=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper_hdmi.bit"

if not exist "%VIVADO_BAT%" (
  echo ERROR: Vivado not found: "%VIVADO_BAT%"
  exit /b 1
)

if not exist "%PROGRAM_TCL%" (
  echo ERROR: Tcl script not found: "%PROGRAM_TCL%"
  exit /b 1
)

if not exist "%BIT_FILE%" (
  echo ERROR: Bitstream not found: "%BIT_FILE%"
  exit /b 1
)

cd /d "%PROJECT_ROOT%"
"%VIVADO_BAT%" -mode batch -source "%PROGRAM_TCL%"
