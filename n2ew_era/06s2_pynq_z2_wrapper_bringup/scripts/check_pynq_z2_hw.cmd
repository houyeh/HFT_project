@echo off
setlocal

set "PROJECT_ROOT=C:\Users\holbo\00VS_projjects\00_HFT_project"
set "VIVADO_BAT=C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
set "CHECK_TCL=%PROJECT_ROOT%\n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\check_pynq_z2_hw.tcl"

if not exist "%VIVADO_BAT%" (
  echo ERROR: Vivado not found: "%VIVADO_BAT%"
  exit /b 1
)

if not exist "%CHECK_TCL%" (
  echo ERROR: Tcl script not found: "%CHECK_TCL%"
  exit /b 1
)

cd /d "%PROJECT_ROOT%"
"%VIVADO_BAT%" -mode batch -source "%CHECK_TCL%"
