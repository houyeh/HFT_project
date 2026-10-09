@echo off
setlocal

for %%I in ("%~dp0..\..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

set "TORCH_PY=C:\Users\holbo\00VS_projjects\02_Michael_score_extractor\.conda-ai\python.exe"

if exist "%TORCH_PY%" (
    "%TORCH_PY%" n2ew_era\06s2_pynq_z2_wrapper_bringup\03_cpu_live_replay_demo.py %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python n2ew_era\06s2_pynq_z2_wrapper_bringup\03_cpu_live_replay_demo.py %*
    ) else (
        py -3 n2ew_era\06s2_pynq_z2_wrapper_bringup\03_cpu_live_replay_demo.py %*
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CPU live replay demo failed. Check that the selected Python environment has torch and numpy installed.
    exit /b %ERRORLEVEL%
) else (
    exit /b 0
)
