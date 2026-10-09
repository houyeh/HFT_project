@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo  Spatial front-end scaling sweep  (PHC, hidden=16 fixed)
echo    C=8   done already (2,420 params, wF1 67.58 +/- 1.00)
echo    C=32  new, ~8,276 params   -- brackets TCN scratch 8,547
echo    C=64  new, ~23,252 params  -- PARAM-MATCHED vs hidden64 23,876
echo  Config: 2 concurrent processes x 2 torch threads = 4 cores
echo ============================================================
echo.
echo  *** BEFORE YOU START ***
echo  The training script aborts if it detects OneDrive / Anaconda
echo  background load. QUIT ONEDRIVE FIRST:
echo    right-click the OneDrive cloud icon in the tray -^> Quit OneDrive
echo  Verify with Task Manager that OneDrive.Sync.Service.exe is gone.
echo.
pause

echo [1/3] Dry run (writes nothing)
python run_spatial_scaling.py --channels 32 --seed 0 --dry-run
if errorlevel 1 goto fail
python run_spatial_scaling.py --channels 64 --seed 0 --dry-run
if errorlevel 1 goto fail
echo.
pause

echo.
echo [2/3] Running 6 jobs as 3 waves of 2 concurrent processes
for %%P in (32 64) do (
  echo   --- C=%%P : seeds 0 and 1 concurrently ---
  start "sp-c%%P-s0" cmd /c "python run_spatial_scaling.py --channels %%P --seed 0 > spatial_c%%P_seed0_console.log 2>&1"
  start "sp-c%%P-s1" /wait cmd /c "python run_spatial_scaling.py --channels %%P --seed 1 > spatial_c%%P_seed1_console.log 2>&1"
  echo   --- C=%%P : seed 2 ---
  start "sp-c%%P-s2" /wait cmd /c "python run_spatial_scaling.py --channels %%P --seed 2 > spatial_c%%P_seed2_console.log 2>&1"
)

echo.
echo [3/3] All launched/finished. Check for aborts:
findstr /C:"RuntimeError" spatial_c*_console.log
if not errorlevel 1 echo   ^<-- ABORTS FOUND ABOVE. Nothing usable from those runs.
echo.
echo Verify results:  python check_spatial_scaling.py
echo.
pause
goto end

:fail
echo Dry run FAILED. Nothing started.
pause
exit /b 1

:end
endlocal
