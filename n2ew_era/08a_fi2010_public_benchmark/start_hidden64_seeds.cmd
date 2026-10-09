@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  hidden64 capacity study - launch seed 1 and seed 2
echo  Config: 2 processes x 2 torch threads = 4 cores total
echo  Outputs isolated: results_seed1 / results_seed2
echo  Existing results\ and models\ are NOT touched
echo ============================================================
echo.

echo [1/4] Dry run - verifying path isolation (writes nothing)
echo.
python run_hidden64_seed.py --seed 1 --dry-run
if errorlevel 1 goto fail
python run_hidden64_seed.py --seed 2 --dry-run
if errorlevel 1 goto fail

echo.
echo CHECK: resolved_RESULTS above must end with results_seed1 / results_seed2.
echo If not, press Ctrl+C now.
echo.
pause

echo.
echo [2/4] Starting two training processes
if exist results_seed1_console.log del results_seed1_console.log
if exist results_seed2_console.log del results_seed2_console.log
start "h64-seed1" cmd /c "python run_hidden64_seed.py --seed 1 > results_seed1_console.log 2>&1"
start "h64-seed2" cmd /c "python run_hidden64_seed.py --seed 2 > results_seed2_console.log 2>&1"

echo.
echo [3/4] Waiting 30s, then verifying both processes survived startup...
timeout /t 30 /nobreak >nul

set STARTFAIL=0
findstr /C:"FATAL" results_seed1_console.log >nul 2>&1 && set STARTFAIL=1
findstr /C:"FATAL" results_seed2_console.log >nul 2>&1 && set STARTFAIL=1
findstr /C:"Traceback" results_seed1_console.log >nul 2>&1 && set STARTFAIL=1
findstr /C:"Traceback" results_seed2_console.log >nul 2>&1 && set STARTFAIL=1
if "%STARTFAIL%"=="1" goto startdied

if not exist results_seed1\runner_banner.json goto startdied
if not exist results_seed2\runner_banner.json goto startdied

echo.
echo [4/4] VERIFIED: both processes are running.
echo.
echo   - Keep the laptop plugged in. Set power plan to NEVER sleep.
echo   - Keep both windows open. Closing a window kills that run.
echo   - Check progress:  python check_hidden64_progress.py
echo.
echo First epoch takes roughly 8-10 minutes. Check back in ~20 minutes.
echo Estimated total runtime: 8-20 hours.
echo.
pause
goto end

:startdied
echo.
echo ************************************************************
echo  STARTUP FAILED - the training processes did NOT survive.
echo  Console output below:
echo ************************************************************
echo.
echo --- seed 1 ---
type results_seed1_console.log 2>nul
echo.
echo --- seed 2 ---
type results_seed2_console.log 2>nul
echo.
echo Nothing is training. Fix the error above before retrying.
echo.
pause
exit /b 1

:fail
echo.
echo Dry run FAILED. Nothing was started.
echo.
pause
exit /b 1

:end
endlocal
