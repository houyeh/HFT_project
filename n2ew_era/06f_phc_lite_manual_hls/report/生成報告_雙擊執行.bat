@echo off
echo ============================================
echo  PHC-lite HLS Report 生成器
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] 安裝 python-docx...
pip install python-docx -q
if errorlevel 1 (
    echo    嘗試使用 pip3...
    pip3 install python-docx -q
)

echo [2/2] 生成 Word 報告...
python generate_report_py.py
if errorlevel 1 (
    echo    嘗試使用 python3...
    python3 generate_report_py.py
)

echo.
echo ============================================
if exist PHC_lite_HLS_Report.docx (
    echo  成功！報告已生成：PHC_lite_HLS_Report.docx
    echo  雙擊開啟即可閱讀
    start PHC_lite_HLS_Report.docx
) else (
    echo  生成失敗，請手動執行：
    echo    pip install python-docx
    echo    python generate_report_py.py
)
echo ============================================
pause
