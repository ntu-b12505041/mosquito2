@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_DEEP_CPU_EXPERIMENT.ps1" -RunName "03_all12_inception1d_raw_zscore" -LeadMode all12 -Preprocess raw_zscore -Model inception1d
echo.
pause
