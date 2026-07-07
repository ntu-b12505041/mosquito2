@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_DEEP_CPU_EXPERIMENT.ps1" -RunName "01_all12_resnet1d_raw_zscore" -LeadMode all12 -Preprocess raw_zscore -Model resnet1d
echo.
pause
