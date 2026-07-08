@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_DEEP_CPU_EXPERIMENT.ps1" -RunName "04_all12_inception1d_bandpass" -LeadMode all12 -Preprocess bandpass_0.5_40_zscore -Model inception1d
echo.
pause
