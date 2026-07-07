@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_DEEP_CPU_EXPERIMENT.ps1" -RunName "02_all12_resnet1d_bandpass" -LeadMode all12 -Preprocess bandpass_0.5_40_zscore -Model resnet1d
echo.
pause
