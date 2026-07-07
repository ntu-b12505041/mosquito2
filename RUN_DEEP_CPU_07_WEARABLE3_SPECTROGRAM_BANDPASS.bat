@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_DEEP_CPU_EXPERIMENT.ps1" -RunName "07_wearable3_spectrogram2d_bandpass" -LeadMode wearable3_vdiff -Preprocess bandpass_0.5_40_zscore -Model spectrogram2d
echo.
pause
