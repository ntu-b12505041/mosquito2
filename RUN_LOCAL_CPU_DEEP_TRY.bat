@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_LOCAL_CPU_DEEP_TRY.ps1"
echo.
pause
