$ErrorActionPreference = "Stop"

# Change this placeholder to your local PTB-XL folder.
# Accepted:
#   1. PTB-XL root folder containing ptbxl_database.csv, scp_statements.csv, records100/
#   2. The records100 folder itself
#
# Example:
#   $PTBXL_DATA_DIR = "D:\datasets\ptb-xl\1.0.3"
#   $PTBXL_DATA_DIR = "D:\datasets\ptb-xl\1.0.3\records100"
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"

if ($PTBXL_DATA_DIR -like "*<<<*") {
    throw "Please edit RUN_LOCAL_CPU.ps1 first and replace <<<PTBXL_DATA_DIR>>> with your local PTB-XL path."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$ResolvedDataDir = (Resolve-Path -LiteralPath $PTBXL_DATA_DIR).Path
if ((Split-Path -Leaf $ResolvedDataDir).ToLower() -eq "records100") {
    $ResolvedDataDir = Split-Path -Parent $ResolvedDataDir
}

Write-Host "Project dir: $ScriptDir"
Write-Host "PTB-XL data dir: $ResolvedDataDir"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating local CPU virtual environment..."
    python -m venv .venv
}

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

Write-Host "Installing CPU requirements..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements_cpu.txt

Write-Host "Checking local PTB-XL records100..."
& $Python scripts\check_local_ptbxl_data.py --data-dir "$ResolvedDataDir"

Write-Host "Running full CPU classical experiments: all12..."
& $Python scripts\run_classical_experiments.py `
    --data-dir "$ResolvedDataDir" `
    --no-download `
    --lead-mode all12 `
    --preprocess raw_zscore bandpass_0.5_40_zscore bandpass_0.5_40_zscore_downsample50 `
    --models logistic_regression random_forest hist_gradient_boosting

Write-Host "Running full CPU classical experiments: wearable3_vdiff..."
& $Python scripts\run_classical_experiments.py `
    --data-dir "$ResolvedDataDir" `
    --no-download `
    --lead-mode wearable3_vdiff `
    --preprocess raw_zscore bandpass_0.5_40_zscore `
    --models logistic_regression random_forest hist_gradient_boosting

Write-Host "Generating report.md..."
& $Python scripts\make_report.py

Write-Host ""
Write-Host "Done."
Write-Host "Metrics:"
Write-Host "  reports\metrics_classical.csv"
Write-Host "  reports\metrics_classical.json"
Write-Host "Report:"
Write-Host "  report.md"
