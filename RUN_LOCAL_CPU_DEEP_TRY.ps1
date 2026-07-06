$ErrorActionPreference = "Stop"

# Change this placeholder to your local PTB-XL folder.
# Accepted:
#   1. PTB-XL root folder containing ptbxl_database.csv, scp_statements.csv, records100/
#   2. The records100 folder itself
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"

# CPU deep-learning trial settings.
# Keep this small first. Set $MAX_RECORDS = 0 to use the full PTB-XL dataset.
$MAX_RECORDS = 1000
$EPOCHS = 2
$BATCH_SIZE = 32

if ($PTBXL_DATA_DIR -like "*<<<*") {
    throw "Please edit RUN_LOCAL_CPU_DEEP_TRY.ps1 first and replace <<<PTBXL_DATA_DIR>>> with your local PTB-XL path."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$ResolvedDataDir = (Resolve-Path -LiteralPath $PTBXL_DATA_DIR).Path
if ((Split-Path -Leaf $ResolvedDataDir).ToLower() -eq "records100") {
    $ResolvedDataDir = Split-Path -Parent $ResolvedDataDir
}

Write-Host "Project dir: $ScriptDir"
Write-Host "PTB-XL data dir: $ResolvedDataDir"
Write-Host "CPU deep trial max records: $MAX_RECORDS"
Write-Host "Epochs: $EPOCHS"
Write-Host "Batch size: $BATCH_SIZE"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment..."
    python -m venv .venv
}

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

Write-Host "Installing CPU requirements..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements_cpu.txt

Write-Host "Installing CPU PyTorch. This can take several minutes the first time..."
& $Python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Checking local PTB-XL records100..."
& $Python scripts\check_local_ptbxl_data.py --data-dir "$ResolvedDataDir"

Write-Host "Running CPU deep-learning trial..."
& $Python scripts\run_deep_experiments.py `
    --data-dir "$ResolvedDataDir" `
    --no-download `
    --device cpu `
    --max-records $MAX_RECORDS `
    --lead-mode wearable3_vdiff `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --preprocess bandpass_0.5_40_zscore `
    --models resnet1d

Write-Host "Generating report.md..."
& $Python scripts\make_report.py

Write-Host ""
Write-Host "Done."
Write-Host "Deep metrics:"
Write-Host "  reports\metrics_deep.csv"
Write-Host "  reports\metrics_deep.json"
Write-Host "Checkpoint:"
Write-Host "  reports\wearable3_vdiff_resnet1d_bandpass_0.5_40_zscore.pt"
Write-Host "Report:"
Write-Host "  report.md"
